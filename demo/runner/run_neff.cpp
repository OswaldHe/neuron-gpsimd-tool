// run_neff - run a trn3 NEFF with libnrt, validate the result bit-exactly, and prove that the
// GPSIMD custom operator ran on the GPSIMD cores.
//
// usage: run_neff <file.neff> [--expect identity|add:N] [--iters N] [--seed S] [--device D]
//                 [--vnc V] [--no-map] [--logs]
//
//   --expect identity   output must equal the input bit-for-bit (control: unmodified NKI NEFF)
//   --expect add:N      output must equal input + N (int32, wrap-around) bit-for-bit, and every
//                       GPSIMD core must have left a valid execution receipt (see sbuf_add.cpp)
//   --no-map            do not map SBUF into the GPSIMD cores' address space
//   --logs              print the GPSIMD cores' kprintf logs
//
// Steps: nrt_init, nrt_load; map SBUF (GPSIMD address-translation entry 38 -> SBUF, via PCIe BAR0);
// fill the input with seeded pseudo-random int32 values; nrt_execute (N times); read the output and
// compare with the host reference with memcmp; read the GPSIMD receipts/logs over BAR0; unmap.
// Exit status: 0 = all checks passed, 1 = a check failed, 2 = usage/runtime error.

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
}

namespace {

// ------------------------------------------------------------------ device access over PCIe BAR0
// aws-neuronx driver ioctls (neuron_ioctl.h)
struct DeviceInfo {
    uint32_t architecture, revision;
    uint32_t connected_device_count;
    uint32_t connected_devices[8];
    uint64_t bar_address[2];
    uint64_t bar_size[2];
};
struct BarRw {
    uint32_t bar;
    uint64_t *address;
    uint32_t *data;
    uint32_t count;
};
const unsigned long IOCTL_DEVICE_INFO = _IOR('N', 3, void *);
const unsigned long IOCTL_BAR_READ = _IOR('N', 11, void *);
const unsigned long IOCTL_BAR_WRITE = _IOW('N', 12, void *);
const uint32_t BAR_CHUNK = 100;   // FW_IO_MAX_READLESS_READ_REGISTER_COUNT

// trn3 (v4) address map
const uint64_t TPB_BAR0_OFFSET = 0xD0000000ull, TPB_BAR0_DIST = 0x4000000ull;   // 64 MB per TPB
const uint64_t SBUF_SOC_BASE[8] = {0x2000000000ull,   0x3000000000ull,   0x6000000000ull,   0x7000000000ull,
                                   0x802000000000ull, 0x803000000000ull, 0x806000000000ull, 0x807000000000ull};
const uint64_t SBUF_BYTES = 128ull * 0x40000;        // 128 partitions x 256 KB stride
// GPSIMD cores: core k of a TPB sees its local address space at TPB + 0x3100000 + k * 0x100000
const uint64_t GPSIMD_BASE = 0x3100000, GPSIMD_STRIDE = 0x100000;
const uint32_t GPSIMD_CORES = 8, DRAM_LO = 0x80000, DRAM_HI = 0xC0000, IRAM_HI = 0x20000;
const uint32_t FW_CTX = 0x80878;                     // firmware context table (word0 = 0x10, word1 = self)
// GPSIMD address-translation block: 40 entries of {ctrl, mask_lo, mask_hi, local_lo, local_hi, soc_lo, soc_hi}
const uint64_t XLAT_OFF = 0x3062000;
const uint32_t XLAT_ENTRY = 38, SBUF_WINDOW = 0xD0000000u;

class Bar0 {
   public:
    explicit Bar0(int dev) {
        std::string path = "/dev/neuron" + std::to_string(dev);
        fd_ = open(path.c_str(), O_RDWR);
        if (fd_ < 0) throw std::runtime_error("cannot open " + path);
        DeviceInfo info{};
        if (ioctl(fd_, IOCTL_DEVICE_INFO, &info)) throw std::runtime_error("NEURON_IOCTL_DEVICE_INFO failed");
        pa_ = info.bar_address[0];
    }
    ~Bar0() { if (fd_ >= 0) close(fd_); }
    std::vector<uint32_t> read(uint64_t off, uint32_t nwords) const {
        std::vector<uint32_t> out(nwords);
        std::vector<uint64_t> addr(BAR_CHUNK);
        for (uint32_t base = 0; base < nwords; base += BAR_CHUNK) {
            uint32_t n = std::min(BAR_CHUNK, nwords - base);
            for (uint32_t k = 0; k < n; ++k) addr[k] = pa_ + off + 4ull * (base + k);
            BarRw rw{0, addr.data(), out.data() + base, n};
            if (ioctl(fd_, IOCTL_BAR_READ, &rw)) throw std::runtime_error("BAR0 read failed");
        }
        return out;
    }
    void write(uint64_t off, const std::vector<uint32_t> &w) const {
        std::vector<uint64_t> addr(w.size());
        std::vector<uint32_t> data(w);
        for (size_t k = 0; k < w.size(); ++k) addr[k] = pa_ + off + 4ull * k;
        BarRw rw{0, addr.data(), data.data(), static_cast<uint32_t>(w.size())};
        if (ioctl(fd_, IOCTL_BAR_WRITE, &rw)) throw std::runtime_error("BAR0 write failed");
    }
    static uint64_t tpb(int t, uint64_t off) { return TPB_BAR0_OFFSET + t * TPB_BAR0_DIST + off; }
    static uint64_t gpsimd(int t, int core, uint32_t local) { return tpb(t, GPSIMD_BASE + core * GPSIMD_STRIDE + local); }

   private:
    int fd_ = -1;
    uint64_t pa_ = 0;
};

bool firmware_up(const Bar0 &b, int t) {
    auto w = b.read(Bar0::gpsimd(t, 0, FW_CTX), 2);
    return w[0] == 0x10 && w[1] == FW_CTX;
}

// Map SBUF of TPB t at core address SBUF_WINDOW. Returns false if the entry is used by someone else.
bool map_sbuf(const Bar0 &b, int t) {
    const uint64_t off = Bar0::tpb(t, XLAT_OFF + 28ull * XLAT_ENTRY);
    auto cur = b.read(off, 7);
    uint64_t soc = cur[5] | (uint64_t)cur[6] << 32;
    if ((cur[0] & 1) && soc != SBUF_SOC_BASE[t]) return false;
    uint64_t mask = (~(SBUF_BYTES - 1)) & 0xFFFFFFFFFFull;
    b.write(off, {0});
    b.write(off + 4, {(uint32_t)mask, (uint32_t)(mask >> 32), SBUF_WINDOW, 0, (uint32_t)SBUF_SOC_BASE[t],
                      (uint32_t)(SBUF_SOC_BASE[t] >> 32)});
    b.write(off, {1});
    return true;
}
void unmap_sbuf(const Bar0 &b, int t) { b.write(Bar0::tpb(t, XLAT_OFF + 28ull * XLAT_ENTRY), {0}); }

// ------------------------------------------------------------------ GPSIMD receipts and logs
struct Receipt {   // must match gpsimd/sbuf_add.cpp
    uint32_t magic0, magic1, prid, executions, first_partition, end_partition, elems_per_partition;
    uint32_t input_checksum, output_checksum, ccount_start, ccount_end, kernel_pc, stack_pointer;
    uint32_t return_address, addend, reserved;
};
static_assert(sizeof(Receipt) == 64, "receipt layout");
const uint32_t RECEIPT_MAGIC0 = 0x4D495350u, RECEIPT_MAGIC1 = 0x43524444u;
const uint32_t KLOG_MAGIC0 = 0x4b4c4f47u, KLOG_MAGIC1 = 0x31534d47u;

struct CoreFindings {
    bool has_receipt = false;
    Receipt receipt{};
    std::string log;
};

// Scan core 0's data RAM for the receipt / log magics; the loadable library is placed at the same
// data-RAM address on all 8 cores, so the other cores are read at the same offsets.
std::vector<CoreFindings> read_gpsimd(const Bar0 &b, int t) {
    std::vector<CoreFindings> out(GPSIMD_CORES);
    auto dram0 = b.read(Bar0::gpsimd(t, 0, DRAM_LO), (DRAM_HI - DRAM_LO) / 4);
    int64_t rcpt = -1, klog = -1;
    for (size_t i = 0; i + 1 < dram0.size(); ++i) {
        if (dram0[i] == RECEIPT_MAGIC0 && dram0[i + 1] == RECEIPT_MAGIC1 && rcpt < 0) rcpt = i;
        if (dram0[i] == KLOG_MAGIC0 && dram0[i + 1] == KLOG_MAGIC1 && klog < 0) klog = i;
    }
    for (uint32_t c = 0; c < GPSIMD_CORES; ++c) {
        if (rcpt >= 0) {
            auto w = b.read(Bar0::gpsimd(t, c, DRAM_LO + 4 * rcpt), 16);
            Receipt r;
            memcpy(&r, w.data(), sizeof r);
            if (r.magic0 == RECEIPT_MAGIC0 && r.magic1 == RECEIPT_MAGIC1) { out[c].has_receipt = true; out[c].receipt = r; }
        }
        if (klog >= 0) {
            auto h = b.read(Bar0::gpsimd(t, c, DRAM_LO + 4 * klog), 8);
            if (h[0] == KLOG_MAGIC0 && h[1] == KLOG_MAGIC1 && h[2] > 0 && h[2] <= 0x10000) {
                uint32_t cap = h[2], pos = h[3];
                auto tw = b.read(Bar0::gpsimd(t, c, DRAM_LO + 4 * klog + 32), (cap + 3) / 4);
                std::string text(reinterpret_cast<const char *>(tw.data()), cap);
                out[c].log = pos <= cap ? text.substr(0, pos) : text.substr(pos % cap) + text.substr(0, pos % cap);
            }
        }
    }
    return out;
}

void clear_gpsimd(const Bar0 &b, int t) {   // invalidate stale receipts/logs from earlier runs
    auto dram0 = b.read(Bar0::gpsimd(t, 0, DRAM_LO), (DRAM_HI - DRAM_LO) / 4);
    for (size_t i = 0; i + 1 < dram0.size(); ++i)
        if ((dram0[i] == RECEIPT_MAGIC0 && dram0[i + 1] == RECEIPT_MAGIC1) || (dram0[i] == KLOG_MAGIC0 && dram0[i + 1] == KLOG_MAGIC1))
            for (uint32_t c = 0; c < GPSIMD_CORES; ++c) b.write(Bar0::gpsimd(t, c, DRAM_LO + 4 * i), {0, 0});
}

// ------------------------------------------------------------------ helpers
void check(NRT_STATUS s, const char *what) {
    if (s != NRT_SUCCESS) throw std::runtime_error(std::string(what) + " failed: NRT_STATUS=" + std::to_string(s));
}
uint32_t xorshift32(uint32_t &s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }

struct Options {
    std::string neff, expect = "identity";
    int iters = 1, device = 0, vnc = 0;
    uint32_t seed = 12345;
    bool map = true, logs = false;
};

Options parse(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--expect") o.expect = val();
        else if (a == "--iters") o.iters = std::stoi(val());
        else if (a == "--seed") o.seed = std::stoul(val());
        else if (a == "--device") o.device = std::stoi(val());
        else if (a == "--vnc") o.vnc = std::stoi(val());
        else if (a == "--no-map") o.map = false;
        else if (a == "--logs") o.logs = true;
        else if (a[0] != '-' && o.neff.empty()) o.neff = a;
        else throw std::runtime_error("unknown argument " + a);
    }
    if (o.neff.empty()) throw std::runtime_error("usage: run_neff <file.neff> [--expect identity|add:N] [--iters N] "
                                                 "[--seed S] [--device D] [--vnc V] [--no-map] [--logs]");
    return o;
}

}  // namespace

int main(int argc, char **argv) {
    Options opt;
    try { opt = parse(argc, argv); } catch (const std::exception &e) { fprintf(stderr, "%s\n", e.what()); return 2; }
    const bool add_mode = opt.expect.rfind("add:", 0) == 0;
    const int64_t addend_arg = add_mode ? std::stoll(opt.expect.substr(4)) : 0;
    const uint32_t addend = static_cast<uint32_t>(addend_arg);
    int failures = 0;
    auto fail = [&](const std::string &m) { printf("  FAIL: %s\n", m.c_str()); ++failures; };

    try {
        std::ifstream f(opt.neff, std::ios::binary);
        if (!f) throw std::runtime_error("cannot read " + opt.neff);
        std::vector<char> neff((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

        check(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""), "nrt_init");
        nrt_model_t *model = nullptr;
        check(nrt_load(neff.data(), neff.size(), opt.vnc, 1, &model), "nrt_load");
        printf("[load] %s (%zu bytes) on vnc %d\n", opt.neff.c_str(), neff.size(), opt.vnc);

        Bar0 bar(opt.device);
        std::vector<int> mapped;
        for (int t = 0; t < 8; ++t) {
            if (!firmware_up(bar, t)) continue;
            clear_gpsimd(bar, t);
            if (opt.map) {
                if (map_sbuf(bar, t)) mapped.push_back(t);
                else printf("[map] TPB%d: translation entry %u busy, not mapped\n", t, XLAT_ENTRY);
            }
        }
        if (opt.map) {
            printf("[map] SBUF mapped at GPSIMD core address 0x%08X (translation entry %u) on TPB", SBUF_WINDOW, XLAT_ENTRY);
            for (int t : mapped) printf(" %d", t);
            printf("\n");
        }

        // tensors
        nrt_tensor_info_array_t *info = nullptr;
        check(nrt_get_model_tensor_info(model, &info), "nrt_get_model_tensor_info");
        nrt_tensor_set_t *ins = nullptr, *outs = nullptr;
        check(nrt_allocate_tensor_set(&ins), "nrt_allocate_tensor_set");
        check(nrt_allocate_tensor_set(&outs), "nrt_allocate_tensor_set");
        std::vector<uint32_t> input;
        std::vector<uint32_t> rows_cols(2, 0);
        struct Out { nrt_tensor_t *t; size_t size; std::string name; };
        std::vector<Out> out_tensors;
        std::vector<nrt_tensor_t *> all;
        for (uint64_t i = 0; i < info->tensor_count; ++i) {
            const nrt_tensor_info_t &ti = info->tensor_array[i];
            if (ti.dtype != NRT_DTYPE_INT32) throw std::runtime_error(std::string("tensor ") + ti.name + " is not int32");
            nrt_tensor_t *t = nullptr;
            check(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, opt.vnc, ti.size, ti.name, &t), "nrt_tensor_allocate");
            all.push_back(t);
            if (ti.usage == NRT_TENSOR_USAGE_INPUT) {
                input.resize(ti.size / 4);
                uint32_t s = opt.seed;
                for (auto &v : input) v = xorshift32(s);   // full-range int32 values (wrap-around exercised)
                if (ti.ndim == 2) { rows_cols[0] = ti.shape[0]; rows_cols[1] = ti.shape[1]; }
                check(nrt_tensor_write(t, input.data(), 0, ti.size), "nrt_tensor_write");
                check(nrt_add_tensor_to_tensor_set(ins, ti.name, t), "nrt_add_tensor_to_tensor_set");
            } else {
                out_tensors.push_back({t, ti.size, ti.name});
                check(nrt_add_tensor_to_tensor_set(outs, ti.name, t), "nrt_add_tensor_to_tensor_set");
            }
            printf("[tensor] %-9s %-6s int32 [%u x %u] %zu bytes\n", ti.name, ti.usage == NRT_TENSOR_USAGE_INPUT ? "input" : "output",
                   ti.ndim > 0 ? ti.shape[0] : 0, ti.ndim > 1 ? ti.shape[1] : 1, ti.size);
        }
        if (input.empty() || out_tensors.size() != 1) throw std::runtime_error("expected one int32 input and one output");

        std::vector<uint32_t> expected(input);
        for (auto &v : expected) v += addend;   // uint32 wrap-around == int32 two's-complement add

        // execute + bit-exact validation
        printf("\n== bit-exact validation (%s, %d execution%s, seed %u)\n", opt.expect.c_str(), opt.iters, opt.iters > 1 ? "s" : "", opt.seed);
        std::vector<uint32_t> output(input.size());
        for (int it = 0; it < opt.iters; ++it) {
            auto t0 = std::chrono::steady_clock::now();
            check(nrt_execute(model, ins, outs), "nrt_execute");
            double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            std::fill(output.begin(), output.end(), 0xA5A5A5A5u);
            check(nrt_tensor_read(out_tensors[0].t, output.data(), 0, out_tensors[0].size), "nrt_tensor_read");
            size_t bad = 0, first_bad = 0;
            for (size_t k = 0; k < output.size(); ++k)
                if (output[k] != expected[k] && bad++ == 0) first_bad = k;
            bool same = memcmp(output.data(), expected.data(), output.size() * 4) == 0;
            printf("  execution %d: %.0f us, %zu/%zu int32 elements bit-exact%s\n", it + 1, us, output.size() - bad, output.size(),
                   same ? "" : " <-- MISMATCH");
            if (!same) {
                fail("output differs from reference");
                printf("    first mismatch at [%zu][%zu]: got %d expected %d (input %d)\n", first_bad / rows_cols[1],
                       first_bad % rows_cols[1], (int32_t)output[first_bad], (int32_t)expected[first_bad], (int32_t)input[first_bad]);
            }
        }
        printf("  sample: input[0][0..3] = %d %d %d %d\n", (int32_t)input[0], (int32_t)input[1], (int32_t)input[2], (int32_t)input[3]);
        printf("          output[0][0..3] = %d %d %d %d\n", (int32_t)output[0], (int32_t)output[1], (int32_t)output[2], (int32_t)output[3]);

        // GPSIMD execution evidence
        std::map<int, std::vector<CoreFindings>> found;
        for (int t = 0; t < 8; ++t) {
            if (!firmware_up(bar, t)) continue;
            auto f2 = read_gpsimd(bar, t);
            bool any = false;
            for (auto &c : f2) any |= c.has_receipt || !c.log.empty();
            if (any) found[t] = f2;
        }
        if (add_mode) {
            printf("\n== GPSIMD execution evidence (receipts read from each GPSIMD core's data RAM over PCIe BAR0)\n");
            if (found.size() != 1) fail("expected receipts on exactly one TPB, found " + std::to_string(found.size()));
            for (auto &[t, cores] : found) {
                printf("  TPB%d  core prid  execs  partitions  input-sum   output-sum  cycles   kernel-pc   stack-ptr   called-from\n", t);
                std::set<uint32_t> prids;
                std::vector<int> covered(rows_cols[0], 0);
                const uint32_t cols = rows_cols[1];
                for (uint32_t c = 0; c < GPSIMD_CORES; ++c) {
                    if (!cores[c].has_receipt) { fail("GPSIMD core " + std::to_string(c) + " left no receipt"); continue; }
                    const Receipt &r = cores[c].receipt;
                    uint32_t in_sum = 0, out_sum = 0;
                    for (uint32_t p = r.first_partition; p < r.end_partition && p < rows_cols[0]; ++p)
                        for (uint32_t i = 0; i < cols; ++i) { in_sum += input[p * cols + i]; out_sum += expected[p * cols + i]; covered[p]++; }
                    printf("        %4u %4u  %5u  %3u..%-3u   0x%08x  0x%08x  %7u  0x%08x  0x%08x  0x%08x\n", c, r.prid, r.executions,
                           r.first_partition, r.end_partition - 1, r.input_checksum, r.output_checksum, r.ccount_end - r.ccount_start,
                           r.kernel_pc, r.stack_pointer, r.return_address);
                    prids.insert(r.prid);
                    if (r.prid != c) fail("receipt in core " + std::to_string(c) + "'s memory has PRID " + std::to_string(r.prid));
                    if (r.executions != (uint32_t)opt.iters) fail("core " + std::to_string(c) + " executed " + std::to_string(r.executions) + " times");
                    if (r.input_checksum != in_sum) fail("core " + std::to_string(c) + " input checksum mismatch (did not read the input tile)");
                    if (r.output_checksum != out_sum) fail("core " + std::to_string(c) + " output checksum mismatch");
                    if (r.elems_per_partition != cols) fail("core " + std::to_string(c) + " element count mismatch");
                    if (r.addend != addend) fail("core " + std::to_string(c) + " used a different addend");
                    if (r.kernel_pc >= IRAM_HI) fail("kernel code address not in GPSIMD instruction RAM");
                    if (r.stack_pointer < DRAM_LO || r.stack_pointer >= DRAM_HI) fail("stack not in GPSIMD data RAM");
                    if (r.ccount_end == r.ccount_start) fail("core cycle counter did not advance");
                }
                for (uint32_t p = 0; p < rows_cols[0]; ++p)
                    if (covered[p] != (int)cols) { fail("partition " + std::to_string(p) + " not processed exactly once"); break; }
                if (prids.size() == GPSIMD_CORES)
                    printf("  -> %u distinct GPSIMD cores (PRID 0..%u) each read and rewrote their own SBUF partitions;\n"
                           "     the checksums of the values they read match the host's random input.\n", GPSIMD_CORES, GPSIMD_CORES - 1);
            }
        } else if (!found.empty()) {
            for (auto &[t, cores] : found)
                for (auto &c : cores)
                    if (c.has_receipt) { fail("control run: unexpected GPSIMD receipt on TPB" + std::to_string(t)); break; }
        } else {
            printf("\n== control run: no GPSIMD custom-operator receipts (as expected for the unmodified NEFF)\n");
        }
        if (opt.logs && !found.empty()) {
            printf("\n== GPSIMD kprintf logs (read from each core's data RAM)\n");
            for (auto &[t, cores] : found)
                for (uint32_t c = 0; c < GPSIMD_CORES; ++c) {
                    size_t pos = 0;
                    const std::string &log = cores[c].log;
                    while (pos < log.size()) {
                        size_t nl = log.find('\n', pos);
                        if (nl == std::string::npos) nl = log.size();
                        printf("  [TPB%d core %u] %s\n", t, c, log.substr(pos, nl - pos).c_str());
                        pos = nl + 1;
                    }
                }
        }

        for (int t : mapped) unmap_sbuf(bar, t);
        for (auto *t : all) { nrt_tensor_t *p = t; nrt_tensor_free(&p); }
        nrt_destroy_tensor_set(&ins);
        nrt_destroy_tensor_set(&outs);
        nrt_free_model_tensor_info(info);
        nrt_unload(model);
        nrt_close();
    } catch (const std::exception &e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    printf("\n%s\n", failures ? "RESULT: FAIL" : "RESULT: PASS");
    return failures ? 1 : 0;
}
