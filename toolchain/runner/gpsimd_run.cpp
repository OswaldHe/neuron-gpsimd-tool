// gpsimd-run - run a trn3 NEFF with GPSIMD custom operators, validate outputs bit-exactly against
// .npy references, and verify that the operators ran on the GPSIMD cores.
//
// usage: gpsimd-run <file.neff> --in NAME=FILE.npy ... [--expect NAME=FILE.npy ...]
//                   [--sites N] [--iters N] [--device D] [--vnc V] [--no-map] [--logs] [--receipts]
//                   [--save DIR] [--quiet]
//
//   --in NAME=FILE      data for input tensor NAME (raw bytes of the .npy array; size must match)
//   --expect NAME=FILE  expected content of output tensor NAME (compared with memcmp: bit-exact)
//   --sites N           number of GPSIMD call sites in the NEFF: every core must report N*iters calls
//                       and a receipt for every site (0 = control run: no receipts allowed)
//   --no-map            do not map SBUF into the GPSIMD cores' address space
//   --logs              print the GPSIMD cores' gp_printf logs
//   --receipts          print the per-core receipt table
//   --save DIR          write every output tensor to DIR/NAME.npy (uint8 bytes when no --expect)
//   --profile FILE      capture a device profile (NTFF) of the last execution; read it with
//                       neuron-explorer view -n <neff> -s FILE --output-format summary-json
//   --hbm TENSOR        map TENSOR's device memory into the GPSIMD cores' address space (one
//                       translation window per tensor, core addresses from 0xE0000000) and write the
//                       core address of the k-th --hbm tensor to element k of the int32 input tensor
//                       --hbm-param (default gp_hbm); element 15 = 0x48424D31 ("HBM1")
//
// Steps: nrt_init, nrt_load; map SBUF (GPSIMD address-translation entry 38 -> SBUF, via PCIe BAR0);
// write inputs; nrt_execute (N times); read and compare outputs; read the receipts/logs that the
// GPSIMD runtime keeps in each core's data RAM (over BAR0); unmap.
// Exit status: 0 = all checks passed, 1 = a check failed, 2 = usage/runtime error.

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
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
#include <nrt/nrt_profile.h>
}

namespace {

// ------------------------------------------------------------------ device access over PCIe BAR0
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
const uint32_t BAR_CHUNK = 100;

const uint64_t TPB_BAR0_OFFSET = 0xD0000000ull, TPB_BAR0_DIST = 0x4000000ull;
const uint64_t SBUF_SOC_BASE[8] = {0x2000000000ull,   0x3000000000ull,   0x6000000000ull,   0x7000000000ull,
                                   0x802000000000ull, 0x803000000000ull, 0x806000000000ull, 0x807000000000ull};
const uint64_t SBUF_BYTES = 128ull * 0x40000;
const uint64_t GPSIMD_BASE = 0x3100000, GPSIMD_STRIDE = 0x100000;
const uint32_t GPSIMD_CORES = 8, DRAM_LO = 0x80000, DRAM_HI = 0xC0000, IRAM_HI = 0x20000;
const uint32_t FW_CTX = 0x80878;
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
// HBM windows: one translation entry per mapped tensor (entries the runtime leaves unused), core
// addresses allocated upwards from 0xE0000000, each window a power of two >= 16 MB aligned to its size.
const uint32_t HBM_ENTRIES[] = {39, 35, 34, 31, 30, 27, 26, 23, 22, 19, 18, 15, 14, 11};
const uint32_t HBM_WINDOW = 0xE0000000u;
bool map_window(const Bar0 &b, int t, uint32_t entry, uint32_t local, uint64_t soc, uint64_t size) {
    const uint64_t off = Bar0::tpb(t, XLAT_OFF + 28ull * entry);
    auto cur = b.read(off, 7);
    if ((cur[0] & 1) && (cur[3] | (uint64_t)cur[4] << 32) != local) return false;     // used by someone else
    uint64_t mask = (~(size - 1)) & 0xFFFFFFFFFFull;
    b.write(off, {0});
    b.write(off + 4, {(uint32_t)mask, (uint32_t)(mask >> 32), local, 0, (uint32_t)soc, (uint32_t)(soc >> 32)});
    b.write(off, {1});
    return true;
}
void unmap_window(const Bar0 &b, int t, uint32_t entry) { b.write(Bar0::tpb(t, XLAT_OFF + 28ull * entry), {0}); }

// ------------------------------------------------------------------ receipts and logs (runtime/rt.c)
struct Receipt {
    uint32_t magic0, magic1, prid, calls, site_mask, last_site, last_cycles, total_cycles;
    uint32_t entry_pc, sp, ra, status, user[4];
};
static_assert(sizeof(Receipt) == 64, "receipt layout");
const uint32_t RECEIPT_MAGIC0 = 0x4D495350u, RECEIPT_MAGIC1 = 0x32504347u;
const uint32_t LOG_MAGIC0 = 0x4b4c4f47u, LOG_MAGIC1 = 0x31534d47u;

struct CoreFindings {
    bool has_receipt = false;
    Receipt receipt{};
    std::string log;
};

std::vector<CoreFindings> read_gpsimd(const Bar0 &b, int t) {
    std::vector<CoreFindings> out(GPSIMD_CORES);
    auto dram0 = b.read(Bar0::gpsimd(t, 0, DRAM_LO), (DRAM_HI - DRAM_LO) / 4);
    int64_t rcpt = -1, klog = -1;
    for (size_t i = 0; i + 1 < dram0.size(); ++i) {
        if (dram0[i] == RECEIPT_MAGIC0 && dram0[i + 1] == RECEIPT_MAGIC1 && rcpt < 0) rcpt = i;
        if (dram0[i] == LOG_MAGIC0 && dram0[i + 1] == LOG_MAGIC1 && klog < 0) klog = i;
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
            if (h[0] == LOG_MAGIC0 && h[1] == LOG_MAGIC1 && h[2] > 0 && h[2] <= 0x10000) {
                uint32_t cap = h[2], pos = h[3];
                auto tw = b.read(Bar0::gpsimd(t, c, DRAM_LO + 4 * klog + 32), (cap + 3) / 4);
                std::string text(reinterpret_cast<const char *>(tw.data()), cap);
                out[c].log = pos <= cap ? text.substr(0, pos) : text.substr(pos % cap) + text.substr(0, pos % cap);
            }
        }
    }
    return out;
}
void clear_gpsimd(const Bar0 &b, int t) {
    auto dram0 = b.read(Bar0::gpsimd(t, 0, DRAM_LO), (DRAM_HI - DRAM_LO) / 4);
    for (size_t i = 0; i + 1 < dram0.size(); ++i)
        if ((dram0[i] == RECEIPT_MAGIC0 && dram0[i + 1] == RECEIPT_MAGIC1) || (dram0[i] == LOG_MAGIC0 && dram0[i + 1] == LOG_MAGIC1) ||
            (dram0[i] == 0x4D495350u && dram0[i + 1] == 0x43524444u))
            for (uint32_t c = 0; c < GPSIMD_CORES; ++c) b.write(Bar0::gpsimd(t, c, DRAM_LO + 4 * i), {0, 0});
}

// ------------------------------------------------------------------ .npy
struct Npy {
    std::string descr;
    std::vector<size_t> shape;
    std::vector<uint8_t> data;
    size_t itemsize() const {
        size_t k = descr.find_first_of("0123456789");
        return k == std::string::npos ? 1 : std::stoul(descr.substr(k));
    }
};
Npy read_npy(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (raw.size() < 10 || memcmp(raw.data(), "\x93NUMPY", 6) != 0) throw std::runtime_error(path + ": not a .npy file");
    size_t hlen, hoff;
    if (raw[6] == 1) { hlen = raw[8] | raw[9] << 8; hoff = 10; }
    else { hlen = raw[8] | raw[9] << 8 | raw[10] << 16 | (size_t)raw[11] << 24; hoff = 12; }
    std::string h(raw.begin() + hoff, raw.begin() + hoff + hlen);
    Npy n;
    auto field = [&](const char *key) {
        size_t k = h.find(key);
        if (k == std::string::npos) throw std::runtime_error(path + ": header lacks " + key);
        return k + strlen(key);
    };
    size_t d = h.find('\'', field("'descr':")) + 1;
    n.descr = h.substr(d, h.find('\'', d) - d);
    if (h.find("True", field("'fortran_order':")) < h.find(',', field("'fortran_order':")))
        throw std::runtime_error(path + ": fortran order not supported");
    size_t s = h.find('(', field("'shape':")) + 1, e = h.find(')', s);
    std::string dims = h.substr(s, e - s);
    for (size_t p = 0; p < dims.size();) {
        size_t q = dims.find(',', p);
        std::string tok = dims.substr(p, q == std::string::npos ? std::string::npos : q - p);
        if (tok.find_first_of("0123456789") != std::string::npos) n.shape.push_back(std::stoul(tok));
        if (q == std::string::npos) break;
        p = q + 1;
    }
    if (n.descr[0] == '>') throw std::runtime_error(path + ": big-endian data not supported");
    n.data.assign(raw.begin() + hoff + hlen, raw.end());
    return n;
}
void write_npy(const std::string &path, const std::string &descr, const std::vector<size_t> &shape, const void *data, size_t bytes) {
    std::string sh = "(";
    for (size_t k = 0; k < shape.size(); ++k) sh += std::to_string(shape[k]) + (shape.size() == 1 || k + 1 < shape.size() ? "," : "");
    sh += ")";
    std::string h = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + sh + ", }";
    while ((10 + h.size() + 1) % 64) h += ' ';
    h += '\n';
    std::ofstream f(path, std::ios::binary);
    f.write("\x93NUMPY\x01\x00", 8);
    uint16_t hl = h.size();
    f.write(reinterpret_cast<const char *>(&hl), 2);
    f.write(h.data(), h.size());
    f.write(static_cast<const char *>(data), bytes);
}
std::string fmt_elem(const std::string &descr, const uint8_t *p) {
    char buf[64];
    char kind = descr.size() > 1 ? descr[1] : 'u';
    size_t sz = Npy{descr, {}, {}}.itemsize();
    uint64_t u = 0;
    memcpy(&u, p, sz);
    if (kind == 'f' && sz == 4) { float f; memcpy(&f, p, 4); snprintf(buf, sizeof buf, "%.9g (0x%08x)", f, (uint32_t)u); }
    else if (kind == 'i') { int64_t v = sz == 1 ? (int8_t)u : sz == 2 ? (int16_t)u : sz == 4 ? (int32_t)u : (int64_t)u; snprintf(buf, sizeof buf, "%" PRId64, v); }
    else snprintf(buf, sizeof buf, "%" PRIu64 " (0x%" PRIx64 ")", u, u);
    return buf;
}

void check(NRT_STATUS s, const char *what) {
    if (s != NRT_SUCCESS) throw std::runtime_error(std::string(what) + " failed: NRT_STATUS=" + std::to_string(s));
}

struct Options {
    std::string neff, save, profile;            // profile: NTFF written for the last execution
    std::map<std::string, std::string> ins, expect;
    std::vector<std::string> hbm;               // tensors mapped into the GPSIMD address space
    std::string hbm_param = "gp_hbm";           // int32 input tensor receiving their core addresses
    int iters = 1, device = 0, vnc = 0, sites = -1;
    bool map = true, logs = false, receipts = false, quiet = false;
};
const char *USAGE =
    "usage: gpsimd-run <file.neff> --in NAME=FILE.npy ... [--expect NAME=FILE.npy ...] [--sites N] [--iters N]\n"
    "                  [--device D] [--vnc V] [--no-map] [--logs] [--receipts] [--save DIR] [--quiet]\n"
    "                  [--hbm TENSOR ...] [--hbm-param NAME] [--profile FILE.ntff]";

Options parse(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        auto kv = [&](std::map<std::string, std::string> &m) {
            std::string v = val();
            size_t eq = v.find('=');
            if (eq == std::string::npos) throw std::runtime_error(a + " expects NAME=FILE");
            m[v.substr(0, eq)] = v.substr(eq + 1);
        };
        if (a == "--in") kv(o.ins);
        else if (a == "--expect") kv(o.expect);
        else if (a == "--iters") o.iters = std::stoi(val());
        else if (a == "--sites") o.sites = std::stoi(val());
        else if (a == "--device") o.device = std::stoi(val());
        else if (a == "--vnc") o.vnc = std::stoi(val());
        else if (a == "--save") o.save = val();
        else if (a == "--profile") o.profile = val();
        else if (a == "--hbm") o.hbm.push_back(val());
        else if (a == "--hbm-param") o.hbm_param = val();
        else if (a == "--no-map") o.map = false;
        else if (a == "--logs") o.logs = true;
        else if (a == "--receipts") o.receipts = true;
        else if (a == "--quiet") o.quiet = true;
        else if (a[0] != '-' && o.neff.empty()) o.neff = a;
        else throw std::runtime_error("unknown argument " + a);
    }
    if (o.neff.empty()) throw std::runtime_error(USAGE);
    return o;
}

}  // namespace

int main(int argc, char **argv) {
    Options opt;
    try { opt = parse(argc, argv); } catch (const std::exception &e) { fprintf(stderr, "%s\n", e.what()); return 2; }
    int failures = 0;
    auto fail = [&](const std::string &m) { printf("  FAIL: %s\n", m.c_str()); ++failures; };

    try {
        std::ifstream f(opt.neff, std::ios::binary);
        if (!f) throw std::runtime_error("cannot read " + opt.neff);
        std::vector<char> neff((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

        check(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""), "nrt_init");
        nrt_model_t *model = nullptr;
        check(nrt_load(neff.data(), neff.size(), opt.vnc, 1, &model), "nrt_load");
        if (!opt.quiet) printf("[load] %s (%zu bytes) on vnc %d\n", opt.neff.c_str(), neff.size(), opt.vnc);

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

        nrt_tensor_info_array_t *info = nullptr;
        check(nrt_get_model_tensor_info(model, &info), "nrt_get_model_tensor_info");
        nrt_tensor_set_t *ins = nullptr, *outs = nullptr;
        check(nrt_allocate_tensor_set(&ins), "nrt_allocate_tensor_set");
        check(nrt_allocate_tensor_set(&outs), "nrt_allocate_tensor_set");
        struct Out { nrt_tensor_t *t; size_t size; std::string name; std::vector<size_t> shape; };
        std::vector<Out> out_tensors;
        std::vector<nrt_tensor_t *> all;
        std::set<std::string> used_in;
        for (uint64_t i = 0; i < info->tensor_count; ++i) {
            const nrt_tensor_info_t &ti = info->tensor_array[i];
            nrt_tensor_t *t = nullptr;
            check(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, opt.vnc, ti.size, ti.name, &t), "nrt_tensor_allocate");
            all.push_back(t);
            std::vector<size_t> shape(ti.shape, ti.shape + ti.ndim);
            if (ti.usage == NRT_TENSOR_USAGE_INPUT) {
                auto it = opt.ins.find(ti.name);
                if (it == opt.ins.end()) throw std::runtime_error(std::string("no --in for input tensor '") + ti.name + "'");
                Npy n = read_npy(it->second);
                if (n.data.size() != ti.size)
                    throw std::runtime_error(std::string("input ") + ti.name + ": " + it->second + " has " + std::to_string(n.data.size()) +
                                             " bytes, tensor has " + std::to_string(ti.size));
                check(nrt_tensor_write(t, n.data.data(), 0, ti.size), "nrt_tensor_write");
                check(nrt_add_tensor_to_tensor_set(ins, ti.name, t), "nrt_add_tensor_to_tensor_set");
                used_in.insert(ti.name);
            } else {
                out_tensors.push_back({t, ti.size, ti.name, shape});
                check(nrt_add_tensor_to_tensor_set(outs, ti.name, t), "nrt_add_tensor_to_tensor_set");
            }
            if (!opt.quiet) {
                std::string sh;
                for (size_t k = 0; k < shape.size(); ++k) sh += (k ? "x" : "") + std::to_string(shape[k]);
                printf("[tensor] %-12s %-6s [%s] %zu bytes\n", ti.name, ti.usage == NRT_TENSOR_USAGE_INPUT ? "input" : "output", sh.c_str(), ti.size);
            }
        }
        for (auto &[name, file] : opt.ins)
            if (!used_in.count(name)) throw std::runtime_error("--in " + name + ": the NEFF has no such input tensor");
        std::vector<int> hbm_mapped;
        if (!opt.hbm.empty()) {
            std::map<std::string, nrt_tensor_t *> by_name;
            for (uint64_t i = 0; i < info->tensor_count; ++i) by_name[info->tensor_array[i].name] = all[i];
            auto pit = by_name.find(opt.hbm_param);
            if (pit == by_name.end()) throw std::runtime_error("--hbm needs an int32 input tensor '" + opt.hbm_param + "'");
            if (opt.hbm.size() > 14) throw std::runtime_error("at most 14 --hbm tensors");
            std::vector<int32_t> params(16, 0);
            uint64_t cursor = HBM_WINDOW;
            for (size_t k = 0; k < opt.hbm.size(); ++k) {
                const std::string &n = opt.hbm[k];
                auto it = by_name.find(n);
                if (it == by_name.end()) throw std::runtime_error("--hbm " + n + ": no such tensor");
                nrt_tensor_device_allocation_info_t ai{};
                check(nrt_tensor_get_device_allocation_info(it->second, &ai), "nrt_tensor_get_device_allocation_info");
                uint64_t win = 16ull << 20, base = 0;
                for (; win <= (1ull << 31); win <<= 1) {
                    base = ai.physical_address & ~(win - 1);
                    if (ai.physical_address + ai.size <= base + win) break;
                }
                cursor = (cursor + win - 1) & ~(win - 1);
                if (win > (1ull << 31) || cursor + win > 0x100000000ull)
                    throw std::runtime_error("--hbm " + n + ": no room in the GPSIMD address space for its window");
                params[k] = (int32_t)(uint32_t)(cursor + (ai.physical_address - base));
                for (int t = 0; t < 8; ++t) {
                    if (!firmware_up(bar, t)) continue;
                    if (map_window(bar, t, HBM_ENTRIES[k], (uint32_t)cursor, base, win)) hbm_mapped.push_back(t * 64 + HBM_ENTRIES[k]);
                    else throw std::runtime_error("translation entry " + std::to_string(HBM_ENTRIES[k]) + " is in use");
                }
                if (!opt.quiet)
                    printf("[hbm] %-12s device 0x%" PRIx64 " (%zu bytes) -> GPSIMD core address 0x%08x (window %" PRIu64 " MB, entry %u)\n",
                           n.c_str(), ai.physical_address, ai.size, (uint32_t)params[k], win >> 20, HBM_ENTRIES[k]);
                cursor += win;
            }
            params[15] = 0x48424D31;
            check(nrt_tensor_write(pit->second, params.data(), 0, 64), "nrt_tensor_write");
        }
        std::map<std::string, Npy> expected;
        for (auto &[name, file] : opt.expect) {
            bool found = false;
            for (auto &o : out_tensors) found |= o.name == name;
            if (!found) throw std::runtime_error("--expect " + name + ": the NEFF has no such output tensor");
            expected[name] = read_npy(file);
        }

        std::vector<std::vector<uint8_t>> last(out_tensors.size());
        for (int it = 0; it < opt.iters; ++it) {
            for (auto &o : out_tensors) {   // poison outputs so stale results cannot pass
                std::vector<uint8_t> poison(o.size, 0xA5);
                check(nrt_tensor_write(o.t, poison.data(), 0, o.size), "nrt_tensor_write");
            }
            const bool prof = !opt.profile.empty() && it == opt.iters - 1;   // profile the last (warm) run
            if (prof) check(nrt_profile_start(model, opt.profile.c_str()), "nrt_profile_start");
            auto t0 = std::chrono::steady_clock::now();
            NRT_STATUS st = nrt_execute(model, ins, outs);
            if (prof) check(nrt_profile_stop(opt.profile.c_str()), "nrt_profile_stop");
            if (st != NRT_SUCCESS) {   // show what the GPSIMD cores logged before failing
                printf("  execution %d failed (NRT_STATUS=%d); GPSIMD logs:\n", it + 1, (int)st);
                for (int t = 0; t < 8; ++t) {
                    if (!firmware_up(bar, t)) continue;
                    auto f2 = read_gpsimd(bar, t);
                    for (uint32_t c = 0; c < GPSIMD_CORES; ++c)
                        if (!f2[c].log.empty()) printf("    [TPB%d core %u] %s\n", t, c, f2[c].log.c_str());
                }
                for (int t : mapped) unmap_sbuf(bar, t);
                for (int te : hbm_mapped) unmap_window(bar, te / 64, te % 64);
                check(st, "nrt_execute");
            }
            double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            std::string line = "  execution " + std::to_string(it + 1) + ": " + std::to_string((int)us) + " us";
            for (size_t k = 0; k < out_tensors.size(); ++k) {
                Out &o = out_tensors[k];
                last[k].assign(o.size, 0);
                check(nrt_tensor_read(o.t, last[k].data(), 0, o.size), "nrt_tensor_read");
                auto e = expected.find(o.name);
                if (e == expected.end()) continue;
                const Npy &ref = e->second;
                if (ref.data.size() != o.size) {
                    fail(o.name + ": expected " + std::to_string(ref.data.size()) + " bytes, tensor has " + std::to_string(o.size));
                    continue;
                }
                size_t isz = ref.itemsize(), n = o.size / isz, bad = 0;
                if (memcmp(ref.data.data(), last[k].data(), o.size) == 0) {
                    line += ", " + o.name + " " + std::to_string(n) + "/" + std::to_string(n) + " bit-exact";
                    continue;
                }
                std::vector<size_t> where;
                for (size_t j = 0; j < n; ++j)
                    if (memcmp(ref.data.data() + j * isz, last[k].data() + j * isz, isz) && bad++ < 5) where.push_back(j);
                line += ", " + o.name + " " + std::to_string(n - bad) + "/" + std::to_string(n) + " bit-exact <-- MISMATCH";
                fail(o.name + ": " + std::to_string(bad) + " of " + std::to_string(n) + " elements differ (execution " + std::to_string(it + 1) + ")");
                for (size_t j : where)
                    printf("    [%zu] got %s expected %s\n", j, fmt_elem(ref.descr, last[k].data() + j * isz).c_str(),
                           fmt_elem(ref.descr, ref.data.data() + j * isz).c_str());
            }
            printf("%s\n", line.c_str());
        }
        if (!opt.save.empty()) {
            mkdir(opt.save.c_str(), 0755);
            for (size_t k = 0; k < out_tensors.size(); ++k) {
                Out &o = out_tensors[k];
                auto e = expected.find(o.name);
                std::string descr = e != expected.end() ? e->second.descr : "|u1";
                std::vector<size_t> shape = e != expected.end() ? e->second.shape : std::vector<size_t>{o.size};
                write_npy(opt.save + "/" + o.name + ".npy", descr, shape, last[k].data(), o.size);
            }
        }

        // GPSIMD execution evidence
        std::map<int, std::vector<CoreFindings>> found;
        for (int t = 0; t < 8; ++t) {
            if (!firmware_up(bar, t)) continue;
            auto f2 = read_gpsimd(bar, t);
            bool any = false;
            for (auto &c : f2) any |= c.has_receipt || !c.log.empty();
            if (any) found[t] = f2;
        }
        if (opt.sites > 0) {
            if (found.size() != 1) fail("expected GPSIMD receipts on exactly one TPB, found " + std::to_string(found.size()));
            const uint32_t want_mask = opt.sites >= 32 ? 0xFFFFFFFFu : (1u << opt.sites) - 1;
            for (auto &[t, cores] : found) {
                uint64_t cycles = 0;
                std::set<uint32_t> prids;
                for (uint32_t c = 0; c < GPSIMD_CORES; ++c) {
                    if (!cores[c].has_receipt) { fail("GPSIMD core " + std::to_string(c) + " left no receipt"); continue; }
                    const Receipt &r = cores[c].receipt;
                    prids.insert(r.prid);
                    cycles += r.total_cycles;
                    std::string who = "TPB" + std::to_string(t) + " core " + std::to_string(c);
                    if (r.prid != c) fail(who + ": receipt written by PRID " + std::to_string(r.prid));
                    if (r.calls != (uint32_t)(opt.sites * opt.iters))
                        fail(who + ": " + std::to_string(r.calls) + " operator calls, expected " + std::to_string(opt.sites * opt.iters));
                    if (r.site_mask != want_mask) fail(who + ": call sites run mask 0x" + std::to_string(r.site_mask));
                    if (r.status) fail(who + ": runtime status " + std::to_string(r.status) + " (1 = barrier timeout, 2 = tile dtype mismatch, 4 = lock timeout, 64 = iDMA error)");
                    if (r.entry_pc >= IRAM_HI) fail(who + ": entry address not in GPSIMD instruction RAM");
                    if (r.sp < DRAM_LO || r.sp >= DRAM_HI) fail(who + ": stack not in GPSIMD data RAM");
                }
                if (opt.receipts) {
                    printf("  TPB%d core prid  calls  sites  last-cycles  entry-pc    stack-ptr   status  user[0..3]\n", t);
                    for (uint32_t c = 0; c < GPSIMD_CORES; ++c) {
                        if (!cores[c].has_receipt) continue;
                        const Receipt &r = cores[c].receipt;
                        printf("       %4u %4u  %5u  0x%03x  %11u  0x%08x  0x%08x  %6u  %08x %08x %08x %08x\n", c, r.prid, r.calls, r.site_mask,
                               r.last_cycles, r.entry_pc, r.sp, r.status, r.user[0], r.user[1], r.user[2], r.user[3]);
                    }
                }
                printf("  GPSIMD: %zu distinct cores (PRID) on TPB%d ran the operators, %u call(s) each, %" PRIu64 " core cycles in total\n",
                       prids.size(), t, cores[0].receipt.calls, cycles);
            }
        } else if (opt.sites == 0) {
            for (auto &[t, cores] : found)
                for (auto &c : cores)
                    if (c.has_receipt) { fail("control run: unexpected GPSIMD receipt on TPB" + std::to_string(t)); break; }
        }
        if (opt.logs && !found.empty()) {
            printf("  GPSIMD logs:\n");
            for (auto &[t, cores] : found)
                for (uint32_t c = 0; c < GPSIMD_CORES; ++c) {
                    size_t pos = 0;
                    const std::string &log = cores[c].log;
                    while (pos < log.size()) {
                        size_t nl = log.find('\n', pos);
                        if (nl == std::string::npos) nl = log.size();
                        printf("    [TPB%d core %u] %s\n", t, c, log.substr(pos, nl - pos).c_str());
                        pos = nl + 1;
                    }
                }
        }

        for (int t : mapped) unmap_sbuf(bar, t);
        for (int te : hbm_mapped) unmap_window(bar, te / 64, te % 64);
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
    printf("%s\n", failures ? "RESULT: FAIL" : "RESULT: PASS");
    return failures ? 1 : 0;
}
