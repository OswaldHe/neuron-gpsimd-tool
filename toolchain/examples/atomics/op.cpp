// atomics: cross-core mutual exclusion (software bakery lock in SBUF), barrier, reductions and a
// dynamic work queue. The trn3 GPSIMD core's exclusive-access instructions (L32EX/S32EX) hang, so
// atomic read-modify-write is built on gp_lock/gp_unlock (Call::atomic_add).
#include <gpsimd/gpsimd.hpp>

GPSIMD_OP(atomics) {
    gp::Call c(call);
    auto x = c.in<int32_t>(0);
    auto out = c.out<int32_t>(0);
    auto sums = c.out<int32_t>(1);
    uint32_t nshared;
    volatile uint32_t *sh = c.shared(&nshared);       // SBUF words shared by the 8 cores, zeroed per call
    volatile uint32_t *counter = &sh[0], *queue = &sh[1];

    // phase 1: contended atomic increments
    for (uint32_t k = 0; k < (c.core() + 1) * 100; ++k) c.atomic_add(counter, 1);
    c.barrier();

    // phase 2: reductions over per-core values
    const int32_t v = static_cast<int32_t>((c.core() * 37 + 11) % 100);
    const uint32_t sum = c.allreduce_add(static_cast<uint32_t>(v));
    const int32_t mx = c.allreduce_max(v), mn = c.allreduce_min(v);
    uint32_t total;
    const uint32_t before = c.exscan_add(static_cast<uint32_t>(v), &total);

    // phase 3: dynamic work queue: claim rows with fetch-and-add until none are left
    uint32_t claimed = 0;
    for (;;) {
        const uint32_t p = c.atomic_add(queue, 1);
        if (p >= x.partitions()) break;
        int32_t s = 0;
        for (uint32_t i = 0; i < x.elems(); ++i) s += x.row(p)[i];
        sums.row(p)[0] = s;
        ++claimed;
    }
    const uint32_t claimed_total = c.allreduce_add(claimed);
    gp_receipt_user(0, c.core());
    gp_receipt_user(1, claimed);
    if (c.core() == 3) out.row(0)[5] = static_cast<int32_t>(before);
    if (c.core() == 0) {
        out.row(0)[0] = static_cast<int32_t>(*counter);
        out.row(0)[1] = 0;
        out.row(0)[2] = static_cast<int32_t>(sum);
        out.row(0)[3] = mx;
        out.row(0)[4] = mn;
        out.row(0)[6] = static_cast<int32_t>(total);
        out.row(0)[7] = static_cast<int32_t>(claimed_total);
        gp_printf("counter %u, rows claimed (all cores) %u\n", *counter, claimed_total);
    }
}
