// addone: y = x + 1 (int32), each core its share of the partitions
#include <gpsimd/gpsimd.hpp>

GPSIMD_OP(addone) {
    gp::Call c(call);
    auto x = c.in<int32_t>(0);
    auto y = c.out<int32_t>(0);
    const gp::Range rows = c.split(x.partitions());
    for (uint32_t p = rows.lo; p < rows.hi; ++p)
        for (uint32_t i = 0; i < x.elems(); ++i) y.row(p)[i] = x.row(p)[i] + 1;
}
