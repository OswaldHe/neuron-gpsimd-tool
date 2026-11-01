// GPSIMD custom op: print the operand values it receives (raw, and reduced through the
// core's Vision-P TIE vector unit), then pass them through unchanged.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <torch/torch.h>
#include <xtensa/tie/xt_ivpn.h>
#include "neuron/neuron-utils.hpp"

static constexpr int LANES = 16;  // xb_vecN_2xf32 = 16 x fp32 (512-bit vector)

// Sum and max of n floats using TIE vector intrinsics (n must be a multiple of LANES).
static void tie_row_stats(const float* src, int n, float* sum, float* mx) {
    alignas(64) float buf[LANES];
    float s = 0.0f, m = -3.4e38f;
    for (int i = 0; i < n; i += LANES) {
        memcpy(buf, src + i, sizeof(buf));
        xb_vecN_2xf32 v = IVP_LVN_2XF32_I(reinterpret_cast<const xb_vecN_2xf32*>(buf), 0);
        s += (float)IVP_RADDN_2XF32(v);
        float vm = (float)IVP_RMAXN_2XF32(v);
        if (vm > m) m = vm;
    }
    *sum = s; *mx = m;
}

torch::Tensor dbg_print(const torch::Tensor& t_in) {
    const int64_t rows = t_in.size(0), cols = t_in.size(1);
    printf("[gpsimd cpu %u/%u] dbg_print: operand shape=(%lld,%lld)\n",
           (unsigned)get_cpu_id(), (unsigned)get_cpu_count(), (long long)rows, (long long)cols);
    torch::Tensor t_out = torch::zeros({rows, cols}, torch::kFloat);
    auto in = t_in.accessor<float, 2>();
    auto out = t_out.accessor<float, 2>();
    alignas(64) float row[256];
    for (int64_t r = 0; r < rows; r++) {
        const int64_t n = cols < 256 ? cols : 256;
        for (int64_t c = 0; c < cols; c++) {
            float v = in[r][c];
            if (c < n) row[c] = v;
            out[r][c] = v;
        }
        if (r < 3) {
            float s, m;
            tie_row_stats(row, (int)(n / LANES) * LANES, &s, &m);
            printf("  row %lld: operand[0..3] = %f %f %f %f | TIE sum=%f max=%f\n", (long long)r,
                   (double)row[0], (double)row[1], (double)row[2], (double)row[3], (double)s, (double)m);
        }
    }
    printf("[gpsimd] dbg_print done\n");
    return t_out;
}
