#!/bin/bash
# run_demo.sh - build and run the trn3 GPSIMD custom-operator demo end to end.
#
#   1. NKI kernel (nki/sbuf_add_nki.py)            -> build/sbuf_add_nki.neff       (neuronx-cc, trn3)
#   2. read the SBUF tile location from the NEFF   -> tile parameters for the custom op
#   3. GPSIMD custom op (gpsimd/sbuf_add.cpp)       -> build/sbuf_add.so             (xt-clang++)
#   4. insert the custom op into the NEFF          -> build/sbuf_add_gpsimd.neff    (tools/neff_tool.py)
#   5. runner (runner/run_neff.cpp)                -> build/run_neff                (g++ + libnrt)
#   6. control: run the unmodified NEFF, expect output == input (bit-exact)
#   7. run the modified NEFF, expect output == input + ADDEND (bit-exact) and GPSIMD execution receipts
#
# Environment (all optional):
#   ADDEND=1000                          constant added by the custom op
#   ITERS=3                              executions of the modified NEFF
#   SEED=12345                           seed of the random int32 input
#   NEURON_RT_VISIBLE_CORES=0            NeuronCore to run on (the runner maps SBUF on device $DEVICE)
#   DEVICE=0                             Neuron device index (/dev/neuron$DEVICE)
#   NEURON_VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0   Python venv with nki + neuronx-cc
#   GPSIMD_TOOLS=/opt/aws/neuron/gpsimd  aws-neuronx-gpsimd-tools install
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=$HERE/build
ADDEND=${ADDEND:-1000}
ITERS=${ITERS:-3}
SEED=${SEED:-12345}
DEVICE=${DEVICE:-0}
NEURON_VENV=${NEURON_VENV:-/opt/aws_neuronx_venv_torch_neuronx_2_14_0}
export GPSIMD_TOOLS=${GPSIMD_TOOLS:-/opt/aws/neuron/gpsimd}
export NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-0}
export NEURON_RT_LOG_LEVEL=${NEURON_RT_LOG_LEVEL:-ERROR}
export NEURON_RT_EXEC_TIMEOUT=${NEURON_RT_EXEC_TIMEOUT:-30}

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
need() { [ -e "$1" ] || { echo "missing: $1 ($2)" >&2; exit 2; }; }

step "0. prerequisites"
need "$NEURON_VENV/bin/python" "Neuron torch venv with nki and neuronx-cc"
need "$GPSIMD_TOOLS/bin/xt-clang++" "aws-neuronx-gpsimd-tools"
need /opt/aws/neuron/lib/libnrt.so "aws-neuronx-runtime-lib"
need /opt/aws/neuron/include/nrt/nrt.h "aws-neuronx-runtime-lib headers"
need "/dev/neuron$DEVICE" "Neuron device"
command -v g++ >/dev/null || { echo "missing: g++" >&2; exit 2; }
echo "venv $NEURON_VENV, gpsimd tools $GPSIMD_TOOLS, device /dev/neuron$DEVICE, NeuronCore $NEURON_RT_VISIBLE_CORES"
mkdir -p "$BUILD"

step "1. compile the NKI kernel for trn3"
PATH="$NEURON_VENV/bin:$PATH" "$NEURON_VENV/bin/python" "$HERE/nki/sbuf_add_nki.py" "$BUILD/sbuf_add_nki.neff" 2>&1 \
    | grep -vE 'FutureWarning|warnings.warn' || true
[ -s "$BUILD/sbuf_add_nki.neff" ] || { echo "NKI compilation failed" >&2; exit 1; }

step "2. locate the SBUF tile in the compiled NEFF"
python3 -I "$HERE/tools/neff_tool.py" info "$BUILD/sbuf_add_nki.neff"
eval "$(python3 -I "$HERE/tools/neff_tool.py" info "$BUILD/sbuf_add_nki.neff" --shell)"
[ "$TILE_DTYPE" = int32 ] || { echo "this demo expects an int32 tile, got $TILE_DTYPE" >&2; exit 1; }

step "3. build the GPSIMD custom operator (C++, Xtensa toolchain)"
"$HERE/gpsimd/build_kernel_lib.sh" "$HERE/gpsimd/sbuf_add.cpp" "$BUILD/sbuf_add.so" \
    -DTILE_SBUF_OFFSET="${TILE_SBUF_OFFSET}u" -DTILE_PARTITIONS="${TILE_PARTITIONS}u" \
    -DTILE_ELEMS="${TILE_ELEMS}u" -DADDEND="$ADDEND"

step "4. insert the custom operator into the NEFF"
python3 -I "$HERE/tools/neff_tool.py" inject "$BUILD/sbuf_add_nki.neff" "$BUILD/sbuf_add.so" sbuf_add \
    "$BUILD/sbuf_add_gpsimd.neff"

step "5. build the runner"
g++ -std=c++17 -O2 -Wall -Wextra -o "$BUILD/run_neff" "$HERE/runner/run_neff.cpp" \
    -I/opt/aws/neuron/include -L/opt/aws/neuron/lib -lnrt -Wl,-rpath,/opt/aws/neuron/lib
echo "runner: $BUILD/run_neff"

step "6. control: unmodified NKI NEFF (expect output == input)"
"$BUILD/run_neff" "$BUILD/sbuf_add_nki.neff" --expect identity --seed "$SEED" --device "$DEVICE"

step "7. NKI NEFF + GPSIMD custom operator (expect output == input + $ADDEND, on the GPSIMD cores)"
"$BUILD/run_neff" "$BUILD/sbuf_add_gpsimd.neff" --expect "add:$ADDEND" --iters "$ITERS" --seed "$SEED" \
    --device "$DEVICE" --logs
