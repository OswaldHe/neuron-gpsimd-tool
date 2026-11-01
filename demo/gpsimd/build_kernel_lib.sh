#!/bin/bash
# build_kernel_lib.sh <src.cpp|src.c> <out.so> [extra compiler flags...]
#
# Builds a trn3 GPSIMD "kernel library" (the format the trn3 runtime/firmware loads for KERNEL_CALL):
#   * Xtensa ELF linked with a position-independent split-load layout (gpsimd/lsp: code at
#     0x01000000, data at 0x02000000, dynamic info at 0x03000000; relocated by the loader),
#   * a GNU build-id note (required by the loader),
#   * one ELF note per kernel (owner "Neuron", type 2: {u32 entry; char name[]}), emitted by
#     NEURON_KERNEL(name) in neuron_kl.h; klpatch.py fills in the entry addresses after linking.
# Only base-ISA code is allowed (-mno-flix ...): the trn3 GPSIMD core does not execute the
# FLIX/vector encodings of the ncore2gp configuration shipped with aws-neuronx-gpsimd-tools.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$1; OUT=$2; shift 2

GPSIMD_TOOLS=${GPSIMD_TOOLS:-/opt/aws/neuron/gpsimd}
export XTENSA_SYSTEM=${XTENSA_SYSTEM:-$GPSIMD_TOOLS/tools/ncore2gp/config}
export XTENSA_CORE=${XTENSA_CORE:-ncore2gp}
export LM_LICENSE_FILE=${LM_LICENSE_FILE:-$GPSIMD_TOOLS/tools/licenses/amzn_vq7_us_582883.out}
export PATH=$GPSIMD_TOOLS/bin:$PATH

COMMON="-O2 -fpic -mno-flix -mno-generate-flix -mno-coproc -ffreestanding -fno-builtin -fno-jump-tables -Wall -Werror -I$HERE/include"
case "$SRC" in
  *.cpp|*.cc|*.cxx) CC="xt-clang++ -std=c++17 -fno-exceptions -fno-rtti -fno-threadsafe-statics -fno-use-cxa-atexit" ;;
  *)                CC="xt-clang" ;;
esac
OBJ="${OUT%.so}.o"
$CC $COMMON "$@" -c "$SRC" -o "$OBJ"
xt-clang -nostdlib -nostartfiles -mlsp="$HERE/lsp" -Wl,--shared-pagesize=128 -Wl,-pie -Wl,--build-id \
         -Wl,-e,_start -o "$OUT" "$OBJ" 2> >(grep -v 'argument unused during compilation' >&2)
xt-nm "$OUT" > "${OUT%.so}.nm"
python3 -I "$HERE/klpatch.py" "$OUT" "${OUT%.so}.nm"
# sanity: only base-ISA code (no FLIX bundles: first byte low nibble 0xE/0xF would be a bundle)
if xt-objdump -d "$OUT" | grep -qE '^\s+[0-9a-f]+:\s+[0-9a-f]{16,}\s'; then
    echo "error: $OUT contains FLIX bundles, which do not run on the trn3 GPSIMD core" >&2; exit 1
fi
echo "kernel library: $OUT ($(stat -c %s "$OUT") bytes)"
