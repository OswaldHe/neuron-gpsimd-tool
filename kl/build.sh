#!/bin/bash
# build.sh <src.c> <out.so>: build a GPSIMD kernel library for the trn3 nrtucode loader
set -e
source $(cd "$(dirname "$0")" && pwd)/../tools/xtenv.sh
D=$(cd "$(dirname "$0")" && pwd)
CFLAGS="-I$D -O2 -fpic -mno-flix -mno-generate-flix -mno-coproc -ffreestanding -fno-builtin -fno-jump-tables -Wall -Werror ${EXTRA_CFLAGS}"
xt-clang $CFLAGS -c "$1" -o "${2%.so}.o"
RT=$D/../toolchain/runtime/builtins.c; RTO="${2%.so}.rt.o"
xt-clang $CFLAGS -fno-builtin -c "$RT" -o "$RTO"
xt-clang -nostdlib -nostartfiles -mlsp=$D/lsp_kl -Wl,--shared-pagesize=128 -Wl,-pie -Wl,--build-id -Wl,-e,_start -o "$2" "${2%.so}.o" "$RTO" ${EXTRA_LIBS}
xt-nm "$2" > "${2%.so}.nm"
python3 -I $D/klpatch.py "$2" "${2%.so}.nm"
