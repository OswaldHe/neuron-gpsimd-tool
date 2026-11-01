#!/bin/bash
# xdis.sh <elf> <out_prefix> [sym@addr ...]: disassemble stripped .text as code (base 0x01000000)
set -e
f=$1; o=$2; shift 2
xt-objcopy -O binary -j .text "$f" $o.text.bin
args=()
for s in "$@"; do n=${s%@*}; a=${s#*@}; args+=(--add-symbol "$n=.text:$(printf '0x%x' $((a - 0x01000000))),function,global"); done
xt-objcopy -I binary -O elf32-xtensa-le -B xtensa --rename-section .data=.text,alloc,code,readonly,contents "${args[@]}" $o.text.bin $o.text.o
xt-objdump -d $o.text.o | awk 'NR>6' | sed -E 's/^ +([0-9a-f]+):/\1:/' > $o.dis
