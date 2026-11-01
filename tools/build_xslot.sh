#!/bin/bash
# build tools/xslot (host x86 tool linked against the Xtensa ISA library)
set -e
D=$(cd "$(dirname "$0")" && pwd)
T=/opt/aws/neuron/gpsimd/tools/XtensaTools
gcc -O2 -o "$D/xslot" "$D/xslot.c" -I$T/include -L$T/lib -lxtisa -Wl,-rpath,$T/lib -ldl
