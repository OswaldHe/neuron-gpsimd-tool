# neuron-gpsimd-tool: custom GPSIMD operators for NKI on AWS Trainium 3

Each Trainium 3 NeuronCore has an 8-core GPSIMD engine next to its Tensor, Vector and Scalar engines. All of
these engines share the on-chip SBUF. NKI reaches the GPSIMD cores only through fixed built-in instructions,
and the official custom-operator toolchain does not support Trainium 2 or 3. This repository provides a
toolchain that compiles C++ operators for the GPSIMD cores and injects them into a compiled NKI kernel (NEFF)
as `KERNEL_CALL` instructions. The operators read and write the same SBUF tiles as the NKI code around them.

```
NKI kernel (gp.call markers) ──neuronx-cc──► kernel.neff ──gpsimd-neff analyze──► call-site tiles
C++ operators (op.cpp) ──────gpsimd-cc (xt-clang)──► ops.so ◄────────────────────────┘
kernel.neff + ops.so ──gpsimd-neff inject──► final.neff ──gpsimd-run──► bit-exact check + per-core receipts
```

The paper (`paper/main.pdf`) and the blog post (`BLOG.md`) describe the design and the results. This README
covers two tasks: setting up the toolchain on a fresh instance, and reproducing the measurements.

| Directory | Contents |
|---|---|
| `toolchain/` | the toolchain (`bin/`), the operator runtime and headers, and the examples. API reference: `toolchain/README.md` |
| `nkiopt/` | benchmark cases: a pure-NKI baseline against a GPSIMD + NKI version, run by `bench.py`. All results: `nkiopt/README.md` |
| `paper/`, `BLOG.md` | the paper and the blog post |
| `figs/` | figures (`figs/src/make_figs.py`) and the profile timelines they use (`figs/data/`) |
| `demo/` | the first minimal end-to-end demo (one fixed operator, no toolchain) |
| `PROGRESS.md`, `re/`, `probes/`, `kl/`, `nki/`, `tools/`, `vecre/` | reverse-engineering notes and experiments behind the toolchain |

## 1. Set up the toolchain on a fresh DLAMI instance

### Requirements

* A **trn3** instance (tested on trn3.48xlarge) started from the **Deep Learning AMI Neuron (Ubuntu 24.04)**
  with Neuron SDK 2.33. The DLAMI already provides the Neuron driver, runtime, profiler, `g++` and the
  `torch-neuronx` virtual environment with `nki`, `neuronx-cc` and the NKI Library (`nkilib`).
* The DLAMI does not include the Xtensa compiler for the GPSIMD cores (`aws-neuronx-gpsimd-tools`). Step 2
  installs it.

The toolchain relies on undocumented interfaces of the runtime and the GPSIMD firmware. It was verified with
exactly these versions:

| Component | Version |
|---|---|
| OS / kernel | Ubuntu 24.04.5, Linux 7.0 (`linux-aws`) |
| `aws-neuronx-dkms` | 2.31.7.0 |
| `aws-neuronx-runtime-lib` (libnrt) | 2.35.30.0 |
| `aws-neuronx-tools` (`neuron-explorer`) | 2.33.38.0 |
| GPSIMD firmware | 1.25.36.0 |
| venv `/opt/aws_neuronx_venv_torch_neuronx_2_14_0` | `neuronx-cc` 2.28.7631, `nki` 0.7.0 |
| `aws-neuronx-gpsimd-tools` | 0.25.0.0 (Xtensa RI-2022.9, core `ncore2gp`) |

Other versions may change the NEFF format or the firmware behavior that the toolchain depends on.

### Step 1: Check the instance

```bash
neuron-ls | head -3                                         # instance-type: trn3.48xlarge
dpkg -l | grep aws-neuronx                                  # runtime 2.35.30, tools 2.33.38, dkms 2.31.7
/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/pip list 2>/dev/null | grep -E '^(nki|neuronx-cc) '
```

### Step 2: Install the GPSIMD Xtensa toolchain

The package comes from the Neuron apt repository, which the DLAMI already configures. Its tools link
against `libtinfo5` and `libncursesw5`, which Ubuntu 24.04 no longer ships. Therefore, install both from the
Ubuntu 22.04 archive. The DLAMI puts the Neuron packages on hold, so the command needs
`--allow-change-held-packages`.

```bash
mkdir -p ~/debs && cd ~/debs
wget http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.3_amd64.deb \
     http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libncursesw5_6.3-2ubuntu0.3_amd64.deb
sudo apt-get update
sudo apt-get install -y --allow-change-held-packages \
     ./libtinfo5_6.3-2ubuntu0.3_amd64.deb ./libncursesw5_6.3-2ubuntu0.3_amd64.deb \
     aws-neuronx-gpsimd-tools=0.25.0.0-3d77332bc
```

Check that the compiler runs. `gpsimd-cc` sets these three variables itself; you need them only for this
manual check.

```bash
G=/opt/aws/neuron/gpsimd
XTENSA_SYSTEM=$G/tools/ncore2gp/config XTENSA_CORE=ncore2gp \
LM_LICENSE_FILE=$G/tools/licenses/amzn_vq7_us_582883.out $G/bin/xt-clang --version
# XtensaTools-14.09 clang version 10.0.1
```

### Step 3: Clone the repository and run the examples

```bash
cd ~
git clone https://github.com/OswaldHe/neuron-gpsimd-tool.git
cd neuron-gpsimd-tool/toolchain
./run_examples.sh vadd_i32      # one example, about 12 s
./run_examples.sh               # every example (about 4 min); prints a PASS/FAIL table
```

Each example compiles its NKI kernel, builds the operator library, injects it into the NEFF, and runs the
result on NeuronCore 0. The outputs must match a numpy reference bit for bit, and every GPSIMD core must
leave an execution receipt. A working setup ends with `passed N, failed 0`. The first run also compiles the
runner (`toolchain/runner/gpsimd_run.cpp`) against libnrt.

To build and run one example by hand:

```bash
bin/gpsimd-build examples/vadd_i32                              # -> examples/vadd_i32/build/final.neff
bin/gpsimd-run-example examples/vadd_i32/build --receipts --logs
```

To write your own operator, see `toolchain/README.md` (the `gp.call` interface, the device API and the
restrictions) and copy one of `toolchain/examples/*/`.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `NEURON_VENV` | `/opt/aws_neuronx_venv_torch_neuronx_2_14_0` | venv with `nki` and `neuronx-cc` (used by `gpsimd-build`) |
| `GPSIMD_TOOLS` | `/opt/aws/neuron/gpsimd` | install location of `aws-neuronx-gpsimd-tools` |
| `NEURON_RT_VISIBLE_CORES` | `0` | NeuronCore that runs the NEFF |
| `NEURON_RT_EXEC_TIMEOUT` | `30` | seconds before the runtime reports a hung execution |
| `SEED` | `1` | seed of the example's random inputs |

## 2. Run the nkiopt cases and reproduce the results

### How a case runs

Each case directory, `nkiopt/cases/<case>/`, holds a `case.py` and an `op.cpp`. `case.py` defines a
pure-NKI `baseline_kernel`, a `gpsimd_kernel` that replaces one step with `gp.call`, the inputs and a
reference. `op.cpp` is the GPSIMD operator. For each version, `bench.py` performs four steps:

1. It compiles the version (and, for the GPSIMD version, runs analyze, `gpsimd-cc` and inject).
2. It checks the outputs against the reference. Most cases must match bit for bit. The GPSIMD version must
   also leave a receipt on every GPSIMD core.
3. It executes the NEFF 5 times and profiles the last (warm) execution, 5 times in total.
4. It reads `total_exec_time` from each profile with `neuron-explorer` and reports the median.

`bench.py` imports NKI, so it runs in the venv:

```bash
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
cd ~/neuron-gpsimd-tool/nkiopt
python bench.py cases/argsort --set N=1024
#   ... (correctness and one line per profiled repeat for each version), then:
#   == argsort [N1024]: baseline 308.8 us -> GPSIMD 47.1 us  (6.56x)
```

`--set KEY=VAL` sets a problem-size variable that `case.py` reads (listed in its docstring). The results go
to `cases/<case>/build_<tag>/results.json`, together with the NEFFs, the operator library and the staged
inputs. Other options: `--repeats R`, `--iters N`, `--only baseline|gpsimd`, `--no-build` (rerun existing
NEFFs) and `--seed S`.

Medians usually vary by less than 1% between captures. A rerun on another day can differ by a few percent
(for example, the decode hashing has measured 47.9 to 48.5 us).

### Paper Table 4: Engram in DeepSeek-V4.1-Flash

The baseline is the hand-optimized NKI code of the vLLM Neuron DeepSeek-V4.1-Flash model. The DeepSeek
cases carry unmodified copies of the helper modules they use (`cases/ds41_*/ds41?/`; origin commit in
`ds41k/SOURCE`). All runs use one NeuronCore.

```bash
# n-gram hashing (both Engram layers' row ids): decode = 10 sequences x 6 rows (T = 60), prefill = 1 x T
python bench.py cases/ds41_ngram  --set B=10 --set L=6    --tag hash_t60
python bench.py cases/ds41_ngram  --set B=1  --set L=1024 --tag hash_t1024
python bench.py cases/ds41_ngram  --set B=1  --set L=2048 --tag hash_t2048
# table retrieval alone (STAGE=gather: the stage that the GPSIMD operator changes)
python bench.py cases/ds41_engram --set T=60   --set STAGE=gather --tag gather_t60
python bench.py cases/ds41_engram --set T=1024 --set STAGE=gather --tag gather_t1024
python bench.py cases/ds41_engram --set T=2048 --set STAGE=gather --tag gather_t2048
# one Engram layer: retrieval + fp8 attention GEMM + gate
python bench.py cases/ds41_engram --set T=60   --tag layer_t60
python bench.py cases/ds41_engram --set T=1024 --tag layer_t1024
python bench.py cases/ds41_engram --set T=2048 --tag layer_t2048
```

Each command takes about 3 to 7 minutes. Each `ds41_engram` run stages the rank's 3.1 GB table shard for both
versions and leaves about 6 GB in its build directory. Therefore, the nine runs need about 40 GB of free disk.

Expected results (median `total_exec_time`):

| Computation | Size | Baseline | GPSIMD + NKI | Speedup |
|---|---|---:|---:|---:|
| n-gram hashing | decode, T = 60 | 86.95 us | 48.45 us | 1.79x |
| | prefill, T = 1024 | 531.1 us | 449.7 us | 1.18x |
| | prefill, T = 2048 | 1055.6 us | 890.6 us | 1.19x |
| table retrieval | decode, T = 60 | 62.1 us | 24.7 us | 2.51x |
| | prefill, T = 1024 | 506.9 us | 102.4 us | 4.95x |
| | prefill, T = 2048 | 980.5 us | 200.7 us | 4.89x |
| one Engram layer | decode, T = 60 | 697.1 us | 681.0 us | 1.02x |
| | prefill, T = 1024 | 3362.9 us | 2968.9 us | 1.13x |
| | prefill, T = 2048 | 6766.4 us | 5926.4 us | 1.14x |
| whole Engram | decode, T = 60 | 1481.2 us | 1410.5 us | 1.05x |
| | prefill, T = 1024 | 7256.9 us | 6387.5 us | 1.14x |
| | prefill, T = 2048 | 14588.4 us | 12743.4 us | 1.14x |

The whole Engram computation is the hashing plus two Engram layers (the model has Engram in layers 1 and 14).
The following snippet computes it from the runs above:

```bash
python - <<'EOF'
import json
med = lambda c, t, v: json.load(open(f"cases/{c}/build_{t}/results.json"))[v]["total_exec_time_us"]["median"]
for t in (60, 1024, 2048):
    b, g = (med("ds41_ngram", f"hash_t{t}", v) + 2 * med("ds41_engram", f"layer_t{t}", v) for v in ("baseline", "gpsimd"))
    print(f"whole Engram, T = {t}: {b:.1f} us -> {g:.1f} us ({b / g:.2f}x)")
EOF
```

### Paper Table 5: NKI Library kernels

The baselines are the NKI Library kernels as shipped in the venv (`nkilib`). BM25 has no library kernel, so
its baseline is our own pure-NKI version.

```bash
python bench.py cases/argsort --set N=1024                          # 6.6x  (expert ids)
python bench.py cases/argsort --set N=4096                          # 39.9x (expert ids)
python bench.py cases/argsort --set N=4096 --set KEYS=random        # 21.8x (random fp32 keys)
python bench.py cases/bm25                                          # 15.3x (14.0K postings a query)
python bench.py cases/bm25 --set QDF=3000,100000  --tag q3000       # 21.0x (41.6K postings)
python bench.py cases/bm25 --set QDF=10000,300000 --tag q10000      # 27.0x (137.3K postings)
python bench.py cases/scatter_add                                   # 4.45x (duplicate indices)
```

`nkiopt/README.md` lists the other cases (MoE permute and dispatch, prior index, graph ANN and calibration
probes), including those where GPSIMD does not win, with their commands.

### Profile timelines and figures

Figures 9, 10 and 11 of the paper show one profiled execution of each version. To regenerate them, keep the
last profile of a run (`KEEP_NTFF=1`), export it with `neuron-explorer`, and convert it to a timeline CSV:

```bash
KEEP_NTFF=1 python bench.py cases/ds41_ngram  --set B=10 --set L=6 --tag hash_t60
KEEP_NTFF=1 python bench.py cases/ds41_engram --set T=1024 --set STAGE=gather --tag gather_t1024
KEEP_NTFF=1 python bench.py cases/ds41_engram --set T=1024 --tag layer_t1024

export_tl() {   # export_tl <build dir> <csv name prefix>
  for v in baseline gpsimd; do
    /opt/aws/neuron/bin/neuron-explorer view -n $1/$v/final.neff -s $1/$v/profile.ntff \
        --output-format json --output-file /tmp/tl_$v.json
    python ../figs/src/extract_profile.py /tmp/tl_$v.json ../figs/data/$2_$v.csv
  done
}
export_tl cases/ds41_ngram/build_hash_t60       hash_decode_bs10     # Figure 9
export_tl cases/ds41_engram/build_gather_t1024  gather_t1024         # Figure 10
export_tl cases/ds41_engram/build_layer_t1024   readout_t1024        # Figure 11
gzip -f ../figs/data/gather_t1024_*.csv ../figs/data/readout_t1024_*.csv   # make_figs.py prefers the .gz files

python ../figs/src/make_figs.py                                      # writes figs/*.png
```

The bar charts take their numbers from `nkiopt/README.md`, which are written into `make_figs.py`. To rebuild
the paper, install LaTeX once and run `make`:

```bash
sudo apt-get install -y texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended \
     texlive-fonts-extra latexmk
cd ../paper && make
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `xt-clang not found` | `aws-neuronx-gpsimd-tools` is not installed (Step 2), or `GPSIMD_TOOLS` points elsewhere. |
| `error while loading shared libraries: libtinfo.so.5` | Install the two Ubuntu 22.04 packages of Step 2. |
| `License checkout failed` | `LM_LICENSE_FILE` must point to the license file in the package; `gpsimd-cc` sets it unless you override it. |
| `nrt_execute failed: NRT_STATUS=5` (timeout) | An operator hung, for example on an unmapped address or a misaligned vector access. |
| The runtime cannot load the NEFF | Another process (for example a vLLM server) holds NeuronCore 0. Stop it before running the cases. |
| `ops.so: contains N FLIX bundle(s)` | The operator uses `float`/`double` or vector code that the trn3 cores cannot execute. Use `gp::f32` (see `toolchain/README.md`). |
| The disk fills up | Build directories hold the staged inputs. Remove them with `rm -rf nkiopt/cases/*/build*`. |
