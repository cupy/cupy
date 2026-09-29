<div align="center">

### numpy for Ascend NPU: fork of CuPy, backend-extended, API-compatible

By Qingfeng Xia

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![CANN](https://img.shields.io/badge/CANN-9.0%20%7C%209.5-orange.svg)]()
[![Platform](https://img.shields.io/badge/platform-Ascend%20910B-green.svg)]()
[![Version](https://img.shields.io/badge/version-0.1.0-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/pytest%20pass%20rate-90%25-orange)]()

[API Docs](https://numpy.org/doc/2.4/reference/index.html) | [Developer Docs](docs/ascend/DeveloperNotes.md) | [Benchmark](tools/benchmark.py) | [Architecture](docs/image/numpy-xpu-architecture.jpg) | [Issues](https://github.com/qingfengxia/numpy-ascend/issues)

</div>

**numpy-ascend** brings the CuPy ndarray API to Huawei Ascend NPUs. It is a fork
of CuPy (v14 lineage, NumPy 2.x) with the CUDA backend replaced by an Ascend
backend built on CANN's `aclnn` operator library, so that existing NumPy/CuPy
code and the wider CuPy/SciPy ecosystem run on the NPU with `import cupy`.

## Features

1. Default dtype `float32` (same as CuPy on GPU) — 10x–200x NPU acceleration for
   math workloads.
2. `float64` and `complex128` are supported, with **three processing modes**
   (see below) because the NPU has no native double-precision throughput and
   some `aclnn` ops reject `DOUBLE`/`COMPLEX128` outright.
3. `int64` and `int32` have hardware acceleration for add/subtract/multiply/divide.
4. API-compatible with CuPy (GPU) and mostly compatible with NumPy (CPU); works
   with the CuPy and SciPy ecosystem.

### float64 / complex128 processing modes

Selected via the environment variable `CUPY_ASCEND_FLOAT64_MODE` (read once at
first use; legacy switch `CUPY_ASCEND_ENABLE_FLOAT64_TO_FLOAT32=1` is equivalent
to `float32`):

| Mode | Behavior | Precision | Speed |
|---|---|---|---|
| `float32` (**default**) | Demote `float64`/`complex128` operands to `float32`/`complex64` at the dispatch layer, compute on NPU, cast results back. | loses ~9 decimal digits per operation | fast (NPU) |
| `cpu` | Whole-op interception: D2H → NumPy in true double precision → H2D. Works for ufuncs, reductions and most general ops (unsupported ones fail loudly instead of silently). | exact (matches NumPy) | slow (PCIe transfers per op) |
| `off` | No demotion, no fallback: ops that cannot run in double precision raise an error. | exact or error | — |

```sh
export CUPY_ASCEND_FLOAT64_MODE=cpu     # exact double precision, host-computed
export CUPY_ASCEND_FLOAT64_MODE=float32 # default: NPU speed, demoted precision
export CUPY_ASCEND_FLOAT64_MODE=off     # never silently lose precision
```

**Why the default `float32` mode can be wrong for you** — error accumulation:

- `float32` carries a 24-bit mantissa: ~7 significant decimal digits
  (`float64`: ~16). Each arithmetic op rounds to the nearest representable
  value, and the rounding errors **accumulate across the computation**.
- Reductions are the worst case: summing `N` values has worst-case relative
  error ~`N·eps` — summing 10^8 random `float64` samples in `float32` mode can
  drift by 1e-2 relative, versus ~1e-16 in `cpu` mode. Long chains of
  elementwise ops (iterative solvers, `cumsum`, normalization) compound the
  same way; values above 2^24 (≈1.7e7) lose integer precision entirely.
- `complex128` demotes to `complex64`, so both real and imaginary parts share
  the same 24-bit mantissa — the relative error applies to magnitude and phase
  alike.
- Rule of thumb: for ML training/inference and image data, `float32` mode is
  the intended trade-off. For numerical verification, financial/scientific
  accumulation (large sums, variance over big arrays, eigen/solvers where the
  residual matters), use `cpu` mode — it is exact and easy to switch per
  process; or keep `float64` data in NumPy and move only the `float32`-hot
  path to the NPU.

## Support Matrix

**OS: Linux only.** Any Linux distribution should work provided the C++
runtime (`libstdc++`, glibc >= 2.17) is new enough; CI is done on x86_64 Linux.

| CANN | x86_64 | aarch64 | Notes |
|---|---|---|---|
| 9.5 | source | source | **not tested**; may work, no CI coverage |
| 9.0 | binary wheel + source | source | primary test target (Ascend 910B) |
| 8.5 | binary wheel + source | source | primary test target (Ascend 910B) |
| 8.2 | source build | source build | **not tested**; may work, no CI coverage |

| Python | Binary wheel | Source build |
|---|---|---|
| 3.10 | yes | yes |
| 3.11 | yes | yes |
| 3.12 | no package yet | yes |

- Binary wheels: see the [Releases page](https://github.com/qingfengxia/numpy-ascend/releases).
  Wheel naming: `numpy_ascend_cann<XY>-<ver>-cp3XX-cp3XX-manylinux_<glibc>_<arch>.whl`
  — pick the wheel matching your interpreter (`cp311` = Python 3.11), CPU
  architecture and CANN release train.
- The wheel does **not** bundle the CANN SDK: the target machine needs a CANN
  toolkit of the same release train plus `set_env.sh`; see
  [docs/ascend/Package.md](docs/ascend/Package.md) for the hard runtime
  prerequisites (libstdc++ version, optional ops-fft, etc.).
- Only one package providing `cupy` may be installed per environment: never
  co-install this with upstream `cupy` / `cupy-cudaXX` or a different CANN
  train — they would overwrite each other's files.

## Installation

### 1. From a binary wheel (recommended)

```sh
pip install numpy_ascend_cann90-...whl          # pick from the Releases page

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

python -c "import cupy; print(cupy.__version__, cupy.backends.ascend.check_cann_version())"
```

### 2. From source (new platform / CANN version, or development)

```sh
git clone git@github.com:qingfengxia/numpy-ascend.git
cd numpy-ascend && git checkout ascend      # the working branch is ascend, not main
export CUPY_INSTALL_USE_ASCEND=1
python setup.py build_ext --inplace         # incremental in-place build (dev)
python -c "import cupy._core"               # L2/L3 check: link + op registration OK
python -m build --wheel                     # optional wheel (cannX.Y tag auto-detected)
python -m build --sdist                     # optional sdist (one file, CPython/CANN agnostic)
```

Environment setup (CANN 8.2/8.5/9.0 installation and switching, development on
machines **without** an NPU, porting to new platforms, troubleshooting) is
documented in [DeveloperNotes.md](docs/ascend/DeveloperNotes.md) §1–§3; wheel
and sdist packaging strategy is in
[docs/ascend/Package.md](docs/ascend/Package.md).

### 3. Verify

```sh
pytest tests/ascend -q                     # no NPU needed: registry / composed ops / kernel parsing
pytest tests/cupy_tests -q                 # needs NPU; unsupported dtypes skipped by default
pytest tests/cupy_tests -q --ascend-dtype-filter=off   # full dtype matrix (shows real failures)
python tools/benchmark.py --list           # no NPU needed: print the op x dtype matrix
python tools/benchmark.py --csv result.csv # needs NPU

# minimal numerical check (needs NPU)
python -c "
import numpy as np, cupy as cp
x = np.random.rand(1000).astype(np.float32)
assert np.allclose(cp.asnumpy(cp.asarray(x).sum()), x.sum(), rtol=1e-5); print('ok')
"
```

Verification levels: **L1** Cython generates `.cpp` → **L2** links → **L3**
`import cupy` + op-registry audit → **L4** numerical results on real hardware.
A machine without an NPU can only reach **L3**; **L4 must run on a 910B**.

## Project Status

### API coverage

See [Progress.md](Progress.md) for the full snapshot:

- **Python Array API standard: > 98% covered** (2 gaps: `i0` — NumPy itself
  recommends `scipy.special.i0` — and `nextafter`).
- CuPy top-level API: > 98% covered.
- Operator-level facts are auto-generated by
  [tools/scan_ops.py](tools/scan_ops.py) into [tools/cst_db.md](tools/cst_db.md).

### Test results (on Ascend 910B)

| Suite | Pass rate |
|---|---|
| `tests/cupy_tests/math_tests` (math_test) | **100%** |
| full `tests/cupy_tests` + `tests/ascend` | **90%** |

Failures concentrate in the documented limitation areas below, not in the math kernels.

### Completed

1. Custom kernels (AscendC, and Triton-Ascend in Python).
2. All major CuPy features: custom kernels, profiler, stream/device management —
   except operator fusion (use a Triton kernel instead).
3. scipy ecosystem support via array_api, see [scipy benchmark with numpy-ascend](tools/)

### Limitations

1. `uint64` is not supported (`int64` is used, with hardware acceleration for add/multiply).
2. Operator fusion.
3. Sparse array/matrix and some `cupy.random` distributions are supported only partially (low priority).
4. Multi-NPU is not ported/tested yet.

## Tools

| Tool | Purpose |
|---|---|
| [`tools/benchmark.py`](tools/benchmark.py) | CLI benchmark: NumPy (CPU) vs numpy-ascend (NPU) speedup over an op × dtype matrix (unary/binary/reduction/matmul/manipulation/sort). `--list` prints the matrix without an NPU; `--category`, `--dtype`, `--repeat`, `--csv`, `--strict` filter and report. Exit code 2/3 signal import/device errors. |
| [`tools/numpy_ascend_migration_helper/`](tools/numpy_ascend_migration_helper/) | Static migration analyzer for NumPy/CuPy source: reports unsupported APIs/dtypes, semantic differences and CPU-fallback candidates (AST-based, not string matching); `--fix` applies AUTO_SAFE rewrites, `--fail-on-error` gates CI. See its [README.md](tools/numpy_ascend_migration_helper/README.md). |
| [`tools/scan_ops.py`](tools/scan_ops.py) | Scans the built backend and generates the operator/coverage database ([tools/cst_db.md](tools/cst_db.md)). |

## Examples

### Python Array API standard

<https://github.com/data-apis/array-api>

```python
import cupy as cp
import cupy.array_api as cpx   # CuPy's Array API namespace

x_gpu = cpx.asarray([1, 2, 3, 4], device='cuda')  # device slot, NPU here
y_gpu = cpx.reshape(x_gpu, (2, 2))
z_gpu = cpx.matmul(y_gpu, y_gpu)
```

If you are writing new data-processing code and do not need NumPy/CuPy
compatibility, `torch_npu`'s array API (`torch._numpy`) is an alternative.

## Documentation Index

| Document | Content |
|---|---|
| [Progress.md](Progress.md) | Status snapshot: coverage, operator counts, milestones, remaining gaps |
| [Memory.md](Memory.md) | Developer handbook: command cheat-sheet, architecture key points, verification levels, pitfalls |
| [DeveloperNotes.md](docs/ascend/DeveloperNotes.md) | Environment setup (incl. no-NPU development), CANN 8.2/8.5/9.0 install, conditional compilation, FFT |
| [docs/ascend/Package.md](docs/ascend/Package.md) | Wheel packaging (cann tag), RPATH strategy, runtime hard prerequisites |
| [install/README.md](install/README.md) | Build system architecture (features / backends) and op registration flow |
| [docs/ascend/](docs/ascend/) | Custom AscendC kernels, FFT, matmul, code reviews, NumPy/CuPy/PyTorch API diffs |
| [tools/cst_db.md](tools/cst_db.md) | Auto-generated operator/coverage database (`python tools/scan_ops.py`) |
| [Roadmap.md](Roadmap.md) · [TODO.md](TODO.md) | Roadmap and task list |

## Contributing

Issues and PRs are welcome at
<https://github.com/qingfengxia/numpy-ascend>. For a code change: rebuild with
`CUPY_INSTALL_USE_ASCEND=1`, verify at least to L3 (`pytest tests/ascend -q`),
and state honestly which verification level was reached (see
[Memory.md](Memory.md)). Commit messages use the `ASCEND [AI]:` prefix on the
`ascend` branch.

## License

MIT — see [LICENSE](LICENSE). This project forks CuPy; upstream CuPy is
MIT-licensed and its copyright notice is preserved in the source tree.

## 中文文档

简体中文说明见 [Readme_ZH.md](Readme_ZH.md)。
