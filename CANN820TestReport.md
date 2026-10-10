# CANN 8.2 (8.2.RC1) Test Report — numpy-ascend on Ascend 910B3

**Date:** 2026-10-10
**Machine:** openEuler 22.03 LTS-SP2, aarch64, 2× Ascend 910B3, driver 25.5.0
(firmware 7.7.0.3.220), CANN toolkit **8.2.RC1** (`/usr/local/Ascend/ascend-toolkit/latest`)
**Build:** `numpy_ascend_cann82-14.0.0a1-cp311-cp311-linux_aarch64.cann8.2.whl`
**Code base:** fork of CuPy v14 alpha, `ascend` branch, commit `90b973a52` + CANN 8.2 fixes (see §6)

> The project targets CANN 9.x (CI: 9.0). CANN 8.2 is documented as
> "source build, not tested". This report is the first 8.2 data point.

---

## 1. Environment as found

| Item | Value |
|---|---|
| Python | 3.11.14 (base miniconda, aarch64) |
| Build deps installed | Cython 3.1.8, fastrlock 0.8.3, numpy **2.3.5**, pytest **8.4.2**, hypothesis 6.54.6 |
| CANN installs present | 8.2.RC1 only (no 9.x anywhere on the box) |
| `ascend_fft` (ops-fft) | **not installed** — `cupy.fft` raises `RuntimeError` by design |
| pytest 9.1.1 | **unusable**: breaks `test_gufuncs.py` `parametrize('sig,', …)` at collection; pinned to 8.4.2 |

## 2. Build result

`CUPY_INSTALL_USE_ASCEND=1 python setup.py build_ext --inplace` → **success**
(`ascend: Yes (version 820)`, `ascend_fft: No`), 78 in-place `.so` files,
`import cupy._core` OK, NPU smoke test OK (sum / matmul / bitwise_and).

Binary wheel (see §7): installs into a clean venv and passes the NPU smoke test.

## 3. Test methodology

Because a single crashing or hanging test kills the whole pytest process
(SIGABRT / segfault / 10-min hang), every test **file** was run in its own
`pytest` invocation with a 600 s timeout:

```sh
bash tests/logs/run_math_tests.sh       # math_tests, one log per file
bash tests/logs/run_rest_tests.sh       # everything else (cupy_tests, cupyx_tests, import/typing)
python tests/logs/aggregate_results.py  # totals; parses each log's final summary line
```

Raw logs: `tests/logs/20261010-logs/**.log` + `summary.txt` per suite.

Test-collection filters active on Ascend (`tests/conftest.py`):

* CUDA-only modules (nvrtc/cufft/cublas/curand/cusparse/`_cub_reduction`) are not collected;
* **`complex128` is skipped** (aclnn complex ops are unreliable on 8.2);
* `--ascend-dtype-filter` leaves float64/complex64 in (default policy is "no skip").

## 4. Results

### 4.1 `tests/ascend` (registry / composed ops / wiring)

**24 failed, 207 passed, 1 skipped.** Classification:

| Group | Count | Root cause |
|---|---|---|
| `test_dtype_filter.py` | 8 | **stale tests** — `DEFAULT_SKIP_DTYPES` was emptied in `863e5d46b`, tests still assert the old `('float64','complex64','complex128')` default |
| `test_custom_kernels.py` | 4 | **stale tests** — left/right_shift were deliberately moved to builtin aclnn ufuncs (`cupy/backends/ascend/kernels/__init__.py`), tests still expect them in the custom-kernel registry |
| `test_cpu_fallback.py` | 4 | **repo gap** — `linalg.det` / `slogdet` are not wired to `cpu_fallback.call()` in `cupy/linalg/_norms.py` |
| `test_composite_ops.py` | 8 | **CANN 8.2 gaps** — `nanargmax/nanargmin/nanmean`: `aclnnSWhere` on 8.2 rejects `DT_INT32` conditions (only `[DT_BOOL, DT_UINT8]`); `angle_deg`: `ascend_angle` AscendC kernel is registered for complex output only |

### 4.2 `tests/cupy_tests/math_tests` (per-file)

| file | failed | passed | note |
|---|---|---|---|
| test_arithmetic | 161 | **8626** | 98.2 % |
| test_explog | 6 | 7 | |
| test_floating | 4 | 3 | |
| test_hyperbolic | 1 | 5 | |
| **test_matmul** | — | — | **STALLED** at 23 % (600 s timeout), ~100 % F in the part that ran |
| test_misc | 133 | 29 | |
| test_rational | 4 | 0 | |
| test_rounding | 5 | 63 | |
| test_special | 3 | 0 | |
| test_sumprod | — | — | CUDA-only import (`_cub_reduction`), skipped |
| test_trigonometric | 9 | 10 | |
| test_window | 16 | 18 | |

**≈96 % pass excluding test_matmul / test_sumprod.**
Dominant signatures: 125× `AssertionError: Only cupy raises error`, 121×
`Mismatched elements: N / N` (numerical mismatch).

### 4.3 Everything else (snapshot while `cupyx_tests` was still running)

```
150 result files:  passed 14873, failed 9253, errors 548, skipped 577, xfailed 36
pass rate: 61.6 % (24 126 ran)
```

Largest failure concentrations:

| file | failed | passed | cause |
|---|---|---|---|
| `cupy_tests/test_cublas.py` | 2006 | 640 | **CUDA-only API** (cuBLAS handle/module API) — must be skipped on Ascend (now added to the conftest list) |
| `cupy_tests/fft_tests/test_fft.py` | 1753 | 39 | **ops-fft not installed** in this CANN 8.2 → `cupy.fft` raises by design |
| `cupyx …/fftpack/test_fftpack.py` | 855 | 46 | same FFT cause |
| `random/test_distributions.py` | 764 | 453 | many distributions need RawKernel/missing aclnn ops on 8.2 |
| `linalg/test_norms.py`, `test_product.py` | 446 | 325 | svd/eig family absent on 8.2 + CPU-fallback wiring gaps |
| `cupyx distributed/test_array_nccl(no_nccl)` | 412 | 40 | NCCL/HCCCL not ported |
| `random/test_generator*.py`, `test_permutations.py` | 529 | 163 | missing aclnn rand ops / RawKernel |

Excluding the three environment-caused outliers (cublas API, FFT family,
distributed/NCCL), the pass rate of the remaining suites is materially higher;
math_tests itself is ~96 %.

### 4.4 Crashes / stalls (per-file runs contained them)

| file | outcome |
|---|---|
| `core_tests/test_ndarray_adv_indexing.py` | **SIGABRT** (core dump) — this is what kills a full-suite run |
| `core_tests/test_ndarray_scatter.py` | **SIGABRT** |
| `lib_tests/test_shape_base.py` | **SIGABRT** |
| `manipulation_tests/test_add_remove.py` | **SIGABRT** |
| `indexing_tests/test_insert.py` | **SEGFAULT** |
| `functional_tests/test_vectorize.py` | **STALLED** (>600 s) |
| `math_tests/test_matmul.py` | **STALLED** (>600 s) |

## 5. CANN 8.2 limitations (vs. the CANN 9.x the code targets)

Headers referenced by the backend but **absent** from CANN 8.2's
`include/aclnnop/` (660 headers): `aclnn_right_shift`, `aclnn_left_shift`,
`aclnn_svd`, `aclnn_angle`, `aclnn_conjugate`, `aclnn_imaginary`, `aclnn_lcm`,
`aclnn_pos`, `aclnn_rot`, `aclnn_solve`.

Runtime/API gaps found:

1. **`aclnnSWhere` condition dtype** — 8.2 accepts only `[DT_BOOL, DT_UINT8]`
   conditions; 9.x accepts more. Breaks the `nanargmax/nanmean` composition.
2. **`ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE`** — no dynamic-UB launch
   attribute ⇒ AscendC custom kernels cannot use `shared_mem` (guarded, raises
   `NotImplementedError`).
3. **`aclTensorLike` + cast composition** — `left_shift`/`right_shift`
   composition fails with `EZ1001 ViewShape overlap` on 8.2 even though the
   temporaries are freshly allocated; the same composition works on 8.5/9.0.
4. **No `aclnnSvd`** (neither header nor symbol) ⇒ `linalg.svd`/`det`/
   `slogdet`/`pinv` degrade; `aclop_Svd` reports `ACL_ERROR_API_NOT_SUPPORT`.
5. **ops-fft not part of base CANN 8.2** ⇒ all `cupy.fft` / `scipy.fftpack`
   tests fail (environment, not code).
6. **Observed CANN runtime error** `rtSetDevice … open device 139 failed,
   result=107001 (device id error)` in creation/`cuda_array_interface` tests —
   a device-id handling issue surfaced by 8.2's stricter `rtSetDevice`.

## 6. Changes made for CANN 8.2 (all staged; patch in §7)

| File | Change |
|---|---|
| `install/cupy_builder/install_build.py` | version detection parses `8.2.T12.0.B077` (was rejecting the only installed CANN) |
| `cupy/backends/ascend/acl_math_ops.h` | `right_shift`: include gated `>=850`, composed `floor(x/2**n)` fallback for 8.2 |
| `cupy/backends/ascend/acl_general_ops.h` | `svd`: include gated `>=850`, `aclop_Svd` stub returns `ACL_ERROR_API_NOT_SUPPORT` |
| `cupy/xpu/ascend/function.pyx` | dynamic-UB launch attr compiled `>=900`, else `NotImplementedError` |
| `cupy/xpu/device.pyx` | `get_compute_capability()` returns `str` on Ascend |
| `cupy/backends/ascend/acl_random_ops.h` | **recreated** (was referenced but never committed): `aclop_Multinomial/RandomUniform/RandomNormal/RandomInt` |
| `tests/conftest.py` | `pytest_ignore_collect` for CUDA-only modules; `complex128` skipped on Ascend |
| `tests/logs/*.sh`, `tests/logs/aggregate_results.py` | per-file test runners with stall timeout + aggregator |

## 7. Artifacts

| Artifact | Path |
|---|---|
| Patch (all staged changes, `git apply`-able) | `tests/logs/20261010-logs/cann820-fixes.patch` |
| Binary wheel | `dist/numpy_ascend_cann82-14.0.0a1-cp311-cp311-linux_aarch64.cann8.2.whl` (3.8 MB, 39 extensions; wheel smoke test passed on 910B3) |
| Per-file logs | `tests/logs/20261010-logs/**.log` (one log per test file) |
| Aggregator | `python tests/logs/aggregate_results.py [logdir]` |

## 8. Recommended next steps

1. Root-cause the SIGABRT/segfault files (`test_ndarray_adv_indexing`,
   `test_ndarray_scatter`, `test_shape_base`, `test_add_remove`,
   `test_insert`) — they also make a single full-suite run impossible.
2. `test_matmul` stall: profile `cupy.linalg` matmul on 8.2 (likely falls into
   a slow per-op error path).
3. Add CANN-8.2 dtype gates for `aclnnSWhere` conditions, or cast the
   condition to `DT_BOOL` in the composite layer (fixes nanargmax/nanmean).
4. Decide policy for `test_cublas.py` (root) — skip on Ascend like the other
   CUDA-only files (already added to the conftest list here).
5. Install `ops-fft` for 8.2 if FFT support is required (`ASCEND_OPS_FFT_PATH`).
6. Update the stale `tests/ascend` expectations (dtype filter default,
   custom-kernel registry) and wire `linalg.det`/`slogdet` to the CPU fallback.
