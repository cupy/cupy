# Progress

Coverage baseline: **Array API standard**, 130 public functions
(`cupy/array_api/`). Registry snapshot taken from
`cupy/backends/ascend/api/acl_utils.pyx` (`register_acl_ufunc("ascend_*", ...)`).

---

## Done

### 1.1 Progress
Year 2025
- Oct 12: MVP for add, cos, matmul, benchmark 10-100X acceleration

- Oct 23: benchmark.py 经过xpu重构后 (重构代码在 xpu branch), NPU测试可以运行
  backends -> cupy.backends,  git mv + _features.py
  disable _preflight, so ignore Cutensor submodule

- Nov 08: reduction op such as `sum()` is working,
    90% math ops ACLOP supported has been added into numpy-ascend
    UnitTest: `pytest tests/cupy_tests/logic_tests/test_truth.py `

- Nov 15: concatenate(), clip(), copy(), non-math/irregular ops initially supported
    + but `array()` seems not working properly 
    > reason is async_copy, two arrays created without print the first will have the same value as the second array
    + scalar 转化 not working, 可能是exp scalar op 没有注册   DONE

- Dec 06: 
    + creation apis:  `arrange()` added, but test failed
    + `concatenate` test passed
- Dec 13:  sorting API
    + `sort()/argosrt()` added, but no `partition()` alcop, 
    + sort also depending on `rollaxis()`
    + fill_kernel() -> aclnnop_FillScalar  GeneralOp 类型

- BitwiseAddScalar op register, _kernel.pyx need update (done)
- reduction kernel, replaced by aclnnop

### 1.2 Session 2026-09-12 (today)

Two commits: `78abec5b8` (math ops + dtype conversion) and `68c8738a5`
(batched matmul + dot).

**Newly registered ufuncs (38 call sites, 33 unique names)** — all in
`acl_utils.pyx`:

```
arccos arcsin arctan arctan2 arccosh arcsinh arctanh      (inverse/hyperbolic trig)
inplace_arccos inplace_arcsin inplace_arctan inplace_arccosh
inplace_arcsinh inplace_arctanh                           (inplace variants)
cbrt exp2 sqrt inplace_exp2 inplace_sqrt                  (power/root)
fabs copysign hypot lcm power float_power                 (binary/composite)
fmax fmin positive invert                                 (elementwise)
rint trunc round                                          (rounding)
inplace_rint inplace_trunc                                (inplace rounding)
nan_to_num nan_to_num_ cast permute roll real             (general/irregular)
```

**Bug fixes (4 wrong ufunc names)**: `ascend_acos`→`ascend_arccos`,
`ascend_asin`→`ascend_arcsin`, `ascend_atan`→`ascend_arctan`,
`ascend_tan2`→`ascend_arctan2`. These were silently registering under names
that no `cupy_*` ufunc ever looked up.

**`argsort` dispatch fix**: `aclop_Sort` was registered for both `sort` and
`argsort`; `argsort` now uses `aclop_Argsort`.

**New aclop wrappers** (10): `aclop_Cbrt`, `aclop_Copysign`, `aclop_Fabs`,
`aclop_Hypot`, `aclop_InplaceRint`, `aclop_NanToNum`, `aclop_Permute`,
`aclop_Rint`, `aclop_Roll`, `aclop_Round`. New aclnn includes:
`aclnn_real`, `aclnn_roll`, `aclnn_round`, `aclnn_s_where`, `aclnn_trunc`.

Notable ones are **composite**: `aclop_Hypot` = x²+y²→sqrt (uses the ternary
`aclnnAdd(self, other, alpha, out)` form); `aclop_Copysign` = abs + neg +
`aclnnGeScalar` + `aclnnSWhere`; `aclop_Cbrt` = `aclnnPowTensorScalar` with 1/3.

**Errors fixed during the session**

1. `if constexpr` on a runtime `dtype` in `acl_scalar_arg.h` → plain `if`.
2. `aclnnAddGetWorkspaceSize(self, other, alpha, out, ...)` needs an
   `aclScalar` for `alpha` → routed through `aclTernaryOpRun`.
3. `aclop_lcm` vs `aclop_Lcm` naming mismatch.
4. `aclCreateScalar(&zero, ACL_FLOAT)` needed explicitly in `aclop_Copysign`.
5. Added unsigned-scalar support, complex (`kind == 'c'/'C'`), and `float16`
   via `uint16_t` in `acl_utils.pyx`.
6. Strides are now passed in **element units** (`_strides[i] // item_size`).
7. Fortran-ordered arrays now raise `NotImplementedError` instead of
   silently producing wrong results.
8. Added `cupy_acl_tensor_owners` registry — `aclDestroyTensor` frees only the
   tensor *metadata*, not the data, so the source ndarray must be pinned.
9. `numpy_to_acl_dtype` → `numpy_dtype_to_acl_dtype` (rename requested in
   `plan.md` §D-3), call sites updated in `_dtype.pyx`.
10. Added `py_list_acl_ufuncs()` — introspection hook that enumerates the whole
    `_builtin_operators` map, enabling device-free (no-NPU) registry testing.

**Linalg** (`_routines_linalg.pyx`)

- `dot()`: 1-D·1-D (`aclnnDot`, 0-D out), 2-D@2-D, 1-D·n-D, n-D·1-D;
  otherwise explicit `NotImplementedError`.
- `matmul()`: 2-D@2-D fast path, else new
  `_ascend_batched_matmul()` which loops `_ascend_matmul` over batch dims.

**Sorting** (`_routines_sorting.pyx`, unstaged)

- `_ndarray_sort` / `_ndarray_argsort` now pass `-1` as the sort axis (the
  array is `rollaxis`'d so the target axis is always last); rolls back only
  when `axis != ndim-1`.
- `argsort` allocates `numpy.int64` (`aclnnArgsort` requires int64) and
  returns `idx_view.copy()`.
- `partition` / `argpartition` fall back to sort/argsort.

**Packaging (this session)**

- `embed_sdk_in_rpath = False` for Ascend → no CANN absolute path in any `.so`.
- `DT_RUNPATH` (modern default) instead of `DT_RPATH`, so `LD_LIBRARY_PATH`
  and `source set_env.sh` take effect. `CUPY_INSTALL_LEGACY_RPATH=1` restores
  the old behaviour. `CUPY_INSTALL_NO_RPATH=1` disables RPATH entirely.
- Wheel platform tag from the CANN version: `cann8.5` / `cann9.0`
  (`get_wheel_platform_tag()`), giving
  `…manylinux_2_17_x86_64.cann8.5.whl`.
- Build metadata written to `cupy/.data/_wheel.json`, re-validated at import by
  `cupy/backends/ascend/__init__.py:check_cann_version()` (release-train
  comparison, warns only, `CUPY_ASCEND_SKIP_VERSION_CHECK=1` to silence).
- See `Package.md` for the full write-up and `install/README.md` §4.4 for the
  build-system view.

### 1.2b Session 2026-09-13 (Level A + B 实施)

**新增 aclnn 包装（12 个）** — 全部为 CANN 8.5.1 已有算子：

| 包装 | 对应 API | 备注 |
|---|---|---|
| `aclop_Unique2` | `unique_all/counts/inverse/values` | 一个包装覆盖 4 个 Array API 函数 |
| `aclop_Trace` / `aclop_Tril` / `aclop_Triu` | `trace`/`tril`/`triu` | Array API linalg + creation |
| `aclop_Qr` / `aclop_Svd` / `aclop_Inverse` | `qr`/`svd`/`svdvals`/`inv` | Array API linalg |
| `aclop_Aminmax` | `ptp` | 三个变体 `aminmax`/`_all`/`_dim` |
| `aclop_Histc` | `histogram` | |
| `aclop_Complex` | `complex(real, imag)` | |
| `aclop_IsNan` | `isnan` | **组合实现**：`x != x`（CANN 无 `aclnnIsNan`） |
| `aclop_RightShift` | `right_shift` | CANN 只有 `aclnnRightShift`，无左移 |
| `aclop_NanMin` / `aclop_NanMax` | `nanmin`/`nanmax` | **组合**：`nan_to_num(±inf)` + `min`/`max` |

**linalg 打通（`cupy/linalg`）**：`qr()`/`svd()`/`inv()` 原本走 `cupyx.lapack`
（Ascend 上是 stub），现在在 `cupy/linalg/_decomposition.py` 与 `_solve.py`
里加了 Ascend 分支，调用新增的 `_ascend_qr` / `_ascend_svd` / `_ascend_inv`
（定义在 `cupy/_ascend/_core/_routines_linalg.pyx`，通过
`_routines_linalg.pxd` 暴露）。

**修复 2 个静默失效的 dispatch bug**：

1. **`isnan`/`isfinite`/`isinf` 名字错误**：`cupy/_logic/content.py` 生成的 ufunc
   名为 `cupy_isfinite` → 派发查 `ascend_isfinite`，但注册的是
   `ascend_is_finite`（多了下划线），**三个函数全部从未派发成功**。
   现在两个拼法都注册。
2. **位运算前缀反转**：`cupy/_core/_routines_binary.pyx` 的 `OP_PREFIX` 逻辑写反了
   ——Ascend 构建（`CUPY_CANN_VERSION > 0`）用了 `"cupy_"` 前缀，导致
   `bitwise_and`/`left_shift`/`right_shift` 等**全部无法派发**。已修正为统一
   `"cupy_"`（由 Ascend 派发器自行改写为 `ascend_`）。

**核实并纠正 plan.md 的 2 处误判**：
- `unique_all/counts/inverse/values` 与 `ptp`、`tril`、`triu`、`trace`
  **本来就能用** —— 它们是纯 Python 组合实现（基于已工作的
  `sort`/`argsort`/`cumsum`/`where`/`diagonal`+`sum`），不需要 aclnn 包装。
  （`unique2`/`tril`/`triu`/`trace`/`aminmax` 包装仍加上了，可用于加速，但
  不是覆盖率瓶颈。）
- `aclnn_shift_left` 确实不存在，但 **`aclnn_right_shift` 存在**（前一轮误判为
  "无任何 shift 算子"）→ `right_shift` 已实现，`left_shift` 仍需组合。

### 1.3 Registry totals

| Metric | Value |
|---|---|
| `register_acl_ufunc` call sites | **183** |
| unique registered names | **165** |
| `aclop_*` C++ wrappers | **80** |
| aclnn headers included | **147** |
| aclnn headers available in CANN 8.5.1 | **756** |

### 1.4 Array API coverage (measured, 2026-09-12)

Counted against the 129 public functions in `cupy/array_api/` (private helpers
such as `_check_valid_dtype` / `_solve` are excluded), treating a function as
covered when it is reachable through a registered ascend ufunc or a host-side
(Cython/Python) implementation.

| Category | Covered / Total | Missing |
|---|---|---|
| creation | 10 / 17 | `empty_like`, `full_like`, `ones_like`, `zeros_like`, `meshgrid`, `from_dlpack` |
| elementwise | 55 / 56 | `bitwise_left_shift` |
| statistical | 7 / 7 | — |
| manipulation | 8 / 8 | — |
| searching | 4 / 4 | — |
| set | 4 / 4 | — |
| sorting | 2 / 2 | — |
| indexing | 1 / 1 | — |
| data_type | 7 / 7 | — |
| utility | 2 / 2 | — |
| linalg | 17 / 21 | `cholesky`, `det`, `eigh`, `eigvalsh` |
| **TOTAL** | **117 / 129 = 90.7 %** | |

变化（相对 2026-09-12 的 86.8 %）：
- elementwise 54→55：`right_shift` 补齐（`aclnnRightShift` 存在）
- linalg 13→17：`qr`/`svd`/`svdvals`/`trace` 打通
- 剩余 4 个 linalg（`cholesky`/`det`/`eigh`/`eigvalsh`）**CANN 8.5.1 无算子**，
  需自研算法或等上游；`bitwise_left_shift` 同理。

> The 7 `*_like` creation functions are trivial aliases of `empty`/`full` +
> `broadcast_to` and are host-side (no aclnn needed) — the real device-side
> coverage is higher still.

> Caveat: "covered" means *a code path exists*. For the ~33 ufuncs added today
> the evidence level is **L3 (compiles + registers + imports)** only — there is
> no NPU on this machine, so **L4 numerical correctness is unverified**
> (`plan.md` §1.4).

### 1.5 Runtime/tooling

- Wheel importer verified: `import cupy` → `14.0.0a1`, 0 SDK RPATH leaks across
  all 37 `cpython-311` `.so`.
- `check_cann_version()` detects `8.5.1`; warns on a simulated 9.0 wheel;
  silent on a match.
- No NPU on this host: device ops raise `EL0003 Invalid_Argument`.
  `LD_PRELOAD=<cann>/opp/…/liboptiling.so` is required to import (CANN 8.5.1
  `libop_common.so` undefined-symbol defect).

---

## 2. Short-term TODO

1. creation/manipulation/indexing/linalg ops

2. statistics ops: passing string arg, it has issue
    it may need CANN 8.5 to construct aclScalar of string type

3. ~~matmul~~ **FIXED**: 
   - 根因: `matmul()` 的 ascend 分支调用 `_ascend_matmul(a, b, out)` 后**缺少 `return`**,
     fall-through 到 cuBLAS 专用代码块的 `a, b = b, a` 转置技巧, 又重新算了 `B @ A`,
     导致 `matmul(a,b)` 返回 `np.matmul(b,a)`。
   - 修复: ascend 分支提前 `return _ascend_matmul(a, b, out)`;
     并把下方 CUDA/cuBLAS 转置代码块包进 `IF CUPY_CANN_VERSION <= 0:` 守卫, 使其对 ascend 成为死代码。
   - 同时实现 `dot()`: 1-D·1-D -> `aclnnDot`(0-D 输出), 2-D@2-D -> `aclnnMatmul`;
     其余维度显式 `NotImplementedError`。
   - `_ascend_dot`/`_ascend_matmul` 统一用 `launch_general_func(..., [], {}, 0)`; 清理了 debug print。
   - `aclop_Matmul` 的 `math_type` 由 `uint8_t` 修正为 aclnn 签名要求的 `int8_t`。

4. concat/pad/reshape op
   numpy_to_acl_dtype ->  numpy_dtype_to_acl_dtype   **DONE (today)**

6. triton-fusion (add data adaptor API)
 or once CANN 8.5 stable released, and pyPTO will be used to write customised kernel

7. aclBlas integration

8. test build on diff OS, currently focus on Ubuntu

10. inplace op :  _kernel.pyx need update

11. **NEW**: build the two wheels (`python -m build --wheel`) — the RPATH fix
    is verified but `dist/` still only holds a stale 2025-10-07 stub wheel.

---

## 3. 核心op支持情况 ( see also Array API standard)
https://data-apis.org/array-api/latest/API_specification/index.html

### 3.1 math ops (updated 2026-09-12)

Now registered: `arccos arcsin arctan arctan2 arccosh arcsinh arctanh cbrt
exp2 sqrt fabs copysign hypot lcm power float_power fmax fmin positive invert
rint trunc round nan_to_num` (+ all inplace variants).

Remaining gaps:

+ **`bitwise_left_shift` / `bitwise_right_shift`** — the only two elementwise
  Array API functions still missing. **Verified: CANN 8.5.1 has NO shift
  operator at all** (`aclnn_shift_left.h` does not exist) → Stage-B, compose
  from `pow(2, n)` + `multiply`/`floor_divide`, or host fallback.
+ `einsum` — `aclnn_einsum.h` **is available and already included**, but no
  `aclop_` wrapper exists → Stage-A.
+ missing 数值计算: `gradient, interp, trapezoid, diff` (compose from
  subtract/divide/take)
+ missing: `frexp, modf` — **no aclnn op**; `ldexp` no aclnn op.
+ complex: `conj`/`conjugate`/`angle`/`imag` have **no aclnn op**;
  `real` (registered today) and `aclnn_complex` exist. `angle` = `atan2(imag,
  real)` composition.
+ `cupy.math_op(scalar, tensor)`: can aclop kernel broadcast deal with this?

### 3.2 indexing ops

- slicing ? working, but it does not use `Slice` aclop
- `math.scan()` is a dummy/empty func, no such aclop
- aclop has `take, put(InplacePut), slice`, but no `choose`
- **NEW**: `aclnn_index` and `aclnn_gather` exist in CANN; `aclnn_index` header
  is already included but unwrapped.

### 3.3 manipulation ops

可能有大量不兼容, 测试工作量不小
+ CUPY `reshape, split` does not need kernel, it is done in cython code on host (Reshape api)
+ ACLOP having: `roll, permute, flip, repeat` , while `repeat/rollaxis()` is written in cython, no kernel needed
  → `roll` and `permute` now have `aclop_` wrappers **DONE (today)**;
    `aclnn_repeat.h` exists but is **not included** → Stage-A
+ cupy uses `concatenate` to impl vstack, stack, hstack without using CUDA kernel
+ `_manipulation/rearange.py`  slicing is used to flip, rotate
+ `squeeze`: Removes size-one axes from the shape of an array

### 3.4 logical/bitwise ops:  
+ ACLOP misses numpy op: `_left_shift`, `_right_shift` (`aclnn_shift_left` exists)
+ `cupy_is_close` should be used as `a.isclose(b)`
+ `is_nan()`: **no aclnn op**; `isnan` = `not_equal(x, x)` composition → Stage-B

TODO   but why `aclnnEqual`has no tensor-scalar version?

### 3.5 statistics reduction ops: 
+ registered: median, var, mean, std,  bincount, histgram (histc), 主要是看nan怎么处理, 部分做了注册
  → verified: `aclop_Median` / `aclop_Std` / `aclop_Var` wrappers **do exist**
    (`acl_reduction_ops.h:73,84,90`) and `aclnn_histc.h` is included, but
    `histc` has **no** `aclop_` wrapper yet.

+ missing: average, quantile,  percentile, vecter op实现难度应该不太大
+ ptp (Range of values (maximum - minimum) along an axis.) -> Aminmax
  → `aclnn_aminmax.h` / `aclnn_aminmax_all.h` / `aclnn_aminmax_dim.h` all exist;
    `aminmax` header already included but **unwrapped** → Stage-A, cheap win

TODO: passing keyword args

### 3.6 set op
+ `is1d`
Array Std API support only:
+ unique_all
+ unique_counts
+ unique_inverse
+ unique_values
→ `aclnn_unique2.h` exists and is **included but unwrapped** → Stage-A,
  4 Array API functions for the price of one wrapper.

### 3.7 random and distribution

AsNumpy project has impl
https://gitcode.com/cann/asnumpy

### 3.8 similar ops 需要验证numpy行为是否一致
1. fmin, nanmin, min, amin
2. remainder, fmod, modf
3. rint, round, around
4. dot, matmul, mm, gemm, inner
5. fabs(real number only), abs
