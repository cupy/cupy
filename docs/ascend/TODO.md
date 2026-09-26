# TODO

## Short term todo
moved to [Progress.md](./Progress.md)

## intermediate (within 6 months)
1. templated ascend kernel JIT
2. compile customer kernel
3. impl missing numpy op for ascend
4. random
5. FFT (if CANN toolkit release ops)
6. single node multiple NPU distribution test

## Longterm (within one year)

1. multi-node multiple NPU
2. double datatype (float32, int64) emulation/wait for hardware suport of future NPU
2. sparse matrix

---

## Stage-A/B 实施进展 (2026-09-13 更新)

> 已实施并 L3 验证（编译 + 链接 + import + 注册表），**数值正确性未验证（无 NPU）**。

### 已完成的 Stage-A 包装（12 个，全在 `cupy/backends/ascend/*.h`）

| 包装 | API | 备注 |
|---|---|---|
| `aclop_Unique2` | `unique_*` ×4 | |
| `aclop_Trace` / `Tril` / `Triu` | `trace`/`tril`/`triu` | |
| `aclop_Qr` / `Svd` / `Inverse` | `qr`/`svd`/`svdvals`/`inv` | |
| `aclop_Aminmax` | `ptp` | |
| `aclop_Histc` | `histogram` | |
| `aclop_Complex` | `complex` | |
| `aclop_IsNan` | `isnan` | **组合** `x != x` |
| `aclop_RightShift` | `right_shift` | CANN 无左移算子 |
| `aclop_NanMin` / `NanMax` | `nanmin`/`nanmax` | **组合** nan_to_num + min/max |

### 已打通 linalg（`cupy/linalg/*.py` Ascend 分支）

`qr()` / `svd()` / `inv()` 原走 `cupyx.lapack`（Ascend stub），现经
`_ascend_qr`/`_ascend_svd`/`_ascend_inv`（`_ascend/_core/_routines_linalg.pyx`）
路由到 aclnn。batched（ndim>2）与 `mode='raw'` 显式 `NotImplementedError`。

### 修复 2 个"静默失效"的 dispatch bug（重要）

1. **`isnan`/`isfinite`/`isinf` 从未派发**：ufunc 名 `cupy_isfinite` → 查
   `ascend_isfinite`，但注册名是 `ascend_is_finite`。已两种拼法都注册。
2. **位运算前缀反转**：`_routines_binary.pyx` 的 `OP_PREFIX` 在 Ascend 构建下
   误用 `"cupy_"`，使 `bitwise_and`/`left_shift`/`right_shift` 等**全部无法派发**。已修正。

### 核实后的结论修正

- `unique_all/counts/inverse/values`、`ptp`、`tril`、`triu`、`trace`
  **本来就可用**（纯 Python 组合，基于已工作的 sort/argsort/where/diagonal+sum），
  不是覆盖率瓶颈 —— 前一轮 plan.md 把它们列为"必须做的 A 档"是高估了。
- `aclnn_right_shift` **存在**（前一轮误判"无任何 shift 算子"）；
  仅 `shift_left` 缺失。

### 剩余（CANN 8.5.1 确实无算子）

`cholesky` / `det` / `eigh` / `eigvalsh` / `bitwise_left_shift` /
`conj` / `angle` / `imag` / `frexp` / `modf` / `ldexp` / `i0` / `nextafter`
→ 需组合实现或等 CANN 上游；且**无 NPU 无法验证数值正确性**，建议在有 910B
的环境里逐个落地并配单测。

---

## Stage-A 算子封装机会分析 (2026-09-12)

"Stage A" = **CANN 8.5.1 已有 aclnn 算子，只需写 `aclop_` 包装 + 注册**，
无需自研算法、无需等上游。数据来源：
`ls $ASCEND_HOME_PATH/include/aclnnop/`（**756** 个头文件）与
`grep aclop_ cupy/backends/ascend/*.h`（当前 ~44 个包装）。

当前状态：aclnn 头文件已 include **129** 个（占可用量的 17%），
其中相当一部分 **include 了但没有 `aclop_` 包装、没有注册**，
属于零风险的 Stage-A 机会。

### A1. 已 include 但未包装 —— 最高优先级（零成本）

这些头文件已经在 `acl_math_ops.h` / `acl_general_ops.h` /
`acl_reduction_ops.h` 里 `#include` 了，只是没有对应的 `aclop_` 函数：

| aclnn 算子 | 对应 numpy / Array API | 备注 |
|---|---|---|
| `aclnn_einsum` | `einsum`, `vecdot`, `tensordot` | 头文件已 include；实现后可一次覆盖 3 个高层 API |
| `aclnn_aminmax` / `_all` / `_dim` | `ptp`, `min`/`max` 合并 | 三个变体都在，`ptp = max-min` |
| `aclnn_unique2` | `unique_all/counts/inverse/values` | 一个包装覆盖 **4 个** Array API 函数 |
| `aclnn_diag` | `diagonal`, `diag`, `diagflat` | |
| `aclnn_tril` / `aclnn_triu` | `tril`, `triu` | Array API linalg 函数 |
| `aclnn_trace` | `trace` | Array API linalg 函数 |
| `aclnn_index` | 高级索引 / `take_along_axis` | 头文件已 include |
| `aclnn_histc` | `histogram` | 头文件已 include，无 `aclop_` |
| `aclnn_complex` | `complex` / 从 real+imag 构造 | 头文件已 include |
| `aclnn_real` | `real` | 今天已注册 ufunc，待确认 `aclop_` 是否独立 |

### A2. CANN 有算子但尚未 include —— 低风险

| aclnn 算子 | 对应 API | 备注 |
|---|---|---|
| `aclnn_repeat` / `aclnn_repeat_interleave` | `repeat`, `tile` | CuPy 目前在 host 侧实现，可下沉到 aclnn |
| `aclnn_expand` | `broadcast_to`, `expand_dims` | |
| `aclnn_topk` | `argpartition`（真正的 topk 语义） | 可替换当前的 sort/argsort 回退 |
| `aclnn_gather` | `take`, `compress`, `take_along_axis` | 注意与已有 `aclop_Take` 区分 |
| `aclnn_isin` / `aclnn_isin_tensor_scalar` | `isin`, `in1d` | 头文件存在，未 include |
| `aclnn_searchsorted` | `searchsorted`, `digitize` | 头文件存在，未 include |
| `aclnn_unique_consecutive` | `unique(..., return_index)` 语义 | 头文件存在，未 include；补充 `unique2` |
| `aclnn_nonzero_v2` | `nonzero`, `argwhere` | 已用 v1，v2 可能更快 |

### A3. linalg 缺口 —— CANN 有算子，可大幅提升覆盖率

Array API linalg 目前 13/21（`cupy/array_api/linalg.py` 中 21 个公开函数），
缺口 8 个。**已核实** CANN 8.5.1 的头文件存在性（`ls include/aclnnop/`）：

| 缺 API | CANN 算子 | 状态 |
|---|---|---|
| `qr` | `aclnn_qr.h` ✅ | **已 include，未包装** → Stage-A |
| `trace` | `aclnn_trace.h` ✅ | **已 include，未包装** → Stage-A |
| `svd` / `svdvals` | `aclnn_svd.h` ✅ | **已 include，未包装** → Stage-A |
| `inv` | `aclnn_inverse.h` ✅ | **已 include，未包装** → Stage-A |
| `solve` | `aclnn_triangular_solve.h` ⚠️ | 无 `aclnn_solve.h`；只能覆盖三角求解，一般求解需组合 |
| `det` | ❌ **无 `aclnn_det.h`** | Stage-B：LU 分解组合，或等 CANN 上游 |
| `cholesky` | ❌ **无 `aclnn_cholesky.h`** | Stage-B 或等上游 |
| `eigh` / `eigvalsh` | ❌ **无** | Stage-B 或等上游 |

> `acl_math_ops.h:437` 里 `aclop_Det` 是**被注释掉**的（`// aclError aclop_Det(...)`），
> 印证了 CANN 8.5.1 没有 det 算子。
> `aclnn_eigh` / `aclnn_cholesky` / `aclnn_solve` / `aclnn_shift_left` 均**不存在**，
> 不要在 756 个头文件里反复找了。

### A4. 确认 CANN 无算子（不要浪费时间找）

以下 op 在 CANN 8.5.1 的 756 个头文件中**不存在**，只能走 Stage-B
（Python/Cython 组合实现）或长期等待：

| 缺 API | 替代方案 |
|---|---|
| `conj` / `conjugate` | 手动构造虚部取负：`complex(real, -imag)` |
| `angle` | `atan2(imag, real)` 组合 |
| `imag` / `imaginary` | 从 `aclnn_complex` 的反向视图或 stride 技巧取 |
| `frexp` | `log2` + `floor` + `pow` 组合 |
| `modf` | `trunc` + `subtract` 组合 |
| `ldexp` | `pow` + `multiply` 组合 |
| `isnan` | `not_equal(x, x)` |
| `bitwise_left_shift` / `bitwise_right_shift` | **无任何 shift 算子**（`aclnn_shift_left` 不存在）→ 整数域用 `pow(2, n)` + `multiply`/`floor_divide` 组合，或 host fallback |
| `hypot` | 已用组合实现（x²+y²→sqrt），无需 CANN 算子 |

### A5. 建议执行顺序

1. **`aclnn_unique2`** → 一次补齐 `unique_all/counts/inverse/values`
   （set 类别 4/4 → 直接完工，投入产出比最高）
2. **`aclnn_qr` / `aclnn_inverse` / `aclnn_trace`** → linalg 覆盖率
   （三者头文件已 include，只差包装）
3. **`aclnn_aminmax`** → `ptp`（极便宜，statistical 类锦上添花）
4. **`aclnn_einsum`** → `einsum` / `vecdot` / `tensordot`（高价值）
5. **`aclnn_topk`** → 替换 `argpartition` 的 sort 回退（性能收益）
6. `aclnn_expand` / `aclnn_repeat` → 把 host 侧实现下沉到 aclnn

> 已排除：`aclnn_shift_left`、`aclnn_det`、`aclnn_cholesky`、`aclnn_solve`、
> `aclnn_eigh` 在 CANN 8.5.1 中**不存在**，不属于 Stage-A。

---

## `ndarray.fill()` 对 strided view 静默写坏内存 (2026-09-26, TODO)

`cupy/_core/core.pyx` `fill()` 的 Ascend 分支中，非 C-连续路径对
**strided view（既非 C- 也非 F-连续）会静默写坏底层数组**：

- 复现：`a = cupy.zeros((4, 4)); a.diagonal(1).fill(1)`
  —— `diagonal(k)` 返回 strided view（stride = N+1），非 C/F 连续。
- 根因：该分支用 `numpy.full(self.shape, ...)`（**视图的 shape**）在 host
  生成数据，再 `self.data.copy_from_host_async(ptr, self.nbytes)` 从视图
  基址开始**线性连续**拷贝 —— 对角线上的值被顺序写到 base 数组开头，
  完全不按 stride 散射。
- 影响面：任何对 strided view 调用 `fill()` 的代码在 Ascend 上都会数据
  损坏（已知的受害者：`eye()`，已改走 `fill_diagonal` →
  `aclnnInplaceFillDiagonal`，见 `cupy/_creation/basic.py`）。
- 修复方向：
  1. 按 inverse-stride 散射写入；或
  2. strided view 直接抛 `NotImplementedError`（宁可报错不可写坏）；
  3. 或改用 aclnnInplaceFillTensor + 一个与视图同布局的 host 侧镜像。
- 无 NPU，修复后数值正确性需在有 910B 的环境验证。

---

## Polynomial 模块备注 (2026-09-20)

- `cupy.polynomial`（新式 polynomial API：`polynomial.py` / `polyutils.py`）
  已基本实现并从 `cupy` 顶层导出；`cupy/lib/__init__.py` 现在也导出
  `poly1d poly polyadd polyder polydiv polyfit polyint polymul polysub
  polyval roots`（对齐 `numpy.lib` 的 re-export 集）。
- 新实现 `polyder` / `polydiv` / `polyint`（`cupy/lib/_routines_poly.py`，
  纯 cupy 算子组合，镜像 `numpy/lib/_polynomial_impl.py`）。
  **没有走 CustomKernel/ElementwiseKernel 路线**：Ascend 上
  `ElementwiseKernel` 仍是 stub（见下方 T4），纯算子组合两条后端都能跑。
  `poly1d.deriv` / `poly1d.integ` 同步接通（需重新编译
  `cupy/lib/_polynomial.pyx`）。
- **Legendre**（`numpy.polynomial.legendre` 的 Legendre 类及 leg* 系列函数）
  cupy upstream 和本仓库均未实现，暂时跳过；等有需求或 aclnn 有对应支持
  再按 `polynomial.py` 的模式补。

---

## TODO (详细方案)

[CuPy在AMD ROCm平台上的数组创建问题分析与解决方案 - GitCode博客](https://blog.gitcode.com/d12d5e41c894b5e8803c5e39838f621d.html)

### 自定义算子JIT  ：编译和动态加载Kernel

不清楚CANN 和 triton-ascend  路标

ElementwiseKernel 可以做到类似 CUDA的层面,  就是有些工作量. 

### 随机数 (不紧急)

可以numpy来生成随机数, AsNumpy has impl

rand： `#include <aclnnop/aclnn_rand.h>`  有算子

```python
    HIP_random = {
        'name': 'random',
        'required': True,
        'file': [
            'cupy.random._bit_generator',
            ('cupy.random._generator_api',
             ['cupy/random/cupy_distributions.cu']),
        ],
        'include': [
            'hiprand/hiprand.h',
        ],
        'libraries': [
            # Dependency from cuRAND header files
            'amdhip64',  # was hiprtc and hip_hcc before ROCm 3.8.0
            'hiprand',
        ],
        'check_method': build.check_hip_version,
        'version_method': build.get_hip_version,
    }
```


### hccl 多卡支持 (rocm 没有做迁移)

CUPY的多卡, 本身需要测试工作量.  

HCCL和NCCL的API兼容性, 粗看很相似. 

### 稠密和稀疏矩阵求解 (cupyx.scipy)

cusparse + cusolver：  aicpu， torch-npu

[我中心在稀疏矩阵乘算子研发中取得新进展--中国科学院计算机网络信息中心](https://cnic.cas.cn/gzdt/202411/t20241120_7442650.html)

```c
#ifdef defined(CUPY_USE_ASCEND)
// not sure if CANN support solver, leave it later
//#include "ascend/cupy_ascend_solver.h"
#include "stub/cupy_cusolver.h"  // gracefully give error message

```

#### TODO:  CUBLAS_COMPUTE ,  cudaDataType 下沉到backend层次

`_numpy_to_backend_dtype()`  不同的backend实现不同的pyx文件

####  TODO: HCCL NCCL, MPI, RCCL 应该可以抽象

sparse, 等ascend缺失的暂不做中性化处理, 直接放入cuda/libs

#### split out cuda enum from backend

这些基本GPU专用, 或者不是核心必须得代码, 拆出到cuda backend去维护
cupy.backends.backend._runtime.pyx 

```python
IF CUPY_CANN_VERSION <= 0:
    # Provide access to constants from Python.
    # TODO(kmaehashi): Deprecate aliases above so that we can just do:
    # from cupy.backends.cuda.api._runtime_enum import *
    # from cupy.backends.cuda.api._device_prop import *
    def _export_enum():
        import sys
        import cupy.backends.backend.api._runtime_enum as _runtime_enum
        this = sys.modules[__name__]
        for key in dir(_runtime_enum):
            if not key.startswith('_'):
                setattr(this, key, getattr(_runtime_enum, key))

    _export_enum()
ELSE:
    # in  the future, add ascend cann enum here
```

### cuTensor： 昇腾CANN对应？

```
#ifdef CUPY_USE_HIP

// Since ROCm/HIP does not have cuTENSOR, we simply include the stubs here
// to avoid code dup.
#include "stub/cupy_cutensor.h"
```

---

## 包级 API 缺口盘点：`cupy/__init__.py` vs `cupy/__init__cupy_original.py`

> 2026-09-18。方法：对两个文件做 **AST 顶层绑定对比**（`ImportFrom` 名字、
> 赋值目标、`def`/`class`），比运行时 `hasattr` 更准确——不受"导入失败被静默
> 跳过"影响。原始 upstream 版本：cupy v14 alpha（2025-10-23）。

### 总量

> **2026-09-18 更新：T1 + T2 已全部导出并验证**（AST 复盘缺失 91 → **16**，
> 运行时 `hasattr` 全部可达）。同时移除了两个非 cupy 包级 API 的导出：
> `matrix_transpose`、`vecdot`（upstream `cupy/__init__` 从未导出它们；
> `matrix_transpose` 仍可经 `x.mT` / `cupy.array_api` 使用，
> `vecdot` 实现保留在 `cupy.linalg._product` 未导出）。

| | 数量 | 状态 |
|---|---|---|
| 原版公开 API | **477** | |
| 当前公开 API | 390 → **461** | T1+T2 补齐后 |
| **缺失（原始 − 当前）** | 93 → **16** | 仅剩 T3/T4/T5 |
| 新增（Ascend 专属） | 4：`cupy_backends`, `is_ascend`, `xpu`, `xpu_backend` | `matrix_transpose`/`vecdot` 已撤下 |

> 2 个误报已剔除：`cuda`、`linalg` 经 `import cupy.cuda` 绑定，运行时存在。
> 剩余 16 个：`angle conj conjugate imag interp real_if_close nanargmax
> nanargmin trapz fuse`（T3）+ `ElementwiseKernel RawKernel RawModule
> ReductionKernel`（T4）+ `random einsum`（T5）。
> 注：`trapz = trapezoid` 实为 1 行别名（`trapezoid` 已存在），可随时补。

### 缺口分档（按实施难度）

**T1 —— 一行导出，内部已是已派发的 ufunc（零风险，立即可做）**

`ascend_*` 注册已存在，只补 `cupy/__init__.py` 的 import：

```
fabs  fmax  fmin  maximum  minimum  clip  nan_to_num  cbrt  heaviside  real
```

**T2 —— 一行导出，内部为纯 Python/host 组合（低风险）**

依赖的都是已派发算子（sort/argsort/where/put/dot/matmul/sum）：

```
argwhere flatnonzero searchsorted        # take/nonzero 组合
append delete resize trim_zeros          # manipulation 组合
lexsort msort sort_complex argsort partition argpartition
cross inner outer kron vdot trace        # linalg 组合（vdot 同 vecdot 模式）
corrcoef correlate cov                   # mean/std 组合
ravel_multi_index unravel_index indices ix_ c_ r_ mask_indices
tril_indices tril_indices_from triu_indices triu_indices_from diag_indices diag_indices_from
diag diagflat tri vander                 # tri + where 组合（tri 亦在此档）
diagonal take_along_axis choose compress extract select   # take + where 组合
unique_all unique_counts unique_inverse unique_values     # 已实现，仅缺导出
bincount histogram histogram2d histogramdd
packbits unpackbits piecewise vectorize
place putmask may_share_memory shares_memory who flatiter
alltrue sometrue                         # = all / any 的别名
```

**T3 —— 导出即用，但内部依赖 CANN 8.5.1 缺失的 aclnn 算子（部分场景不可用）**

```
conj conjugate angle imag               # 无 conjugate aclnn 算子；real 已可用
interp real_if_close                    # interp 需组合；real_if_close 依赖 complex 判断
nanargmax nanargmin                     # 无 nan-arg aclnn；需 nan_to_num + argmin/max 组合
fuse                                    # fusion 在 Ascend 上关闭（fusion_stub.py）
trapz                                   # 依赖 trapezoid（需另行组合实现）
```

**T4 —— Kernel 编写 API（Ascend 上为 stub，需 JIT/编译器路线）**

```
ElementwiseKernel RawKernel RawModule ReductionKernel
```
对应 `TODO.md` 的 JIT/自定义 kernel 长期项；`RawKernel` 目前是
`raw_kernel_stub.pyx`。

**T5 —— 整个子系统未移植**

```
random                        # cupy.random 完全未导出；依赖 generator 体系
einsum                        # aclnn_einsum.h 存在但无 aclop_ 包装（见 Stage-A）
```

### 统计小结

| 档 | 数量 | 说明 |
|---|---|---|
| T1 一行导出（ufunc 已注册） | 10 | ✅ **DONE 2026-09-18** |
| T2 一行导出（纯 host 组合） | 65 | ✅ **DONE 2026-09-18** |
| T3 依赖缺失 aclnn / 需另行组合 | 10 | 等 CANN 上游或写组合实现 |
| T4 kernel API | 4 | JIT 路线 |
| T5 子系统 | 2 | random / einsum |
| **合计** | **91**（原 93 − 2 误报） | T1+T2 已消掉，剩 **16** |

> 注意：T1/T2 属于**导出缺口**而非实现缺口——多数函数在 `cupy/_math`/
> `cupy/_sorting`/`cupy/_indexing` 等模块中已完整实现且可被 `import cupy`
> 的内部路径使用，只是没有提升到包命名空间。因此建议与
> `cupy/__init__cupy_original.py` 逐段对照，按原文件的组织顺序批量补
> `# NOQA` import，避免逐个手写。
