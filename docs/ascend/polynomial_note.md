# cupy / cupyx polynomial 迁移 Ascend 方案

> 范围：`cupy.poly1d` / `cupy.poly` / `polyadd` / `polysub` / `polymul` /
> `polyfit` / `polyval` / `roots`，以及 `cupy.polynomial`（NumPy 风格多项式包）
> 和其上的 `cupyx.scipy.interpolate._polyint`、`cupyx.scipy.signal._polyutils`。

## 1. 关键结论：polynomial 是"纯组合层"

与 FFT 不同，polynomial **没有独立的 CUDA kernel**，全部由已有
ufunc / linalg 例程组合而成，因此不需要新增 `acl_utils.pyx` 算子注册：

```
cupy/lib/_polynomial.pyx    poly1d（cdef class，仅属性/运算符转发）
cupy/lib/_routines_poly.py  poly / polyadd / polysub / polymul / polyfit / polyval / roots
cupy/polynomial/            polyvander / polycompanion / as_series（纯 Python）
cupy/_math/misc.py          convolve（polymul 与 poly 的底层）
```

依赖树与 Ascend 后端状态：

| 底层依赖 | 用于 | Ascend 状态 |
|---|---|---|
| `pad` / `trim_zeros` / `power` / `sum` / `add` / `sub` | polyadd/polysub/polyval/_polypow | 已注册 ufunc 组合，直接可用 |
| `convolve` → `_dot_convolve`（as_strided + matmul） | polymul、poly | `dot`/`matmul` 已注册，可用 |
| `convolve` → `_fft_convolve`（`cupy.fft`） | 大尺寸 polymul、poly 的根合并树 | aclfft 仅 FP32（本次已打通） |
| `eigvalsh` | `roots`、`poly(2D Hermitian)` | **CANN 8.5.1 无 aclnn eig/eigh**（已验证 aclnnop 头文件目录） |
| `lstsq`（gelsd） | `polyfit` | **无 aclnn gels/lstsq**；用已注册的 `svd` 组合替代 |
| `svd` / `qr` / `inverse` / `dot` / `matmul` | lstsq 组合、polyfit 协方差 | 已注册（`_ascend_svd` 见 `cupy/_ascend/_core/_routines_linalg.pyx:651`） |

另一障碍：`cupy.linalg` 包当前在 Ascend 构建下**不可导入**（`cupy/linalg/_solve.py:8`
的 `from cupy.cuda import device`，而 Ascend 无 `cupy.cuda`），所以 polyfit 不能
直接引用 `cupy.linalg.lstsq/inv`，需要解析层。

## 2. 实施阶段（P0–P3）

### P0 —— 直接点亮（已实施）
1. `install/cupy_builder/features/ascend.py`：`ascend_files` 加入
   `cupy.lib._polynomial`（该 pyx 只 cimport `_ndarray_base`，无 CUDA 依赖）。
2. `cupy/__init__.py`：按 FFT 的 try/except 模式放开
   `from cupy import polynomial` 与 `cupy.lib._polynomial import poly1d`、
   `cupy.lib._routines_poly import {poly, polyadd, polysub, polymul, polyfit, polyval, roots}`
   （上游导出点见 `cupy/__init__ copy.py:543-550`）。
3. 无需新算子：polyadd/polysub/polyval/_polypow/polymul(direct) 全部走已注册 ufunc。

### P1 —— convolve 的 FFT 路径守卫（已实施）
`cupy/_math/misc.py` 新增 `_fft_convolve_ok(a1, a2)`：Ascend 下
`result_type` 不是 float32/complex64 时 FFT 卷积不可用（aclfft 无双精度），
`convolve()` 自动回落 `_dot_convolve`。
`_routines_poly.poly()` 在 Ascend + 非 FP32 时改用逐根累加
（`convolve([1], [1,-r])` 循环，O(n²)，根数通常很小），绕开 `_fft_convolve`
的 2-D 批处理路径（`_dot_convolve` 仅支持 1-D）。

### P2 —— polyfit 的 lstsq（已实施）
调研发现本仓库的 `cupy.linalg` 已经做过后端化改造，可直接复用：
- `cupy/linalg/_solve.py::lstsq` 已改为 **SVD 伪逆组合实现**（无 gels/cusolver），
  经 `cupy.linalg.svd` 的 Ascend 分支落到 `_ascend_svd`；
- `cupy/linalg/_solve.py::inv` 已有 `is_ascend()` 分支调用 `_ascend_inv`（aclnnInverse）；
- `cupy/__init__.py:11-16` 通过 `sys.modules` 别名让 `cupy_backends.cuda` 指向
  中性后端模块，因此 `import cupy.linalg` 在 Ascend 构建下是可行的。

因此 `cupy/lib/_routines_poly.py` 顶部采用解析层：
优先 `from cupy.linalg import lstsq/inv`（Ascend 分支已就绪），
ImportError 时回落到新增的 `cupy/_ascend/_poly_linalg.py`
（同样基于 `_ascend_svd`/`_ascend_inv` 的纯组合实现，作为防御性兜底）。
注意 `_ascend_svd` 的 Vh 方向解释与 `cupy/linalg/_decomposition.py` 的
Ascend 分支保持一致（`v.transpose().conj()`），需 L4 硬件验证。

### P3 —— roots / poly(2D)（未实施）
`roots` 与 `poly(2D Hermitian)` 依赖 `eigvalsh`，CANN 8.5.1 无对应 aclnn 算子。
现状：`roots()` 在 Ascend 下抛出明确的 `NotImplementedError`。
后续路线：companion 矩阵很小（deg×deg），可做 `.get()` → `numpy.linalg.eigvalsh`
CPU 回退；或等 CANN 提供 `aclnnLinalgEigh` 后按 skill 五层流程接入。

### P4 —— cupyx 侧
`cupyx.scipy.interpolate._polyint`、`cupyx.scipy.signal._polyutils`
构建在上述原语上，随 P0–P2 自动可用。

## 3. 已知限制（Ascend）

- **运行期库路径**：开发环境用源码构建的 ops-fft 时，需把其构建目录加入
  `LD_LIBRARY_PATH`（`export LD_LIBRARY_PATH=~/repos/ops-fft/build:$LD_LIBRARY_PATH`），
  否则 `cupy.backends.ascend.api.aclfft` 导入时报
  `libcann_ops_fft.so.1: cannot open shared object file`。官方 `.run` 安装到
  CANN `lib64`（已在 `set_env.sh` 的搜索路径内）则无此问题。
- `roots` / `poly(2D)`：未支持（无特征值算子，见 P3）。
- `polymul`/`poly` 的 FFT 加速路径仅 FP32/complex64；其余 dtype 走 direct。
- `polyfit` 的 `lstsq` 走 `aclnnSvd`，双精度是否被 910B 支持需硬件验证（L4）。
- 验证等级：无 NPU 环境完成 L3（import + registry）；数值验证需 910B。

## 4. 参考

- 上游导出点：`cupy/__init__ copy.py:44,543-550`
- convolve：`cupy/_math/misc.py`（`convolve` / `_fft_convolve` / `_dot_convolve`）
- Ascend SVD：`cupy/_ascend/_core/_routines_linalg.pyx:651`、`cupy/linalg/_decomposition.py`（svd 的 Ascend 分支）
- 测试：`tests/cupy_tests/lib_tests/test_polynomial.py`
