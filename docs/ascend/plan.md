# numpy-ascend 迁移工作计划 (plan.md)

> 基于 AGENTS.md / Progress.md / TODO.md 与当前代码库实际状态整理。
> 分支: `ascend`  |  基线: CuPy v14 alpha (NumPy 2.1)  |  目标后端: CANN 8.5 (aclnn)

---

## 1. 现状分析 (AGENTS.md)

### 1.1 架构分层 (自上而下)

| 层 | 位置 | 状态 |
|----|------|------|
| 1. NumPy API (Python) | `cupy/*.py` | 约 70% 已适配 |
| 2. `cupy._core` (Cython) | `cupy/_core/` + `cupy/_ascend/_core/` | 核心 ufunc/reduction 已替换为 aclnn 派发 |
| 3. `cupy.xpu` (高层后端 API) | `cupy/xpu/` (`device/stream/memory/pinned_memory`) | 已可用；`function.pyx` / `graph.pyx` 对 ascend 禁用为 stub |
| 4. `cupy.backends.backend` (低层后端抽象) | `cupy/backends/backend/api/{runtime,driver}.pyx` | 已中性化，`is_ascend()` 生效 |
| 5. `cupy.backends.ascend` (aclnn 实现) | `cupy/backends/ascend/` | 算子包装 + 注册中心已完成 |

### 1.2 Ascend 后端关键文件

- 算子包装 (C++): `acl_math_ops.h` (~87 个 aclnn include), `acl_general_ops.h` (~22), `acl_reduction_ops.h` (~11)
- 模板引擎: `acl_op_template.h` (`DECLARE_ACL_*` 宏 + `aclUnaryOpRun/BinaryOpRun/ReductionOpRun/IrregularOpRun`)
- **算子注册与派发中心**: `cupy/backends/ascend/api/acl_utils.pyx`
  - `_builtin_operators` (OpInfo→函数指针 union)
  - `register_math_operators()` / `register_reduction_operators()` / `register_irregular_operators()`
  - 已注册 **134 处** 调用 (约 108 个唯一 `ascend_*` op 名，含 inplace / scalar 变体)
- 分发消费方: `cupy/_ascend/_core/_kernel.pyx` (elementwise/ufunc), `_routines_linalg.pyx`, `_routines_sorting.pyx`
- 融合 stub: `cupy/_ascend/_core/fusion_stub.py` (fusion 关闭), `raw_kernel_stub.pyx`

### 1.3 已完成 (Done)

- backend 中性化重构 (`cupy.cuda`→`cupy.xpu`, `cupy_backends.cuda`→`cupy.backends.backend`)
- elementwise / reduction / comparison / bitwise / matmul / dot / sort / argsort
- creation: `array/arange/linspace/eye` (部分), manipulation: `concatenate/stack/flip`
- casting/fill/copy/nonzero/round/clip/isclose/divmod/heaviside
- **2026-09-12 新增**（详见 `Progress.md` §2 里程碑）：
  - 三角/双曲反函数族 + `cbrt`/`exp2`/`fabs`/`copysign`/`hypot`/`lcm`/
    `positive`/`invert`/`rint`/`trunc`/`nan_to_num`/`permute`/`roll`/`real`/`cast`
  - `dot()` 与 `_ascend_batched_matmul()`
  - sorting/argsort 轴与 dtype 修正
  - 4 个静默失效的 ufunc 名拼写 bug 修复
  - `py_list_acl_ufuncs()` 内省钩子（支持无 NPU 下的注册表测试）
  - 打包：RPATH 修复 + wheel 平台标签 + CANN 版本校验（见 `docs/ascend/Package.md`）
- 无 NPU 环境可 `import cupy._core`（`initialize_backend(0)` 需注释）

### 1.3.1 覆盖率快照 (2026-09-12)

**Array API: 112 / 129 = 86.8 %**
（registry: 164 处注册 / 148 唯一名 / 117 base 名；aclnn 头文件 include 129/756）

| 类别 | 覆盖 |
|---|---|
| elementwise | 54/56 |
| statistical / manipulation / searching / set / sorting / indexing / data_type / utility | **全部 100 %** |
| creation | 10/17 |
| linalg | 13/21 |
| **合计** | **112/129 = 86.8 %** |

> 证据级别：新增算子的验证止于 **L3（编译 + 注册 + import）**；
> 本机无 NPU，**L4 数值正确性未验证**。

### 1.4 环境约束

- CANN 8.5 已装但**无 NPU**，只能验证"编译 + import"，不能跑数值测试 / benchmark
- 有 NPU 验证环境: ModelArts EulerOS + 910B, CANN 8.2, Python 3.9
- `double` / `int64` 多数 aclnn 不支持或走 AICPU（慢）；bfloat16 非标准 numpy 类型不支持

---

## 2. 可迁移但未完成的 Array API 清单（2026-09-12 重新核实）

> 本节已按**实测数据**重写。依据：`ls $ASCEND_HOME_PATH/include/aclnnop/`
> （CANN 9.0.1 共 **814** 个 `aclnn_*.h`）、`tools/scan_ops.py` 的 CST 统计
> （190 条注册 / 174 唯一名 —— 不要用 grep：注释掉的注册会被算进去），
> 以及 `cupy/array_api/` 的 129 个公开函数。
> 详见 `Progress.md` §3/§4 与 `TODO.md` 的 Stage-A 分析。

**当前覆盖率：118 / 129 = 91.5 %**（详见 `Progress.md` §4；2026-09-16 更新）。

> 2026-09-13 实施结果：**A 档 14 个包装 + linalg 打通 + 2 个 dispatch bug 修复**，
> 覆盖率 86.8 % → **90.7 %**。详见 `Progress.md` §2 里程碑 / `TODO.md` 开头。
> 剩余 12 项缺口全部是"**CANN 8.5.1 无对应算子**"，需组合实现或等上游，
> 且无 NPU 无法验证数值正确性。

### 2.0 自 plan.md 初版以来已完成（本节原列的 A 档条目）

以下条目**已经做完**，从待办中移除（证据：`acl_utils.pyx` 注册计数 ≥ 1）：

| 原 A 档条目 | 状态 |
|---|---|
| `cbrt`, `rint`/`round`, `trunc`/`fix`, `fabs` | ✅ 已注册（+ inplace 变体） |
| `exp2` 注册缺失 | ✅ 已注册（+ `inplace_exp2`） |
| `reciprocal` 缺 inplace | ✅ 已有 |
| `atan2` 拼写 `tan2` 待核对 | ✅ **确认为 bug 并修复** → `ascend_arctan2` |
| `real` | ✅ 已注册 |
| `roll` / `permute` / `flip` | ✅ 已注册（`aclop_Roll`/`aclop_Permute`） |
| `cast`（`astype` 依赖） | ✅ 已注册，`aclop_Copy` 已改造为 dtype 转换路径 |
| `ptp` → `Aminmax` | ⚠️ 仍未做（`aclnn_aminmax.h` 已 include，零成本） |
| `unique2` → 4 个 set 函数 | ⚠️ 仍未做（头文件已 include） |
| `imag` / `conj` / `angle` | ❌ **CANN 无算子**，实为 B 档（原判错误） |
| `repeat` | ⚠️ 仍未做（`aclnn_repeat.h` 存在但未 include） |
| `put` | ⚠️ `aclop_Put` 存在但未注册 |
| `einsum` | ❌ 原判为 B 档（组合），**实为 A 档**（`aclnn_einsum.h` 存在） |

### A 档 — CANN 8.5.1 已有算子，只需包装 + 注册（零算法风险）

**A1. 头文件已 include，只差 `aclop_` 包装**（成本最低）

| Array API / numpy API | CANN 算子 | 备注 |
|---|---|---|
| `unique_all` / `unique_counts` / `unique_inverse` / `unique_values` | `aclnn_unique2` | 一个包装覆盖 **4 个** Array API 函数 |
| `einsum`, `vecdot`, `tensordot` | `aclnn_einsum` | 一个包装覆盖 **3 个** 高层 API |
| `tril`, `triu` | `aclnn_tril` / `aclnn_triu` | Array API linalg |
| `trace` | `aclnn_trace` | Array API linalg |
| `qr` | `aclnn_qr` | Array API linalg |
| `svd` / `svdvals` | `aclnn_svd` | Array API linalg |
| `inv` | `aclnn_inverse` | Array API linalg |
| `diagonal` / `diag` | `aclnn_diag` | |
| `ptp` | `aclnn_aminmax`(+`_all`/`_dim`) | `ptp = max - min` |
| `histogram` | `aclnn_histc` | 头文件已 include，无 `aclop_` |
| 高级索引 / `take_along_axis` | `aclnn_index` | |

**A2. CANN 有算子，需补 `#include` + 包装**

| API | CANN 算子 | 备注 |
|---|---|---|
| `repeat`, `tile` | `aclnn_repeat` / `aclnn_repeat_interleave` | 可从 host 侧下沉到 aclnn |
| `expand_dims`, `broadcast_to` | `aclnn_expand` | |
| `meshgrid` | `aclnn_expand` + reshape | 纯组合，但算子已具备 |
| `argpartition` | `aclnn_topk` | 可替换当前的 sort/argsort 回退（性能） |
| `take`, `compress`, `take_along_axis` | `aclnn_gather` | 与已有 `aclop_Take` 区分 |
| `isin`, `in1d` | `aclnn_isin` / `aclnn_isin_tensor_scalar` | |
| `searchsorted` | `aclnn_searchsorted` | |
| `nonzero` / `argwhere` | `aclnn_nonzero_v2` | 已用 v1，v2 可能更快 |
| `unique(..., return_index)` | `aclnn_unique_consecutive` | 补充 `unique2` |
| `solve`（三角部分） | `aclnn_triangular_solve` | ⚠️ **无 `aclnn_solve`**，一般求解需组合 |

### B 档 — CANN 无算子，需在 Python/Cython 层组合实现

> **2026-09-13 更新**：elementwise 部分已改用**自定义 AscendC 内核**实现
> （M1/M2 基础设施落地，详见 `docs/ascend/CustomKernel.md` 与 `Memory.md` §4.5）：
> `conj`/`angle`/`imag`/`frexp`/`modf`/`ldexp`/`left_shift`/`right_shift` 共 8 个内核
> 编译通过并注册派发（L1-L3；数值验证待 910B）。表中的"组合实现"策略仅作为
> 9.0 以下 CANN 或内核异常时的 fallback 参考。

| 缺失 API | 实现策略 | 依赖的已注册算子 |
|---|---|---|
| ~~`bitwise_left_shift`, `bitwise_right_shift`~~ | **DONE (2026-09-13, 自定义 AscendC 内核 `ascend_left/right_shift`，i32)**；CANN 9.0.1 亦有 `aclnn_left_shift/right_shift` 头，可择优切换 | — |
| `cholesky` | 无 `aclnn_cholesky`；需自研或等上游 | — |
| `det` | 无 `aclnn_det`（`aclop_Det` 在 `acl_math_ops.h:437` 被注释）→ LU 分解组合 | `tril`/`triu`/`prod` |
| `eigh`, `eigvalsh` | 无算子，需自研或等上游 | — |
| `conj` / `conjugate` | CANN 无 `conj` → `complex(real, -imag)` | `complex`, `negative` |
| `angle` | `atan2(imag, real)` 组合 | `arctan2` |
| `imag` / `imaginary` | 从 `aclnn_complex` 反向视图 / stride 技巧 | `complex` |
| `frexp` | `log2` + `floor` + `pow` | `log2`, `floor`, `pow` |
| `modf` | `trunc` + `subtract` | `trunc`, `subtract` |
| `ldexp` | `pow` + `multiply` | `pow`, `multiply` |
| `isnan` | `not_equal(x, x)` | `not_equal` |
| `convolve` | `matmul`（im2col）或 FFT（若 CANN 提供） | `matmul` |
| `gradient`, `diff` | `subtract` + 切片 | `subtract` |
| `interp`, `trapezoid` | `subtract` / `divide` / `take` 组合 | 已有 |
| `average`, `quantile`, `percentile` | 基于 `sort` + `take` 组合 | `sort`, `take` |
| `choose` | `take` + `where` 组合 | `take`, `s_where` |
| `from_dlpack` | 已有 DLPack 通路（`cupy/_core/dlpack.pyx`），host 侧接线即可 | — |
| `*_like`（`empty_like`/`full_like`/`ones_like`/`zeros_like`） | `empty`/`full` + `broadcast_to`，纯 host 侧 | — |

> **原 plan.md 的两处误判已修正**：
> 1. `imag`/`conj`/`angle` 原列 A 档，实际 CANN 8.5.1 **无这些算子** → B 档。
> 2. `einsum` 原列 B 档（"分解为 matmul"），实际 `aclnn_einsum.h` **存在** → A 档，
>    不需要手工分解。

### C 档 — 需 CANN 上游或长期投入（暂缓）

- `random` / 随机分布：`aclnnRand` 存在，但 generator/bit_generator 体系工作量大；可先用 numpy 生成（AsNumpy 有 impl）
- `FFT`：等 CANN nnal/asdsip 发布对应算子
- `matmul` 支持 dtype 扩展到 fp16/bf16（当前仅 `float32`）
- `double`/`int64` emulation（等硬件）
- sparse / cusolver / cuTensor 类：CANN 暂无对应，保持 stub

### D 档 — 已知 Bug / 技术债

1. ~~**matmul 结果转置 bug**~~ **✅ FIXED**（根因：ascend 分支缺 `return`，fall-through 到 cuBLAS 转置代码块）
2. `runtime._ensure_context` / `_register_acl_ufunc` 在无 NPU 下的初始化路径
3. ~~`numpy_to_acl_dtype` 重命名~~ **✅ DONE** → `numpy_dtype_to_acl_dtype`
4. ~~`cupy_scalar_to_acl_scalar` 的 unsigned / complex / string 分支~~ **部分 DONE**
   （unsigned / complex / fp16 已完成；**string 分支仍未完成**）
5. `_kernel.pyx` 中 `_reduce_dims`/`indexer` 逻辑对 ascend 未启用，shape 语义待确认
6. `is_ump_supported` 等 `IF CUPY_CANN_VERSION <= 0` 分支需补齐 ascend 侧
7. **NEW**: `_routines_sorting.pyx` 的改动尚未 commit（`partition`/`argpartition`
   目前回退到 sort/argsort）
8. **NEW**: `dist/` 里只有 2025-10-07 的旧 stub wheel，需要实际构建
   `cann8.5` / `cann9.0` 两个包（见 `docs/ascend/Package.md`）

---

## 3. 分阶段实施计划

### 阶段 P0 — 修 Bug + 补测试基线（1-2 周）
- [x] ~~修 matmul 转置 bug~~ **DONE**
- [x] ~~重命名 `numpy_to_acl_dtype` → `numpy_dtype_to_acl_dtype`~~ **DONE**
- [x] ~~补齐 `cupy_scalar_to_acl_scalar` 的 unsigned/complex 分支~~ **DONE**
- [ ] 补齐 `cupy_scalar_to_acl_scalar` 的 **string** 分支（`Memory.md` §6 P1）
- [ ] 在 ModelArts 910B 上跑通 `pytest tests/cupy_tests/math_tests` 建立基线
- [ ] 提交 `_routines_sorting.pyx` 的未提交改动

### 阶段 P1 — A 档快速补齐（3-4 周）— 按投入产出比排序

**P1a. 免费收益（头文件已 include，只差包装）**
- [ ] `unique2` → 一次覆盖 `unique_all`/`unique_counts`/`unique_inverse`/
      `unique_values`（**set 类别直接 4/4 完工**）
- [ ] `qr` / `inverse` / `trace` → linalg 覆盖率 +3
- [ ] `svd` → `svd` + `svdvals`（linalg +2）
- [ ] `einsum` → `einsum` / `vecdot` / `tensordot`（一个包装，三个 API）
- [ ] `aminmax` → `ptp`
- [ ] `diag` / `tril` / `triu` → `diagonal` / `tril` / `triu`（creation + linalg）
- [ ] `histc` 包装 + 注册
- [ ] `index` → 高级索引

**P1b. 需补 include**
- [ ] `repeat` / `expand` / `topk` / `gather` / `isin` / `searchsorted` /
      `nonzero_v2` / `unique_consecutive`
- [ ] `triangular_solve` → `solve`（三角部分）

**P1c. 回归验证**
- [ ] `_kernel.pyx` 的 inplace / scalar 路径回归
- 验证: `pytest tests/cupy_tests/{math,logic,creation,manipulation}_tests`

> 预期效果：P1 完成后 Array API 覆盖率可从 **86.8 % → ~95 %**
> （补齐 elementwise 最后 2 个 + linalg 8 个中的 6 个 + creation 的 `*_like`）。

### 阶段 P2 — B 档组合实现（6-8 周）
- [x] `isnan` = `not_equal(x,x)`（aclnn 组合）
- [x] shift ops / `frexp` / `modf` / `ldexp` / `conj` / `angle` / `imag`：
  **改用自定义 AscendC 内核实现**（2026-09-13，M1/M2 基础设施落地，
  8 个内核编译通过并注册派发，L1-L3；数值验证待 910B。见
  `docs/ascend/CustomKernel.md` 与 `Memory.md` §4.5）
- [ ] `diff` / `gradient` / `interp` / `trapezoid`
- [ ] `average` / `quantile` / `percentile`（基于 sort+take）
- [ ] `choose`（take + where）
- [ ] `*_like` 系列 + `from_dlpack` 接线（纯 host 侧，成本极低）
- [ ] `cholesky` / `det` / `eigh` / `eigvalsh`（无 CANN 算子，需自研或等上游）

### 阶段 P3 — JIT / kernel（与 CANN 路标对齐）
- [x] **自定义 kernel 编译**（2026-09-13）：不等 pyPTO/triton-ascend —— bisheng
  直接编译 AscendC + aclrt 内存加载（`BinaryLoadFromFile`/`LaunchKernelWithConfig`），
  JIT + 磁盘缓存 + 注册派发（详见 `docs/ascend/CustomKernel.md`）；
  路线 A only（custom aclnn 插件裁决不做）
- [x] **triton-ascend 桥接 M6**（2026-09-13）：`triton_bridge.py`（零拷贝 adapter +
  流桥接 + `@jit_ufunc` 降级）+ `tests/ascend/` 31 用例；
  真内核端到端待 triton-ascend × CANN 9.0.1 版本矩阵实测
- [ ] `RawKernel`/`RawModule` 公开 API（`raw_kernel_stub.pyx` 仍是空壳；
  底座 aclrt 加载/启动链路已就绪，M3 剩余工作）
- [ ] `ElementwiseKernel` 基于模板的 JIT（`acl_op_template.h` 思路扩展）

### 阶段 P4 — 随机 / FFT / 多卡（长期）
- [ ] 随机分布（可先 numpy 生成 + 搬运）
- [ ] FFT（等 CANN 算子）
- [ ] HCCL 多卡（抽象 NCCL/RCCL/HCCL）

---

## 4. Cython 代码能否用 tree-sitter 分析？

**结论: 可以，但需要自备 Cython grammar，当前环境未装。**

### 4.1 现状核查

- 已安装: `tree_sitter 0.26.0` (Python 绑定)
- **未安装**: `tree_sitter_cython` / `tree-sitter-cython` grammar，也无 `tree_sitter_languages`
- 即: 当前**只能解析 `.c`/`.cpp`/`.py` 等已有 grammar，不能直接解析 `.pyx`/`.pxd`**

### 4.2 可行方案

| 方案 | 说明 | 评价 |
|------|------|------|
| **A. 用 community `tree-sitter-cython` grammar** | GitHub `tree-sitter/tree-sitter-cython`(或 fork)，编译成 `.so` 后用 `Language(...)` 加载 | 推荐，能真正解析 `.pyx` 语法 (cdef/cimport/IF/宏) |
| B. 先把 `.pyx` 当 Python 用 `tree-sitter-python` 解析 | 能解析大部分 body；`cdef/cimport/IF ...:` 会报错/recovery | 快速但不可靠，仅适合粗统计 |
| C. 解析生成的 `.cpp` | 用 `tree-sitter-cpp` 解析 `acl_utils.cpp` 等生成文件 | 稳定，但丢失 Cython 语义（宏展开后 AST 巨大） |
| D. 不依赖 tree-sitter，用正则 + Cython 自带 `cython --parse`/AST | `register_acl_ufunc` 提取这类任务是**纯文本模式**，grep 已足够 | 本次分析即用此方式 (grep 出 **164** 处注册) |

### 4.3 建议落地方式 (方案 A)

```bash
# 1. 拉取 Cython grammar 并编译为 Python 可加载的 .so
git clone https://github.com/tree-sitter/tree-sitter-cython
cd tree-sitter-cython
# 需本机有 C 编译器 + tree-sitter CLI
tree-sitter build --output cython.so        # 或用 pip 安装带 wheel 的第三方包
```

```python
# 2. 解析 .pyx 示例
import tree_sitter_cython
from tree_sitter import Language, Parser

CYTHON = Language(tree_sitter_cython.language())
parser = Parser(CYTHON)
src = open("cupy/_ascend/_core/_kernel.pyx", "rb").read()
tree = parser.parse(src)
root = tree.root_node
print(root.has_error)          # 检查语法错误
```

### 4.4 在本项目中的实际用途 (用 tree-sitter 之后能做什么)

1. **自动盘点算子覆盖率**: 扫描 `acl_utils.pyx` 的 `register_acl_ufunc("ascend_*")` 调用 → 生成"已注册 op 集合"；扫描 `cupy/**/*.py(x)` 中 `create_ufunc(...)` / `_kernel` 调用 → 对比得出未迁移 op，自动产出 Progress 表。
2. **发现未迁移的 CUDA 残留**: 解析 all `.pyx` 中 `cimport cupy.cuda` / `cupy_backends` / `cuBLAS` 等 symbol，定位尚未中性化的代码。
3. **批量重构/迁移辅助**: 结合 `cupy/backends/api_replace_tool.py`（已有 CUDA→XPU 名称替换脚本），用 AST 精确定位 `cuXXX` 调用点后做安全替换，避免正则误伤注释。
4. **静态校验**: 检测 `IF CUPY_CANN_VERSION <= 0:` 分支是否成对覆盖 ascend 侧。

> 说明: 因 tree-sitter-cython 维护度一般，建议**先试方案 A**；若 grammar 编译受限，退化到方案 C（解析生成的 `.cpp`）或方案 D（grep/正则）也能满足"算子覆盖率统计"的目标。

---

## 5. 下一步 (本 plan 的 Immediate Next Actions)

> 更新时间: 2026-09-13（晚）。P0/P1/P2(elementwise)/P3(M1/M2/M6) 大部分已完成。

1. 在 910B 机器上建立 `pytest` 基线（本机无 NPU，只能到 L3）。
   **新增**：`tests/ascend/`（31 用例：自定义内核注册回归 + triton 桥接）应纳入基线。
2. ~~按 P0 修 matmul 转置 bug + 补齐 scalar/unsigned 分支~~ **DONE**。
   剩余：`cupy_scalar_to_acl_scalar` 的 **string** 分支。
3. ~~**按 P1a 做"免费收益"批量注册**~~ **DONE (2026-09-13)**：
   覆盖率 86.8 % → **90.7 %**。
3b. ~~**自定义 AscendC 内核基础设施 + B 档 elementwise 8 内核 + triton 桥接**~~
   **DONE (2026-09-13 晚)**：见 `docs/ascend/CustomKernel.md`、`Memory.md` §4.5/§4.6；
   覆盖率 90.7 % → **91.5 %**（elementwise 56/56）。
4. 剩余（**CANN 无算子**，需组合或等上游）：`cholesky`/`det`/`eigh`/`eigvalsh`/
   `i0`/`nextafter`（后两者 AscendC 亦无原语，需多项式/位技巧内核）。
   **数值验证**：8 个自定义内核（conj/angle/imag/frexp/modf/ldexp/shift×2）
   须在 910B 上逐个验证 + 单测。
4b. 可选加速项（非阻塞）：`repeat`/`expand`/`topk`/`gather`/`isin`/
   `searchsorted`/`nonzero_v2`/`unique_consecutive`/`triangular_solve`/
   `einsum`（头文件已 include，只差包装）。
5. ~~引入 `tree-sitter-cython`，写一个 `tools/scan_ops.py` 自动生成算子覆盖率
   报告（替代手工维护 Progress.md）~~ **DONE (2026-09-16)**：`tools/scan_ops.py`
   已落地（方案 A：tree-sitter-python / -cython / -cpp），生成
   `tools/cst_db.json`（机器可读）+ `tools/cst_db.md`（报告），一条命令刷新：
   `python tools/scan_ops.py --runtime`；`--check` 可在 CI 里判断 db 是否过期。
   CST vs 运行时注册表交叉校验 `only_static/only_runtime` 均为空（174 个算子）。
6. 构建 `cann8.5` / `cann9.0` 两个 wheel（见 `docs/ascend/Package.md`）。
   注：wheel 只带内核源码（`kernels/*.cpp`），运行时 bisheng JIT；
   AOT 开关 `CUPY_ASCEND_AOT_KERNELS=1` 尚未实现（见 CustomKernel.md §3.2）。
7. **M3 收尾**：`RawKernel`/`RawModule` 公开 API（`raw_kernel_stub.pyx` 重写，
   aclrt 底座已就绪）；自定义内核的标量操作数支持。

