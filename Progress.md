# Progress — numpy-ascend

> **状态快照: 2026-09-16**。本文件只回答"现在到哪了"；
> 操作手册 / 环境 / 命令 / 坑见 [Memory.md](Memory.md)；
> 算子事实数据由 [`tools/scan_ops.py`](tools/scan_ops.py) 自动生成到
> [`tools/cst_db.md`](tools/cst_db.md)（本节表格的数字即来自该数据库）。

## 1. 一句话现状

CuPy（v14.0.0a1, fork 自 v14 alpha / NumPy 2.x）后端已从 CUDA 换为 **CANN aclnn**：

| 维度 | 现状 |
|---|---|
| Array API 标准覆盖（`cupy/array_api/`，129 个公共函数） | **118 / 129 = 91.5 %** |
| 注册算子（`ascend_*`，唯一名） | **176**（public 149 / inplace 27） |
| `cupy/_core` 派发链 ufunc 覆盖 | **110 / 151 = 72.8 %** |
| 剩余可移植缺口 | **2**（`i0`、`nextafter`，见 §4.1） |
| 本机验证等级 | **L3**（编译 + import + 注册）；**无 NPU，L4 数值未验证** |

## 2. 里程碑（压缩）

| 时间 | 内容 |
|---|---|
| 2025-10-12 | MVP：add / cos / matmul 在 910B 上 10–100× 加速 |
| 2025-10-23 | 后端重构：`backends → cupy.backends` + `_features.py`；benchmark 可在 NPU 跑 |
| 2025-11-08 | reduction（`sum`）打通；~90 % math ops 有 aclop |
| 2025-11-15 | irregular ops：`concatenate / clip / copy`；scalar 算子补齐 |
| 2025-12-06 | creation（`arange`）+ `concatenate` 测试通过 |
| 2025-12-13 | sorting（`sort / argsort`）；`fill_kernel → FillScalar`（GeneralOp） |
| 2026-09-12 | math/dtype 大补（38 处注册、33 个新名）+ `matmul`/`dot` 修复；修正 4 个错误 ufunc 名 |
| 2026-09-13 | CANN 9.0.1 适配 + Level A/B（14 个 aclnn 包装）+ 自定义 AscendC 内核（M1/M2）+ triton 桥接（M6）+ 打包/RPATH |
| 2026-09-16 | `_core` 检视 P0/P1 落地；`_ascend → _core/_ascend` 迁移；import 免 `LD_PRELOAD`；**benchmark 重构**；**CST 数据库自动化**；**pytest dtype 过滤**；**6 个缺失算子的组合实现**（缺口 8 → 2） |

## 3. 算子覆盖（实测 2026-09-16，CST 与运行时注册表交叉验证）

| 指标 | 值 |
|---|---|
| `register_acl_ufunc` 注册条目（op × OpType） | **192** |
| 唯一 `ascend_*` 算子 | **176**（public 149 / inplace 27） |
| 运行时 `py_list_acl_ufuncs()` | 192 条 → 去重 176，**与静态 CST 完全一致**（互查 `only_static/only_runtime` 均为空） |
| cupy ufunc 声明（`cupy/_core` + `cupy/_math` 顶层） | **151** |
| 已覆盖 | **110 = 72.8 %**（builtin 98 + AscendC 自定义内核 7 + 宿主端组合 5） |
| 未覆盖：可移植缺口 / 内核噪声 | **2 / 39** |
| `aclop_*` C++ 包装 | **83** |
| 宿主端组合实现 | **5**（`nanargmax nanargmin nanmean choose angle_deg`）|
| aclnn 头文件 include / CANN 9.0.1 可用 | **143 / 814** |
| AscendC 自定义内核 | **8**（`angle conjugate frexp imag ldexp left_shift modf right_shift`）|
| `IF CUPY_CANN_VERSION` 分支 | 66 |
| CST 解析文件（cython / python / cpp） | 140 / 748 / 23（21 处 grammar 报错，均为 tree-sitter 语法限制，见 `tools/cst_db.md` §7） |

> 数据来源：`python tools/scan_ops.py --runtime` → `tools/cst_db.json|md`。
> 手工 grep 得到的旧数字（183 调用/165 唯一名）已作废——grep 会把**注释掉的注册**算进去
> （例：`#register_acl_ufunc("ascend_nanprod", ...)`），CST 不会。

## 4. Array API 覆盖（基线：`cupy/array_api/` 129 个公共函数）

| 类别 | 覆盖 / 总数 | 缺失 |
|---|---|---|
| creation | 10 / 17 | `*_like` ×4、`meshgrid`、`from_dlpack` |
| elementwise | 56 / 56 | — |
| statistical | 7 / 7 | — |
| manipulation | 8 / 8 | — |
| searching | 4 / 4 | — |
| set | 4 / 4 | — |
| sorting | 2 / 2 | — |
| indexing | 1 / 1 | — |
| data_type | 7 / 7 | — |
| utility | 2 / 2 | — |
| linalg | 17 / 21 | `cholesky`、`det`、`eigh`、`eigvalsh` |
| **TOTAL** | **118 / 129 = 91.5 %** | |

- `*_like` 系列是 `empty/full + broadcast_to` 的宿主端别名，**无需 aclnn**，实际设备端覆盖更高。
- 剩余 4 个 linalg 在 **CANN 无算子**，需自研算法或等上游。

### 4.1 仍缺的 2 个 ufunc（可移植缺口）

`i0`、`nextafter` —— AscendC 无原语，需多项式/位技巧内核 → 暂缓（见 `docs/ascend/CustomKernel.md`）。

**已补齐的 6 个（2026-09-16，全部"组合实现"）**：

| 算子 | 实现位置 | 组合方式 |
|---|---|---|
| `nanprod` | C++ `aclop_NanProd`（`acl_reduction_ops.h`）| `NanToNum(nan=1)` + `Prod`；整型跳过 NanToNum 直接 `Prod`（同一 `aclop_NanMin/NanMax` 套路）|
| `nanargmax` / `nanargmin` | `cupy/_core/_ascend/composite.py` | `where(isnan→±inf)` + `argmax/argmin` |
| `nanmean` | 同上 | `nansum / 非NaN计数`（计数 = "1 数组 + NaN 占位" 再 `nansum`，避免 bool 归约/Cast）|
| `choose` | 同上 | 逐 choice `where(index == k, candidate, result)`（`cupy_choose` 是带 `raw` 指针的 kernel，aclnn 无法表达）|
| `angle(deg=True)` | 同上 | `angle(z) * 180/pi`（`cupy_angle` 已由自定义内核覆盖）|

依赖清单是**机器可读**的（`composite.py: REQUIRED_OPS`），
`tools/scan_ops.py` 会解析它并与运行时注册表对账（`missing_deps` 必须为空），
`tests/ascend/test_composite_ops.py` 也会校验同一份清单。

## 5. 疑难 bug 修复清单（按"静默失效"优先）

| 症状 | 根因 | 修复 |
|---|---|---|
| `matmul(a,b)` 返回 `B@A` | ascend 分支缺 `return`，fall-through 到 cuBLAS 转置块（`a,b=b,a`） | 提前 `return`；CUDA 块用 `IF CUPY_CANN_VERSION <= 0` 隔离 |
| `isnan/isfinite/isinf` 从不派发 | 注册名 `ascend_is_finite` ≠ ufunc `cupy_isfinite` | 两种拼法都注册 |
| `bitwise_*` 从不派发 | `OP_PREFIX` 在 ascend 构建取成 `"cupy_"` | 统一 `"cupy_"`（派发器自行改写为 `ascend_`） |
| `argsort` 结果错 | 与 `sort` 共用 `aclop_Sort` | 改用 `aclop_Argsort` |
| 4 个三角/反三角从不派发 | `ascend_acos/asin/atan/tan2` 拼写错 | → `arccos/arcsin/arctan/arctan2` |
| `put_raise` 从不派发 | 注册成 `ascend_raise_put` | → `ascend_put_raise` |
| `scan` 静默不干活 | `pass` 桩 | 用 aclnn 一步完成扫描 |
| `concatenate` >8 操作数结果错 | 宿主指针数组当数据的技巧 | `IF` 整段禁用（显式报错） |
| `matmul` 数值/签名不符 | `math_type` 用 `uint8_t` | → `int8_t`（aclnn 签名要求） |
| `import cupy` 失败（undefined symbol） | `libop_common.so` 未声明 `liboptiling.so` 依赖 | `cupy/__init__.py` 里 `ctypes` 预加载（见 Memory §3.3） |
| `argwhere` 计数未赋值 | 用了未注册的 `count_nonzero` | 改 `cupy.sum(mask)` |
| `_kernel.pyx` 里 `cimport` 新函数即 SIGSEGV | `acl_utils.pxd` 的 `__pyx_capi__` 静态解析 | 逻辑放进 `acl_utils` 内部或走 `launch_general_func` |

## 6. 工具链（2026-09-16 新增，均已落地）

| 工具 | 用途 | 命令 |
|---|---|---|
| `tools/scan_ops.py` | **CST 数据库**：算子注册/覆盖缺口/`aclop→aclnn` 映射/CUDA 残留/`IF` 分支/导出缺口/组合依赖对账/解析自检 | `python tools/scan_ops.py --runtime`（`--check` 判断是否过期） |
| `cupy/_core/_ascend/composite.py` | **宿主端组合算子**（NPU 无 aclnn 实现时用已注册算子拼出来），`REQUIRED_OPS` 是机器可读依赖清单 | — |
| `benchmark.py` | CPU(numpy) vs XPU(cupy) 加速比，**op × dtype 两个维度** | `python benchmark.py [--list\|--csv\|--category\|--matrix-elementwise]` |
| `cupy/testing/_ascend_dtypes.py` | pytest 的 dtype 过滤策略（去掉 NPU 不支持的 float64/complex） | 默认 auto；`--ascend-dtype-filter=off` 跑全量 |
| `tests/ascend/` | 无需 NPU 的回归测试（111 passed, 10 skipped） | `pytest tests/ascend -q` |

**benchmark 要点**：12 组 / 119 个算子条目，默认 133 个用例（`--matrix-elementwise` → 203）；
vector 10M、matrix 4K（matmul）；float32 全量 + float64/int64 四则运算 + int32 位运算 + bool；
每例独立容错（`UNSUPPORTED` / `FAIL` / `MISMATCH` 分开统计）；无 NPU 时 exit 3、
`--list` 可离线打印 op×dtype 矩阵。

**实测降噪效果**（本机无 NPU，`test_arithmetic.py -k "float64 and floor_divide"`）：
dtype 过滤 off → **1312 failed**；on → **1312 skipped, 0 failed**。

## 7. 下一步

优先级清单在 **[Memory.md §6](Memory.md)**（P0 打 910B 基线 → P1 补算子 → P2 组合实现 → P3 长期）。
最高优先级仍是：**在 910B 上跑出 `pytest` 基线，并验证 matmul/dot、8 个 AscendC 自定义内核与
5 个宿主端组合算子的数值正确性**（本机只能到 L3；组合算子的对拍用例已在
`tests/ascend/test_composite_ops.py` 里写好，`has_npu` 为真时自动执行）。
