# Ascend 后端的 dtype promotion：code review 与落地计划

> 状态：**方案（未落地）**，2026-09-20。
> 触发问题：`promote_types` 到底在哪算？`float32 tensor op int32` 的结果 dtype 是
> launch 前算好并插 cast，还是 ascend 内核自己处理 operand dtype 不一致？
> 参考文档：`docs/references/numpy_cupy_pytorch_dtype_promotion_comparison.md`
> （NEP 50 weak/strong scalar、CuPy 0-D 差异、PyTorch 差异；下称「参考文档」）。
>
> 相关兄弟文档：`arg_passing_plan.md`（§2.2.1 的 M2 约束——本计划的 cast 插入点
> 必须保持「操作数与参数分区」）、`refactor_exception.md`（错误传播，本计划依赖它
> 把 aclnn 的 dtype 报错变成 Python 异常）。

---

## 0. 结论（三条设计不变式）

| # | 不变式 | 现状 |
|---|---|---|
| 1 | **promote 在 host 侧、launch 之前完成**：结果 dtype 由 loop 选择（`can_cast`/`result_type` + NEP 50 weak scalar）决定，`out` 按它分配 | ✅ 已实现且与 numpy 2 对拍一致（§1） |
| 2 | **送到 aclnn 的 `(self, other, out)` dtype 一致**：`scalar op tensor` 的标量在 host 侧按 loop dtype 物化（`f32 tensor + 2.0` → f32 aclScalar，即参考文档 §1 的 weak 语义）；`tensor op tensor` 不一致时**我们显式插 cast**，不依赖 CANN 的「互相推导」 | ◐ 标量已归一（M-D1 ✅，§2 问题 1）；数组不一致零处理（§2 问题 2，待 M-D2） |
| 3 | **插入的 cast 遵守内存布局约束**：cast 的目的 buffer 一律新分配（天然连续）；非连续输入靠 aclTensor strides 表达，aclnnCast 拒绝时先连续化；dtype 已一致时零 cast | ❌ 尚无 cast 插入点（§2 问题 3） |

一句话：**ascend 内核内部不做 cast**——cast 是 host 侧的显式决策，而且次数最少。

---

## 1. 现状：host 侧 promote 已经是 numpy 语义（对拍过的部分）

### 1.1 机制

ufunc 的 loop 表没有混合类型 loop（`_routines_math.pyx:363-372`：
`'??->?'`/`'bb->b'`/…/`'ff->f'`/`'dd->d'`/…），所以每个调用必然落到某个 loop：

1. `_preprocess_args(..., use_c_scalar=False)` 把 Python/NumPy 标量转成 **numpy 标量**，
   并记录 `weak_t = type(arg) if type(arg) in [int, float, complex] else False`
   ——**只有 Python 标量是 weak**（`_kernel.pyx:69-85`），与 NEP 50 一致；
2. `_Ops.guess_routine → _guess_routine_from_in_types`：逐 loop 用 `numpy.can_cast` 判，
   weak 标量加一道 `numpy.result_type(weak_t(0), ot) == ot` 复核
   （`_kernel.pyx:1199-1231`；`_check_should_use_weak_scalar:657`）；
3. `op.out_types` → `_get_out_args_from_optionals` 按 it 分配 `out`；用户给了 `out=` 时
   `_raise_if_invalid_cast(out_type, arr.dtype, casting, ...)` 校验（`_kernel.pyx:576-600`）。

### 1.2 对拍（本机 numpy 2.4.3，全部实测）

| 表达式 | 选中 loop | 本项目 out | numpy 2 | 一致 |
|---|---|---|---|---|
| `f32_arr + 2`（Python int） | `ff->f` | float32 | float32 | ✓ |
| `f32_arr + 2.0`（Python float） | `ff->f` | float32 | float32 | ✓ |
| `f32_arr + np.float64(2.0)` | `dd->d` | float64 | float64 | ✓ |
| `f32_arr + np.int32(2)` | `dd->d` | float64 | float64 | ✓ |
| `f32_arr + i32_arr` | `dd->d` | float64 | float64 | ✓ |
| `f16_arr + 3.0` | `ee->e` | float16 | float16 | ✓ |
| `int32_0D * f32_1D` | `dd->d` | **float64** | float32（0-D 特殊规则） | ✗ 已知差异 |

> 最后一行就是参考文档 §3 的 CuPy 已知差异（0-D array 不享受 scalar-like 提升），
> **与上游 CUDA CuPy 行为相同，不修**，但要在 M-D4 测试里作为「已知差异」锁住，
> 防止将来有人当 bug 修。

`float32 tensor op float64 -> float32` 这个用户预期，指的就是第一/二行：
**Python 层的 float64 值是 weak scalar**，按 NEP 50 不抬升结果——本项目已正确。

---

## 2. 现状 code review：问题表

| # | 级别 | 问题 | 证据 | 影响 |
|---|---|---|---|---|
| 1 | **P1 → ✅ 已修复（M-D1，2026-09-20）** | ufunc 路径的**死分支**：`elif isinstance(x, _ndarray_base)` 与上一行条件重复 → `CScalar.from_numpy_scalar_with_dtype(x, t)` 永不执行；所有标量走 `scalar_to_c_scalar(x)` **保留自身 dtype**，loop dtype `t` 被忽略（修复前：`f32_arr + np.int64(2)` 传出 int64 aclScalar，靠 CANN 隐式转换） | 修复前 `_kernel.pyx:903`；CUDA 原版是 `x if isinstance(x, _ndarray_base) else from_numpy_scalar_with_dtype(x, t)`（`_gpu/_kernel.pyx:1374-1379`） | **修复后**：`elif isinstance(x, numpy.generic)` 分支按 loop dtype 物化标量（`f32_arr + 2.0` → f32 aclScalar；`+ np.int64(2)` → loop `dd->d` → f64 aclScalar），与 ElementwiseKernel 路径的 `apply_dtype(in_types[i])`（`_kernel.pyx:1430-1432`）语义一致；ascend 内核内部不再做标量 cast |
| 2 | **P1** | `tensor op tensor` dtype 不一致时**零处理**：直接 `aclnnAdd(f32, i32, out f64)` 交给 CANN | CANN 文档（`aclnnop/aclnn_add.h:29-38`）：self/other「数据类型需要与 other 构成**互相推导关系**」，out「需要是 self 与 other **推导之后可转换**的数据类型」 | ① CANN 推导结果若 ≠ loop dtype（f32 而非 f64），先低精度算再上转 → 与 numpy 数值分歧（int32 大值丢精度），**只有 910B 可验证**；② CANN 白名单（「整型、浮点」）不含 bool → `add(bool_arr, f32_arr)` 运行期失败（loop 选得出来，op 不收） |
| 3 | P2 | CUDA 的 kernel 内转换机制在 Ascend 是**死代码**：`type_map` 算出来只喂给一行注释掉的 `#kern = self._get_elementwise_kernel(dev_id, arginfos, type_map)` | `_kernel.pyx:1410-1457` | 维护误导（读代码会以为存在类型转换层）；也是「cast 插入点缺失」的根源 |
| 4 | P2 | 派发层没有 dtype 一致性检查：`launch_acl_func_raw` / general 路径不校验 `(self, other, out)` 的 dtype 组合 | `acl_utils.pyx:1072+` | 错误组合只能靠 aclnn 的 EL 报错（消息在 CANN 内部，定位差、时机晚） |
| 5 | P3 | 用户 `out=` 时 loop dtype 与 out dtype 可不同（`same_kind` 允许降位） | numpy 2.4 实测：`f32_arr += i64_arr` 与 `np.add(f32, i64, out=f32)` 均通过、结果 f32 | **与 numpy 一致** ✓；但意味着 aclnn 必须接受 out dtype ≠ 推导 dtype（仍属问题 2 的范畴） |
| 6 | 顺带 → ✅ 已修复（2026-09-20） | `aclop_IsClose` 参数读取错位：`atol` 读 `args[0]/"rtol"`、`rtol` 读 `args[1]/"atol"`、`equal_nan` 读 `args[1]/"order"`——三者交叉，`equal_nan` 实际拿到 atol 的值当 bool | 修复前 `acl_general_ops.h:445-447`；cupy 侧 ufunc 是 `_is_close(a, b, rtol, atol, equal_nan)`（nin=5，`cupy/_logic/comparison.py:132`），标量操作数按序落在 `args[0..2]` | **修复后**：`rtol/atol/equal_nan` 分别读 `args[0/1/2]`（键名 `rtol/atol/equal_nan`），aclnnIsClose 按签名 `(self, other, rtol, atol, equal_nan, out)` 传参；删除死代码 `indices`（nout 恒为 1）；注释说明 CANN 头文件里 rtol/atol 的中文描述写反、按参数名对齐 numpy。数值验证（rtol/atol 语义）归入 M-D5 |

---

## 3. 目标设计

### 3.1 数据流（cast 只出现在显式标注的一处）

```
Python 调用
  → _preprocess_args            标量 → numpy scalar；记录 weak_t（Python type）
  → guess_routine               host 侧 promote（can_cast / result_type）→ op(in_types, out_types)
  → [M-D1] 标量物化             from_numpy_scalar_with_dtype(x, in_types[i])
                                → aclScalar 的 dtype == loop dtype（host 转换，无设备 kernel）
  → [M-D2] 数组 cast            x.dtype != in_types[i] 时插一次显式 cast（新分配连续 buffer）
  → out = out_types[i]          用户 out → _raise_if_invalid_cast(casting)
  → aclnn 收到 (in_types…, out_types) —— 与 loop 完全一致 ⇒ 内核内零 cast
```

`scalar op tensor` 三种情形（对应参考文档 §7 的四元组）：

| 表达式 | loop | 标量物化成 | 依据 |
|---|---|---|---|
| `f32_arr + 2` / `+ 2.0` | `ff->f` | **f32** aclScalar | NEP 50：Python 标量是 weak，值 `2.0` 在 host 上按 f32 舍入（`1e300` → inf，与 numpy 一致，因为结果本来就是 f32） |
| `f32_arr + np.float64(2.0)` | `dd->d` | **f64** aclScalar | NumPy 标量是 strong |
| `f32_arr + np.int32(2)` | `dd->d` | **f64** aclScalar | 同上（先升位再计算，这正是「避免内核内 cast」的含义：cast 已在 host/显式步骤做完） |

### 3.2 cast 的内存布局规则（不变式 3 的细则）

1. **cast 结果一律新分配**（连续 buffer），不原地 cast——原地会破坏 `out=` 的别名语义与
   `_copy_in_args_if_needed` 的 `may_share_bounds` 判断。
2. **非连续输入**：aclTensor 自带 strides，多数 aclnn 接受；cast 的输入也走 strides。
   若 910B 实测 `aclnnCast` 拒绝非连续 src（EL0003 一类），则先连续化（`ascontiguousarray`
   语义）再 cast。既有范式可参照：
   * `aclop_LeftShift` 的组合实现用 `aclTensorLike`（连续分配）做 cast 中转
     （`acl_math_ops.h:246-264`）；
   * `aclop_Copy` 已有防御：src/out dtype 元数据为 `ACL_DT_UNDEFINED` 直接拒绝
     （`acl_general_ops.h:458-473`，NPU 实测 `aclnnInplaceCopy` 不可靠、统一走 `aclnnCast`）。
3. **零 cast fast path**：所有输入 dtype == out dtype 时不插任何 cast（绝大多数调用的热路径）。
4. **次数最小化**：同一输入只 cast 一次；`out` 永不 cast（dtype 由 loop 决定，写不进来的情况
   由 `_raise_if_invalid_cast` 在 host 侧拒绝）。
5. **生命周期**：cast 临时 buffer 用 host 侧 cupy 数组持有（方案 A，见下）或 C++ RAII
   （`AclScalarTensorGuard` / `aclTensorLike` + `DestroyTensorLike` 范式），禁止裸指针。

### 3.3 实现载体：A（host 侧 Cython 插 cast，推荐）vs B（C++ guard）

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **A（推荐）** | `_kernel.pyx` 的 inout_args 构建处：`if isinstance(x, _ndarray_base) and x.dtype != t: x = x.astype(t)` | 复用 cupy 现有 `elementwise_copy`（→ `aclop_Copy` → `aclnnCast`）：连续性、生命周期、缓存全白得；无新 C++；与 M2 无耦合 | dtype 不一致时多一次 kernel launch + 一份临时显存（仅混合 dtype 路径） |
| B | dispatch/C++ 侧 `AclCastGuard`，解包时按需 cast | host 开销更小；M2 的 `AsGeneralOp<Fn>` 里可复用 | 要自己管理 RAII/连续性/所有权；在 M2 落地前没有干净的插入点 |

> 方案 A 的插入点在 `_kernel.pyx`（操作数通道），不碰注册表与参数区，与
> `arg_passing_plan.md` §2.2.1 约束 1（操作数与参数分区）天然兼容。

---

## 4. 落地计划

| 步骤 | 内容 | 工作量 | 验证 |
|---|---|---|---|
| **M-D1 ✅** | 修死分支：`elif isinstance(x, numpy.generic)` → `from_numpy_scalar_with_dtype(x, t)`，对齐 CUDA 语义；顺带核对 ElementwiseKernel 路径（已一致）与 reduction 路径是否也有标量归一缺口 | 0.5d（已实施） | 编译通过 + `pytest tests/ascend` 212 passed（5 个失败为本修复前已存在的过期断言）；行为级验证（aclScalar dtype == loop dtype，用 `ascend_dump_args` 探针）需要设备，归入 M-D5 |
| **M-D2** | 数组 cast 插入（方案 A）+ 零 cast fast path + §3.2 规则落地 | 1d | 无 NPU：探针/mock 断言「cast 只在 dtype 不一致时插入、目标 dtype == loop dtype、每输入至多一次」 |
| **M-D3** | 派发层 dtype 一致性 guard：wrapper 收到混合 dtype 且该 op 未声明支持 → 报错（过渡开关 `CUPY_ASCEND_ALLOW_MIXED_DTYPE=1`，仿 M1 严格模式哲学：宁可响亮失败） | 0.5d | 单测：混合 dtype 默认抛、开关放行 |
| **M-D4** | 对拍测试 `tests/ascend/test_dtype_promotion.py`：§1.2 表全量（loop 选择 vs numpy）+ 0-D 已知差异 + inplace/`out=` 降位（numpy 允许，实测）+ weak 极端值（`f32_arr + 1e300` → inf） | 0.5d | 无 NPU |
| **M-D5** | 910B 数值基线：`f32+i32` 大整数精度、bool 组合、非连续输入 cast、cast 性能开销 | — | 与 numpy 数值对拍 |

依赖关系：M-D1 独立可先行（纯行为对齐 + 修 bug）；M-D2 依赖 M-D1 的探针设施；
M-D3 在 M-D2 之后才有意义（那时混合 dtype 本来就不该到达 aclnn）。

---

## 5. 风险与开放问题

| 项 | 说明 |
|---|---|
| 性能 | 混合 dtype 每输入 +1 次 cast kernel + 1 份临时显存；**同 dtype 热路径必须零开销**（fast path 只是一次 dtype 比较）。若实测 cast 开销显著，再考虑方案 B 或白名单跳过（如 CANN 已证实某组合推导无损）。 |
| CANN 隐式推导与 numpy 的差异 | 目前只能 910B 实测枚举（哪些组合 CANN 推导成什么、是否有精度分歧）。M-D2 落地后此风险**整体消失**（不再依赖隐式推导），但 M-D2 之前的过渡期里错误仍然可能静默——这是 M-D3 guard 存在的理由。 |
| bool | 即使 M-D2 落地（bool → f32 的 cast），也要确认 `aclnnCast` 对 bool 输入的支持——`acl_reduction_ops.h:349` 已注明「bool->int aclnnCast 未验证」。`add(bool, bool)`（loop `??->?`）则是另一个问题：aclnnAdd 不收 bool，bool 变体应注册到 `aclnnLogicalOr/And/…`（cupy 语义 `+` 即 `|`），属算子注册问题，另行处理。 |
| `astype` 的 order='K' | 非连续输入 cast 后保持布局类别；aclnnCast 对非连续 src 的支持要 910B 实测（拒绝则按 §3.2 规则 2 连续化）。 |
| weak 标量极端值 | `f32_arr + 1e300`：host 物化成 f32 → inf，与 numpy（NEP 50 不看值）一致；M-D4 用测试锁住。 |
| 大整数标量 | `int64_arr + 2**63` 的 loop 选择与 numpy 的报错行为需在 M-D4 对拍（低风险，列出备查）。 |
| inplace 的 out 降位 | `f32_arr += i64_arr`：numpy 2 允许（f64 里算、写回 f32），本项目 loop `dd->d` + `same_kind` 亦放行——行为一致，但意味着 aclnn 拿到 `out=f32` 而 loop 是 f64，M-D2 的 cast 规则要保证此时 `out` 不被 cast、输入按 in_types 处理。 |

---

## 6. 参考

* `docs/references/numpy_cupy_pytorch_dtype_promotion_comparison.md`（NEP 50 weak/strong、
  CuPy 0-D 差异、PyTorch 差异——§1.2 表的「已知差异」出处）
* `cupy/_core/_ascend/_kernel.pyx`（loop 选择 `:1199-1231`、weak 判定 `:69-85`/`:657`、
  inout 构建 `:899-908`、`apply_dtype` `:1430-1432`、死掉的 type_map `:1410-1457`）
* `cupy/_core/_gpu/_kernel.pyx:1374-1379`（CUDA 原版：数组原样、标量按 loop dtype 物化）
* `cupy/_core/_routines_math.pyx:355-372`（loop 表）
* CANN 头文件：`aclnnop/aclnn_add.h:29-38`（「互相推导关系」）、`aclnnop/aclnn_cast.h`
* 既有 cast 范式：`acl_math_ops.h:246-264`（`aclTensorLike` 连续中转）、
  `acl_general_ops.h:458-473`（`aclop_Copy` 的 ACL_DT_UNDEFINED 防御）
* `docs/ascend/arg_passing_plan.md`（§2.2.1 M2 约束：操作数与参数分区；
  本计划的 cast 插入点遵守该约束）
