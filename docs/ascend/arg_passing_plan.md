# 参数传递（args / kwargs）实现计划 —— Ascend 后端

> 上下文：`docs/ascend/code_review_ascend_backend.md` §2.2 指出
> **`launch_acl_func` 完全丢弃 `args` / `kwargs`**（生成代码里是 `CYTHON_UNUSED`）。
> 本文给出修复计划，并回答四个相关问题：
> ① aclScalar 是否支持 string？② 是否需要更抽象的 arg？
> ③ aclnn 算子是否一般不支持 str 参数？④ 是否要额外向 `launch_general_func` 传字符串？

---

## 0. 结论先行

> 状态（2026-09-20）：①③④ 的结论已由**代码**回答，②「类型化 arg」已实现（§2.4）。
> 每条结论后面标注了落地位置。

| 问题 | 结论 |
|---|---|
| ① `aclScalar` 支持 string 吗？ | **支持但有代价**：`ACL_STRING = 13` 存在，但 `aclScalar` 的值存在**内联 union** 里，没有 `char*` 槽位 → 只能指针语义，host 侧必须保活。**本次改为不走 aclScalar**：`ARG_STRING` 在 C++ 侧用 `std::string` 自持所有权（`acl_scalar_arg.h`），彻底回避 pointer-keyed keepalive。消费状态见附录 A（唯一真实用户是 `einsum` 的 equation）。 |
| ② 需要更抽象的 arg 吗？ | **需要，且已经实现为"类型化"而不是"更泛化"**：`AclArgKind {ARG_NONE, ARG_SCALAR, ARG_INT_ARRAY, ARG_STRING, ARG_TENSOR}` + `AclArg`（§2.4）。每算子 arg spec 声明表仍然是 TODO（当前用类型驱动 + `_KNOWN_SCALAR_KEYS` 白名单）。 |
| ③ aclnn 是否一般不支持 str？ | **实践上是的**。CANN 9.0.1 的 817 个 aclnn 头里 **110 个**声明了 `const char*`，其中 numpy 风格的只有 `einsum`（equation）/ `resize`（mode）/ `roi_align`，其余全是 quant / attention / 通信 / norm 类；我们派发链路上的 elementwise / reduction / sorting 算子全部是数值/布尔。完整分类见附录 A。 |
| ④ 是否要额外传 string 给 `launch_general_func`？ | **不建议作为常规路径**。字符串应在 **host 侧解析成 int/bool**（wrapper 已经这么做：`order`/`stable`/`descending`/`some`/`sorted` 都是 int 编码）。`ARG_STRING` 作为逃生舱已打通（白名单 `_STRING_ARG_OPS` + 探针 op 验证），等 `einsum` 落地时直接用。 |

---

## 1. 现状（实测证据）

### 1.1 只有 2/10 个函数指针槽位有参数通道

`ctypedef union FuncPtrUnion`（`acl_utils.pyx:556`）10 个成员中：

| 槽位 | 签名是否含 args/kwargs |
|---|---|
| `reduction_op` | ✅ `aclReductionOpRun(self, dim, keepdim, out, wsfunc, kfunc, stream, kwargs)` |
| `general_op` | ✅ `aclop_X(ins, outs, args, kwargs, stream)` |
| `unary_op` / `inplace_unary_op` / `binary_op` / `inplace_binary_op` / `scalar_binary_op` / `inplace_scalar_binary_op` / `tri_op` / `inplace_tri_op` | ❌ 固定签名，**没有参数入口** |

规模：`register_acl_ufunc` 中走「窄签名」路径的注册 **137 处**，`GENERAL_OP` **31 处**。
即**绝大多数算子今天无法接收任何额外参数**。

### 1.2 实际会出现的参数类型（而非猜测）

| 来源 | 参数内容 | 类型 |
|---|---|---|
| ufunc 路径 `_kernel.pyx:795-800` | `out` / `_where` / `dtype` / `casting` **已 pop 掉**，不会进入后端 | — |
| ufunc 路径 `_kernel.pyx:913` | `kwargs["where"] = <ndarray>`（`has_where` 分支） | **ndarray** |
| ufunc/reduction 路径 | `pos_args = args[nin+nout:]`（正常调用为空） | 标量 |
| Ascend 原生 routines | `[offset]`、`[k]`、`[not complete]`、`[axis, 1, 0]` | int/bool |
| C++ 侧 `GetScalarArg` 已消费的 key | `atol axis bins descending dim full_matrices k keepdim max min nan neginf order posinf rtol shift some sorted stable start step stop` | **全部数值/布尔**（枚举已 int 编码） |

> 关键观察：**没有任何字符串**出现在当前 args/kwargs 中；唯一非标量是 `where` 的 ndarray，
> 而它今天被 `if ascalar: push_back` 静默丢弃（review §2.5）→ `cupy.add(a, b, where=mask)`
> 会在 mask=False 的位置也写结果，**静默错误**。

### 1.3 三种静默丢参表现

1. 窄签名槽位：`launch_acl_func` 的 `args`/`kwargs` 参数在生成代码里是 `CYTHON_UNUSED`（review §2.2）。
2. `_convert_arg_to_acl_scalar` 返回 `NULL` 时被 `if ascalar: push` 忽略（只有一行 print）。
3. 类型不支持（ndarray / str / None）时同样静默 `NULL`。

---

## 2. 设计

### 2.1 阶段 1 —— 止血（P1，小改动，无 NPU 可验证）  DONE!

1. **修 review 指出的一并缺陷**（同批提交，都是静默错结果）：
   - reduction 注册三元错位：`ascend_argmax → aclop_ArgMin`、`ascend_argmin → aclop_Mean`、`ascend_mean → aclop_ArgMax`（review §2.1）
   - `_create_ops_vector(ops, outs)` 重复创建 out 张量（review §2.3）
2. **参数严格模式**：`if ascalar: push_back` → 不可转换即
   `raise ValueError(f"{opname}: unsupported arg #{i} {type(arg).__name__}")`；
   临时放宽开关 `CUPY_ASCEND_LENIENT_ARGS=1`（默认关闭），便于渐进迁移。
3. **arg spec 白名单**：为每个算子声明其接受的 key 集（从 `GetScalarArg` 调用点自动提取，
   当前共 22 个 key）；未声明的 key → 报错而非丢弃。
4. **测试**：注册一个只记录收到的 `(ins, outs, args, kwargs)` 的测试用 `general_op`，
   断言参数确实到达 C++ 层。无 NPU 可跑（`tests/ascend/`）。

### 2.2 阶段 2 —— 统一参数通道（P1-P2，主改动）

**推荐方案 B：全量收敛到 `general_op` 单签名。** 保留 `OpType` 仅作元数据（arity / inplace /
scalar 判定），新增一个 union 成员 `unified_op`，用 C++ 模板把现有窄签名 wrapper **零改动**地
适配成 general 形态：

```cpp
// 新增：把任意窄签名 wrapper 包装成 general 形态（绑定发生在注册时）
template <auto Fn>
aclError AsGeneralOp(const vector<const aclTensor*>& ins, const vector<aclTensor*>& outs,
                     const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
    // 窄签名的一元/二元/… 在这里解包 ins/outs/args（含 scalar 兜底）
}
...
func_union.unified_op = &AsGeneralOp<aclop_Sqrt>;   // 具体实例化 → 可放入 union
```

收益：
- **只有一条参数通路**：arg 校验、`where`、nogil 改造、RAII 只需做一次；
- **所有既有注册不用改写**（实测 222 行调用 / 188 个名字 / 110 个 `aclop_*` wrapper，见 §2.2.1；
  旧记的 137 应是只算了窄签名那批）。注册代码仍写 `register_acl_ufunc("ascend_sqrt", UNARY_OP, ...)`，
  由注册函数内部按 `OpType` 选择 `unified_op = &AsGeneralOp<Fn>`。
- `launch_acl_func` 退化为 `launch_general_func` 的薄包装（或直接删除）。

**首个真实用例：`where`（ndarray 参数）**打通全链路。
> 注意：把 mask 送进 C++ 只是第一步，"按 mask 选择输出"的语义实现（`aclnnSWhere` 或
> mask 合成）是独立任务，需在 910B 上验证。

### 2.2.1 M2 的 3 条硬约束（2026-09-20 评审补充）

> 起因：讨论「`REVERSE_SCALAR_BINARY_OP` 能否被 `AclArg` 里的位置信息取代」时发现：全量
> general_op 有 3 条必须显式写下来的约束，否则 M2 会把 M1 刚消灭的**静默错误结果**以新形式引回来。

**现状规模（实测，供估算 M2 代价）**

| 项 | 数量 |
|---|---|
| `register_acl_ufunc(...)` 调用 | 222 行 |
| distinct 注册名 | 188 |
| `aclop_*` wrapper | 110 |
| OpType 分布 | UNARY 56 / GENERAL 41 / BINARY 37 / REDUCTION 25 / INPLACE_UNARY 21 / SCALAR 18 / REVERSE 12 / INPLACE_BINARY 7 / INPLACE_SCALAR 1 |
| 同时注册 SCALAR + REVERSE 的名字（= 真正需要「方向」的） | **12**：`subtract`/`sub`、`true_divide`、`floor_divide`、`fmod`、`power`、`float_power`、`remainder`、`greater`、`greater_equal`、`less`、`less_equal` |
| 按位置读参数的调用点 `GetScalarArg<T>(args, k, ...)` | 34 |

#### 约束 1（必须）：操作数与参数**分区**，`ArgsType` 的位置语义固定为「参数区下标」

`FindArg` 的取值顺序是「kwargs → `args[argIndex]`」（`acl_scalar_arg.h:194`），而 `GetScalarArg`
**只校验 kind**（`acl_scalar_arg.h:555`）：

```cpp
// acl_scalar_arg.h:194  —— kwargs 缺失时回退到位置参数
if (argIndex >= 0 && argIndex < static_cast<int>(args.size())) { return &args[argIndex]; }
// acl_scalar_arg.h:555  —— 只要 kind 是 ARG_SCALAR 就照单全收
if (arg->kind != ARG_SCALAR) { /* warn 或 throw */ }
return ToScalarArg<ToScalarType>(arg->scalar);
```

而 pyx 侧今天已经会往 `acl_args` 里塞**标量操作数**（非 ndarray 的 `ins` 元素），位置参数随后追加
（`acl_utils.pyx:857-869`）。于是「操作数 scalar 占 `args[0]`，参数被挤到 `args[1…]`」——此时那 34 处
`GetScalarArg<T>(args, 0, kwargs, "axis", -1)` 会**安静地把操作数的值当成 `axis`**：kind 检查通过
（它确实是 `ARG_SCALAR`），没有 warning、没有异常。例：`aclop_Sort` 的 `axis` 读的就是 `args[0]`。

**规则**：

1. 操作数（ndarray **和** scalar）一律不进 `ArgsType`：tensor 走 `intensors`，scalar 操作数继续留在
   `ins` 里（由 pyx 的 `scalar_index` 记录位置，dispatch 决定是 `SCALAR_BINARY_OP` 还是 LHS 形态）。
   **M2 不改这个布局**：`AsGeneralOp<Fn>` 解包时按 `ins` 取操作数。
2. 若将来确实要让 C++ 侧看见操作数位置（例如自证方向），必须**另开区域或加显式标记**
   （独立的 `operands` 向量，或 `AclArg::is_operand`），并保证 `FindArg` 的 `argIndex` 只指向参数区。
   **不允许**用「前 N 个槽位是操作数」这类隐式约定。
3. **自检**：import 期对每个注册的 unified op 做一次「只解包、不执行 aclnn」的 dry-run，校验 arg 形状
   与 `Fn` 的签名一致（无 NPU 可跑）。

#### 约束 2：方向走「元数据位 + 适配器模板」，不在 wrapper 里靠 `args[k].kind` 猜

- `OpType` 继续作元数据，但把 `REVERSE_SCALAR_BINARY_OP` 表述为 **`SCALAR_BINARY_OP | SCALAR_IS_LHS`**
  （方向是标志位，不是独立槽位），`FuncPtrUnion` 相应删掉 `reverse_scalar_binary_op` 成员；
- 12 个双向算子的 `aclop_R*` 手写 wrapper 用一个适配器模板收敛掉（每算子零手写代码）：

```cpp
template <auto ForwardFn, auto ReverseFn>
aclError AsScalarBinaryDirectional(const std::vector<const aclTensor*>& ins,
                                   const std::vector<aclTensor*>& outs,
                                   const ArgsType& params, const KwargsType& kwargs,
                                   aclrtStream stream, bool scalar_is_lhs);
```

- **反例（不要做）**：把方向判断塞进每个 wrapper（`if (args[0].kind == ARG_SCALAR) aclop_Rsubs(...)`）。
  代价有三：① 12 个算子各写一遍运行时分支；② 把「方向」和「实现策略（原生 ScalarTensor / 换边 /
  物化标量）」两个正交维度压进同一个函数；③ aclnn 层面的不对齐不会因为 wrapper 统一而消失，只会从
  「注册表分工」变成「函数内分支」——反向能力只有 12/188 个名字需要，不值得让 188 个都改协议。

#### 约束 3：类型安全来自 `Fn` 的函数类型，不靠运行时 `kind` 判断

窄签名 wrapper 的参数表本身就是「签名描述」。`AsGeneralOp<Fn>` 应从 `Fn` 的函数类型推导期望的
（arity + 每个位置的 kind），在解包时做静态/解包期校验；wrapper 保持窄签名**零改动**。

| 做法 | wrapper 改动 | 新增机制 | 静默错误风险 |
|---|---|---|---|
| **A（推荐）** `AsGeneralOp<Fn>` 自动解包 + 操作数/参数分区 | 0 | 1 个模板 + `Fn` trait 校验 | 低（解包期挡住） |
| B 同上，但操作数混进 `args` | 0 | 「前 N 槽是操作数」隐式约定 | **高**（约束 1） |
| C 手写 unified wrapper（每算子吃 `AclArg` 判方向） | 12 个算子各 +5~15 行 | 每函数内 `if` 分支 | 中（运行时才暴露） |

#### 附：方向问题的 aclnn 事实（供 M2 定接口时参考）

| numpy 运算 | `(t,t)` | `(t,s)` | `(s,t)` |
|---|---|---|---|
| subtract | `aclnnSub` | `aclnnSubs` | ✅ 原生 `aclnnRsubs`（`out = other - self*alpha`） |
| multiply / divide / floor_divide | `aclnnMul` / `aclnnDiv` / `aclnnFloorDivide` | `aclnnMuls` / `aclnnDivs` / `aclnnFloorDivides` | ❌ 无 |
| fmod | `aclnnFmodTensor` | `aclnnFmodScalar` | ❌ 无 |
| remainder / power | `...TensorTensor` | `...TensorScalar` | ✅ 原生 `...ScalarTensor` |
| gt / ge / lt / le | `aclnnGtTensor` … | `aclnnGtScalar` … | ❌ 无（reverse = **换边**：`scalar > x` ≡ `aclnnLtScalar(x, scalar)`） |
| maximum / minimum | `aclnnMaximum` / `aclnnMinimum` | ❌ **无 scalar 变体** | ❌ 无 |

> 注：比较运算的 `(s,t)` 形态**只有显式 ufunc 调用才会到达**——CPython 的 `do_richcompare` 会把
> `1 > x` 交换成 `x.__richcmp__(1, Py_LT)` → `cupy.less(x, 1)`（标量仍在右）。详见 §2.4(c) 的备注。

整个 CANN 里 `ScalarTensor` 家族只有 4 个（`Rsubs`、`PowScalarTensor`、`RemainderScalarTensor`、
`IsInScalarTensor`）⇒ 反向实现只有三条路：**调原生 / 换边 / 物化标量（`AclScalarTensorGuard`）**。
这三条路的分支必须留在**实现层**（wrapper / 适配器模板），不能变成「注册表键的第三维」。

### 2.3 阶段 3 —— 类型化 arg（"abstract arg"）

跨边界的参数改为 **tagged 容器**，而不是继续传 Python 对象：

```cython
cdef enum AclArgKind: ARG_NONE, ARG_SCALAR, ARG_INT, ARG_BOOL,
                      ARG_INT_ARRAY, ARG_SHAPE, ARG_ARRAY, ARG_STRING, ARG_FLOAT
cdef struct AclArg:            # C 侧
    AclArgKind kind
    aclScalar* scalar          # ARG_SCALAR/ARG_INT/ARG_BOOL/ARG_FLOAT
    aclIntArray* int_array     # ARG_INT_ARRAY/ARG_SHAPE
    aclTensor* tensor          # ARG_ARRAY (e.g. where)
    const char* cstr           # ARG_STRING
```

- **声明表**：`ARG_SPECS = {'ascend_sort': {'axis': ARG_INT, 'stable': ARG_INT, ...}, ...}`
  （可由 `GetScalarArg` 调用点自动生成，避免手工漂移）
- **C++ 消费**：`GetScalarArg<T>` 升级为 `GetArg<T>(args, index, kwargs, key, default)`，
  支持 `const char*` / `aclIntArray*` / `aclTensor*` 三个新增重载
- **保活**：统一 `_acl_arg_owners`（bytes / ndarray / aclIntArray 各一类），
  与 patch 里 `_acl_scalar_owners` 的机制合并
- **字符串策略（正式规则）**：
  1. 能在 host 侧判定的一律 host 侧解析（`kind='stable'` → 1、`mode` → 枚举、`order` → flag）；
  2. op spec 未声明 `ARG_STRING` 而收到 `str` → **抛 `NotImplementedError`**（响亮失败）；
  3. 仅当 aclnn 签名确为 `const char*` 时才允许 `ARG_STRING` 透传
     （当前 1 个在用：`ascend_einsum`，见 §4 风险表）。

---

## 2.4 本次实现（M3-lite + reverse scalar）

### (a) 统一参数通道：typed `AclArg`

`cupy/backends/ascend/acl_scalar_arg.h`（**唯一**定义处；`acl_op_template.h` 里重复的一份
`using KwargsType/ArgsType` 已删除并改为 include）：

```cpp
enum AclArgKind { ARG_NONE, ARG_SCALAR, ARG_INT_ARRAY, ARG_STRING, ARG_TENSOR };
struct AclArg {
    AclArgKind kind = ARG_NONE;
    const aclScalar* scalar = nullptr;   // ARG_SCALAR
    const aclTensor* tensor = nullptr;   // ARG_TENSOR（预留 where=）
    std::string str;                     // ARG_STRING（自持所有权）
    std::vector<int64_t> ints;           // ARG_INT_ARRAY（**按值**）
};
using ArgsType = std::vector<AclArg>;
using KwargsType = std::unordered_map<std::string, AclArg>;
```

读取接口（类型不符 → 缺省值 / 报错，**不做指针重新解释**）：

| 接口 | 用途 |
|---|---|
| `GetScalarArg<T>(args, i, kwargs, key, default)` | 签名与语义不变（37 处调用点零改动），内部走 tag |
| `TryGetInt64List(args, i, kwargs, key, out)` | **int / 序列[int] / None 三态**：multi-axis 参数入口 |
| `GetStringArg` / `GetTensorArg` / `HasArg` / `HasIntArrayArg` | 其余 tag |
| `AclIntArrayGuard` | 把 `vector<int64_t>` 现场变成 `aclIntArray*` 并在析构时释放 |

两个关键设计决定：

1. **`ARG_INT_ARRAY` 传值不传 `aclIntArray*`**：CANN 头里 `aclIntArray` 是**不透明类型**
   （只有 `aclCreateIntArray/aclDestroyIntArray`，没有 `Size()/operator[]`，见
   `aclnn/acl_meta.h`），C++ 侧读不出内容；传值后由需要 `aclIntArray*` 的 aclnn 接口
   现场构造（`AclIntArrayGuard`，空序列 → `nullptr`，正是 `axis=None` 的语义）。
2. **`ARG_STRING` 用 `std::string`**：避免 pointer-keyed keepalive（同一 `bytes` 被两个
   参数复用时 pop 会提前释放）。

pyx 侧：`_convert_arg(opname, name, value)` 是**唯一**转换入口（类型驱动：`None`→NONE，
`str`→STRING（白名单）、`list/tuple[int]`→INT_ARRAY、数值/布尔→SCALAR；序列里混入
float/str 直接报错而不是截断）；`_destroy_arg` 只对 `ARG_SCALAR` 做释放（其余按值持有）。

### (b) 已接入 IntArray 的算子

`aclop_Flip(axis=)`、`aclop_Roll(shift=, axis=)`、`aclop_Permute(dims=)`、
`aclop_Aminmax(dim=)`（`acl_general_ops.h`）。这四处原先的注释写的是
「Multi-axis tuples cannot be expressed through the positional-scalar arg channel yet
(see arg_passing_plan.md M3)」——现在 `flip(a, (0, 2))` / `roll(a, (1, 2), axis=(0, 1))`
可达。（`Aminmax` 的 aclnn 接口只吃单个 `int64_t dim`，序列取第一个轴，已在注释中说明。）

### (c) reverse scalar binary（`scalar <op> tensor`）

问题（review 未覆盖的一类**静默错误结果**）：

| 调用 | 旧行为 |
|---|---|
| `x - 1` | `("ascend_subtract", SCALAR)` 未注册（scalar 变体被注册成了 `ascend_sub`）→ 报错 |
| `1 - x` | 同上 → 报错（若注册了会算成 `x - 1`） |
| `2 / x` | `("ascend_true_divide", SCALAR)` 已注册 → `Divs(x, 2)` = **`x / 2`（静默算错）** |
| `2 ** x` | `("ascend_power", SCALAR)` 已注册 → **`x ** 2`（静默算错）** |
| `cupy.greater(1, x)`（**显式 ufunc 调用**） | `GtScalar(x, 1)` = **`x > 1`（静默算错）** |
| `x // 2` | `("ascend_floor_divide", SCALAR)` 未注册 → 报错 |

> **触发条件（易错点，2026-09-20 订正）**：只有 `ins[0]` 是标量时才走 REVERSE。**算术**的反射调用
> 会把**原始顺序**交给右操作数类型的同一个 `nb_*` 槽——`core.pyx:77-85` 的注释（"extension types
> in Cython 0.x shares implementations of op and rop"）+ 生成的 C 代码可证：`nb_subtract` 就是
> `__sub__` 的 wrapper 本身，没有任何交换：
>
> ```
> /* cupy/_core/core.cpp */
> #define __pyx_nb_subtract_4cupy_5_core_4core__ndarray_base __pyx_pw_...(ndarray_base)_121__sub__
> static PyObject *__pyx_pw_..._121__sub__(PyObject *__pyx_v_x, PyObject *__pyx_v_y) { ... }
> ```
>
> 所以 `1 - x` / `2 / x` / `2 ** x` 的 `ins` 确实是 `(标量, 张量)` → REVERSE ✓。
>
> **比较运算的运算符写法不走这条**：CPython 的 `do_richcompare` 会**交换操作数并反转比较符**
> （`_Py_SwappedOp`），`1 > x` 实际到达的是 `__richcmp__(x, 1, Py_LT)` →
> `numpy.less(x, 1)`（`core.pyx:1253-1257`）→ 标量在**右**，走 `SCALAR_BINARY_OP`，
> **旧代码也不会算错**。比较类的 REVERSE 只能由**显式 ufunc 调用**触发：
> `cupy.greater(1, x)`、`np.less(1, x)`、`cupy.less_equal(2, x)`、`cupy.not_equal(1, x)` …

实现：

- `acl_opinfo.h`：新增 `REVERSE_SCALAR_BINARY_OP = 10`；`FuncPtrUnion` 新增
  `reverse_scalar_binary_op`，签名 `(const aclScalar* self, const aclTensor* other,
  aclTensor* out, aclrtStream)`（**标量在前**，避免与 `scalar_binary_op` 混淆）。
- `launch_acl_func` 记录标量位置（`scalar_index`/`n_scalars`）→ `get_op_type(..., scalar_is_lhs)`；
  交换律算子（`_COMMUTATIVE_OPS`：add/multiply/maximum/minimum/logical_*/bitwise_*/
  hypot/logaddexp/gcd/lcm/equal/not_equal）在没有 reverse 实现时**回退**到 `SCALAR_BINARY_OP`；
  非交换律算子没有 reverse 实现时**报错**（不再算错）。
- C++ 实现（`acl_math_ops.h`）：

| wrapper | 实现方式 |
|---|---|
| `aclop_Rsubs` | CANN 原生 `aclnnRsubs(self, other, alpha, out)`，语义 `out = other - self*alpha` |
| `aclop_RPowScalar` | CANN 原生 `aclnnPowScalarTensor(self_scalar, exponent_tensor, out)` |
| `aclop_RRemainderScalar` | CANN 原生 `aclnnRemainderScalarTensor` |
| `aclop_RDivs` / `aclop_RFloorDivides` / `aclop_RFmodScalar` | 无 ScalarTensor 版本 → `AclScalarTensorGuard` 把标量物化成 **1 元素张量**（`aclrtMalloc` + `aclnnInplaceFillScalar`，RAII 释放），再走 `aclnnDiv`/`aclnnFloorDivide`/`aclnnFmodTensor` 的 broadcast |
| `aclop_RGtScalar` / `RGeScalar` / `RLtScalar` / `RLeScalar` | 比较运算的 reverse 就是**换边**：`scalar > tensor` ≡ `aclop_LtScalar(tensor, scalar)` |

- 注册名修正：`cupy_subtract`（`create_arithmetic('subtract', ...)` 生成的 ufunc 名）的
  scalar 变体原来注册成 `ascend_sub` → 补 `ascend_subtract`（两个名字都保留）；
  `ascend_floor_divide` 补 `SCALAR_BINARY_OP`（`aclop_FloorDivides`）。
- **仍未覆盖**：`atan2` / `copysign` 的 scalar 形态（aclnn 无对应算子，需要 materialization 或组合实现）；
  `maximum` / `minimum` 今天 `cupy.maximum(x, 1)` 会报「未注册」，但**不是补一行注册能解决的**：
  实测 CANN 只有 `aclnnMaximumGetWorkspaceSize` / `aclnnMinimumGetWorkspaceSize`（都是 tensor-tensor），
  **没有 scalar 变体**，必须先物化标量（`AclScalarTensorGuard`）再走 `aclBinaryOpRun`；
  `hypot` 连 aclnn 对应都没有（现为组合实现，见 `aclop_Hypot`）。另外 `_COMMUTATIVE_OPS` 的回退
  只覆盖 `scalar <op> tensor → SCALAR_BINARY_OP` **一个方向**，反向缺注册不会回退（见 §2.2.1 约束 2）。

### (d) 测试与验证

- `tests/ascend/test_unified_args.py`（新增，**无 NPU**）：探针 op `ascend_dump_args`
  断言 scalar/int_array/string/none 按 tag 送达 C++；`py_get_op_type` 断言标量位置 → OpType；
  `py_is_registered` 断言 reverse 变体存在（名字 + OpType 双向核对）。
- `tests/ascend/test_arg_passing.py`：随语义更新（`None` 与 `[1, 2]` 不再是 unsupported）。
- 构建：`CUPY_INSTALL_USE_ASCEND=1 python setup.py build_ext --inplace`；`pytest tests/ascend`
  （190 passed，5 个 pre-existing failure 见提交说明）。

### (e) 与 §2.2「方案 B」的关系

本次**没有**做「137 处窄签名全量收敛到 `unified_op` + `AsGeneralOp<>`」（M2）。
typed channel 是 M2 的前置：`AsGeneralOp<Fn>` 解包时可以直接复用
`GetScalarArg` / `TryGetInt64List`，不必再造一套参数表示。

---

## 2.5 补充需求：launch 原语必须把 Ascend 错误码交给 caller（noexcept 方向）

**约定**（本次已落地结构，属于 refactor_exception.md 的 S2）：

1. `launch_general_func` / `launch_acl_func` / `launch_reduction_op` 三个**原语**返回
   `aclError`（**不是 `void`**），acl/aclnn 失败时**不抛异常**，把错误码原样返回给 caller
   —— 这样这一层将来可以是 `noexcept`，C/C++ 调用方不承担跨语言异常。
2. Python 路径统一用 `launch_general_func_checked` / `launch_acl_func_checked` /
   `launch_reduction_op_checked`：内部调用原语，非 0 时 `raise_acl_op_error()`（带
   `aclGetRecentErrMsg()`）。**抛不抛由 caller 决定**。
3. 仍会抛的是**调用方 bug**：参数不可转换、未知 key、op 未注册（`NotImplementedError`/
   `ValueError`）—— 这些不属于 acl 错误码体系。

已完成：三个原语去 raise；新增三个 `*_checked`；`acl_utils.pxd` 导出两者；**18 处调用点**
（`_core/_ascend/_kernel.pyx`、`_core/_ascend/_routines_linalg.pyx`、
`_core/_ascend/_routines_sorting.pyx`、`_core/_reduction.pyx`、`_core/_routines_indexing.pyx`、
`_core/_routines_math.pyx`）改用 `*_checked`，Python 行为与上一版完全一致。

**noexcept 化的剩余工作**（这是「本次重估暂不实施」的部分，刻意不做）：

| # | 项 | 说明 |
|---|---|---|
| 1 | C++ 层仍会 throw | `acl_scalar_arg.h` 的 `CheckIntegerArg/CheckFloatArg/ToScalarArg/GetScalarArg`；这是 noexcept 的**前置**（否则 C++ 异常会穿过 C 边界 abort）→ refactor_exception.md S3 |
| 2 | 参数校验仍抛 Python 异常 | `_convert_arg`/`_create_keyword_args` 若要知道「返回码」语义，需改成 `bint`/错误码出参（`ACL_ERROR_INVALID_PARAM`），由 caller 决定抛不抛 |
| 3 | 没有真正的 `noexcept` 标注 | `cdef` 函数目前是 `except *`（Python 异常可穿透）。Cython ≥ 3.0 支持 `noexcept`，但要求整条调用链不抛，且调用点不得依赖 Python 异常 |
| 4 | C/C++ 调用方尚不存在 | 现在「返回错误码」这条通路只有 pyx 内部 + `*_checked` 在用；C 侧入口（若将来暴露 `cupy_ascend_*` C API）可直接用原语 |
| 5 | 建议顺序 | S3（C++ 去异常）→ ②（校验改返回码）→ 标注 `noexcept` → 统一 `check_acl_status()` 收口 |

---

## 附录 A：numpy 里带**字符串参数**的 API 分类与处置

（回答「有哪些 numpy 函数需要传递 string」；结论：**只有 A3 类必须跨边界传字符串**）

| 类别 | 代表 API（字符串参数） | 字符串语义 | 处置 |
|---|---|---|---|
| **A1 枚举/模式** | `sort/argsort/partition(kind='quicksort'|'stable')`、`searchsorted(side='left')`、`take/put(mode='raise'|'wrap'|'clip')`、`pad(mode='constant'|'edge'|'reflect'|..., reflect_type='even'|'odd')`、`fft.*(norm='ortho')`、`linalg.qr(mode='reduced')`、`linalg.eigh(UPLO='L')`、`linalg.norm(ord='fro'|'nuc')`、`quantile/percentile(method='linear')`、`histogram(bins='auto'|'fd'|'scott')`、`correlate/convolve(mode='same')`、`reshape/ravel/flatten/asarray(order='C')`、`packbits(bitorder='big')`、`einsum(optimize='greedy')`、`busday_offset(roll='forward')` | 取值有限（≤10 个），可映射到 int/bool/枚举 | **host 侧解析**（首选，已有实践：`order`/`stable`/`descending`/`some`/`sorted` 都是 int 编码）。**不进 arg 通道** |
| **A2 dtype / casting 名** | `astype('f4')`、`dtype('f8')`、`can_cast('i4','f8',casting='safe')`、`newbyteorder('<')` | 类型名 | host 侧 `numpy.dtype()` → `numpy_dtype_to_acl_dtype()`；不进 arg 通道 |
| **A3 承载算子语义的字符串** | **`numpy.einsum('ij,jk->ik', a, b)` 的 `equation`** | 字符串本身就是算子（无法用有限枚举表示） | **唯一必须走 `ARG_STRING` 的用例**：CANN 对应 `aclnnEinsumGetWorkspaceSize(const aclTensorList* tensors, const char* equation, aclTensor* output, ...)`（`aclnnop/aclnn_einsum.h`）。当前 Ascend **没有 einsum 实现**（无 `ascend_einsum` 注册），通道已就绪 |
| **A4 aclnn 原生 mode 字符串** | `aclnnResize(self, scales, const char* mode, out)`（对应 interp/resize 类 API） | 3~4 个取值 | 既可 host 解析（A1 做法），也可走 `ARG_STRING`；取决于是否同时需要传别的参数 |
| **A5 字符串数据（不是参数）** | `numpy.char.*` / `numpy.strings.*`（`upper`/`strip`/`find`…）、`numpy.str_` 数组 | **数组元素**是字符串 | Ascend 算子不支持 `ACL_STRING` tensor → **CPU fallback**；不进参数通道 |
| **A6 与算子无关（host 行为）** | `savetxt(fmt=)`、`loadtxt/genfromtxt(delimiter=,comments=)`、`memmap(mode='r+')`、`savez`、`array2string`、`set_printoptions` | 文件/打印格式 | 与 aclnn 无关，不跨边界 |

> 旁证：CANN 9.0.1 `include/aclnnop/` 共 817 个头，110 个出现 `const char*`；去掉 quant /
> attention / 通信 / norm 类后，numpy 风格的只有 `einsum` / `resize` / `roi_align`。

---

## 3. 里程碑

| 里程碑 | 内容 | 工作量 | 验证 |
|---|---|---|---|
| M1 ✅ | 阶段 1（错位修复 + 严格校验 + arg spec + 测试 op） | 0.5d | `tests/ascend/` 新增用例；无 NPU |
| **M3-lite ✅** | tagged `AclArg`（`ARG_SCALAR/INT_ARRAY/STRING/NONE`）+ `TryGetInt64List` + `AclIntArrayGuard` + 白名单字符串通道 + 探针 op + **reverse scalar**（§2.4） | 1d | `tests/ascend/test_unified_args.py`（无 NPU） |
| **S2-lite ✅** | launch 原语返回错误码、`*_checked` 归位到 caller（§2.5） | 0.5d | 重编 + `pytest tests/ascend`；C 侧入口待建 |
| M2 | 阶段 2（`unified_op` + `AsGeneralOp<>` 适配模板 + `where` 送达）；**必须满足 §2.2.1 的 3 条约束**（操作数与参数分区 / 方向走元数据位 + 适配器模板 / 类型安全来自 `Fn` trait） | 2-3d（+0.5d 启动期 dry-run 自检） | 重编 + import + 参数到达断言 + dry-run；`where` 语义待 910B |
| M3-rest | `ARG_SPECS` 自动声明表（`GetScalarArg` 调用点生成）、`ARG_TENSOR` 启用（`where=`） | 1d | 类型矩阵单测 |
| M5 | noexcept 深化：C++ 去异常（S3）→ 校验改返回码 → 统一 `check_acl_status`（§2.5 表 1-5） | 2-3d | 无 NPU 可测到「构造越界标量不再 abort」 |
| M4 | 910B 基线：`tril(k=)`、`trace(offset=)`、`nan_to_num(nan=)`、`histc(bins=)`、`add(where=)`、`flip(axis=(0,1))`、`roll(shift,axis)`、`1-x`、`2/x`、`1>x` 逐个回归 | — | 数值正确性 |

---

## 4. 风险与开放问题

| 项 | 说明 |
|---|---|
| 严格校验会暴露既有静默错误 | 这是目的；但需一次性审计窄签名注册的 arg 需求（实测 188 个名字 / 222 行调用，见 §2.2.1；可用 `GetScalarArg` 调用点自动提取，成本可控） |
| **操作数混入 `ArgsType`（M2 高危）** | `FindArg` 是「kwargs → `args[k]`」回退、`GetScalarArg` 只校验 kind，所以标量操作数一旦进了参数向量，34 处 `GetScalarArg<T>(args, k, ...)` 会**静默读到操作数的值**（kind 恰好也是 `ARG_SCALAR`，无 warning）。布局规则见 §2.2.1 约束 1。 |
| M2 的「不丢类型安全」 | 窄签名的参数表是编译期约束；收敛到 general 形态后要保住它，只能靠 `AsGeneralOp<Fn>` 从 `Fn` 的函数类型推导签名 + 启动期 dry-run（§2.2.1 约束 3）。退化成 wrapper 内运行时 `kind` 判断 = 把 M1 消灭的一类错误重新打开。 |
| 反向能力只有 12/188 需要 | 别为「方向」把 188 个注册都改成运行时协议：用适配器模板（`AsScalarBinaryDirectional<Fwd, Rev>`）覆盖那 12 个即可（§2.2.1 约束 2 与附录）。 |
| `where` 语义 | 参数送到 C++ ≠ 语义正确；Ascend 无直接对应的一元 where，需合成或 `aclnnSWhere`。`ARG_TENSOR` 的 tag / `GetTensorArg` 已就绪，但 pyx 侧仍**显式拒绝** ndarray 参数（避免「送到了但不生效」的静默错误） |
| reverse scalar 的设备侧正确性 | `AclScalarTensorGuard`（div/floor_divide/fmod）引入了一次 1 元素 `aclrtMalloc` + fill kernel，**只有 910B 上才能验证**；若 aclnn 对 1 元素广播有额外约束，需要回退到「Reciprocal+Muls」组合（已在 §2.4(c) 表里列出） |
| executor 泄漏（review §L6） | 与 arg 无关，但 M2 改 C++ 模板时**顺带**补 `aclDestroyAclOpExecutor`（只调一段就 return 的路径） |
| nogil 计划耦合 | `AsGeneralOp` 的模板形态正好是 `ascend_backend_nogil_plan.md` §3 想要的 RAII guard 落点，两者应合并推进 |
| 字符串需求将来出现 | ✅ **首个消费者已落地**：`ascend_einsum`（`aclnnEinsum`：tensor list + `const char*` equation + out），equation 走 ARG_STRING（`_STRING_ARG_OPS` 白名单，`aclop_Einsum` + `AclTensorListGuard`）。python 侧快速路径在 `cupy/linalg/_einsum.py`，**默认关闭**（`CUPY_ASCEND_NATIVE_EINSUM=1` 启用；dtype 不在 CANN 白名单/带 dtype kwarg/dtype 不统一时回退 python 组合实现）；CANN 白名单无 DOUBLE/INT8/BOOL，910B 数值验证待做 |

---

## 5. 参考

- `docs/ascend/code_review_ascend_backend.md` §0（所有权契约表）、§2.1-2.5（功能缺陷）
- `docs/ascend/refactor_exception.md`（异常/错误码体系，本文 §2.5 是它的 S2）
- `docs/ascend/patches/memory_leak_fix.patch`（已提交的泄漏修复，含 `_acl_scalar_owners` 保活范式）
- `docs/ascend/ascend_backend_nogil_plan.md`（RAII / 全局状态，与 M2 合并推进）
- `cupy/backends/ascend/acl_scalar_arg.h`（`AclArg` 定义、`GetScalarArg`/`TryGetInt64List`、`AclIntArrayGuard`）
- `cupy/backends/ascend/acl_math_ops.h`（`aclop_R*` reverse scalar 实现、`AclScalarTensorGuard`）
- `cupy/backends/ascend/api/acl_utils.pyx`（`_convert_arg`、三个 launch 原语与 `*_checked`、`py_dump_args`）
- `tests/ascend/test_unified_args.py`（无 NPU 的类型矩阵 / dispatch / 注册表测试）
