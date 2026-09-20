# Ascend 后端：把错误/异常传递到 Python 的重构计划

- 目标读者：维护 `cupy/backends/ascend/` 与 `cupy/_core/_ascend/` 的开发者。
- 相关文档：`docs/ascend/code_review_ascend_backend.md`（风险表 B 条目）、
  `docs/ascend/ascend_backend_nogil_plan.md`（§2.B / S3 阶段）、
  `docs/ascend/arg_passing_plan.md`（参数通道）。
- 状态：**S1 已落地**（见 §5「本次已完成」），S2~S5 待做。

---

## 1. 现状：三类错误，两种会丢

| # | 错误来源 | 现状（重构前） | 后果 |
|---|---|---|---|
| A | **我们自己的 C++ 异常**：`acl_scalar_arg.h` 的 `ToScalarArg`（NaN/Inf）、`CheckIntegerArg`/`CheckFloatArg`（越界）、`AclScalarToDouble`（dtype 不支持）、`GetScalarArg`（缺参） | `cdef extern from "../acl_*.h"` 声明**没有 `except +`**；Cython 不生成 try/catch，也不设置 `PyErr` | 异常穿过 Cython/C 栈 → `std::terminate()` / UB（**进程退出，不是 Python 异常**）。触发条件很普通：`cupy.tril(a, k=nan)`、`cupy.histc(a, bins=1e30)` |
| B | **aclnn/acl 返回码**（`aclnnStatus` / `aclError`） | `CHECK_STATUS` 宏只 `std::cerr`；`launch_general_func`/`launch_acl_func`/`launch_reduction_op` 只 `print` 一行后 `return ret`，而调用方（`cupy/_core/_ascend/_kernel.pyx`）连返回值都丢掉 | 「算子没跑成」= 「结果是对的」：输出数组是未初始化/半算的 → **静默错误结果** |
| C | Python 层参数校验（`_convert_arg_strict`、`_KNOWN_SCALAR_KEYS`、`_no_ascend_impl_msg`） | 已经 `raise NotImplementedError/RuntimeError/ValueError` | 唯一正确的一类，是 S2 要吸收进异常体系的存量 |

B 类的一个真实复现路径（已修）：`CreateAclScalar(double, aclDataType)` 只处理
float/double/int16/int32/int64/bool，`uint8/uint16/uint32/uint64/float16/bf16/complex`
全部 `return nullptr`；`Arange/Histc/Fill/MaskedScatter/ternary-alpha` 都是「用 tensor 的
dtype 造 scalar」，于是把 `nullptr` 交给了 aclnn。

---

## 2. 参考实现：cupy 的 CUDA 后端是怎么做的

cupy 的 CUDA 侧事实标准是 **错误码 + 单一收口 + 专用异常类型**，不依赖跨语言异常：

```134:152:cupy/backends/backend/api/runtime.pyx
class CUDARuntimeError(RuntimeError):

    def __init__(self, status):
        self.status = status
        cdef bytes name = cudaGetErrorName(<Error>status)
        cdef bytes msg = cudaGetErrorString(<Error>status)
        super(CUDARuntimeError, self).__init__(
            '%s: %s' % (name.decode(), msg.decode()))

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        # to reset error status
        cudaGetLastError()
        raise CUDARuntimeError(status)
```

可以直接借鉴的四点：

1. **异常继承 `RuntimeError`**（不是自定义 BaseException）→ 上层 `except RuntimeError`
   的既有代码（含 cupy 自己的）继续可用；同时保留 `status` 属性便于分类。
2. **`__reduce__` 支持 pickle**：异常只持有 `status`（整数）/字符串，**绝不持有裸指针**，
   否则跨进程/子解释器会失效（与 review D 条同一类问题）。
3. **单一收口函数** `check_status(status)`：所有 C API 调用点 `return status`，
   由一处完成「取名字、取描述、清错误状态、抛异常」。C 函数本身**从不抛 C++ 异常**。
4. 设备端异步错误在**同步点**检查（`cudaStreamSynchronize` 之后），而不是每个 kernel 都查。

Ascend 侧的差别：CANN 的错误描述不是按 `ret` 查表，而是「最近一次错误」的
`aclGetRecentErrMsg()`（全局/线程本地状态），所以收口函数必须**在失败点立刻取**，
不能延后（延后会被后续调用覆盖）。

---

## 3. Cython 最佳实践（本项目必须遵守的几条）

1. **任何可能抛 C++ 异常的 `cdef extern` 声明都必须显式标注**：`except +`
   （标准异常）、`except +*`（连非 `std::exception` 派生也捕获）、
   或 `except +ValueError` 之类的映射（Cython ≥ 3.0）。
   **不写 = Cython 不知道 = 不捕获**，异常会穿透到 C 栈。
   受影响文件：`cupy/backends/ascend/api/acl_utils.pyx` 里所有
   `cdef extern from "../acl_*.h"` 块（当前只有 `OpInfo` 构造函数带 `except +`）。
2. **`with nogil` 段内调用 `except +` 函数会延迟抛出**（Cython 3 会挂起异常，重新持有
   GIL 后再 raise）。因此 nogil 段里**不能**做 Python 侧 cleanup，资源回收必须靠
   C++ RAII 或回到 GIL 段处理 —— 这也是 `ascend_backend_nogil_plan.md` §2.B 把
   「C++ 层不抛」列为首选的原因。
3. `except *` 只做「返回非 Python 类型时检查 `PyErr`」，它**不会**捕获 C++ 异常；
   不要指望 `except *` 兜住 A 类问题。
4. cdef 函数的异常规格要有意选择：`except *`（返回裸指针/`aclError` 时）、
   `except? -1`（返回整数、想省一次 `PyErr_Occurred`）。
5. **异常消息构造只能在持 GIL 时做**（`aclGetRecentErrMsg()` 的字符串解码、
   f-string/`format`），nogil 段里只允许记录 `ret` 这个整数。
6. C++ 析构（RAII guard、`aclTensorList` 的 `aclDestroyTensorList`）必须 `noexcept`：
   析构里抛出的异常同样会穿过边界；`AclTensorListGuard` 是正确写法。
7. `std::cerr`/`std::cout` 用于**诊断**可以，但不能作为错误传递手段（当前
   `CHECK_STATUS` 就是这个问题：它在 `acl_op_template.h` 里只打印）。

---

## 4. 目标设计

### 4.1 异常类型（对齐 `CUDARuntimeError`）

新增 `cupy/backends/ascend/api/errors.pyx`（或直接放在 `acl_utils.pyx` 顶部先用着）：

```
AscendError(RuntimeError)                            # 基类，便于整体捕获
├── AscendRuntimeError(AscendError)                  # aclrt/acl 返回码（≙ CUDARuntimeError）
├── AscendOpError(AscendError)                       # 算子级：ACLNN_ERR_PARAM_INVALID（dtype/参数不支持）
└── AscendNotSupportedError(AscendError, NotImplementedError)   # 未注册/未移植的 op
```

* 全部继承 `RuntimeError` → 现有 `except RuntimeError` / `except NotImplementedError`
  代码语义不变；
* 只保存 `opname: str`、`ret: int`、`detail: str`，支持 `__reduce__`。

### 4.2 单一收口函数

```cython
cdef inline void check_acl_status(str opname, long ret) except *:
    """所有 aclop_*/aclrt* 调用点的唯一出口。"""
    if ret == ACL_SUCCESS:
        return
    detail = _acl_recent_errmsg()          # 必须在失败点立刻取
    if ret == ACLNN_ERR_PARAM_INVALID:     # 161001：dtype/参数不支持
        raise AscendOpError(opname, ret, detail)
    raise AscendRuntimeError(opname, ret, detail)
```

* C++ 侧 `CHECK_STATUS` 退化为「只打印诊断」，不再承担错误传递；
* `_launch_custom_ufunc` 已抛 `RuntimeError`，一并改走此函数；
* `_no_ascend_impl_msg` 的裸 `NotImplementedError` → `AscendNotSupportedError`。

### 4.3 C++ 层不再抛异常（首选路线）

`acl_scalar_arg.h` 的会抛函数改成错误码/`optional`：

| 现 API | 目标 API | 说明 |
|---|---|---|
| `ToScalarArg<T>(s, throw_on_error)` | `bint TryToScalarArg<T>(const aclScalar* s, T* out)` | NaN/Inf/dtype 不支持 → `false` |
| `CheckIntegerArg/CheckFloatArg` | 饱和 + `bint* ok` | 越界不再 `throw std::out_of_range` |
| `GetScalarArg<T>(..., optional<T>)` | `bint TryGetScalarArg<T>(..., T* out)` | 缺参 → `false`（配合 `arg_passing_plan.md` M2 的统一参数通道，返回 `ACL_ERROR_INVALID_PARAM`） |
| `AclScalarToDouble` | 保留，但失败返回 `bint`/`optional<double>` | dtype 不支持是**调用方**该报的错 |

迁移策略（增量、每步可编译可测）：

1. 先加 `Try*` 新 API，旧 API 内部调用新 API：失败时 `std::cerr` + 返回默认值
   （与当前行为一致，不改变调用点）；
2. 逐个把调用点换成 `Try*` + `return ACL_ERROR_INVALID_PARAM`；
3. 删掉会抛的旧 API，并在 CI 加一条 grep 规则：`cupy/backends/ascend/*.h`
   不允许新增 `throw`（`acl_*.h` 是唯一会被 extern 边界调用的地方）。

> 已经完成的第一步见 §5：`CreateAclScalar` 的越界改为「饱和 + stderr」，
> 不再有任何抛异常路径。

### 4.4 备选路线：给 extern 加 `except +`

若短期不方便改 C++，则必须**成组**改（残缺的 `except +` 比没有更危险）：

```cython
cdef extern from "../acl_math_ops.h" nogil:
    aclError aclop_Add(...) except +
    ...
```

* 支持 `except +*`（非 std 异常）与 `except +ValueError`（类型映射）；
* `with nogil` 内的延迟抛出需 Cython ≥ 3.0（本项目 3.3.0）；
* 缺点：异常会带着「半清理」的 C++ 状态穿过 Cython 栈，`finally` 语义与 Python 不同，
  且 `print`/iostream 混在一起难定位 —— 只作为 S3 完成前的兜底。

### 4.5 补上检查点

* **异步 kernel 错误**：`aclrtSynchronizeStream` 之后 check（`aclfft` / `matmul` /
  `cupy_ascend_runtime.h` 的 stream 封装）；
* **runtime 层**：`cupy/backends/backend/api/runtime.pyx`（Ascend 分支）的
  `aclrt*` 调用统一走 4.2 的收口；
* **Python 层入口**：`cupy/_core/_ascend/_kernel.pyx` 目前丢弃
  `launch_general_func` 的返回值 —— 收口后 `ret` 恒为 0，可以保留不动，
  但建议改为「不做任何假设」的写法：`launch_general_func(...)` 返回非 0 时也 raise。

---

## 5. 本次已完成（S1）

1. `cupy/backends/ascend/acl_type_traits.h`
   * `CreateAclScalar(double, aclDataType)` 补齐 int8/uint8/int16/**uint16**/int32/**uint32**/
     int64/uint64/float16/bf16/complex64/complex128/bool（uint16/uint32 原先返回 `nullptr`）；
   * 越界/NaN → 饱和 + `std::cerr`（**不 throw**，见 §3.2 的约束）；
   * `aclDtypeToString` / `PrintScalarValue` / `TypeToAclDataType` 同步补全，便于定位问题；
   * 新增 `FloatToHalfBits`（round-to-nearest-even）。
2. `cupy/backends/ascend/api/acl_utils.pyx`
   * 新增 `_acl_recent_errmsg()` / `raise_acl_op_error()`；
   * `launch_general_func` / `launch_acl_func` / `launch_reduction_op` 的
     `ret != 0` 从 `print` 改为 `raise RuntimeError(opname, ret, aclGetRecentErrMsg())`；
   * 抛出点一律放在 `finally` 之后，保证 tensor/scalar/kwargs 已回收。

**行为变化**：以前「打印一行然后继续」的算子失败，现在会抛 `RuntimeError`。
迁移期仍可用 `CUPY_ASCEND_LENIENT_ARGS=1` 关掉参数校验（它对本次的返回码检查无效，
若要回滚返回码检查，请显式加临时开关并在下一个 commit 里删掉）。

---

## 需留意, 是否要重构scalar转化

```c++
// ---------------------------------------------------------------------------
// double -> 指定 dtype 的 aclScalar 创建
//
// 调用点（Arange / Histc / Fill / MaskedScatter / ternary 的 alpha / Clamp ...）
// 都是「用 tensor 的 dtype 造 scalar」，所以这里**必须**覆盖全部基础类型：
// 少一个分支就会返回 nullptr，而 nullptr 交给 aclnn 轻则
// ACLNN_ERR_PARAM_INVALID（以前只剩 stdout 一行 WARNING），重则崩溃。
// uint16/uint32 原先就落在这个坑里 —— 注意 aclScalar 本身是支持它们的
// （common_types.h: v_t::ui16/ui32 + ToUint16()/ToUint32()，
// opdev/data_type_utils.h 的 TypeSize/IsBasicType 也包含 DT_UINT16/32），
// 缺的只是本函数。
//
// 另一条硬约束：本函数在「没有 `except +` 的 extern 边界」内被调用
// （见 docs/ascend/refactor_exception.md），因此**不能 throw**：C++ 异常不会
// 被翻译成 Python 异常，只会穿过 Cython 栈导致 std::terminate。越界/NaN
// 一律饱和 + stderr 警告。
// 返回值仍可能是 nullptr（dtype 真的无法用 double 表达，或 aclCreateScalar
// 分配失败），调用方需要检查 —— 派发层已把 nullptr 引发的 aclnn 失败升级为
// Python 异常。
// ---------------------------------------------------------------------------
namespace acl_type_traits_detail {
   
```


## 6. 分阶段落地顺序

| 阶段 | 内容 | 无 NPU 下的验证 |
|---|---|---|
| **S1（已完成）** | 派发层返回码 → `RuntimeError`；`CreateAclScalar` dtype 完备 | `cython -3 --cplus api/acl_utils.pyx`；`build_ext --inplace`；`pytest tests/ascend`；C++ 端直接调 `aclCreateScalar`（见 §7） |
| S2 | 新增 `errors.pyx`（`AscendOpError` 等）+ `check_acl_status` 收口；`_no_ascend_impl_msg` 改用新类 | 新增 pytest：断言异常类型/`ret`/消息（不需要设备） |
| S3 | `acl_scalar_arg.h` 去异常（`Try*` API + 调用点返回 `ACL_ERROR_INVALID_PARAM`） | C++ 单测 + `pytest`；CI grep 禁止新增 `throw` |
| S4 | `except +` 兜底（若仍有 throw 路径）或彻底删除；`_kernel.pyx` 不再忽略返回值 | 同上 |
| S5 | 设备侧回归：stream sync 后的异步错误、`aclnn` dtype 支持矩阵实测 | 需要 NPU（或用 acl stub 库） |

---

## 7. 无 NPU 环境下的验证手段（本次实际用过）

`aclScalar` 的创建/销毁是**纯 host 侧**行为，可以脱离 NPU 直接验证
（这是本次能确认「`aclScalar` 支持 uint16/uint32」的依据）：

```bash
# 1) 只做语法/语义检查
g++ -std=c++17 -fsyntax-only -I cupy/backends/ascend \
    -I "$ASCEND_HOME_PATH/include" -I "$ASCEND_HOME_PATH/include/aclnn" t.cpp

# 2) 链接 CANN 的 host 侧库，真跑 aclCreateScalar / aclScalar::GetData()
g++ -std=c++17 -O0 -I cupy/backends/ascend \
    -I "$ASCEND_HOME_PATH/include" -I "$ASCEND_HOME_PATH/include/aclnn" t.cpp \
    -L"$ASCEND_HOME_PATH/lib64" -lnnopbase -lopapi -lascendcl \
    -lexe_graph -lgraph -lgraph_base -lregister -o t
LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH" ./t
```

建议把 `t.cpp` 固化成 `tools/ascend/check_acl_scalar.cpp`（枚举所有 dtype，
打印 `GetDataType()` + `GetData()` 解析值 + 越界/NaN 行为），
让「scalar 类型表」有回归测试；半精度位模式可以直接和
`numpy.float16(x).view(numpy.uint16)` 对照（本次已逐值比对通过）。
