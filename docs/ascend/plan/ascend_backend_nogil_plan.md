# Ascend 后端 nogil / free-threading 改造计划

> **进度（2026-10-24）**：
> - S0 内存泄漏修复：已合入（核实：destroy-then-pop / try-finally / raise 前回收）。
> - S1 Phase-2 抽取：已合入（commit `a2538b956`）——11 个函数指针 ctypedef 声明 `nogil`，
>   general/scalar/reduction 三条 launch 路径的二段调用包进 `with nogil`。
> - S2 注册表 init-only：已合入（同上）——`register_acl_ufunc` 去 nogil、
>   `FuncPtrUnion` memset 零初始化、运行期注册入口文档化。
> - S3 C++ 异常翻译：已合入（commit `38332af2c`，路线 2）——ctypedef 加 `except +`，
>   Cython 3.2 在 nogil 段生成 catch + GIL 重取 + `__Pyx_CppExn2PyErr`。
>   路线 1（`acl_scalar_arg.h` 错误码化）仍为可选后续优化。
> - S4 smoke：已通过——CPython 3.14.9t（conda `py314t`）全量构建 + `import cupy`/
>   `cupy.fft`/`cupyx.scipy.sparse.linalg`/`cupyx.scipy.linalg` + 8 线程注册表并发探测 OK。
>   注意：free-threaded 下 `cythonize()` 进程池会因 forkserver 递归执行 setup.py，
>   已在 `_command.py` 按 `Py_GIL_DISABLED` 强制串行；TSan/多线程压测需 NPU 机器。
> - 遗留：fork 侧裸全局缓存迁 `cached_object`（`xpu/device.pyx`、`memory.pyx` 等）、
>   S5 设备上下文 per-thread 化与 aclfft plan 亲和性、NaN 标量运行时验证（需 NPU）。

- 目标读者：维护 `cupy/backends/ascend/` 的开发者。
- 相关文档：`docs/ascend/code_review_ascend_backend.md`（检视报告）、
  `docs/ascend/patches/memory_leak_fix.patch`（内存泄漏修复，是本文档的前置条件）。
- 结论：**在内存泄漏（尤其 `cupy_acl_tensor_owners` 的生命周期协议）与 C++ 异常翻译问题修掉之前，
  不要开始释放 GIL。** 否则会把「偶发泄漏」变成「确定性崩溃」。

---

## 1. 现状盘点

### 1.1 已经释放 GIL 的地方（很少）

| 位置 | 说明 |
|---|---|
| `api/aclfft.pyx:206`、`217`、`268`、`284-294` | `aclfftPlan1d` / `aclfftDestroy` / `aclfftSetStream` / `aclfftExec*`；传进去的 host 指针在 `with nogil` 之前取好，生命周期由局部 `numpy` 数组保证 —— 这是**正确写法**，可以作为模板 |
| `api/acl_utils.pyx:670` | `_launch_custom_ufunc` 的 `aclop_LaunchCustomKernel` |

### 1.2 完全没有释放 GIL 的地方（主路径）

`launch_general_func` / `launch_acl_func` / `launch_reduction_op` 都不是 `nogil` 函数，因此：

- aclnn 一阶段 `*GetWorkspaceSize`（算子选择 + IR 构建，CPU 密集，毫秒级）与二阶段 kernel 下发
  全程持有 GIL；
- `aclrtSynchronizeStream`（部分算子 `sync=true`）也持有 GIL。

也就是说，**今天 GIL 事实上充当了全局互斥锁**：`_builtin_operators` 注册表、`cupy_acl_tensor_owners`
这两个 C 全局状态之所以没出问题，只是因为「同一时刻只有一个线程在跑 Python 代码」。多线程下
（`threading` 并发调用 cupy、或 CPython 3.13t free-threaded 构建）这个假设会直接失效。

---

## 2. 六类风险与对应修改

### A. `PyMem_Malloc/PyMem_Free` 不能在 nogil 段调用

现状：`acl_utils.pyx:323-325`（三个维度数组）、`143-215`（标量 buffer）都在将来最想放进 nogil 的
「Python → C 转换」阶段；free-threaded 构建下 obmalloc 内部虽加锁，但 API 契约要求持有线程状态，
而且每次分配都去拿全局分配器锁，会把 nogil 的收益吃掉。

修改：

1. 这些 buffer **从不交给 Python**，改用 `libc.stdlib.malloc/free`，或 C++ 侧 `std::vector` /
   `std::unique_ptr<int64_t[]>`（推荐 C++ RAII，见 §3 的 guard）。
2. 分配与释放都必须发生在 GIL 段；nogil 段只持有裸指针。
3. patch 已先修掉泄漏；改成 `malloc/free` 属于同一处的后续清理，建议一起做。

### B. C++ 异常没有被翻译成 Python 异常（最致命）

现状：`acl_utils.pyx:809`、`1273`、`1327` 都是

```cython
cdef extern from "../acl_math_ops.h" nogil:
    aclError aclop_Add(const aclTensor* self, ...)
```

**没有 `except +`**。而这些头文件里的 C++ 是**会抛异常**的（`acl_scalar_arg.h`）：

- `ToScalarArg`：NaN → `throw std::runtime_error`（246 行），Inf → `throw std::runtime_error`（254 行）；
- `CheckIntegerArg/CheckFloatArg`：越界 → `throw std::out_of_range`（70、182 行）；
- `AclScalarToDouble`：类型不支持 → `throw std::runtime_error`（167 行）。

触发路径很现实：`cupy.tril(a, k=nan)`、`cupy.histc(a, bins=1e30)`、`nan_to_num(a, nan=nan)`
（这些 op 都走 `GetScalarArg<int64_t/float>`）。异常从没有 `except +` 的边界逃走后，会穿过
Cython 生成的 C++ 栈（不执行任何 cleanup、不设置 `PyErr`），最终 `std::terminate` 或 UB。

这在**当前 GIL 模式下就会崩**，只是概率低；nogil 化后触发面更大。

修改（两条路线，优先选 1）：

1. **让 C++ 层不再抛异常**：`acl_scalar_arg.h` 的 `CheckIntegerArg/CheckFloatArg/ToScalarArg/
   AclScalarToDouble` 改成返回错误码或 `std::optional`，由调用方转成 aclError。这样 `nogil` 边界上
   不存在跨语言异常，最省心。
2. 保留异常：给所有可能抛的 extern 声明加 `except +`，并确认所用 Cython 版本对
   「`with nogil` 内调用 `except +` 函数」的延迟抛出支持（Cython 3.x 支持把异常挂起到重新获取 GIL 时
   再抛，但需要在实现里确保 nogil 段结束后立刻回到 GIL 段处理）。

> 注意 `print` / `std::cout`：`CHECK_STATUS` 用的是 `std::cerr`（安全），但 pyx 里的 `print(...)`
> 在 nogil 段是非法的，必须移出。

### C. 全局可变注册表 → 数据竞争

现状：`register_acl_ufunc` 被声明为 `except * nogil`（`acl_utils.pyx:486`）并**写**
`_builtin_operators`（`cdef cpp_map[OpInfo, FuncPtrUnion, OpInfo.Hash]`，442 行），而 dispatch 路径
同时在**读**同一个 map。`std::unordered_map` 的并发读写是 UB（rehash 时读线程会拿到已释放的桶）。

修改：

1. 注册只允许发生在 import 阶段（`init_builtin_operators()`，1533-1539 行）——即**任何算子调用之前**；
   给 `register_acl_ufunc` 去掉 `nogil`，并在 `_builtin_operators` 上禁止运行期写入。
2. 若确实需要运行期注册（比如 `py_register_acl_ufunc` 的扩展场景），加 `std::mutex`，
   dispatch 端在 GIL 段一次性取出 `FuncPtrUnion`（当前已经是这么做的，保持即可）。
3. `_custom_kernel_specs` 是 Python dict：读取时先取局部引用再判断，避免
   「判断存在 → 使用」之间被其他线程改写。

### D. `cupy_acl_tensor_owners` 的生命周期协议

现状：模块级 `cdef dict`，键是 `aclTensor*`，值是保活对象；`cupy_destroy_acl_tensor` 做
「pop + aclDestroyTensor」。它依赖三件事：

1. 创建与 pop 严格成对（patch 已用 `try/finally` 保证）；
2. **pop 与 `aclDestroyTensor` 的顺序**（patch 已改为先 destroy 后 pop）；
3. dict 的强引用在 nogil 段不会失效。

nogil 化后的新风险：

- 「pop 掉最后一个 Python 引用 → ndarray 被 GC → 显存归还内存池 → aclnn 还在用这块显存」
  的窗口期：任何顺序错误都会变成 UAF，且因为异步执行而极难复现。
- PEP 684 子解释器：这个 C 全局变量被多个解释器共享，会把 A 解释器的 `PyObject*` 交给 B 解释器
  （直接崩）。

修改：

1. 把 `cupy_acl_tensor_owners` 从「模块级 C 全局 + 手工 pop」改成 **RAII guard 对象**（见 §3），
   生命周期严格绑定 aclTensor，nogil 段完全不触碰它。
2. 顺序固定为：`aclDestroyTensor()` → 再放开 Python 引用。
3. 中期把该状态迁到 **module state**（`PyModule_GetState` / `__pyx_mstate`），为子解释器与
   free-threading 做准备；`_builtin_operators` / `_custom_kernel_specs` 同理。

### E. 接口签名必须从「Python 容器」改成「C 结构体」

现状：

```cython
cdef aclError launch_general_func(str opname, sequence ins, sequence outs,
    list args, dict kargs, intptr_t stream_ptr) except *
```

（`acl_utils.pxd:21`；`sequence` 还是一个只有 `list` 一个成员的 fused type，白白生成 `[list, list]`
特化并强制所有调用方先 `list(...)`。）

nogil 段的铁律：**不出现任何 Python 对象/API**。因此 `str opname`、`list/dict args`、
`ndarray._shape/_strides/dtype/data.ptr`、`type()/issubclass`、`print()`、f-string、异常构造
全部要在 GIL 段完成转换。

修改：引入 C 侧请求结构体（例如 `AclLaunchArgs { const aclTensor** ins; int nin; ... }`），
或者接受「多次 cdef 函数调用，每次只传 C 值」的风格。前者更接近 §3 的三段式。

### F. Cython / nogil 细节清单

| 项 | 说明 |
|---|---|
| `with nogil` 段内不能 `raise` | nogil 段只返回 `aclError`；回到 GIL 段再 `raise`（patch 已把 `print` 移出 `finally`，为后续铺路） |
| `FuncPtrUnion` 未初始化 | `register_*` 里反复复用同一个 `FuncPtrUnion func_union`，只赋值一个成员就注册；建议 `memzero` 或显式初始化，避免将来误读未初始化成员（优化器下是 UB） |
| stream 必须提前取 | `stream_module.get_current_stream_ptr()` 是 thread-local；`aclfft.pyx:269` 的 `aclfftSetStream` 要在 nogil 之前算好 |
| `aclfft` 的 plan 是共享可变状态 | `plan->stream` 会被 `aclfftSetStream` 改写 → 多线程共享同一个 `Plan1d` 会互相覆盖；需要文档化「一个 plan 只在一个线程/stream 上用」或改成每线程一 plan |
| 设备上下文 | `runtime._ensure_context()` 在 free-threaded 下不再是「每进程一次」，`aclrtSetDevice` 等需改成 per-thread 惰性初始化 |
| module init | `init_builtin_operators()` 在 import 时执行（1539 行）；free-threading 下 import 仍是串行的，可放心把「注册」放这里 |

---

## 3. 目标结构：三段式 + RAII guard

```cython
# ---------------------------------------------------------------------------
# Phase 1（GIL）：Python 对象 -> C 值；所有分配在这里，全部登记进 guard
# ---------------------------------------------------------------------------
cdef class _AclLaunchGuard:
    """一次算子调用的全部 C 侧资源；__dealloc__ 里统一回收。

    关键点：nogil 段（Phase 2）绝不触碰本对象内部的 Python 引用，
    所有 Python 引用只在 GIL 段被创建/释放。
    """
    cdef vector[aclTensor*] tensors
    cdef vector[aclScalar*] scalars
    cdef aclIntArray* dim
    cdef object keep_alive          # 保活 ndarray 列表（可选，避免全局 dict）

    def __dealloc__(self):
        # 顺序：先销毁 acl 对象（可能仍引用 dim/buffer），再放开 Python 引用
        for t in self.tensors:
            cupy_destroy_acl_tensor(t)
        for s in self.scalars:
            _destroy_acl_scalar(s)
        if self.dim != NULL:
            aclDestroyIntArray(self.dim)

# ---------------------------------------------------------------------------
# Phase 2（nogil）：只调 C，只返回 aclError，不碰任何 Python 对象
# ---------------------------------------------------------------------------
    with nogil:
        ret = func_ptr.general_op(intensors, outtensors, acl_args, acl_kwargs, stream)

# ---------------------------------------------------------------------------
# Phase 3（GIL）：错误转换 + 清理（guard 出作用域时自动完成）
# ---------------------------------------------------------------------------
    if ret != 0:
        raise RuntimeError(f'ascend op {opname!r} failed: {ret} ({aclGetRecentErrMsg()})')
```

落地要点：

1. **Phase 1 与 Phase 3 允许用 Python**：`type()/issubclass`、属性访问、`PyMem/Malloc`、异常、日志。
2. **Phase 2 只能是 C**：函数指针调用、`clock`、C++ 容器操作；不允许 `print`、不允许构造 Python 异常。
3. 把 §2.A 的 `PyMem_Malloc` 换成 C++ `std::vector` 或 `malloc/free`，让 guard 的析构成为
   唯一的释放点（这样也彻底解决 L1/L4）。
4. 异常：Phase 2 的 extern 声明要么全部 `except +`（并在 Phase 2 结束后立刻处理），要么让 C++
   层不抛（推荐）。
5. 建议把 `_builtin_operators` 的读取全部收进一个 `cdef inline` 取值函数，保持「nogil 只读快照」。

### 分阶段落地顺序

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| S0 | 合入 `memory_leak_fix.patch` | 现有测试不变绿→不变红；RSS/显存不再随调用次数线性增长 |
| S1 | Phase 2 抽取：只把「二段接口调用」包进 `with nogil`，Phase 1/3 保持持 GIL | 单线程功能等价；`absl`/日志无变化 |
| S2 | 注册表改 init-only（`register_acl_ufunc` 去 nogil，或加 mutex）；`__pyx_mstate` 化全局变量 | 代码中不再有运行期写注册表 |
| S3 | C++ 层去异常（或全量 `except +`）；`trim` 掉 nogil 段里的 Python 调用 | 构造 NaN/越界标量参数时得到 Python 异常而不是进程退出 |
| S4 | 在 CPython 3.13t/3.14t 上跑 `import cupy` + 单线程 smoke；再上多线程 + TSan/helgrind | 无 TSan 报告；无跨解释器对象共享告警 |
| S5 | 设备上下文 per-thread 化；`aclfft` plan 线程亲和性文档化 | 多线程并发调用不再互相覆盖 stream |

---

## 4. 与 free-threaded CPython（PEP 703）的关系

1. **锁粒度**：free-threading 下不再有 GIL，`threading` 并发调用 cupy 会真正并行进入 C 层，
   §2.C/§2.D 的全局状态是新瓶颈与新的崩溃源；用 §3 的 guard + init-only 注册表把共享状态压到零。
2. **引用计数**：free-threading 下 refcount 是原子的，把对象放进 dict / 局部变量仍能保活，
   但**不要依赖「函数返回时局部变量才释放」这种隐式时序** —— 显式化（`Py_INCREF/Py_DECREF`
   或 RAII `PyObject*` 包装）。
3. **子解释器（PEP 684）**：模块级 C 全局变量（`_builtin_operators`、`cupy_acl_tensor_owners`、
   `_acl_scalar_owners`、`_custom_kernel_specs`）被所有解释器共享，其中的 `PyObject*` 属于某一个
   解释器 → 必须迁到 module state，或做成 per-interpreter 初始化。
4. **`PyMem_Malloc` in nogil**：即便 free-threaded 的 obmalloc 是线程安全的，也**不要**在 nogil 段
   调用它；用 `malloc/free` 或 C++ 容器。
5. **`cdef` 全局变量的初始化**：module 级 `cdef dict x = {}` 在 free-threaded 下的创建时机与
   多线程首次 import 需要复核（建议放在 `init_*` 函数里显式创建）。
6. **测试**：`python3.13t` + `-X gil=0`；多线程压测同一组 ufunc；TSan 构建 `acl_utils.pyx` 生成物；
   关注 `cupy` 自身内存池（`MemoryPool`）在无 GIL 下的线程安全性（属于 cupy 上游范围，需要单独评估）。

---

## 5. 附：nogil 改造前的「不做清单」

在 S0–S1 完成前，以下改动**不要**做，否则会把可复现的 bug 变成不可复现的崩溃：

- 不要把 `cupy_ndarray_to_acl_tensor` 整个包进 `with nogil`（里面有 `PyMem_Malloc`、
  `_shape/_strides` 属性访问、异常构造）。
- 不要让 `launch_general_func` / `launch_acl_func` / `launch_reduction_op` 变成 `nogil` 函数
  （签名里全是 Python 容器）。
- 不要给 `register_acl_ufunc` 之外的路径开放运行期注册。
- 不要在 `with nogil` 段内 `print` 或构造 Python 异常。
