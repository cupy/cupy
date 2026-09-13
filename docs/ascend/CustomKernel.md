# 自定义 Ascend Kernel（Custom Kernel）基础设施设计

> 状态：**计划（待确认）**
> 目标：在 numpy-ascend 中提供与 `cupy.RawKernel` / `cupy.RawModule` 对标的自定义
> kernel 能力，用 bisheng 编译 AscendC 内核；同时让内置缺失算子（Progress.md 中的
> 缺口）可以"源码内置 + 运行时编译"的方式补齐；并作为客户编写自定义 kernel/ufunc
> 的扩展点。

---

## 0. 一句话结论

Ascend 的运行时 API（CANN 9.0.1 已验证存在）构成了与 CUDA 驱动 API 完全对齐的链路：

```
CUDA:   nvrtc 编译 → cuModuleLoadData → cuModuleGetFunction → cuLaunchKernel
Ascend: bisheng 编译 → aclrtCreateBinary/aclrtBinaryLoad
        → aclrtBinaryGetFunction    → aclrtLaunchKernel(WithConfig)
```

`cupy.xpu.function` 的 `Module`/`Function` 抽象与 `cupy/_ascend/_core/raw_kernel_stub.pyx`
（目前是空壳）就是为这条链路预留的插入点。

---

## 1. 已验证的 CANN 9.0.1 事实（本机探查结果）

| 能力 | API / 工具 | 位置 |
|---|---|---|
| 内核编译器 | `bisheng`（`--cce-aicore-only`、`--cce-aicore-jit`、`--soc-version`） | `<CANN>/tools/bisheng_compiler/bin/bisheng` |
| AscendC 内核头 | `kernel_operator.h` | `<CANN>/x86_64-linux/ascendc/include/basic_api/` |
| 二进制载入内存 | `aclrtCreateBinary(data, len)` / `aclrtDestroyBinary` | `acl/acl_rt.h:2786/2797` |
| 设备模块加载 | `aclrtBinaryLoad(binary, &binHandle)` / `aclrtBinaryUnLoad` | `acl/acl_rt.h:2810/2822` |
| 取内核句柄 | `aclrtBinaryGetFunction(binHandle, kernelName, &funcHandle)` | `acl/acl_rt.h:2835` |
| 内核启动 | `aclrtLaunchKernel(funcHandle, numBlocks, argsData, argsSize, stream)` | `acl/acl_rt.h:2850` |
| 带配置启动 | `aclrtLaunchKernelWithConfig(funcHandle, numBlocks, stream, cfg, argsHandle, reserve)` / `V2` | `acl/acl_rt.h:3223/4427` |
| 启动属性 | `aclrtLaunchKernelAttr`（localMemorySize 等） | `acl/acl_rt.h:448` |
| 已有后端接入 | bisheng 已接入 `_compiler.py` 的 device 编译路径 | `get_ascendcc_path()`（本会话已修好多候选探测） |

与 CUDA 的语义差异（决定封装方式）：

- **启动粒度是 `numBlocks`（一维 block 数）**，没有 CUDA 的 grid×block 二维/三维；
  `cupy.RawKernel` 的 `grid, block` 参数需要映射（见 §4.3）。
- **参数是序列化的 `argsData` 缓冲**（对标 CUDA 的 kernel params 打包），而非
  `void* args[]` 指针数组；`aclrtLaunchKernelWithConfig` 走 `aclrtArgsHandle`。
- **没有 NVRTC 等价物**：JIT = 子进程调用 bisheng（CUDA 是进程内 libnvrtc），
  因此 JIT 有进程/文件系统开销，磁盘缓存（§3.3）是强需求。

---

## 2. 技术路线（已确认：只做路线 A）

### 路线 A：RawKernel 直启（对标 `cupy.RawKernel`）—— 唯一路线 ✅

```
AscendC 源码（用户/内置 .cpp，extern "C" kernel 函数）
   │  bisheng --cce-aicore-only --soc-version=<ascend910b> -O3
   ▼
.o (aicore fatbin, host 无需链接)
   │  读入内存 → aclrtCreateBinary → aclrtBinaryLoad → aclrtBinaryGetFunction
   ▼
aclrtLaunchKernel(funcHandle, numBlocks, argsData, stream)
   argsData = 打包的 (设备指针, 标量…) —— kernel 侧以 __gm__ 指针接收
```

- 优点：最薄、与 cupy.RawKernel 心智模型 1:1；缺失算子直接手写 kernel 补齐；
  不依赖 aclnn 的算子注册体系。
- 限制：内核要自己管 tiling/多核切分（无 host tiling）；调试手段有限。

### ~~路线 B：Custom aclnn 算子~~ —— 已裁决：不做（out of scope）

**用户决策（2026-09）**：只做路线 A。custom aclnn 算子插件整体移出本计划范围，
包括此前拟预留的 `CUSTOM_OP` OpType —— 不预留。若未来确有性能关键算子需要
tiling 框架与 ufunc 派发接入，另立项目重新评估。

**结论：基础设施只按路线 A 建设。缺失算子全部以内置 AscendC 内核
（§3.1 `kernels/` 注册表）或现有 aclnn 组合的方式补齐。**

---

## 3. 方案设计

### 3.1 源码放哪里

| 类别 | 位置 | 说明 |
|---|---|---|
| 内置补缺 kernel 源码 | `cupy/backends/ascend/kernels/*.cpp` | 随源码树走 git；纯 AscendC，不参与 host 编译 |
| 内置 kernel 索引/注册 | `cupy/backends/ascend/kernels/__init__.py` | `KERNELS = {name: {'src': ..., 'entry': ..., 'options': ...}}` |
| 用户 kernel | 用户代码字符串（`cupy.RawKernel(code, ...)`）或任意文件路径 | 不进包 |
| JIT 缓存 | `~/.cache/cupy/cann_kernel_cache/` | 对标 cupy 的 `~/.cupy/kernel_cache/`；可用 `CUPY_CACHE_DIR` 覆盖 |

缓存键 = sha256(源码 + options + SoC 型号 + CANN series + bisheng 版本)，
产物 `{key}.o` + `{key}.json`（元数据：entry 名、SoC、编译命令、时间戳）。
用 **CANN series**（major.minor）而非 patch 做键，保证 9.0.1 编的缓存对 9.0.2 仍可复用，
跨 series 自动失效重编。

### 3.2 什么时候编译（三种时机并存）

1. **运行时 JIT（默认，对标 cupy RawModule 首次调用）**：
   首次 `RawKernel.__call__` 时检查磁盘缓存 → 未命中则 spawn bisheng。
   bisheng 在装了 CANN 的机器上必然存在（复用 `get_ascendcc_path()`），
   所以**不需要分发 .so，只分发源码**。
2. **构建时 AOT（可选开关，针对内置 kernel）**：
   `CUPY_ASCEND_AOT_KERNELS=1` 时 setup.py 把 `kernels/*.cpp` 预编译进
   `cupy/backends/ascend/kernels/_aot/<soc>/`，import 时优先加载 AOT 产物。
   面向"wheel 想避免首调编译延迟"的场景；默认关（wheel 无 .so 分发负担）。
3. **显式预热 API**：`cupy.RawModule(...).compile()`（对标 cupy 语义），供打包工具调用。

> 分发策略结论：**wheel 只带源码**（与 cupy wheel 只带 .cu 源码、运行时 nvrtc 编译
> 同构）。AOT 是优化项不是必需项——因为 CANN 机器必有 bisheng。

### 3.3 如何加载（运行时，无需 dlopen）

与 CUDA 不同，aicore fatbin **不经过 dlopen**，直接走 aclrt 内存加载：

```cython
# cupy/_ascend/_core/raw_kernel_stub.pyx 的实现要点
buf = open(cached_path, 'rb').read()
binary = aclrtCreateBinary(<const void*>buf, len(buf))     # 内存 → aclrtBinary
aclrtBinaryLoad(binary, &self._bin_handle)                 # 设备模块
aclrtBinaryGetFunction(self._bin_handle, entry, &self._func)
# __call__:
aclrtLaunchKernelV2(self._func, num_blocks, args_data, args_size, &cfg, stream)
# 析构 / 缓存淘汰:
aclrtBinaryUnLoad(self._bin_handle); aclrtDestroyBinary(binary)
```

- `Module` 缓存于 `cupy.xpu.function` 的模块表（复用现有 `get_cached_module` 机制）。
- 生命周期挂接设备清理（对标 `cupy.cuda` 的 per-device module 表）。

### 3.4 Python API（对标 cupy，最小差异）

```python
# 与 cupy 相同的构造方式（backend 参数忽略/可选 'bisheng'）
kern = cupy.RawKernel(r'''
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void ascend_add(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t n) {
    // AscendC: GlobalTensor + Pipe/Que/LocalTensor ...
}
''', 'ascend_add')
kern(grid, block, (args...))            # 映射语义见 §3.4.1

mod  = cupy.RawModule(code=..., options=('--soc-version=ascend910b',))
fn   = mod.get_function('ascend_add')
```

- 参数打包：ndarray → 设备指针 + shape/dtype 元信息由打包器生成 `argsData`
  （对标 cupy 的 CPointer 机制，`cupy.xpu.function` 已有 `_pointer()`）。

### 3.4.1 `grid/block → numBlocks` 乘积映射（已确认）及其影响 ✅

**已确认的映射规则**：`numBlocks = grid × block`（各维乘积展开为一维），
并且把 **block 维度作为内核首参数传入**，内核用它反解 CUDA 语义的索引。
若内核签名声明了 `bx, by, bz`（或 `blockDim`）前导参数，打包器自动注入。

**Ascend 侧事实（影响分析的基准）**：
- 一次 `aclrtLaunchKernel(func, numBlocks, ...)` 会启动 `numBlocks` 个**内核实例**，
  每个实例占一个 AI Core 任务槽（910B 约有 20+ AI Core）。超过核数的实例排队时分复用
  —— 这与 CUDA"block 排队上 SM"同构，所以 **numBlocks 语义上对应 CUDA 的
  grid 总尺寸，而不是 grid×block**。
- AscendC 实例之间**没有 CUDA 意义上的 thread**：实例内用向量指令（LocalTensor，
  一个向量指令处理 128 个 float）做数据并行；`GetBlockIdx()`/`GetBlockNum()`
  对应 CUDA 的 `blockIdx`/`gridDim`。

**影响 1 —— 语义：每个 AscendC 实例 = 一个 CUDA thread（可精确反解）**

设 `B = bx*by*bz`，`G = gx*gy*gz`，`numBlocks = G*B`。内核内：

```cpp
uint64_t tid = GetBlockIdx();            // [0, G*B)
uint64_t cuda_block = tid / B;           // 反解 CUDA blockIdx（按需再拆 x/y/z）
uint64_t cuda_thread = tid % B;          // 反解 CUDA threadIdx
```

这样从 CUDA 移植的**索引数学**可以逐行照搬（每个实例处理原来 1 个 thread 的活）。
这是乘积映射买到的东西：源码级语义兼容。

**影响 2 —— 性能：超订阅因子 = B（必须正视）**

- CUDA 上典型 `block=(256,)`：同样的 N 元素工作量，Ascend 侧实例数是
  CUDA grid 的 256 倍。每个实例有固定开销（任务派发、Pipe/UB 初始化、
  tiling 边界计算），向量单元一次能处理 128 lane —— **每实例只算 1 个元素
  是对向量单元的严重浪费**。
- 结论：乘积映射保证**正确**，但直接把 CUDA 内核逐 thread 翻译通常**不达标**。
  推荐的内核写法是 AscendC 原生风格：实例数取 `numBlocks = G`（或核数相关值），
  每实例用向量指令处理一段连续数据（`ceil(N/GetBlockNum())`）。
- **API 兼容性与性能的取舍**：内核代码反正必须用 AscendC 重写（CUDA 源码
  不能直接编译），所以保留 cupy 的 `(grid, block)` 签名买到的只是
  *调用方*代码兼容，内核本身仍需按影响 1/2 两种风格之一编写。文档将明确：
  - 逐 thread 翻译风格（乘积映射 + 反解）：移植快，适合访存/控制流复杂、
    计算量小或作为正确性基准；
  - AscendC 原生风格（`numBlocks = grid`，调用时传 `block=(1,)` 即退化）：
    性能正解，M4 内置算子一律用这种。

**影响 3 —— 资源与上限**

- `numBlocks` 上限未知（硬件/驱动队列深度，CUDA 有 2^31-1/65535 限制）：
  M1 spike 需实测大 `G*B`（如 10^8）是否报错，打包器按结果加**钳制与告警**。
- 每实例的 UB/L1 占用由内核 tiling 决定，实例数 × 实例占用 > 片上资源时
  由运行时分时换入换出，吞吐下降 —— 原生风格下实例数取核数量级最稳。

**影响 4 —— 同步与共享内存**

- CUDA `__syncthreads()` / `__shared__` 在"每实例 = 一个 thread"的反解风格下
  **没有对应物**（实例间无共享内存，只有 GM 原子操作/Que 通信）。
  依赖块内同步的 CUDA 内核无法用乘积映射自动移植 —— 这是乘积映射的硬边界，
  此类内核必须改写为 AscendC 原生风格（块内 = 实例内 UB 搬入搬出）。

**文档承诺**：`RawKernel.__call__(grid, block, args)` 保持 cupy 签名；
映射规则（乘积 + block 维度首参注入）写进 docstring 与用户指南，
并在运行时对 `G*B` 超过实测上限的情况给出明确报错。

### 3.5 客户编写 kernel / ufunc 的扩展点（需求 2）

| 层级 | 客户写什么 | 得到什么 |
|---|---|---|
| RawKernel | AscendC 内核源码 | 手动 `kern(grid, block, args)` 启动 |
| RawModule | 多内核 + `get_function` | 模块级缓存复用 |
| ElementwiseKernel（现有 API，升级） | `in_types/out_types + 运算表达式` | M4 起把表达式模板化成 AscendC elementwise 内核自动 JIT，替换当前 fallback |

~~自定义 aclnn op 插件进 ufunc 派发~~ —— 随路线 B 裁决一并取消（§2）。

`ElementwiseKernel` 升级是"客户写 ufunc"的主路径：客户代码零改动，
后端自动从"Python 循环 fallback"切换到"AscendC JIT 内核"。
内置算子补缺（M4）一律用 AscendC 原生风格 + `numBlocks = grid`，
不走乘积反解风格。

### 3.6 目录 / 模块落点汇总

```
cupy/backends/ascend/
  kernels/                    # 内置 AscendC 源码（新增）
    __init__.py               # KERNELS 注册表
    add.cpp, ...              # 缺失算子内核（M4 起逐个补）
  api/
    acl_utils.pyx             # 派发表（不改动；无 CUSTOM_OP 预留）
cupy/_ascend/_core/
  raw_kernel_stub.pyx         # 替换为真实现（RawKernel/RawModule，已在 ascend_files 列表）
  _compiler.pyx?              # BishengKernelCompiler（新，对标 cupy.cuda.compiler）
cupy/backends/ascend/bisheng.py  # bisheng 子进程封装 + 缓存管理（新增，Python）
cupy/backends/ascend/triton_bridge.py  # M6：triton-ascend 零拷贝适配 + jit_ufunc（新增，可选依赖）
install/cupy_builder/
  backends/ascend.py          # AOT 开关时复用 get_device_compile_args()
```

---

## 3A. 补充设计 1：Python 写内核 —— triton-ascend 前端（可选层）

### 3A.1 定位与分层

triton-ascend（华为对 OpenAI Triton 的 Ascend 移植）是**Python 前端 → AscendC/IR → 二进制**
的生成器，它**不替代本计划的路线 A，而是叠加在路线 A 之上**：

```
层级 3（可选）：@triton.jit Python 内核        ← triton-ascend 生成 AscendC/IR 并编译
层级 2（本计划核心）：RawKernel/RawModule      ← AscendC 源码，bisheng JIT，aclrt 加载
层级 1（共用底座）：aclrtBinaryLoad → aclrtLaunchKernelV2  ← 两条路径汇聚于此
```

两条路径的产物最终都走 `aclrtBinaryLoad/GetFunction/LaunchKernel` ——
M1-M3 建成的底座对 triton-ascend 直接可用；差异只在"源码从哪来"。

### 3A.2 零拷贝（no-copy）内存衔接 ⭐

cupy 的内存池最终走 `aclrtMalloc`，与 triton-ascend 期望的设备内存**同源**，
因此零拷贝只需传指针、不搬数据：

```python
class _CuPyTensorAdapter:
    """把 cupy.ndarray 适配成 triton-ascend 期望的 tensor-like 接口。
    纯视图：只暴露 data_ptr/shape/strides/dtype，绝无数据拷贝。"""
    def __init__(self, arr: cupy.ndarray):
        self._arr = arr
    def data_ptr(self):        return self._arr.data.ptr
    @property
    def shape(self):           return self._arr.shape
    @property
    def stride(self):          return tuple(s // self._arr.itemsize for s in self._arr._strides)
    @property
    def dtype(self):           return self._arr.dtype
```

必须桥接的三件事（风险集中在流，不在内存）：

| 事项 | 做法 | 风险 |
|---|---|---|
| 设备上下文 | cupy 与 triton 都基于 `aclrtSetDevice` 的进程级上下文 | 低；确认无二次 init |
| **流** | 启动前把 cupy 当前流注入 triton launcher（或经 adapter 传 stream 句柄） | **高**：各用各的流 = 数据竞争 |
| 指针生命周期 | 同流序保证：kernel 排队在流上，pool 释放也在流序之后（与 CUDA 同理） | 低；跨流用法需用户自管 |

非连续数组：adapter 传真实 strides，由 triton kernel 索引公式消化（triton 天然支持）；
不复制、不 contiguous 化 —— 这是相对"先 `.ascontiguousarray()` 再传"的零拷贝卖点。

### 3A.3 Python ufunc 计划（客户视角的完整样例）

```python
import cupy
from triton_ascend import tl                      # triton-ascend（可选依赖）
import cupy.backends.ascend.triton_bridge as tb   # 本计划新增

# ---- 客户只写这一层（纯 Python）----
@tb.jit_ufunc(('ff->f', 'dd->d', 'll->l'))        # dtype 签名表，对标 create_ufunc
def my_add(x, y, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(z_ptr := offs, tl.load(x_ptr := offs, mask) + tl.load(y_ptr := offs, mask), mask)

# ---- 使用：与 cupy 原生 ufunc 无差别 ----
a = cupy.arange(10, dtype=cupy.float32)           # 零拷贝直传
my_add(a, a, out=z)                               # 派发：triton JIT → 缓存 → aclrt 启动
```

`triton_bridge.jit_ufunc` 的职责（新建 `cupy/backends/ascend/triton_bridge.py`）：
1. 适配器包装入参/出参（零拷贝）+ 注入当前流；
2. 调 triton-ascend 编译缓存（它自带 `~/.triton/cache`，键含 SoC/CANN 版本）；
3. 把 `(name, dtype签名)` 注册进 Python 层 ufunc 表，与 ElementwiseKernel 派发并存；
4. **可选依赖策略**：`import triton_ascend` 失败 → 该 ufunc 回退到现有
   ElementwiseKernel/Python fallback，行为降级不报错（对标 cupy 对 cupy.cuda.jitify 的态度）。

### 3A.4 排期与依赖

- **不阻塞 M1-M5**：底座（§2/§3）先落地，triton-ascend 前端排为 **M6**，
  与 Memory.md P3 的"等 triton-ascend 成熟"节奏一致。
- 前置条件：triton-ascend 与 CANN 9.0.1 的版本矩阵实测（它自身对 CANN/SoC 版本敏感）；
  M6 开工前先做一个 30 分钟的 spikes：`pip install triton-ascend` + 零拷贝 adapter POC。
- 内置算子补缺（M4）**不依赖** triton-ascend：AscendC 手写内核是确定项，
  triton 路线是给客户的"低门槛写 ufunc"通道。

---

## 3B. 补充设计 2：是否复用 cupy 的 compiler 基础设施？

### 3B.1 盘点：cupy 现有"编译器栈"的三块资产

| 资产 | 位置 | 耦合度 | 对 Ascend 的可复用性 |
|---|---|---|---|
| 编译/缓存骨架 | `cupy/cuda/compiler.py`（纯 Python：`_compile_module_with_cache`、缓存键、`CompileException`、日志捕获、预处理） | 高（nvrtc/nvcc/hipcc 内嵌） | **骨架可移植，代码不可 import** |
| Module/Function 抽象 | `cupy/xpu/function.pyx`（CPointer、模块表；但 `Module.load` 走 `driver.linkAddData(CU_JIT_INPUT_PTX)`） | 中（driver 调用点集中、已有 `IF` 分离先例） | **直接复用抽象**，仅替换加载实现 |
| RawKernel 替换槽位 | `cupy/_ascend/_core/raw_kernel_stub.pyx`（已在 `ascend_files` 列表） | 无 | **就是为此预留的** |

### 3B.2 决策：分两步走

**短期（M2，确定执行）—— 移植而非 import**：
新建 `cupy/backends/ascend/bisheng.py`，从 `cupy/cuda/compiler.py` 抄骨架
（缓存键哈希、磁盘缓存两段式、异常类、日志透传约 150-200 行），编译实现换成
spawn bisheng。理由：
- `compiler.py` 的函数体与 NVRTC/nvcc 深度耦合（arch 探测、jitify、devlink），
  改造成中性层要动上游文件，违背"ascend 改动不回灌 CUDA 路径"的项目约定；
- 骨架本身很小，复制成本低；
- CUDA 侧行为零变化，回归风险为零。

**中期（M5+，视需要）—— 收敛为中性骨架**：
若将来 CUDA 侧也要动缓存层（如新缓存键策略），再把两份实现上提为
`cupy/backends/backend/compiler.py`（模板方法：backend 只提供
`compile(source, options) -> bytes` 与缓存目录），与本仓库已有的
`install/cupy_builder/{features,backends}/` 抽象同构。**现在不做**。

**Module/Function 层**：继续用 `cupy.xpu.function` 的抽象（`CPointer` 打包、
per-device 模块表、`get_cached_module`），在 `.pyx` 内用
`IF CUPY_CANN_VERSION > 0:` 把 `Module.load` 的实现分叉为
aclrt 三连（`aclrtCreateBinary/BinaryLoad/BinaryGetFunction`）——
该文件已有 `IF` 分离 texture 等 CUDA 特性的先例（`function.pyx:118`），
是既定模式，不新造抽象。

### 3B.3 对比结论一句话

> **复用"抽象与骨架"，不复用"实现代码"**：抽象层（xpu.function + raw_kernel_stub）
> 是现成的插座位；`compiler.py` 的缓存骨架以移植方式进入 `bisheng.py`；
> 中性化收敛推迟到 CUDA 侧也有收益时再做。

---

## 4. cupy/nvcc 与 numpy-ascend/bisheng 工作流对比

| 维度 | cupy (CUDA) | numpy-ascend (Ascend) | 备注 |
|---|---|---|---|
| 内核语言 | CUDA C++ | AscendC（C++ 子集 + GM_ADDR/LocalTensor） | 心智差异最大处 |
| 编译器 | nvcc / NVRTC（进程内 JIT） | bisheng（子进程 JIT） | JIT 有 fork 开销 → 缓存必须 |
| 头文件 | cuda_runtime.h | kernel_operator.h（`x86_64-linux/ascendc/`） | |
| 编译选项 | `-arch=sm_xx` | `--soc-version=ascend910b` 等 | SoC 从驱动探测 |
| 产物 | PTX/cubin/fatbin | aicore fatbin (.o) | |
| 内存加载 | `cuModuleLoadData` | `aclrtCreateBinary` + `aclrtBinaryLoad` | |
| 取函数 | `cuModuleGetFunction` | `aclrtBinaryGetFunction` | |
| 启动 | `cuLaunchKernel(grid, block, args[], stream)` | `aclrtLaunchKernelV2(func, numBlocks, argsData, cfg, stream)` | grid×block → numBlocks；args[] → argsData |
| grid 语义 | 二维/三维 grid×block | 一维 numBlocks | 需映射层 |
| 缓存目录 | `~/.cupy/kernel_cache` | `~/.cache/cupy/cann_kernel_cache` | |
| wheel 分发 | 只带源码，运行时 JIT | 同（AOT 可选） | |
| host 端包装 | Cython 直调 driver API | Cython 直调 aclrt API | 对称 |
| 调试 | cuda-gdb / printf | printf + msdebug（弱） | 风险项 |

---

## 5. 分阶段实施计划

### M1 — Spike：打通全链路（最高风险优先）⭐
- **素材已在树内**（本计划评审时发现）：`cupy/backends/ascend/demo/` 下
  `cos_custom.cxx`（完整 AscendC 内核：CopyIn/Compute/CopyOut 流水线）、
  `load_bin_main.cxx`（已走通 `aclrtBinHandle→aclrtBinaryGetFunction→
  ArgsHandle/ParamHandle→aclrtLaunchKernelWithConfig`）、`CMakeLists.txt`+`build.sh`
  （aic/aiv/host 三段 CMake 工程，含 build/ 产物）。
- M1 范围收窄为：① 跑通 `demo/build.sh`（bisheng 路径已修复）；② 把
  `load_bin_main.cxx` 的加载/启动调用序列整理成 Cython extern 声明草案；
  ③ 确认 `argsData`/`aclrtArgsHandle` 参数打包布局与 numBlocks 实测上限；
  ④ 笔记回写本文档 §4 差异表。
- **验证分级**：本机（无 NPU）到 L2（编译成功 + aclrt API 链接通过）；
  数值验证需 910B（与现有约定一致，不谎称数值正确）

### M2 — JIT 编译器 + 缓存（`cupy/backends/ascend/bisheng.py`）
- `BishengKernelCompiler`：spawn bisheng、缓存键、磁盘缓存、并发锁、错误透传
- 缓存骨架**移植自** `cupy/cuda/compiler.py`（不 import，见 §3B.2）
- SoC 型号探测（驱动 → ascend910b 等映射表）
- 纯 Python，可本机单测（mock bisheng + 真 bisheng 编译两档）

### M3 — RawKernel/RawModule 真实现（`raw_kernel_stub.pyx` 重写）
- `aclrt` API 的 Cython extern 声明（新增 `cupy/backends/ascend/api/aclrt_driver.pyx`
  或并入 acl_utils）
- `cupy.xpu.function.Module/Function` 对接、per-device module 表、二进制卸载
- 本机可测 L1-L3（编译+加载路径 mock 设备部分）

### M4 — 用基础设施补缺失算子
- 按 Progress.md 缺口清单，优先"无 aclnn 等价"的：
  `conjugate/imag/angle/frexp/ldexp/modf/nextafter/choose/...`
- 每个算子 = `kernels/<op>.cpp` + 注册表项 + （必要时）Cython fallback 组合逻辑
- 910B 上批量数值验证

### M5 — 客户扩展点与文档
- ElementwiseKernel → AscendC 模板化 JIT（表达式 → 内核）
- 用户指南：`docs/ascend/CustomKernelUsage.md`
  （含 §3.4.1 两种内核编写风格的选型指引与移植陷阱清单）

### M6 — triton-ascend Python 前端（可选，见 §3A）
- 前置 spike：triton-ascend × CANN 9.0.1 版本矩阵实测 + 零拷贝 adapter POC（30 分钟级）
- `triton_bridge.py`：`_CuPyTensorAdapter`（零拷贝）、流桥接、`jit_ufunc` 注册、
  依赖缺失回退
- 风险最高项：**流桥接**（cupy 流 ↔ triton launcher 流不一致 = 数据竞争）

### 里程碑依赖与风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| `aclrtLaunchKernelV2` 参数布局/numBlocks 语义与假设不符 | M3 API 设计 | M1 spike 先行确认 |
| `numBlocks = G*B` 超过运行时上限 | 大网格场景 | M1 实测上限，打包器钳制 + 报错 |
| 乘积映射下逐 thread 翻译风格性能不达标 | M4 内置算子 | 内置算子一律 AscendC 原生风格（numBlocks=grid） |
| AscendC 语言版本随 CANN 变化 | 缓存键/源码兼容 | 缓存键含 CANN series；`IF` 条件编译 |
| 本机无 NPU，仅能 L1-L2 | 数值正确性验证 | M1/M4 的数值验证集中在 910B 批量做 |
| bisheng 子进程慢（fork + 编译秒级） | 首调延迟 | 磁盘缓存 + AOT 开关 + 显式预热 |
| kernel 参数打包 ABI（GM_ADDR 序列） | M3 | spike 确认 `argsData` 布局后再定稿 CPointer 映射 |

---

## 6. 决策记录（2026-09 确认）✅

| # | 决策点 | 结论 |
|---|---|---|
| 1 | 技术路线 | **只做路线 A**（RawKernel 直启）；路线 B（custom aclnn 插件）及 `CUSTOM_OP` 预留整体取消 |
| 2 | 编译时机 | **JIT 优先 + AOT 可选**（`CUPY_ASCEND_AOT_KERNELS=1` 构建期预编译内置内核） |
| 3 | grid/block 映射 | **`numBlocks = grid × block` 乘积映射**，block 维度作内核首参注入；影响详见 §3.4.1（语义可反解 / 性能需原生风格 / 上限待 spike / 块内同步无对应物） |
| 4 | 启动时机 | 文档先行评审（本文档），评审通过后启动 M1 |
| 5 | 内置内核位置 / 缓存目录 | 按文档 §3.1（`cupy/backends/ascend/kernels/`、`~/.cache/cupy/cann_kernel_cache/`），评审时如有异议一并调整 |
| 6 | Python 内核（triton-ascend） | **M6 可选层**：triton-ascend 为 Python 前端，产物仍走 aclrt 底座；零拷贝经 `_CuPyTensorAdapter`（data_ptr + strides，不搬数据）；流桥接是主要风险点；缺依赖时回退 ElementwiseKernel fallback（§3A） |
| 7 | compiler 基础设施复用 | **复用抽象与骨架，不复用实现代码**：抽象层用 `cupy.xpu.function` + `raw_kernel_stub` 现成插座位；`cupy/cuda/compiler.py` 缓存骨架移植进 `bisheng.py`（不 import）；中性化收敛推迟到 CUDA 侧有收益时（§3B） |
