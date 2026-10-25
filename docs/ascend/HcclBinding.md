# HCCL 绑定（`cupy.cuda.nccl` → `cupyx.distributed`）

> 状态：**活跃 / 方案已落地（实现完成，L4 待有卡验证）** · 2026-10-25
> 关联：[Readme.md](Readme.md) §0 速查、[Roadmap.md](Roadmap.md)（`cupyx.distributed` 行）、
> [TODO.md](TODO.md)「hccl 多卡支持」、[DeveloperNotes.md](DeveloperNotes.md)（五层架构）
>
> 涉及文件：`cupy/backends/ascend/cupy_hccl.h`、`cupy/backends/cupy_xccl.h`、
> `install/cupy_builder/features/ascend_hccl.py`、`install/cupy_builder/_features.py`、
> `cupyx/distributed/_klv_utils.py`、`cupyx/distributed/_store.py`、`setup.py`

---

## 1. 目标与范围

让 `cupyx.distributed`（`NCCLBackend` / `DistributedArray`）在 Ascend 上可用，
**复用上游 CuPy 的 NCCL 代码路径**，只把「NCCL 实现」替换成 HCCL。

做（本次）：

* `cupy/backends/ascend/cupy_hccl.h`：完整的 **NCCL API 兼容层**，逐个函数映射到 HCCL；
  仿照 `cupy/backends/hip/cupy_rccl.h` 的位置与职责（`cupy_rccl.h` 只 include + typedef，
  因为 RCCL 就是 NCCL 的 API 兼容实现；HCCL **不是**，所以本文件是一个真正的 shim）。
* 构建集成：新增可选 feature `ascend_hccl`，把 `nccl.pyx` 编成
  `cupy.backends.cuda.libs.nccl`（运行时经 `sys.modules` 别名暴露为 `cupy_backends.cuda.libs.nccl`）。
* Cython/Python 侧适配：HCCL 的 **root info 是 4108 B**（NCCL uniqueId 只有 128 B），
  `cupyx.distributed` 的 TCP store 值缓冲从 256 B 扩到 8 KiB。
* `setup.py`：`cupyx.distributed` 从 Ascend wheel 的排除列表中移除。

不做（明确留给后续，见 §7）：

* `ncclCommSplit` / `NcclConfig`（需要 Cython 层 Ascend 分支，见 §4.1）。
* fp64、`NCCL_AVG` 归约（HCCL 无对应）。
* 自环 P2P（rank i → rank i）的等价替换（`_nccl_comm.py` 的 scatter/gather/all_to_all 会用到）。

---

## 2. 调用链与分层原理

```
cupyx.distributed.init_process_group()            cupyx/distributed/_init.py
        │  backend='nccl'
        ▼
NCCLBackend / _DenseNCCLCommunicator              cupyx/distributed/_nccl_comm.py
        │  nccl.get_unique_id() / nccl.NcclCommunicator / comm.allReduce(...)
        ▼
cupy.cuda.nccl                                    cupy/cuda/nccl.py（薄壳）
        │  from cupy_backends.cuda.libs.nccl import *
        ▼
nccl.pyx  (module: cupy.backends.cuda.libs.nccl)  cupy/backends/cuda/libs/nccl.pyx
        │  cdef extern from '../../cupy_xccl.h'
        ▼
cupy_xccl.h                                       后端选择 + _ncclAllReduce/… 转发
        │  #if defined(CUPY_USE_ASCEND) → ascend/cupy_hccl.h
        ▼
cupy_hccl.h   NCCL 名字 + NCCL 数值约定            类型/错误码/组语义翻译
        ▼
HCCL: HcclGetRootInfo / HcclCommInitRootInfo / HcclAllReduce / HcclBatchSendRecv …
```

**为什么不去写一套新的 `cupyx.distributed` HCCL backend？**

1. `nccl.pyx` 只是 C API 的 Cython 声明，本身与设备无关；`_nccl_comm.py` 只用到
   NCCL 的 8 个集合通信 + 3 个 P2P（`send/recv/groupStart|End`），语义与 HCCL 一一对应。
2. 上游一旦改动 `_nccl_comm.py`（例如新增后端或改 dtype 表），shim 方案零成本跟进；
   自建 backend 则会分叉成需要长期维护的第二套实现。
3. 与 ROCm 的既有做法一致（`cupy_rccl.h`），reviewer 心智负担最低。

代价是 **数值约定必须在 shim 里翻译**：`cupy/backends/cuda/libs/nccl.pxd` 把 NCCL 的枚举
数值（`NCCL_INT8 = 0` … `NCCL_BFLOAT16 = 9`）硬编码给 Python 层，Python 再把 int 原样传回
C++。所以 `cupy_hccl.h` 里自定义的 `ncclDataType_t` / `ncclRedOp_t` 必须使用 **NCCL 的数值**，
HCCL 的数值只在调用点翻译（见 §4.2）。

---

## 3. 兼容层的类型约定

| 概念 | NCCL（Cython/Python 期望） | 本 shim | 说明 |
|---|---|---|---|
| 通信域 | `ncclComm_t`（`void*`） | `typedef HcclComm ncclComm_t` | 同为 `void*`，`<intptr_t>self._comm` 可直接用 |
| bootstrap id | `ncclUniqueId` 128 B | `typedef HcclRootInfo ncclUniqueId` **4108 B** | 字段名 `internal` 一致，Cython 侧无需改结构 |
| id 长度常量 | `NCCL_UNIQUE_ID_BYTES` | `#define NCCL_UNIQUE_ID_BYTES HCCL_ROOT_INFO_BYTES` | 见 §4.3（`cdef enum` 在 `cdef extern` 内不落盘，宏生效） |
| stream | `cudaStream_t` | `typedef aclrtStream cudaStream_t` | 与 `hip_rccl.h` 的 `typedef hipStream_t` 同构；`runtime.Stream` 是 `ctypedef void*` |
| 结果码 | `ncclResult_t` | 自定义 enum，值域 0..6 | 见 §4.4 |
| dtype / redop | NCCL 数值 | 自定义 enum（NCCL 数值） | 调用点 `ToHcclDataType()` / `ToHcclReduceOp()` |

---

## 4. 兼容层实现要点

### 4.1 版本号策略：`NCCL_VERSION_CODE = 21700`（2.17.0）

`nccl.pyx` 的每个 API 都由这个宏守卫，选择哪一条「兼容分支」直接决定 shim 要提供什么：

| 报告版本 | 触发的分支 | 对本 shim 的影响 |
|---|---|---|
| `>= 2.2` | `ncclBroadcast` 允许 send≠recv | 需要在 root 上先 `aclrtMemcpyAsync`（§4.6） |
| `>= 2.4` | `abort()` / `checkAsyncError()` 可用 | 提供（§4.7） |
| `>= 2.7` | `send` / `recv` 可用 | 提供（P2P 是 `cupyx.distributed` 的散射/聚集基础） |
| `< 2.18` | `cupy_xccl.h` 用 **mock** `_ncclConfig_t{int splitShare;}` | **必须选这条** |
| `>= 2.18` | `_ncclConfig_t` 变成真正的 `ncclConfig_t` | `NcclConfig.__init__` 里 `self._config.splitShare = …`（`nccl.pyx:278`）会编译失败 |

即：**2.17.0 是当前 Cython 层允许的最高「完整可用」等级**。副作用（已实测确认）：

* `NcclConfig(split_share)` **可以**构造（守卫是 `< 2.17`），`split_share` 可读写；
* 但把它传给 `NcclCommunicator(..., config=cfg)` 会 raise
  `RuntimeError: NcclConfig is not supported in this version`（该守卫是 `< 2.18`）；
* `commSplit()` 同样 raise `RuntimeError: ncclCommSplit is not available in this version`。

若将来要支持 `commSplit`，正确做法是给 `nccl.pyx` 加 `IF CUPY_CANN_VERSION > 0` 分支，
把 `_ncclConfig_t` 换成 `HcclCommConfig`（`HcclCommConfigInit()` 已在 `hccl_comm.h` 内联提供），
并把 `_ncclCommSplit` 指向 `HcclCreateSubCommConfig`。

### 4.2 枚举/错误码映射表

dtype（**左列是 NCCL 数值 = `nccl.pxd` 暴露给 Python 的数值**）：

| NCCL | 值 | HcclDataType | 备注 |
|---|---|---|---|
| `ncclInt8` | 0 | `HCCL_DATA_TYPE_INT8` (0) | |
| `ncclUint8` | 1 | `HCCL_DATA_TYPE_UINT8` (7) | |
| `ncclInt32` | 2 | `HCCL_DATA_TYPE_INT32` (2) | |
| `ncclUint32` | 3 | `HCCL_DATA_TYPE_UINT32` (9) | |
| `ncclInt64` | 4 | `HCCL_DATA_TYPE_INT64` (5) | |
| `ncclUint64` | 5 | `HCCL_DATA_TYPE_UINT64` (6) | |
| `ncclFloat16` | 6 | `HCCL_DATA_TYPE_FP16` (3) | |
| `ncclFloat32` | 7 | `HCCL_DATA_TYPE_FP32` (4) | |
| `ncclFloat64` | 8 | `HCCL_DATA_TYPE_FP64` (10) | HCCL 多数集合算子文档只列到 fp32/bfp16，**实测待定**（§7） |
| `ncclBfloat16` | 9 | `HCCL_DATA_TYPE_BFP16` (11) | |

归约：`SUM/PROD/MAX/MIN` → `HCCL_REDUCE_SUM/PROD/MAX/MIN`（数值恰好也是 0..3）；
`NCCL_AVG(4)` → `HCCL_REDUCE_RESERVED` → 返回 `ncclInvalidUsage`（不静默算错）。

错误码 `HcclResult` → `ncclResult_t` 是**有损折叠**，且必须落在 0..6：
`nccl.pyx` 用 `ERROR2[status]` 索引，越界会抛 `KeyError` 而不是可读错误。
映射：`PARA/PTR→4`、`NOT_SUPPORT/NOT_FOUND/UNAVAIL→5`、`MEMORY/OOM→2`、
`REMOTE/SYSCALL/TIMEOUT/网络类→6`、其它 `→3`、`SUCCESS→0`。

### 4.3 `NCCL_UNIQUE_ID_BYTES` 为什么改宏就行

`nccl.pyx` 里是：

```cython
cdef extern from '../../cupy_xccl.h':
    cdef enum:
        NCCL_UNIQUE_ID_BYTES = 128
    ctypedef struct ncclUniqueId:
        char internal[NCCL_UNIQUE_ID_BYTES]
```

`cdef enum` 位于 `cdef extern` 块内时 **Cython 只登记名字、不生成定义**（已用最小用例
`cython x.pyx && grep FOO_BYTES x.cpp` 验证：生成代码里出现的是符号 `FOO_BYTES`/`NCCL_UNIQUE_ID_BYTES`，
不是字面量 128）。因此头文件里的 `#define NCCL_UNIQUE_ID_BYTES 4108` 就是唯一事实来源，
`len(commId) != NCCL_UNIQUE_ID_BYTES`、`internal[:NCCL_UNIQUE_ID_BYTES]` 的 memcpy
都自动变成 4108，`nccl.pyx` 一行不用改。头里另有两条 `static_assert` 兜底
（`sizeof(HcclRootInfo) == 4108`、`sizeof(ncclUniqueId) == NCCL_UNIQUE_ID_BYTES`）。

### 4.4 组语义：`ncclGroupStart/End` → `HcclBatchSendRecv`

HCCL **没有**「打开一个组」的原语，P2P 在 issue 时就与对端配对。直接把
`ncclGroupStart/End` 变成空操作会让 `_nccl_comm.py` 的
`scatter / gather / all_to_all / send_recv`（group 内多条 send+recv）**死锁**。

所以 shim 里做一个小缓冲（`cupy_hccl::Group`，`thread_local`）：

* `ncclGroupStart()` → `active = true`，清空缓冲；
* `ncclSend/ncclRecv()` 在 group 内 → 追加 `HcclSendRecvItem{sendRecvType, buf, count, dataType, remoteRank}`；
* `ncclGroupEnd()` → 一次 `HcclBatchSendRecv(items, n, comm, stream)` 发出；
* **comm 变化时先 flush**（一次 batch 绑定一个 comm/stream）；
* 集合通信（AllReduce/…/Broadcast）**不缓冲**，按原顺序立即下发 —— `HcclBatchSendRecv`
  只收 P2P，而 `_SparseNCCLCommunicator` 会在 group 内调用 `broadcast`，缓冲会改变语义。

未开组时 `ncclSend/ncclRecv` 直接走 `HcclSend/HcclRecv`。

### 4.5 弱符号保护

HCCL 把「通信域管理」类接口声明为 `HCOMM_WEAK_SYMBOL`：8.5 在 `libhccl_fwk.so`，
9.x 在 `libhcomm.so`（`libhccl_fwk.so` 变成指向它的符号链接）。若驱动/库版本不含某符号，
链接不会失败，但运行期是 `nullptr` —— 直接调用就是段错误。
`GetRootInfo/CommInitRootInfo/CommInitAll/CommDestroy/GetRankSize/GetRankId/`
`GetCommAsyncError/CommSuspend` 全部先 `Available(fn)` 判空，缺失时返回
`ncclInvalidUsage`（`HcclGetCommAsyncError` 缺失时退化为「无错误」，保持可轮询）。

### 4.6 `ncclBroadcast` 非原地

`HcclBroadcast` 只有原地版本。NCCL 允许 `sendbuff != recvbuff`（只有 root 的
`sendbuff` 有效）。shim 在 root 上先 `aclrtMemcpyAsync(recv ← send, D2D, stream)`，
再对 `recvbuff` 做原地广播。`cupyx.distributed` 走的是 `send == recv` 的路径，
这条分支只在用户直接调用 `nccl.NcclCommunicator.broadcast(a, b, …)` 时才触发。

### 4.7 `ncclCommAbort` = suspend + destroy

HCCL 没有 abort。`abort()` 在 NCCL 里是「不等待就丢弃通信域」；这里实现为
`HcclCommSuspend`（尽力而为，停止在途请求）+ `HcclCommDestroy`（释放资源）。
这是必要的：`nccl.pyx` 的 `abort()` 之后会把 `_comm` 置 0，`__dealloc__` 不再 destroy，
若只 suspend 就会泄漏通信域。

### 4.8 参数校验

所有入口先查 `comm == nullptr`、`peer/root < 0`、dtype/op 是否可映射，返回
`ncclInvalidArgument` / `ncclInvalidUsage`，而不是把非法枚举丢给 HCCL
（HCCL 对非法 dtype 的返回码在不同版本不一致，且部分会直接 abort 进程）。

---

## 5. 构建集成

### 5.1 `cupy_xccl.h` 的分支顺序修复（必要）

原顺序是

```c
#if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)   // ← 先命中
#elif defined(CUPY_USE_HIP)
#elif defined(CUPY_USE_ASCEND)
```

而 Ascend 构建里 `CUPY_NO_CUDA` **不会**被定义（`cupy_setup_build.py:356`
`no_cuda = ctx.use_stub`，只对 stub 构建为真）。结果 Ascend 会走进 CUDA 分支，
`#include <nccl.h>` 直接报错。已把 `CUPY_USE_ASCEND` 提到最前，并加注释说明原因。

### 5.2 新增 feature `ascend_hccl`（可选、可降级）

`install/cupy_builder/features/ascend_hccl.py`：

| 项 | 值 |
|---|---|
| 模块 | `cupy.backends.cuda.libs.nccl`（源 = `cupy/backends/cuda/libs/nccl.pyx`，未改） |
| include 探针 | `hccl/hccl.h`（`<sdk>/include` 已在 include path 上） |
| 链接库（按存在性过滤） | `hccl` + 「框架库」(`hcomm` 优先，否则 `hccl_fwk`) + `hccl_alg` + `hccl_plf` + `ascendcl` |
| 版本 | 无（`get_version()` 返回 `None`，summary 显示 `ascend_hccl: Yes`） |
| 环境变量 | `CUPY_ENABLE_HCCL=0` 跳过检测；`=1` 缺失即构建失败（CI 用） |

要点：

* `libhccl.so` 只有集合/P2P（`HcclAllReduce`…`HcclSend`）；
  `HcclGetRootInfo`/`HcclCommInitRootInfo` 等在「框架库」里，名字随版本变（§4.5）。
* 9.x 的 `libhcomm.so` 自身存在未定义引用（`hccl::CollAlgOperator::GetBlockDim` 等），
  必须把 `hccl_alg` / `hccl_plf` 一起放上链接行，否则 `ld` 直接报 undefined reference
  （8.5.1 同样需要）。已在两套 CANN 上验证过链接行。
* 每个模块只链接 **自己 feature 的** `libraries`（`cupy_setup_build.py:461`
  `s['libraries'] = module.libraries`），所以 `ascendcl` 必须在本 feature 里重复声明。
* 找不到头文件 → `modules = []` + 保留一个库名做探针 → 走「optional module skipped」
  流程（summary 里 `ascend_hccl: No`），`import cupy` 不受影响，
  `cupy.cuda.nccl.available = False`，`init_process_group` 抛原有错误。

### 5.3 root info 4108 B 过 TCP store

`cupyx.distributed` 用 TCP store 交换 bootstrap id，而 KLV 消息体是定长 256 B
（`_klv_utils._VALUE_BUFFER_SIZE`，`_store.py` 里还有一处硬编码 `> 256` 校验）。
4108 B 装不下 → 握手必然 `ValueError: Invalid length for message`。

改动（三处，最小面）：

* `cupyx/distributed/_klv_utils.py`：`_VALUE_BUFFER_SIZE = 256 → 8192`（带注释说明 HCCL）；
* `cupyx/distributed/_store.py`：长度校验改用 `_klv_utils._VALUE_BUFFER_SIZE`，去掉魔法数；
* `cupyx/distributed/_store.py`：新增 `_recv_exact()` 并替换两处裸 `socket.recv`。
  帧变成 8 KB 后，单次 `recv` 不再保证返回完整一帧（原实现默认「小包一次到齐」），
  否则偶发 `from_buffer_copy` 报 `Buffer size too small`。

不影响 CUDA/ROCm 语义：只是每条 rendezvous 消息变大，该通道仅用于初始化，且读端改为
「读满一帧」后对小包同样正确。

### 5.4 wheel 侧

`setup.py::_CUPYX_EXCLUDED_ON_ASCEND` 原先把 `cupyx.distributed` 整个从 Ascend wheel 剔除
（理由：NCCL-based，CUDA-only）。现已移除该条并加注释：没有 HCCL 时它仍可 import，
行为与「上游 CuPy 未编译 NCCL」一致。

---

## 6. `nccl.pyx` 为什么一行都不用改

| 潜在问题 | 结论 |
|---|---|
| `NCCL_UNIQUE_ID_BYTES = 128` 与实际 4108 冲突 | 不冲突，§4.3 |
| `ctypedef struct ncclComm` / `ncclComm_t` 与 `typedef HcclComm` 冲突 | 只声明不生成；Cython 始终使用 typedef 名 |
| `<runtime.Stream>stream` 类型不匹配 | `runtime.Stream` 是 `ctypedef void* 'cudaStream_t'`，`aclrtStream` 也是 `void*` |
| `ncclCommDestroy` 返回 `HcclResult`，Cython 声明 `void` | 与上游 NCCL 情况相同（`void` 调用方忽略返回值） |
| `>= 2.0` 的 `ERROR1/ERROR2` 选表 | `NCCL_VERSION_CODE` 常量宏，编译期决定；错误码已折叠到 0..6 |

---

## 7. 已知缺口与风险（有卡后逐项确认）

| # | 项 | 现状 | 处理建议 |
|---|---|---|---|
| R1 | fp64 归约 | 映射到 `HCCL_DATA_TYPE_FP64`，但 HCCL 多数算子文档只列 fp16/fp32/bfp16 | 实测；若不支持则在 `_nccl_comm.py` 的 `_nccl_dtypes` 里对 Ascend 屏蔽 `'d'`，抛清晰错误（NCCL 语义允许 float64，用户需降精度） |
| R2 | `NCCL_AVG` | 返回 `ncclInvalidUsage` | cupy 的 `_nccl_ops` 只映射 sum/prod/max/min，当前不可达 |
| R3 | `commSplit` / `NcclConfig` | `NcclConfig` 可构造但传参被拒；`commSplit` 直接 raise（`< 2.18` 分支） | §4.1 的 Cython 分支方案 |
| R4 | 自环 P2P（rank i → rank i） | `_nccl_comm.py` 的 `scatter/gather/all_to_all` 会给自身发/收 | HCCL 对 self-send 支持随版本/拓扑变化，实测确认；不支持则把这 3 个算子改成集合算子实现（`HcclAllGather`+切片 / `HcclScatter` / `HcclAlltoAll`） |
| R5 | `HcclBatchSendRecv` 行为 | 已按 batch 语义实现；单 item 时退化为普通 send/recv | 实测 group 内 N 发 1 收、1 发 N 收（scatter/gather 形态）不超时 |
| R6 | 通信域初始化要求 | `HcclGetRootInfo` 需在 root 上取、其它 rank 收同一份 4108 B；要求先 `aclrtSetDevice` | `_nccl_comm.py` 已要求「每进程先切设备」，与 `init_process_group` 文档一致 |
| R7 | 超时/环境 | HCCL 走 `HCCL_EXEC_TIMEOUT`、rank table / `HCCL_IF_IP` 等 | 在测试计划 T3 里固定 env 并记录 |
| R8 | 错误信息可读性 | `HcclResult` 折叠成 7 类 | 调试时用 `HCCL_*` 的 plog（`~/ascend/log`）看原始码 |

---

## 8. 当前验证等级（无卡环境）

| 等级 | 内容 | 状态 |
|---|---|---|
| L1 | `cython --cplus nccl.pyx` + `g++ -fsyntax-only` 通过 | ✅（CANN 9.0.1 与 8.5.1 头文件各一次） |
| L2 | 编出 `nccl.so` 并链接 `hccl/hcomm|hccl_fwk/hccl_alg/hccl_plf/ascendcl`，HCCL 符号解析成功 | ✅（两套 CANN 布局都验证过链接行；`ldd` 确认 `libhccl/libhcomm/libhccl_alg/libhccl_plf/libascendcl` 全部解析） |
| L2.5 | 独立 smoke 程序链接后运行，打印 `get_version()=21700`、`sizeof(ncclUniqueId)=4108` | ✅ |
| L3 | `build_ext --inplace` 成功、`ascend_hccl: Yes`；`cupy.cuda.nccl.available`、`get_build_version()==21700`、`get_version()==21700`、dtype/op 常量、`import cupyx.distributed` | ✅ 2026-10-25 重跑（见 §8.1） |
| L4 | 真机多卡数值/死锁 | ❌ 无 NPU |

### 8.1 2026-10-25 实测记录（无 NPU，CANN 9.0.1）

构建：

```
$ export CUPY_INSTALL_USE_ASCEND=1 && python setup.py build_ext --inplace
HCCL found: building cupy.backends.cuda.libs.nccl
            (libraries: hccl, hcomm, hccl_alg, hccl_plf, ascendcl)
  ascend    : Yes (version 901)
  ascend_fft: No        -> Cannot link libraries: ['cann_ops_fft']   # 既有的可选缺口
  ascend_hccl: Yes
EXIT=0        # 仅 ascend_fft 探针的 2 行 "error:"，属预期降级
```

| 用例 | 结果 |
|---|---|
| T1 常量/版本 | ✅ `build_version == runtime_version == 21700`；dtype `0..9`、op `0..3` |
| T1b KLV 4108 B 往返 | ✅ 帧 8200 B、payload 4117 B、`Set.from_klv` 还原一致 |
| T1c `NcclError(0..6)` | ✅ 7 个码全部可读，无 `KeyError`（验证了错误码折叠） |
| T1d `get_unique_id()` | ✅ 无卡时 `NcclError: NCCL_ERROR_SYSTEM_ERROR`（HCCL 返回 syscall/memory 错），未崩溃 |
| T1e-1 长度契约 | ✅ `commId length 128 != 4108`（`NCCL_UNIQUE_ID_BYTES` 宏生效） |
| T1e-2 group 原语 | ✅ `groupStart/groupEnd` 空组返回成功且不下发 HCCL |
| T1e-4/5 init / initAll | ✅ 均到达 HCCL，返回可读 `NcclError` |
| T1f `NcclConfig` / `commSplit` | ✅ 构造成功（`split_share=1`）；传参 raise `not supported`；`commSplit` raise `not available` |
| T2 `init_process_group(1, 0)` | ✅ 走通 TCP store → `nccl.get_unique_id()`（取 root info），止于 `HcclCommInitRootInfo` 报 `NCCL_ERROR_SYSTEM_ERROR`（无卡）；即除设备调用外整链路已通 |
| `pytest tests/cupyx_tests/distributed_tests/test_store.py` | ✅ 2 passed（TCP store + `_recv_exact` 回归） |
| `pytest tests/ascend/test_hccl_binding.py` | ✅ 10 passed（构建特性库解析 7 + shim 逻辑编译运行 3 组参数化） |
| `pytest .../test_array_no_nccl.py` | ❌ 228 errors，全部是 `EL0003 Invalid_Argument` / `rtGetDevMsg`（无卡），非代码问题 |

日志：`tests/logs/2026-10-25-logs/{test_hccl_binding,test_store,test_array_no_nccl}.log`。

**未覆盖**：任何真正下发 HCCL 集合/P2P 的路径（T2–T8 全部需要 NPU）。

---

## 9. 测试计划

### 9.0 约定

* 测试日志：`tests/logs/<YYYY-MM-DD>-logs/<pytest_name>.log`（仓库既有约定）。
* 多卡：每进程一张卡，`ASCEND_RT_VISIBLE_DEVICES=<id>`；多卡机器上并行跑不同子集。
* 每步都先 `source $ASCEND_HOME_PATH/set_env.sh`（`LD_LIBRARY_PATH` 需要包含 CANN `lib64`；
  `liboptiling.so` 已由 `cupy/__init__.py` 用 ctypes 预加载，无需 `LD_PRELOAD`）。

### T0 构建/静态（无卡可跑）

```bash
export CUPY_INSTALL_USE_ASCEND=1
python setup.py build_ext --inplace 2>&1 | tee /tmp/hccl_build.log
grep -E "ascend_hccl|HCCL found" /tmp/hccl_build.log     # 期望 ascend_hccl: Yes
grep -c "error:" /tmp/hccl_build.log                     # 期望 0
```

反例（可选）：`CUPY_ENABLE_HCCL=0 python setup.py build_ext --inplace`
→ summary 应显示 `ascend_hccl: No` 且 `import cupy` 仍成功。

注意：只改 `cupy_hccl.h`（纯头文件）**不会**触发 `nccl.cpp` 重编（distutils 只比较
`.cpp` 与 `.o` 的 mtime），必须删掉目标产物再编：

```bash
rm -f build/temp.linux-x86_64-cpython-311/cupy/backends/cuda/libs/nccl.o \
      build/lib.linux-x86_64-cpython-311/cupy/backends/cuda/libs/nccl*.so \
      cupy/backends/cuda/libs/nccl*.so cupy/backends/cuda/libs/nccl.cpp
```

### T0.5 shim 逻辑（无卡可跑，已自动化）

`tests/ascend/test_hccl_binding.py`：

```bash
python -m pytest tests/ascend/test_hccl_binding.py -q
```

两部分：

1. **构建特性**（7 个用例，纯 Python）：`has_hccl_header()` 与
   `select_hccl_libraries()` 在伪造的 SDK 树上——9.x（`libhcomm`）/ 8.5（`libhccl_fwk`）/
   两者共存时优先 `hcomm` / 缺 `hccl_alg|hccl_plf` 时自动剔除 / 带版本号 `libfoo.so.1` /
   SDK 不可探测时保留全集。
2. **shim 翻译逻辑**（`tests/ascend/hccl_shim_test.cpp`，用**记录型假 HCCL** 替换实现，
   不链接任何 CANN 库、不需要 NPU）：dtype/op 映射、`NCCL_AVG` 拒绝、
   `HcclResult`→`ncclResult_t` 折叠、未开组 P2P 直通、**组内 4 条 P2P → 恰好 1 次
   `HcclBatchSendRecv`（顺序/rank/buf/count/dtype 全对）**、comm 变化触发 flush、
   空组零调用、非原地 broadcast 只在 root 拷贝、弱符号缺失降级、abort=suspend+destroy、
   参数校验。全部 11 组断言通过即打印 `ALL PASSED`。

### T1 导入与常量（无卡可跑）

```python
from cupy.cuda import nccl
assert nccl.available
assert nccl.get_build_version() == 21700
assert (nccl.NCCL_INT8, nccl.NCCL_FLOAT64, nccl.NCCL_BFLOAT16) == (0, 8, 9)
assert (nccl.NCCL_SUM, nccl.NCCL_MIN) == (0, 3)
from cupyx.distributed import init_process_group, NCCLBackend  # 仅 import
```

KLV 回归（无需 NPU）：`_klv_utils` 对 4108 B 值做 `Set.klv() → Set.from_klv()` 往返一致。

### T2 单进程单卡（1 卡即可）

* `nccl.get_unique_id()` 返回 4108 B；`NcclCommunicator(1, uid, 0)` 成功；
* `size() == 1`、`rank_id() == 0`、`device_id() == 当前卡`；
* `allReduce/reduceScatter/allGather/broadcast` 在 n=1 上等于恒等（数值对比 numpy）；
* `destroy()` 后进程正常退出（无 HCCL 报错、无 hang）。

### T3 多进程集合通信（≥2 卡）

用 `torch.distributed` 式的「每进程一卡 + TCP store」跑 `NCCLBackend`：

| 用例 | 检查点 |
|---|---|
| `all_reduce`（sum/prod/min/max） | 与 numpy 结果逐位一致；dtype 覆盖 f16/f32/bf16/i8/i32/i64/u8/u32/u64 |
| `reduce`（root=0/root=1） | 非 root 输出不被写（`root != rank` 时 NCCL 语义） |
| `broadcast`（root=0/1，send==recv） | 全 rank 一致 |
| `reduce_scatter`（count=n/ranks） | 与「先 all_reduce 再切片」一致 |
| `all_gather` | 与拼接一致 |
| `barrier` | 不死锁（走 store，不走 HCCL） |
| 交错调用 | 连续 100 次 all_reduce + 5 次 reduce_scatter，无性能退化/超时 |

环境：`HCCL_EXEC_TIMEOUT=600`（首次建链慢）、固定 `HCCL_IF_IP`（多网卡机器必须）。

### T4 P2P / 组语义（≥2 卡，最容易暴露问题）

* `send/recv` 点对点（开组与不开组各一遍）；
* `send_recv`（同组一 send 一 recv，双向互换数据）；
* `scatter` / `gather` / `all_to_all`：**这三条会自环 P2P（R4）**，是首要验证项；
  数值和 `_nccl_comm.py` 期望一致，超时即视为 HCCL 不支持自环 → 走上表 R4 的替代实现；
* group 内混入 `broadcast`（稀疏路径形态）不改变顺序语义。

### T5 `cupyx.distributed` 端到端（≥2 卡）

* `init_process_group(n, rank)` + `DistributedArray` 基本读写；
* 参考上游：`tests/cupyx_tests/distributed_tests/test_comm.py`、
  `test_array_nccl.py`、`comm_runner.py`（`-m` 多进程 runner）；
* 日志写 `tests/logs/<date>-logs/cupyx_distributed_*.log`。

### T6 稀疏通信（可选，需 scipy + sparse）

`_SparseNCCLCommunicator` 的 `send/recv/bcast/gather/alltoall` 元数据交换在 group 内混用
集合与 P2P，验证 §4.4 的「集合不缓冲」假设。

### T7 错误路径

| 场景 | 期望 |
|---|---|
| `NcclConfig(split_share)` | 可构造；传给 `NcclCommunicator(config=…)` 时 `RuntimeError: NcclConfig is not supported in this version`（< 2.18 分支，已实测） |
| `commSplit(...)` | `RuntimeError: ncclCommSplit is not available in this version` |
| `all_reduce` 传非连续/不支持 dtype | 明确 `TypeError`/`RuntimeError`，不静默算错 |
| 非法 `op`（如 f32 prod，HCCL 不支持时） | `NcclError`（`HCCL_E_PARA→ncclInvalidArgument`），不是 `KeyError` |
| 未 `aclrtSetDevice` 就建域 | 可读错误 + 不泄漏 |

### T8 版本矩阵回归

CANN 8.5.1 与 9.0.1 各跑 T0–T4（链接行不同：`hccl_fwk` vs `hcomm`），
确认 `select_hccl_libraries()` 的解析结果和 summary 输出。

---

## 10. 变更清单（供 review）

| 文件 | 变更 |
|---|---|
| `cupy/backends/ascend/cupy_hccl.h` | **新增/补全**：NCCL→HCCL 完整 shim（类型、错误码、dtype、组、P2P、集合、弱符号保护） |
| `cupy/backends/cupy_xccl.h` | `CUPY_USE_ASCEND` 分支提前（否则误走 CUDA `<nccl.h>`） |
| `install/cupy_builder/features/ascend_hccl.py` | **新增**：可选 feature，编 `cupy.backends.cuda.libs.nccl` |
| `install/cupy_builder/_features.py` | 注册 `CUPY_ascend_hccl` |
| `cupyx/distributed/_klv_utils.py` | `_VALUE_BUFFER_SIZE` 256 → 8192（容纳 4108 B root info） |
| `cupyx/distributed/_store.py` | 长度校验改用上述常量 |
| `setup.py` | wheel 不再排除 `cupyx.distributed` |
| `tests/ascend/test_hccl_binding.py` | **新增**：构建特性库解析 + shim 逻辑的 pytest 入口 |
| `tests/ascend/hccl_shim_test.cpp` | **新增**：宿主端 shim 翻译逻辑测试（假 HCCL，无需 NPU） |
| `docs/ascend/HcclBinding.md` | 本文档 |

未改动（有意）：`cupy/backends/cuda/libs/nccl.pyx|.pxd`、`cupyx/distributed/_nccl_comm.py`、
`cupyx/distributed/_init.py`、`cupy/cuda/nccl.py` —— 见 §6。
