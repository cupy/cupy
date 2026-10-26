# Roadmap

Cupy v14 (2025-Sep) 的 NumPy/SciPy 兼容层“基于 NumPy 2.3、SciPy 1.16”实现，

## 1. 功能补全

### blas, sip

[sip:基于华为Ascend AI处理器的信号处理加速库项目 - AtomGit](https://gitcode.com/cann/sip)

complex, signal, fft, solver

### CPU 回退： CPU实现一些NPU没有算法

scipy的cupyx的支持 —— **具体缺口、可达上限与落地顺序见
[CupyxScipyMigration.md](CupyxScipyMigration.md)**（2026-10-18 评估：现状 ~31 %，
加逐函数 host 回退后 ~88-90 %；前置机制是把 `cupy/_logic/truth.py:14` 的逐调用点回退
提升为 `cpu_fallback.host_unless_kernels()`）。

64bit计算

### scalar 传递str arg
statistis: scalar 传递str arg, 是否cupy scalar这里就要patch?
pass_arg.md,  需要重构, 支持string,甚至list的传参

### polynomial (部分支持)
polynomial 是“纯组合层”，没有自己的 kernel
cupy.poly1d 相关代码全部位于 cupy/lib/，没有独立的 CUDA kernel，完全由已有 ufunc/linalg 组合而成：
核心障碍：eigvalsh 与 lstsq 没有原生算子

### 多卡并行

### profiler

`cudaProfilerStart/Stop` 已映射到 ACL profiler（`cupy/backends/ascend/cupy_ascend_profiler.h`），
`cupy.cuda.runtime.profilerStart()` / `cupyx.profiler.profile()` 可直接使用：

- 会话状态机：`aclprofInit(path)` → `aclprofCreateConfig(dev, metrics, dataType)` →
  `aclprofStart` … `aclprofStop` → `aclprofDestroyConfig` → `aclprofFinalize`（每次 Stop 都会 flush）
- 环境变量：`CUPY_ASCEND_PROFILER`(0/1)、`CUPY_ASCEND_PROFILER_PATH`(结果目录)、
  `CUPY_ASCEND_PROFILER_DATA_TYPE`、`CUPY_ASCEND_PROFILER_AICORE_METRICS`
- CANN 9.x 的 `aclprof*` 不在 `libascendcl.so` 中（只在 `libmsprofiler.so`），故用 dlopen/dlsym
  延迟解析，库缺失时退化为 no-op（等价 CUDA 未挂 profiler 的行为）
- TODO：`cupy.cuda.nvtx` 的 mark/range → `aclprofMarkEx/aclprofRangeStart/RangeStop`；
  与 msprof/mindstudio 结果目录联动

### cupyx / cupyx.scipy 移植（2026-10-18 评估，详见 [CupyxScipyMigration.md](CupyxScipyMigration.md)）

已打通：`cupyx.scipy.fft(+fftpack)`、`cupyx.scipy.sparse`（门面）、`cupy.random`、`cupyx.profiler`；
25 个子模块全部可 import。剩余阻塞分两类：

**A 类：子系统级（不可移植，需换实现路线）**

| 子系统 | 卡点 | 路线 |
|---|---|---|
`cupyx.jit` / `cupy.RawKernel` | NVRTC，无 CUDA C 代码生成 | triton-ascend 或 AscendC JIT（[CustomKernel.md](CustomKernel.md)） |
`cupyx.distributed` | NCCL（`_nccl_comm.py:7`） | HCCL 绑定 |
`cupyx.cusolver` / `cupyx.lapack` | 无 cuSOLVER | ops-solver / ops-blas LAPACK |
`cupyx.cutensor` | 无对应物 | 暂不做 |

**B 类：248 个内联 CUDA kernel**（`ElementwiseKernel`/`RawKernel`/`create_ufunc`）——
Ascend 上 kernel 体不编译，只按名字查注册表，未注册即 `NotImplementedError`
（`cupy/_core/_ascend/_kernel.pyx:1036`、`acl_utils.pyx:1822`）。
分布：special 101、signal 41、sparse 37、ndimage 27、interpolate 18、cupyx.signal 12、linalg 4、spatial 3、fft 2。

**覆盖率与优先级**（公共名约 470 个）：

| 档 | 覆盖率 | 动作 |
|---|---|---|
现状 | ~30 % | — |
+R1 | ~51 % | ① 派发器加 `cupyx_scipy_special_<x>` → `ascend_<x>` 归一化，白拿 6 个（erf/erfc/erfinv/cbrt/exp2/expm1）② 其余 93 个加 `cupyx_scipy_special_*` 前缀 host 回退（仿 `acl_utils.pyx:2109` 的 `_COMPLEX_NO_ACLNN_OPS`） |
+R3 | ~86 % | ndimage/signal/interpolate 逐**公共函数**回退（~50 条） |
+R2 | ~88 % | sparse 纯 cupy 回退：`tocsc/tocsr`（[TODO.md](TODO.md) S-1…S-5）+ `toarray` + 910B 缺失的 7 族 aclsparse 算子 |
上限 | ~88-90 % | 余下为 jit/distributed/cusolver/cutensor 与 f64/complex 精度类 |

前置：R0 把 `cupy/_logic/truth.py:14` `_ascend_set_host_fallback` 提升为
`cpu_fallback.host_unless_kernels()`（`kernels=None` = 无条件回退），R1-R3 全部复用它。

> scipy 侧注意：`SCIPY_ARRAY_API=1` 只对 `special` / `signal` / `ndimage` 三个模块把 cupy 数组
> 委托给 `cupyx.scipy.*`；`scipy.sparse` 无任何 array-api 分支（构造走 `np.asarray` → `TypeError`），
> **sparse 必须显式 `import cupyx.scipy.sparse as sparse`**。而被委托的三个模块正是 B 类重灾区。

## 2. Debug和性能优化

### pytest
numpy/cupy的API细微差别, 兼容模式
math_tests 时间很长

`tests/cupyx_tests` 侧（2940 个 test def）：`tests/conftest.py:112` 的 CUDA-only 跳过表目前只覆盖
`cupy_tests`，需补 `cupyx_tests/jit_tests`、`distributed_tests`、`test_cusolver.py`、
`test_cutensor.py`、`test_lapack.py`，以及条件跳过 `fft_tests/test_callbacks.py`，
否则 cupyx 基线会被 4 个未移植子系统掩盖（jit 目前还会在 `test_cooperative_groups.py` 处 collection error）。

### code review
[X] memory leak
[X] cupy/_core下面的 pyx 和 cupy/_core/_ascend下面的核心代码, matmul实测通过了, 重点是数学操作之外的, indexing, sort, pading, rotate等manipulate api without alcop/need special arg passing. ascend op should be more pytorch style, not numpy style, so if there is aclop but hehavior diff from numpy, or  numpy, cupy, pytorch api diff, please give table comparison.  output to code_review_ascend_core.md

## 3. Python的free-thread生态的演进

cupy 上游拉去cupy的更新, 需要skill去

numpy ABI解耦, 独立演进, scipy
生态：NumPy 2.1+ 对 free‑threaded 更友好

### cython no-GIL
cython 3.2 及之后（持续增强 no‑GIL）, python 3.14 no-GIL

默认即使你在 free‑threaded 解释器里 import 一个普通 Cython 模块，解释器也会因“未声明兼容”而重新启用 GIL——模块能跑，但失去 no‑GIL 收益。
想真正不恢复 GIL，模块加声明：
+ 文件头：# cython: freethreading_compatible=True
+ 命令行：cython -X freethreading_compatible=True ...
+ setuptools：cythonize(..., compiler_directives={"freethreading_compatible": True})（3.1+ 才生效）

在 free‑threaded CPython 里多线程调 pure_c_sum，只要 a 不被其它线程同时写、且不碰 Python 对象，就能真正并行；若循环里要创建 Python 对象/抛异常，得回到 with gil 块。

#### 线程安全工具（替代“靠 GIL 保护”）
free‑threaded 下不能再假设任意 Python 操作原子，Cython 给三档 ：
1. with cython.nogil：只放纯 C/已持锁逻辑；碰 Python 对象要 with gil。在 free‑threaded 里“持有 GIL”语义约等于“处于可安全碰 Python 的 thread state”，不是独占全局锁。

2. with cython.critical_section(obj)：基于 CPython critical section，对 obj（或两个 obj、pymutex）加局部锁；非 free‑threaded 构建下为空操作（因为本来有 GIL）。可作装饰器锁 self：
```python
@cython.cclass
class Counter:
    cdef int n
    @cython.critical_section
    def inc(self):
        self.n += 1
```

3. cython.pymutex：更硬的自管互斥，不会因调用别的 Python 对象被临时释放；适合精细临界区，但要自己防死锁。
```python
cdef cython.pymutex m
with cython.pymutex_lock(m):
```

（具体 wrapper 名随版本略有差异，3.1 起以 pymutex 文档为准。）
其它配合：cdef class 的共享可变属性、dict/list 缓存、懒初始化单例、引用计数缓存（freelist）都要按 free‑threaded 重审；CPython 自身对很多内置类型用了 per‑object 锁，但你的 Cython/C 状态不会自动加锁。

## 4. cupy特性跟踪


## 5. XPU抽象backend 架构调整说明 (backend): What has been done after fork

xpu C API;  
hip_rocm抽取

### 5.1 _core/core.pyx 拆分出3个文件, 方便porting

这个已经提交上游社区了 refactor MR, 但是我编译有点问题, 还没有接受

### 5.2 Neutral backend API

#### cuda法律上是禁止二进制的转译

**NVIDIA最终用户许可协议门户**：https://www.nvidia.com/en-us/about-nvidia/eula-agreement/

NVIDIA的EULA**并未禁止重新编译CUDA源代码**

- **合法途径**：像AMD的HIP（HIPIFY工具）和Intel的SYCL（SYCLomatic工具）这类技术，其工作方式是**将CUDA源代码转换为另一种兼容的编程模型代码**，然后使用目标平台自己的编译器和工具链进行编译。这个过程不涉及对CUDA SDK输出成果的逆向或反编译，因此是合规的。
- **核心区别**：关键在于“**转译（Translation）**”与“**移植（Porting）**”的区别。EULA禁止的是对已编译的二进制/PTX代码进行直接转译，但不禁止对源代码进行转换和重新编译。

CUDA的runtime API emulate 可能不违反EULA, 但是没有必要冒着未来的法律, 直接中性化.

#### XPU API 中性化重构
+ cupy.cuda -> cupy.xpu ,  XPU 泛指任何CPU之外的计算加速器

+ cupy_backends.cuda -> cupy.backends.backend , 为什么叫backends, 这是和torch保持移植. 

+ xpuXXX 作为runtime的抽象API

#### 保留Cupy的名称, 致敬Cupy的作者们

同时保持Cupy作者沟通, 确保Cupy是否有商标/著作权, 是否也已授权其他XPU使用. 

架构中性化, 也可以和cupy作者沟通, 看这种架构的重构上游是否可以接受. nvidia在致力做自己官方的pynumeric, 那么社区驱动cupy未来就有不确定性. 

### 5.3 (refactor underway): cuda backend need xpu -> cuda API mapping
有一个python的脚本(api_replace_tool.py)来负责处理. 
这样处理后, cuda backend有工作量, cuda api -> xpu api, 暂时不能编译. 所以我放在新的分支 xpu开发

cudaDataType  -> xpuDataType 这是一个typedef
cuDoubleComplex -> xpuComplex128,  numpy,torch use this style 
enum xpuFunction_attribute
cuGetErrorString -> 

```c++
// Context
xpuBlasStatus cublasCreate(...) {
    return CUBLAS_STATUS_SUCCESS;
}
```
