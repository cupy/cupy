# Roadmap

Cupy v14 (2025-Sep) 的 NumPy/SciPy 兼容层“基于 NumPy 2.3、SciPy 1.16”实现，

## 1. 功能补全

### blas, sip

[sip:基于华为Ascend AI处理器的信号处理加速库项目 - AtomGit](https://gitcode.com/cann/sip)

complex, signal, fft, solver

### CPU 回退： CPU实现一些NPU没有算法

scipy的cupyx的支持

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

## 2. Debug和性能优化

### pytest
numpy/cupy的API细微差别, 兼容模式
math_tests 时间很长

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
