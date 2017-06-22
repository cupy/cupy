# SGEMM example

This example contains implementation of single-precision general matrix-multiplication  (SGEMM).
The implementation is based on the one in [MAGMA](http://icl.cs.utk.edu/magma/).


### How to demo
The demo contains a script that calculates matrix multiplication of A (m x k) and B (k x n).
The demo can be run by the following command.

```
python sgemm.py [--m m] [--n n] [--k k]
```


### What this demo contains

CuPy arrays can be used without abstractions such as `ElementwiseKernel` if convenient `ndarray` interface is not necessary.
In this example, we work at the level of linear arrays (`float*`) and `cuLaunchKernel` to call a SGEMM kernel that exploits the predefined memory layout.

For compilation, `load_kernel` is used to compile a CUDA code written in `sgemm.cu`.
This function takes a text of code and name of the kernel as input and returns `cupy.cuda.Function` object.
The compiled code is cached, and it avoids the compilation process after the first time.
Also, the CUDA code can be modified at Python level because it is simply a text.
In this example, C macros that determines distribution of data to threads are specified at runtime.

Back to `cupy.cuda.Function`, this object allows you to call the kernel with CUDA's `cuLaunchKernel` interface.
In other words, you have control over grid size, block size, shared memory size and stream id.
At this level of interface, it becomes straightforward to transfer `.cpp` code that calls CUDA kernels to CuPy code.

Some points to note.

1. When writing kernels by yourself, remember to put `extern "C"` on top of the kernel that you want to call from Python.
2. When `ndarray` abstraction is not used as is the case in this example, the code behavior can differ depending on memory layout of the input CuPy arrays.
In that case, it is important to enforce the expected memory layout using functions such as `cupy.ascontigousarray` and `cupy.asfortranarray`.
