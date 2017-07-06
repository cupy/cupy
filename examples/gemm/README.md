# SGEMM example

This example contains implementation of single-precision general matrix-multiplication  (SGEMM).
The implementation is based on the one in [MAGMA](http://icl.cs.utk.edu/magma/).


### How to demo
The demo contains a script that calculates matrix multiplication of A (m x k) and B (k x n).
The demo can be run by the following command.

```
python sgemm.py [--gpu GPU_ID] [--m m] [--n n] [--k k]
```


### What this demo contains

In this example, we work on a SGEMM kernel that requires a complete interface to `cuLaunchKernel` (e.g. grid size and size of shared memory), which is not provided by `cupy.ElementwiseKernel`.
CuPy arrays work regardless of the underlying memory layouts thanks to `ndarray` abstraction.
As is the case for this example, `ndarray` abstraction does not need to be used if the underlying memory layouts of arrays match the ones expected by a kernel.
The SGEMM kernel expects input and output arrays to be in Fortran contiguous memory layout, and this layout is enforced by `cupy.asfortranarray`.

#### How to dynamically compile and launch a kernel function written in CUDA C
For compilation, `load_kernel` is used to compile a CUDA code written in `sgemm.cu`.
This function takes a text of code and name of the kernel as input and returns `cupy.cuda.Function` object.
The compiled code is cached, and it avoids the compilation process after the first time.
Also, the CUDA code can be modified at Python level because it is simply a text.
In this example, C macros that determine a distribution of data to threads are specified at runtime.
Note that `"extern C"` needs to be put on top of the kernel that is called.

#### How to supply grid size, block size and shared memory size on launching a kernel function
Back to `cupy.cuda.Function`, this object allows you to call the kernel with CUDA's `cuLaunchKernel` interface.
In other words, you have control over grid size, block size, shared memory size and stream id.
At this level of interface, it becomes straightforward to replace host `.cu` that calls CUDA kernels with Python code.
