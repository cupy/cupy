.. _udkernel:

User-Defined Kernels
====================

CuPy provides easy ways to define three types of CUDA kernels: elementwise kernels, reduction kernels and raw kernels.
In this documentation, we describe how to define and call each kernels.


Basics of elementwise kernels
-----------------------------

An elementwise kernel can be defined by the :class:`~cupy.ElementwiseKernel` class.
The instance of this class defines a CUDA kernel which can be invoked by the ``__call__`` method of this instance.

A definition of an elementwise kernel consists of four parts: an input argument list, an output argument list, a loop body code, and the kernel name.
For example, a kernel that computes a squared difference :math:`f(x, y) = (x - y)^2` is defined as follows:

.. doctest::

   >>> squared_diff = cp.ElementwiseKernel(
   ...    'float32 x, float32 y',
   ...    'float32 z',
   ...    'z = (x - y) * (x - y)',
   ...    'squared_diff')

The argument lists consist of comma-separated argument definitions.
Each argument definition consists of a *type specifier* and an *argument name*.
Names of NumPy data types can be used as type specifiers.

.. note::
   ``n``, ``i``, and names starting with an underscore ``_`` are reserved for the internal use.

The above kernel can be called on either scalars or arrays with broadcasting:

.. doctest::

   >>> x = cp.arange(10, dtype=np.float32).reshape(2, 5)
   >>> y = cp.arange(5, dtype=np.float32)
   >>> squared_diff(x, y)
   array([[ 0.,  0.,  0.,  0.,  0.],
          [25., 25., 25., 25., 25.]], dtype=float32)
   >>> squared_diff(x, 5)
   array([[25., 16.,  9.,  4.,  1.],
          [ 0.,  1.,  4.,  9., 16.]], dtype=float32)

Output arguments can be explicitly specified (next to the input arguments):

.. doctest::

   >>> z = cp.empty((2, 5), dtype=np.float32)
   >>> squared_diff(x, y, z)
   array([[ 0.,  0.,  0.,  0.,  0.],
          [25., 25., 25., 25., 25.]], dtype=float32)


Type-generic kernels
--------------------

If a type specifier is one character, then it is treated as a **type placeholder**.
It can be used to define a type-generic kernels.
For example, the above ``squared_diff`` kernel can be made type-generic as follows:

.. doctest::

   >>> squared_diff_generic = cp.ElementwiseKernel(
   ...     'T x, T y',
   ...     'T z',
   ...     'z = (x - y) * (x - y)',
   ...     'squared_diff_generic')

Type placeholders of a same character in the kernel definition indicate the same type.
The actual type of these placeholders is determined by the actual argument type.
The ElementwiseKernel class first checks the output arguments and then the input arguments to determine the actual type.
If no output arguments are given on the kernel invocation, then only the input arguments are used to determine the type.

The type placeholder can be used in the loop body code:

.. doctest::

   >>> squared_diff_generic = cp.ElementwiseKernel(
   ...     'T x, T y',
   ...     'T z',
   ...     '''
   ...         T diff = x - y;
   ...         z = diff * diff;
   ...     ''',
   ...     'squared_diff_generic')

More than one type placeholder can be used in a kernel definition.
For example, the above kernel can be further made generic over multiple arguments:

.. doctest::

   >>> squared_diff_super_generic = cp.ElementwiseKernel(
   ...     'X x, Y y',
   ...     'Z z',
   ...     'z = (x - y) * (x - y)',
   ...     'squared_diff_super_generic')

Note that this kernel requires the output argument explicitly specified, because the type ``Z`` cannot be automatically determined from the input arguments.


Raw argument specifiers
-----------------------

The ElementwiseKernel class does the indexing with broadcasting automatically, which is useful to define most elementwise computations.
On the other hand, we sometimes want to write a kernel with manual indexing for some arguments.
We can tell the ElementwiseKernel class to use manual indexing by adding the ``raw`` keyword preceding the type specifier.

We can use the special variable ``i`` and method ``_ind.size()`` for the manual indexing.
``i`` indicates the index within the loop.
``_ind.size()`` indicates total number of elements to apply the elementwise operation.
Note that it represents the size **after** broadcast operation.

For example, a kernel that adds two vectors with reversing one of them can be written as follows:

.. doctest::

   >>> add_reverse = cp.ElementwiseKernel(
   ...     'T x, raw T y', 'T z',
   ...     'z = x + y[_ind.size() - i - 1]',
   ...     'add_reverse')

(Note that this is an artificial example and you can write such operation just by ``z = x + y[::-1]`` without defining a new kernel).
A raw argument can be used like an array.
The indexing operator ``y[_ind.size() - i - 1]`` involves an indexing computation on ``y``, so ``y`` can be arbitrarily shaped and strode.

Note that raw arguments are not involved in the broadcasting.
If you want to mark all arguments as ``raw``, you must specify the ``size`` argument on invocation, which defines the value of ``_ind.size()``.


Reduction kernels
-----------------

Reduction kernels can be defined by the :class:`~cupy.ReductionKernel` class.
We can use it by defining four parts of the kernel code:

1. Identity value: This value is used for the initial value of reduction.
2. Mapping expression: It is used for the pre-processing of each element to be reduced.
3. Reduction expression: It is an operator to reduce the multiple mapped values.
   The special variables ``a`` and ``b`` are used for its operands.
4. Post mapping expression: It is used to transform the resulting reduced values.
   The special variable ``a`` is used as its input.
   Output should be written to the output parameter.

ReductionKernel class automatically inserts other code fragments that are required for an efficient and flexible reduction implementation.

For example, L2 norm along specified axes can be written as follows:

.. doctest::

   >>> l2norm_kernel = cp.ReductionKernel(
   ...     'T x',  # input params
   ...     'T y',  # output params
   ...     'x * x',  # map
   ...     'a + b',  # reduce
   ...     'y = sqrt(a)',  # post-reduction map
   ...     '0',  # identity value
   ...     'l2norm'  # kernel name
   ... )
   >>> x = cp.arange(10, dtype=np.float32).reshape(2, 5)
   >>> l2norm_kernel(x, axis=1)
   array([ 5.477226 , 15.9687195], dtype=float32)

.. note::
   ``raw`` specifier is restricted for usages that the axes to be reduced are put at the head of the shape.
   It means, if you want to use ``raw`` specifier for at least one argument, the ``axis`` argument must be ``0`` or a contiguous increasing sequence of integers starting from ``0``, like ``(0, 1)``, ``(0, 1, 2)``, etc.


Raw kernels
-----------

Raw kernels can be defined by the :class:`~cupy.RawKernel` class.
By using raw kernels, you can define kernels from raw CUDA source.

:class:`~cupy.RawKernel` object allows you to call the kernel with CUDA's ``cuLaunchKernel`` interface.
In other words, you have control over grid size, block size, shared memory size and stream.

.. doctest::

   >>> add_kernel = cp.RawKernel(r'''
   ... extern "C" __global__
   ... void my_add(const float* x1, const float* x2, float* y) {
   ...     int tid = blockDim.x * blockIdx.x + threadIdx.x;
   ...     y[tid] = x1[tid] + x2[tid];
   ... }
   ... ''', 'my_add')
   >>> x1 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
   >>> x2 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
   >>> y = cp.zeros((5, 5), dtype=cp.float32)
   >>> add_kernel((5,), (5,), (x1, x2, y))  # grid, block and arguments
   >>> y
   array([[ 0.,  2.,  4.,  6.,  8.],
          [10., 12., 14., 16., 18.],
          [20., 22., 24., 26., 28.],
          [30., 32., 34., 36., 38.],
          [40., 42., 44., 46., 48.]], dtype=float32)

Raw kernels operating on complex-valued arrays can be created as well:

.. doctest::

   >>> complex_kernel = cp.RawKernel(r'''
   ... #include <cupy/complex.cuh>
   ... extern "C" __global__
   ... void my_func(const complex<float>* x1, const complex<float>* x2,
   ...              complex<float>* y, float a) {
   ...     int tid = blockDim.x * blockIdx.x + threadIdx.x;
   ...     y[tid] = x1[tid] + a * x2[tid];
   ... }
   ... ''', 'my_func')
   >>> x1 = cupy.arange(25, dtype=cupy.complex64).reshape(5, 5)
   >>> x2 = 1j*cupy.arange(25, dtype=cupy.complex64).reshape(5, 5)
   >>> y = cupy.zeros((5, 5), dtype=cupy.complex64)
   >>> complex_kernel((5,), (5,), (x1, x2, y, cupy.float32(2.0)))  # grid, block and arguments
   >>> y
   array([[ 0. +0.j,  1. +2.j,  2. +4.j,  3. +6.j,  4. +8.j],
          [ 5.+10.j,  6.+12.j,  7.+14.j,  8.+16.j,  9.+18.j],
          [10.+20.j, 11.+22.j, 12.+24.j, 13.+26.j, 14.+28.j],
          [15.+30.j, 16.+32.j, 17.+34.j, 18.+36.j, 19.+38.j],
          [20.+40.j, 21.+42.j, 22.+44.j, 23.+46.j, 24.+48.j]],
         dtype=complex64)

Note that while we encourage the usage of ``complex<T>`` types for complex numbers (available by including ``<cupy/complex.cuh>`` as shown above), for CUDA codes already written using functions from ``cuComplex.h`` there is no need to make the conversion yourself: just set the option ``translate_cucomplex=True`` when creating a :class:`~cupy.RawKernel` instance.

The CUDA kernel attributes can be retrieved by either accessing the :attr:`~cupy.RawKernel.attributes` dictionary,
or by accessing the :class:`~cupy.RawKernel` object's attributes directly; the latter can also be used to set certain
attributes:

.. doctest::

   >>> add_kernel = cp.RawKernel(r'''
   ... extern "C" __global__
   ... void my_add(const float* x1, const float* x2, float* y) {
   ...     int tid = blockDim.x * blockIdx.x + threadIdx.x;
   ...     y[tid] = x1[tid] + x2[tid];
   ... }
   ... ''', 'my_add')
   >>> add_kernel.attributes  # doctest: +SKIP
   {'max_threads_per_block': 1024, 'shared_size_bytes': 0, 'const_size_bytes': 0, 'local_size_bytes': 0, 'num_regs': 10, 'ptx_version': 70, 'binary_version': 70, 'cache_mode_ca': 0, 'max_dynamic_shared_size_bytes': 49152, 'preferred_shared_memory_carveout': -1}
   >>> add_kernel.max_dynamic_shared_size_bytes  # doctest: +SKIP
   49152
   >>> add_kernel.max_dynamic_shared_size_bytes = 50000  # set a new value for the attribute  # doctest: +SKIP
   >>> add_kernel.max_dynamic_shared_size_bytes  # doctest: +SKIP
   50000

Dynamical parallelism is supported by :class:`~cupy.RawKernel`. You just need to provide the linking flag (such as ``-dc``) to :class:`~cupy.RawKernel`'s ``options`` arugment. The static CUDA device runtime library (``cudadevrt``) is automatically discovered by CuPy. For further detail, see `CUDA Toolkit's documentation`_.

.. _CUDA Toolkit's documentation: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compiling-and-linking

Accessing texture memory in :class:`~cupy.RawKernel` is supported via CUDA Runtime's Texture Object API, see :class:`~cupy.cuda.texture.TextureObject`'s documentation as well as CUDA C Programming Guide. For using the Texture Reference API, which is marked as deprecated as of CUDA Toolkit 10.1, see the introduction to :class:`~cupy.RawModule` below.

.. note::
    The kernel does not have return values.
    You need to pass both input arrays and output arrays as arguments.

.. note::
    No validation will be performed by CuPy for arguments passed to the kernel, including types and number of arguments.
    Especially note that when passing :class:`~cupy.ndarray`, its ``dtype`` should match with the type of the argument declared in the method signature of the CUDA source code (unless you are casting arrays intentionally).
    For example, ``cupy.float32`` and ``cupy.uint64`` arrays must be passed to the argument typed as ``float*`` and ``unsigned long long*``.
    For Python primitive types, ``int``, ``float`` and ``bool`` map to ``long long``, ``double`` and ``bool``, respectively.

.. note::
    When using ``printf()`` in your CUDA kernel, you may need to synchronize the stream to see the output.
    You can use ``cupy.cuda.Stream.null.synchronize()`` if you are using the default stream.


Raw modules
-----------

For dealing a large raw CUDA source or loading an existing CUDA binary, the :class:`~cupy.RawModule` class can be more handy. It can be initialized either by a CUDA source code, or by a path to the CUDA binary. The needed kernels can then be retrieved by calling the :meth:`~cupy.RawModule.get_function` method, which returns a :class:`~cupy.RawKernel` instance that can be invoked as discussed above.

.. doctest::

    >>> loaded_from_source = r'''
    ... extern "C"{
    ...
    ... __global__ void test_sum(const float* x1, const float* x2, float* y, \
    ...                          unsigned int N)
    ... {
    ...     unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    ...     if (tid < N)
    ...     {
    ...         y[tid] = x1[tid] + x2[tid];
    ...     }
    ... }
    ...
    ... __global__ void test_multiply(const float* x1, const float* x2, float* y, \
    ...                               unsigned int N)
    ... {
    ...     unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    ...     if (tid < N)
    ...     {
    ...         y[tid] = x1[tid] * x2[tid];
    ...     }
    ... }
    ...
    ... }'''
    >>> module = cp.RawModule(code=loaded_from_source)
    >>> ker_sum = module.get_function('test_sum')
    >>> ker_times = module.get_function('test_multiply')
    >>> N = 10
    >>> x1 = cp.arange(N**2, dtype=cp.float32).reshape(N, N)
    >>> x2 = cp.ones((N, N), dtype=cp.float32)
    >>> y = cp.zeros((N, N), dtype=cp.float32)
    >>> ker_sum((N,), (N,), (x1, x2, y, N**2))   # y = x1 + x2
    >>> assert cp.allclose(y, x1 + x2)
    >>> ker_times((N,), (N,), (x1, x2, y, N**2)) # y = x1 * x2
    >>> assert cp.allclose(y, x1 * x2)

The instruction above for using complex numbers in :class:`~cupy.RawKernel` also applies to :class:`~cupy.RawModule`.

For CUDA kernels that need to access global symbols, such as constant memory, the :meth:`~cupy.RawModule.get_global` method can be used, see its documentation for further detail.

CuPy also supports the Texture Reference API. A handle to the texture reference in a module can be retrieved by name via :meth:`~cupy.RawModule.get_texref`. Then, you need to pass it to :class:`~cupy.cuda.texture.TextureReference`, along with a resource descriptor and texture descriptor, for binding the reference to the array. (The interface of :class:`~cupy.cuda.texture.TextureReference` is meant to mimic that of :class:`~cupy.cuda.texture.TextureObject` to help users make transition to the latter, since as of CUDA Toolkit 10.1 the former is marked as deprecated.)


Kernel fusion
--------------------

:func:`cupy.fuse` is a decorator that fuses functions.  This decorator can be used to define an elementwise or reduction kernel more easily than :class:`~cupy.ElementwiseKernel` or :class:`~cupy.ReductionKernel`.

By using this decorator, we can define the ``squared_diff`` kernel as follows:

.. doctest::

   >>> @cp.fuse()
   ... def squared_diff(x, y):
   ...     return (x - y) * (x - y)

The above kernel can be called on either scalars, NumPy arrays or CuPy arrays likes the original function.

.. doctest::

   >>> x_cp = cp.arange(10)
   >>> y_cp = cp.arange(10)[::-1]
   >>> squared_diff(x_cp, y_cp)
   array([81, 49, 25,  9,  1,  1,  9, 25, 49, 81])
   >>> x_np = np.arange(10)
   >>> y_np = np.arange(10)[::-1]
   >>> squared_diff(x_np, y_np)
   array([81, 49, 25,  9,  1,  1,  9, 25, 49, 81])

At the first function call, the fused function analyzes the original function based on the abstracted information of arguments (e.g. their dtypes and ndims) and creates and caches an actual CUDA kernel.  From the second function call with the same input types, the fused function calls the previously cached kernel, so it is highly recommended to reuse the same decorated functions instead of decorating local functions that are defined multiple times.

:func:`cupy.fuse` also supports simple reduction kernel.

.. doctest::

   >>> @cp.fuse()
   ... def sum_of_products(x, y):
   ...     return cp.sum(x * y, axis = -1)

You can specify the kernel name by using the ``kernel_name`` keyword argument as follows:

.. doctest::

   >>> @cp.fuse(kernel_name='squared_diff')
   ... def squared_diff(x, y):
   ...     return (x - y) * (x - y)

.. note::
   Currently, :func:`cupy.fuse` can fuse only simple elementwise and reduction operations.  Most other routines (e.g. :func:`cupy.matmul`, :func:`cupy.reshape`) are not supported.
