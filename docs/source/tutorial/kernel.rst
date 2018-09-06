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

   >>> sum_kernel = cp.RawKernel(r'''
   ... extern "C" __global__
   ... void my_add(const float* x1, const float* x2, float* y) {
   ...     int tid = blockDim.x * blockIdx.x + threadIdx.x;
   ...     y[tid] = x1[tid] + x2[tid];
   ... }
   ... ''', 'my_add')
   >>> x1 = cupy.arange(25, dtype=cupy.float32).reshape(5, 5)
   >>> x2 = cupy.arange(25, dtype=cupy.float32).reshape(5, 5)
   >>> y = cupy.zeros((5, 5), dtype=cupy.float32)
   >>> sum_kernel((5,), (5,), (x1, x2, y))  # grid, block and arguments
   >>> y
   array([[ 0.,  2.,  4.,  6.,  8.],
          [10., 12., 14., 16., 18.],
          [20., 22., 24., 26., 28.],
          [30., 32., 34., 36., 38.],
          [40., 42., 44., 46., 48.]], dtype=float32)

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
