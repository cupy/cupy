CuPy - an array library on CUDA with NumPy-subset interface
===========================================================

.. module:: cupy

CuPy is an implementation of NumPy-compatible multi-dimensional array on CUDA.
CuPy consists of the core multi-dimensional array class, :class:`cupy.ndarray`,
and many functions on it. It supports a subset of :class:`numpy.ndarray`
interface that is enough for Chainer.

The following is a brief overview of supported subset of NumPy interface:

- `Basic indexing <http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_
  (indexing by ints, slices, newaxes, and Ellipsis)
- Element types (dtypes): bool_, (u)int{8, 16, 32, 64}, float{16, 32, 64}
- Most of the array creation routines
- Reshaping and transposition
- All operators with broadcasting
- All `Universal functions <http://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_ (a.k.a. ufuncs)
  except those for complex numbers
- Dot product functions (except einsum) using cuBLAS
- Reduction along axes (sum, max, argmax, etc.)

CuPy also includes following features for performance:

- Customizable memory allocator, and a simple memory pool as an example
- User-defined elementwise kernels
- User-defined full reduction kernels
- cuDNN utilities

CuPy uses on-the-fly kernel synthesis: when a kernel call is required, it
compiles a kernel code optimized for the shapes and dtypes of given arguments,
sends it to the GPU device, and executes the kernel. The compiled code is
cached to ``$(HOME)/.cupy/kernel_cache`` directory (this cache path can be
overwritten by setting the ``CUPY_CACHE_DIR`` environment variable). It may
make things slower at the first kernel call, though this slow down will be
resolved at the second execution. CuPy also caches the kernel code sent to GPU
device within the process, which reduces the kernel transfer time on further
calls.


A list of supported attributes, properties, and methods of ndarray
------------------------------------------------------------------

Memory layout
~~~~~~~~~~~~~

===================================== ===================================================
 Attributes of cupy.ndarray            Correspondig attributes of numpy.ndarray
===================================== ===================================================
 :attr:`cupy.ndarray.allocator`
 :attr:`cupy.ndarray.base`             :attr:`numpy.ndarray.base`
 :attr:`cupy.ndarray.ctypes`           :attr:`numpy.ndarray.ctypes`
 :attr:`cupy.ndarray.itemsize`         :attr:`numpy.ndarray.itemsize`
 :attr:`cupy.ndarray.flags`            :attr:`numpy.ndarray.flags`
 :attr:`cupy.ndarray.nbytes`           :attr:`numpy.ndarray.nbytes`
 :attr:`cupy.ndarray.shape`            :attr:`numpy.ndarray.shape`
 :attr:`cupy.ndarray.size`             :attr:`numpy.ndarray.size`
 :attr:`cupy.ndarray.strides`          :attr:`numpy.ndarray.strides`
===================================== ===================================================

Data type
~~~~~~~~~

===================================== ===================================================
 Attributes of cupy.ndarray            Correspondig attributes of numpy.ndarray
===================================== ===================================================
 :attr:`cupy.ndarray.dtype`            :attr:`numpy.ndarray.dtype`
===================================== ===================================================

Other attributes
~~~~~~~~~~~~~~~~

===================================== ===================================================
 Attributes of cupy.ndarray            Correspondig attributes of numpy.ndarray
===================================== ===================================================
 :attr:`cupy.ndarray.T`                :attr:`numpy.ndarray.T`
===================================== ===================================================

Array conversion
~~~~~~~~~~~~~~~~

================================== ================================================
 Methods of cupy.ndarray            Correspondig methods of numpy.ndarray
================================== ================================================
 :meth:`cupy.ndarray.tolist`        :meth:`numpy.ndarray.tolist`
 :meth:`cupy.ndarray.tofile`        :meth:`numpy.ndarray.tofile`
 :meth:`cupy.ndarray.astype`        :meth:`numpy.ndarray.astype`
 :meth:`cupy.ndarray.copy`          :meth:`numpy.ndarray.copy`
 :meth:`cupy.ndarray.view`          :meth:`numpy.ndarray.view`
 :meth:`cupy.ndarray.fill`          :meth:`numpy.ndarray.fill`
================================== ================================================

Shape manipulation
~~~~~~~~~~~~~~~~~~~

================================== ================================================
 Methods of cupy.ndarray            Correspondig methods of numpy.ndarray
================================== ================================================
 :meth:`cupy.ndarray.reshape`       :meth:`numpy.ndarray.reshape`
 :meth:`cupy.ndarray.transpose`     :meth:`numpy.ndarray.transpose`
 :meth:`cupy.ndarray.swapaxes`      :meth:`numpy.ndarray.swapaxes`
 :meth:`cupy.ndarray.ravel`         :meth:`numpy.ndarray.ravel`
 :meth:`cupy.ndarray.squeeze`       :meth:`numpy.ndarray.squeeze`
================================== ================================================

Item selection and manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

================================== ================================================
 Methods of cupy.ndarray            Correspondig methods of numpy.ndarray
================================== ================================================
 :meth:`cupy.ndarray.take`          :meth:`numpy.ndarray.take`
 :meth:`cupy.ndarray.diagonal`      :meth:`numpy.ndarray.diagonal`
================================== ================================================

Calculation
~~~~~~~~~~~

================================== ================================================
 Methods of cupy.ndarray            Correspondig methods of numpy.ndarray
================================== ================================================
 :meth:`cupy.ndarray.max`           :meth:`numpy.ndarray.max`
 :meth:`cupy.ndarray.argmax`        :meth:`numpy.ndarray.argmax`
 :meth:`cupy.ndarray.min`           :meth:`numpy.ndarray.min`
 :meth:`cupy.ndarray.argmin`        :meth:`numpy.ndarray.argmin`
 :meth:`cupy.ndarray.clip`          :meth:`numpy.ndarray.clip`
 :meth:`cupy.ndarray.trace`         :meth:`numpy.ndarray.trace`
 :meth:`cupy.ndarray.sum`           :meth:`numpy.ndarray.sum`
 :meth:`cupy.ndarray.mean`          :meth:`numpy.ndarray.mean`
 :meth:`cupy.ndarray.var`           :meth:`numpy.ndarray.var`
 :meth:`cupy.ndarray.std`           :meth:`numpy.ndarray.std`
 :meth:`cupy.ndarray.prod`          :meth:`numpy.ndarray.prod`
 :meth:`cupy.ndarray.dot`           :meth:`numpy.ndarray.dot`
================================== ================================================

Arithmetic and comparison operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

=================================== =================================================
 Methods of cupy.ndarray            Correspondig methods of numpy.ndarray
=================================== =================================================
 :meth:`cupy.ndarray.__lt__`         :meth:`numpy.ndarray.__lt__`
 :meth:`cupy.ndarray.__le__`         :meth:`numpy.ndarray.__le__`
 :meth:`cupy.ndarray.__gt__`         :meth:`numpy.ndarray.__gt__`
 :meth:`cupy.ndarray.__ge__`         :meth:`numpy.ndarray.__ge__`
 :meth:`cupy.ndarray.__eq__`         :meth:`numpy.ndarray.__eq__`
 :meth:`cupy.ndarray.__ne__`         :meth:`numpy.ndarray.__ne__`
 :meth:`cupy.ndarray.__nonzero__`    :meth:`numpy.ndarray.__nonzero__`
 :meth:`cupy.ndarray.__neg__`        :meth:`numpy.ndarray.__neg__`
 :meth:`cupy.ndarray.__pos__`        :meth:`numpy.ndarray.__pos__`
 :meth:`cupy.ndarray.__abs__`        :meth:`numpy.ndarray.__abs__`
 :meth:`cupy.ndarray.__invert__`     :meth:`numpy.ndarray.__invert__`
 :meth:`cupy.ndarray.__add__`        :meth:`numpy.ndarray.__add__`
 :meth:`cupy.ndarray.__sub__`        :meth:`numpy.ndarray.__sub__`
 :meth:`cupy.ndarray.__mul__`        :meth:`numpy.ndarray.__mul__`
 :meth:`cupy.ndarray.__div__`        :meth:`numpy.ndarray.__div__`
 :meth:`cupy.ndarray.__truediv__`    :meth:`numpy.ndarray.__truediv__`
 :meth:`cupy.ndarray.__floordiv__`   :meth:`numpy.ndarray.__floordiv__`
 :meth:`cupy.ndarray.__mod__`        :meth:`numpy.ndarray.__mod__`
 :meth:`cupy.ndarray.__divmod__`     :meth:`numpy.ndarray.__divmod__`
 :meth:`cupy.ndarray.__pow__`        :meth:`numpy.ndarray.__pow__`
 :meth:`cupy.ndarray.__lshift__`     :meth:`numpy.ndarray.__lshift__`
 :meth:`cupy.ndarray.__rshift__`     :meth:`numpy.ndarray.__rshift__`
 :meth:`cupy.ndarray.__and__`        :meth:`numpy.ndarray.__and__`
 :meth:`cupy.ndarray.__or__`         :meth:`numpy.ndarray.__or__`
 :meth:`cupy.ndarray.__xor__`        :meth:`numpy.ndarray.__xor__`
 :meth:`cupy.ndarray.__iadd__`       :meth:`numpy.ndarray.__iadd__`
 :meth:`cupy.ndarray.__isub__`       :meth:`numpy.ndarray.__isub__`
 :meth:`cupy.ndarray.__imul__`       :meth:`numpy.ndarray.__imul__`
 :meth:`cupy.ndarray.__idiv__`       :meth:`numpy.ndarray.__idiv__`
 :meth:`cupy.ndarray.__itruediv__`   :meth:`numpy.ndarray.__itruediv__`
 :meth:`cupy.ndarray.__ifloordiv__`  :meth:`numpy.ndarray.__ifloordiv__`
 :meth:`cupy.ndarray.__imod__`       :meth:`numpy.ndarray.__imod__`
 :meth:`cupy.ndarray.__ipow__`       :meth:`numpy.ndarray.__ipow__`
 :meth:`cupy.ndarray.__ilshift__`    :meth:`numpy.ndarray.__ilshift__`
 :meth:`cupy.ndarray.__irshift__`    :meth:`numpy.ndarray.__irshift__`
 :meth:`cupy.ndarray.__iand__`       :meth:`numpy.ndarray.__iand__`
 :meth:`cupy.ndarray.__ior__`        :meth:`numpy.ndarray.__ior__`
 :meth:`cupy.ndarray.__ixor__`       :meth:`numpy.ndarray.__ixor__`
=================================== =================================================

Special methods
~~~~~~~~~~~~~~~

=================================== =================================================
 Methods of cupy.ndarray            Correspondig methods of numpy.ndarray
=================================== =================================================
 :meth:`cupy.ndarray.__copy__`       :meth:`numpy.ndarray.__copy__`
 :meth:`cupy.ndarray.__deepcopy__`   :meth:`numpy.ndarray.__deepcopy__`
 :meth:`cupy.ndarray.__getstate__`
 :meth:`cupy.ndarray.__setstate__`   :meth:`numpy.ndarray.__setstate__`
 :meth:`cupy.ndarray.__array__`      :meth:`numpy.ndarray.__array__`
 :meth:`cupy.ndarray.__len__`        :meth:`numpy.ndarray.__len__`
 :meth:`cupy.ndarray.__getitem__`    :meth:`numpy.ndarray.__getitem__`
 :meth:`cupy.ndarray.__setitem__`    :meth:`numpy.ndarray.__setitem__`
 :meth:`cupy.ndarray.__int__`        :meth:`numpy.ndarray.__int__`
 :meth:`cupy.ndarray.__long__`       :meth:`numpy.ndarray.__long__`
 :meth:`cupy.ndarray.__float__`      :meth:`numpy.ndarray.__float__`
 :meth:`cupy.ndarray.__oct__`        :meth:`numpy.ndarray.__oct__`
 :meth:`cupy.ndarray.__hex__`        :meth:`numpy.ndarray.__hex__`
 :meth:`cupy.ndarray.__repr__`       :meth:`numpy.ndarray.__repr__`
 :meth:`cupy.ndarray.__str__`        :meth:`numpy.ndarray.__str__`
=================================== =================================================

Memory transfer
~~~~~~~~~~~~~~~

=================================== =================================================
 Methods of cupy.ndarray            Correspondig methods of numpy.ndarray
=================================== =================================================
 :meth:`cupy.ndarray.get`
 :meth:`cupy.ndarray.set`
=================================== =================================================


A list of supported routines of :mod:`cupy` module
--------------------------------------------------

Array creation routines
~~~~~~~~~~~~~~~~~~~~~~~

================================ ========================================
 Functions of cupy                Correspondig functions of numpy
================================ ========================================
 :func:`cupy.empty`               :func:`numpy.empty`
 :func:`cupy.empty_like`          :func:`numpy.empty_like`
 :func:`cupy.eye`                 :func:`numpy.eye`
 :func:`cupy.identity`            :func:`numpy.identity`
 :func:`cupy.ones`                :func:`numpy.ones`
 :func:`cupy.ones_like`           :func:`numpy.ones_like`
 :func:`cupy.zeros`               :func:`numpy.zeros`
 :func:`cupy.zeros_like`          :func:`numpy.zeros_like`
 :func:`cupy.full`                :func:`numpy.full`
 :func:`cupy.full_like`           :func:`numpy.full_like`

 :func:`cupy.array`               :func:`numpy.array`
 :func:`cupy.asarray`             :func:`numpy.asarray`
 :func:`cupy.ascontiguousarray`   :func:`numpy.ascontiguousarray`
 :func:`cupy.copy`                :func:`numpy.copy`

 :func:`cupy.arange`              :func:`numpy.arange`
 :func:`cupy.linspace`            :func:`numpy.linspace`

 :func:`cupy.diag`                :func:`numpy.diag`
 :func:`cupy.diagflat`            :func:`numpy.diagflat`
================================ ========================================

Array manipulation routines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

================================ ========================================
 Functions of cupy                Correspondig functions of numpy
================================ ========================================
 :func:`cupy.copyto`              :func:`numpy.copyto`

 :func:`cupy.reshape`             :func:`numpy.reshape`
 :func:`cupy.ravel`               :func:`numpy.ravel`

 :func:`cupy.rollaxis`            :func:`numpy.rollaxis`
 :func:`cupy.swapaxes`            :func:`numpy.swapaxes`
 :func:`cupy.transpose`           :func:`numpy.transpose`

 :func:`cupy.atleast_1d`          :func:`numpy.atleast_1d`
 :func:`cupy.atleast_2d`          :func:`numpy.atleast_2d`
 :func:`cupy.atleast_3d`          :func:`numpy.atleast_3d`
 :func:`cupy.broadcast`           :func:`numpy.broadcast`
 :func:`cupy.broadcast_arrays`    :func:`numpy.broadcast_arrays`
 :func:`cupy.squeeze`             :func:`numpy.squeeze`

 :func:`cupy.column_stack`        :func:`numpy.column_stack`
 :func:`cupy.concatenate`         :func:`numpy.concatenate`
 :func:`cupy.dstack`              :func:`numpy.dstack`
 :func:`cupy.hstack`              :func:`numpy.hstack`
 :func:`cupy.vstack`              :func:`numpy.vstack`

 :func:`cupy.array_split`         :func:`numpy.array_split`
 :func:`cupy.dsplit`              :func:`numpy.dsplit`
 :func:`cupy.hsplit`              :func:`numpy.hsplit`
 :func:`cupy.split`               :func:`numpy.split`
 :func:`cupy.vsplit`              :func:`numpy.vsplit`
================================ ========================================

Binary operations
~~~~~~~~~~~~~~~~~

================================ ========================================
 Functions of cupy                Correspondig functions of numpy
================================ ========================================
 :data:`cupy.bitwise_and`         :data:`numpy.bitwise_and`
 :data:`cupy.bitwise_or`          :data:`numpy.bitwise_or`
 :data:`cupy.bitwise_xor`         :data:`numpy.bitwise_xor`
 :data:`cupy.invert`              :data:`numpy.invert`
 :data:`cupy.left_shift`          :data:`numpy.left_shift`
 :data:`cupy.right_shift`         :data:`numpy.right_shift`

 :func:`cupy.binary_repr`         :func:`numpy.binary_repr`
================================ ========================================

Indexing routines
~~~~~~~~~~~~~~~~~

================================ ========================================
 Functions of cupy                Correspondig functions of numpy
================================ ========================================
 :func:`cupy.take`                :func:`numpy.take`
 :func:`cupy.diagonal`            :func:`numpy.diagonal`
================================ ========================================

Input and output
~~~~~~~~~~~~~~~~

================================ ========================================
 Functions of cupy                Correspondig functions of numpy
================================ ========================================
 :func:`cupy.load`                :func:`numpy.load`
 :func:`cupy.save`                :func:`numpy.save`
 :func:`cupy.savez`               :func:`numpy.savez`
 :func:`cupy.savez_compressed`    :func:`numpy.savez_compressed`

 :func:`cupy.array_repr`          :func:`numpy.array_repr`
 :func:`cupy.array_str`           :func:`numpy.array_str`

 :func:`cupy.base_repr`           :func:`numpy.base_repr`
================================ ========================================

Linear algebra
~~~~~~~~~~~~~~

================================ ========================================
 Functions of cupy                Correspondig functions of numpy
================================ ========================================
 :func:`cupy.dot`                 :func:`numpy.dot`
 :func:`cupy.vdot`                :func:`numpy.vdot`
 :func:`cupy.inner`               :func:`numpy.inner`
 :func:`cupy.outer`               :func:`numpy.outer`
 :func:`cupy.tensordot`           :func:`numpy.tensordot`

 :func:`cupy.trace`               :func:`numpy.trace`
================================ ========================================

Logic functions
~~~~~~~~~~~~~~~

================================ ========================================
 Functions of cupy                Correspondig functions of numpy
================================ ========================================
 :data:`cupy.isfinite`            :data:`numpy.isfinite`
 :data:`cupy.isinf`               :data:`numpy.isinf`
 :data:`cupy.isnan`               :data:`numpy.isnan`

 :func:`cupy.isscalar`            :func:`numpy.isscalar`

 :data:`cupy.logical_and`         :data:`numpy.logical_and`
 :data:`cupy.logical_or`          :data:`numpy.logical_or`
 :data:`cupy.logical_not`         :data:`numpy.logical_not`
 :data:`cupy.logical_xor`         :data:`numpy.logical_xor`

 :data:`cupy.greater`             :data:`numpy.greater`
 :data:`cupy.greater_equal`       :data:`numpy.greater_equal`
 :data:`cupy.less`                :data:`numpy.less`
 :data:`cupy.less_equal`          :data:`numpy.less_equal`
 :data:`cupy.equal`               :data:`numpy.equal`
 :data:`cupy.not_equal`           :data:`numpy.not_equal`
================================ ========================================

Mathematical functions
~~~~~~~~~~~~~~~~~~~~~~

================================ ========================================
 Functions of cupy                Correspondig functions of numpy
================================ ========================================
 :data:`cupy.sin`                 :data:`numpy.sin`
 :data:`cupy.cos`                 :data:`numpy.cos`
 :data:`cupy.tan`                 :data:`numpy.tan`
 :data:`cupy.arcsin`              :data:`numpy.arcsin`
 :data:`cupy.arccos`              :data:`numpy.arccos`
 :data:`cupy.arctan`              :data:`numpy.arctan`
 :data:`cupy.hypot`               :data:`numpy.hypot`
 :data:`cupy.arctan2`             :data:`numpy.arctan2`
 :data:`cupy.deg2rad`             :data:`numpy.deg2rad`
 :data:`cupy.rad2deg`             :data:`numpy.rad2deg`
 :data:`cupy.degrees`             :data:`numpy.degrees`
 :data:`cupy.radians`             :data:`numpy.radians`

 :data:`cupy.sinh`                :data:`numpy.sinh`
 :data:`cupy.cosh`                :data:`numpy.cosh`
 :data:`cupy.tanh`                :data:`numpy.tanh`
 :data:`cupy.arcsinh`             :data:`numpy.arcsinh`
 :data:`cupy.arccosh`             :data:`numpy.arccosh`
 :data:`cupy.arctanh`             :data:`numpy.arctanh`

 :data:`cupy.rint`                :data:`numpy.rint`
 :data:`cupy.floor`               :data:`numpy.floor`
 :data:`cupy.ceil`                :data:`numpy.ceil`
 :data:`cupy.trunc`               :data:`numpy.trunc`

 :func:`cupy.sum`                 :func:`numpy.sum`
 :func:`cupy.prod`                :func:`numpy.prod`

 :data:`cupy.exp`                 :data:`numpy.exp`
 :data:`cupy.expm1`               :data:`numpy.expm1`
 :data:`cupy.exp2`                :data:`numpy.exp2`
 :data:`cupy.log`                 :data:`numpy.log`
 :data:`cupy.log10`               :data:`numpy.log10`
 :data:`cupy.log2`                :data:`numpy.log2`
 :data:`cupy.log1p`               :data:`numpy.log1p`
 :data:`cupy.logaddexp`           :data:`numpy.logaddexp`
 :data:`cupy.logaddexp2`          :data:`numpy.logaddexp2`

 :data:`cupy.signbit`             :data:`numpy.signbit`
 :data:`cupy.copysign`            :data:`numpy.copysign`
 :data:`cupy.ldexp`               :data:`numpy.ldexp`
 :data:`cupy.frexp`               :data:`numpy.frexp`
 :data:`cupy.nextafter`           :data:`numpy.nextafter`

 :data:`cupy.add`                 :data:`numpy.add`
 :data:`cupy.reciprocal`          :data:`numpy.reciprocal`
 :data:`cupy.negative`            :data:`numpy.negative`
 :data:`cupy.multiply`            :data:`numpy.multiply`
 :data:`cupy.divide`              :data:`numpy.divide`
 :data:`cupy.power`               :data:`numpy.power`
 :data:`cupy.subtract`            :data:`numpy.subtract`
 :data:`cupy.true_divide`         :data:`numpy.true_divide`
 :data:`cupy.floor_divide`        :data:`numpy.floor_divide`
 :data:`cupy.fmod`                :data:`numpy.fmod`
 :data:`cupy.mod`                 :data:`numpy.mod`
 :data:`cupy.modf`                :data:`numpy.modf`
 :data:`cupy.remainder`           :data:`numpy.remainder`

 :data:`cupy.clip`                :func:`numpy.clip`
 :data:`cupy.sqrt`                :data:`numpy.sqrt`
 :data:`cupy.square`              :data:`numpy.square`
 :data:`cupy.absolute`            :data:`numpy.absolute`
 :data:`cupy.sign`                :data:`numpy.sign`
 :data:`cupy.maximum`             :data:`numpy.maximum`
 :data:`cupy.minimum`             :data:`numpy.minimum`
 :data:`cupy.fmax`                :data:`numpy.fmax`
 :data:`cupy.fmin`                :data:`numpy.fmin`
================================ ========================================

Sorting, searching, and counting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

================================ ========================================
 Functions of cupy                Correspondig functions of numpy
================================ ========================================
 :func:`cupy.argmax`              :func:`numpy.argmax`
 :func:`cupy.argmin`              :func:`numpy.argmin`
================================ ========================================

Statistics
~~~~~~~~~~

================================ ========================================
 Functions of cupy                Correspondig functions of numpy
================================ ========================================
 :func:`cupy.amin`                :func:`numpy.amin`
 :func:`cupy.amax`                :func:`numpy.amax`

 :func:`cupy.mean`                :func:`numpy.mean`
 :func:`cupy.var`                 :func:`numpy.var`
 :func:`cupy.std`                 :func:`numpy.std`
================================ ========================================

Other
~~~~~

================================ ========================================
 Functions of cupy                Correspondig functions of numpy
================================ ========================================
 :func:`cupy.asnumpy`
================================ ========================================
