# distutils: language = c++

import functools
import os
import pickle
import re
import sys
import warnings

import numpy

import cupy
from cupy._core._kernel import create_ufunc
from cupy._core._kernel import ElementwiseKernel
from cupy._core._kernel import ufunc  # NOQA
from cupy._core._ufuncs import elementwise_copy
from cupy._core._ufuncs import elementwise_copy_where
from cupy._core import flags
from cupy._core import syncdetect
from cupy import cuda
from cupy.cuda import memory as memory_module


from cupy_backends.cuda.api.runtime import CUDARuntimeError
from cupy import _util

cimport cpython  # NOQA
cimport cython  # NOQA
from libcpp cimport vector
from libc.stdint cimport int64_t, intptr_t

from cupy._core cimport _carray
from cupy._core cimport _dtype
from cupy._core._dtype cimport get_dtype
from cupy._core._kernel cimport create_ufunc
from cupy._core cimport _routines_binary as _binary
from cupy._core cimport _routines_indexing as _indexing
from cupy._core cimport _routines_linalg as _linalg
from cupy._core cimport _routines_logic as _logic
from cupy._core cimport _routines_manipulation as _manipulation
from cupy._core cimport _routines_math as _math
from cupy._core cimport _routines_sorting as _sorting
from cupy._core cimport _routines_statistics as _statistics
from cupy._core cimport _scalar
from cupy._core cimport dlpack
from cupy._core cimport internal
from cupy.cuda cimport device
from cupy.cuda cimport function
from cupy.cuda cimport pinned_memory
from cupy.cuda cimport memory
from cupy.cuda cimport stream as stream_module
from cupy_backends.cuda cimport stream as _stream_module
from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda.libs cimport cublas


# If rop of cupy.ndarray is called, cupy's op is the last chance.
# If op of cupy.ndarray is called and the `other` is cupy.ndarray, too,
# it is safe to call cupy's op.
# Otherwise, use this function `_should_use_rop` to choose
# * [True] return NotImplemented to defer rhs, or
# * [False] call NumPy's ufunc to try all `__array_ufunc__`.
# Note that extension types (`cdef class`) in Cython 0.x shares
# implementations of op and rop. (i.e. `__radd__(self, other)` is
# `__add__(other, self)`.)
#
# It follows NEP 13 except that cupy also implements the fallback to
# `__array_priority__`, which seems fair and necessary because of the
# following facts:
# * `numpy` : `scipy.sparse` = `cupy` : `cupyx.scipy.sparse`;
# * NumPy ignores `__array_priority__` attributes of arguments if NumPy finds
#   `__array_function__` of `cupy.ndarray`;
# * SciPy sparse classes don't implement `__array_function__` and they even
#   don't set `__array_function__ = None` to opt-out the feature; and
# * `__array_priority__` of SciPy sparse classes is respected because
#   `numpy.ndarray.__array_function__` does not disable `__array_priority__`.
@cython.profile(False)
cdef inline _should_use_rop(x, y):
    try:
        y_ufunc = y.__array_ufunc__
    except AttributeError:
        # NEP 13's recommendation is `return False`.
        xp = getattr(x, '__array_priority__', 0)
        yp = getattr(y, '__array_priority__', 0)
        return xp < yp
    return y_ufunc is None


cdef tuple _HANDLED_TYPES


cdef class ndarray:

    """Multi-dimensional array on a CUDA device.

    This class implements a subset of methods of :class:`numpy.ndarray`.
    The difference is that this class allocates the array content on the
    current GPU device.

    Args:
        shape (tuple of ints): Length of axes.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        memptr (cupy.cuda.MemoryPointer): Pointer to the array content head.
        strides (tuple of ints or None): Strides of data in memory.
        order ({'C', 'F'}): Row-major (C-style) or column-major
            (Fortran-style) order.

    Attributes:
        base (None or cupy.ndarray): Base array from which this array is
            created as a view.
        data (cupy.cuda.MemoryPointer): Pointer to the array content head.
        ~ndarray.dtype(numpy.dtype): Dtype object of element type.

            .. seealso::
               `Data type objects (dtype) \
               <https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html>`_
        ~ndarray.size (int): Number of elements this array holds.

            This is equivalent to product over the shape tuple.

            .. seealso:: :attr:`numpy.ndarray.size`

    """

    def __init__(self, shape, dtype=float, memptr=None, strides=None,
                 order='C'):
        cdef Py_ssize_t x, itemsize
        cdef tuple s = internal.get_size(shape)
        del shape

        cdef int order_char = (
            b'C' if order is None else internal._normalize_order(order))

        # `strides` is prioritized over `order`, but invalid `order` should be
        # checked even if `strides` is given.
        if order_char != b'C' and order_char != b'F':
            raise ValueError('order not understood. order=%s' % order)

        # Check for erroneous shape
        self._shape.reserve(len(s))
        for x in s:
            if x < 0:
                raise ValueError('Negative dimensions are not allowed')
            self._shape.push_back(x)
        del s

        # dtype
        self.dtype, itemsize = _dtype.get_dtype_with_itemsize(dtype)

        # Store shape and strides
        if strides is not None:
            if memptr is None:
                raise ValueError('memptr is required if strides is given.')
            self._set_shape_and_strides(self._shape, strides, True, True)
        elif order_char == b'C':
            self._set_contiguous_strides(itemsize, True)
        elif order_char == b'F':
            self._set_contiguous_strides(itemsize, False)
        else:
            assert False

        # data
        if memptr is None:
            self.data = memory.alloc(self.size * itemsize)
            self._index_32_bits = (self.size * itemsize) <= (1 << 31)
        else:
            self.data = memptr
            bound = cupy._core._memory_range.get_bound(self)
            self._index_32_bits = bound[1] - bound[0] <= (1 << 31)

    cdef _init_fast(self, const shape_t& shape, dtype, bint c_order):
        """ For internal ndarray creation. """
        cdef Py_ssize_t itemsize
        self._shape = shape
        self.dtype, itemsize = _dtype.get_dtype_with_itemsize(dtype)
        self._set_contiguous_strides(itemsize, c_order)
        self.data = memory.alloc(self.size * itemsize)
        self._index_32_bits = (self.size * itemsize) <= (1 << 31)

    @property
    def __cuda_array_interface__(self):
        if runtime._is_hip_environment:
            raise AttributeError(
                'HIP/ROCm does not support cuda array interface')
        cdef dict desc = {
            'shape': self.shape,
            'typestr': self.dtype.str,
            'descr': self.dtype.descr,
        }
        cdef int ver = _util.CUDA_ARRAY_INTERFACE_EXPORT_VERSION
        cdef intptr_t stream_ptr

        if ver == 3:
            stream_ptr = stream_module.get_current_stream_ptr()
            # CAI v3 says setting the stream field to 0 is disallowed
            if stream_ptr == 0:
                stream_ptr = _stream_module.get_default_stream_ptr()
            desc['stream'] = stream_ptr
        elif ver == 2:
            # Old behavior (prior to CAI v3): stream sync is explicitly handled
            # by users. To restore this behavior, we do not export any stream
            # if CUPY_CUDA_ARRAY_INTERFACE_EXPORT_VERSION is set to 2 (so that
            # other participating libraries lacking a finer control over sync
            # behavior can avoid syncing).
            pass
        else:
            raise ValueError('CUPY_CUDA_ARRAY_INTERFACE_EXPORT_VERSION can '
                             'only be set to 3 (default) or 2')
        desc['version'] = ver
        if self._c_contiguous:
            desc['strides'] = None
        else:
            desc['strides'] = self.strides
        if self.size > 0:
            desc['data'] = (self.data.ptr, False)
        else:
            desc['data'] = (0, False)

        return desc

    # The definition order of attributes and methods are borrowed from the
    # order of documentation at the following NumPy document.
    # https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html

    # -------------------------------------------------------------------------
    # Memory layout
    # -------------------------------------------------------------------------
    @property
    def flags(self):
        """Object containing memory-layout information.

        It only contains ``c_contiguous``, ``f_contiguous``, and ``owndata``
        attributes. All of these are read-only. Accessing by indexes is also
        supported.

        .. seealso:: :attr:`numpy.ndarray.flags`

        """
        return flags.Flags(self._c_contiguous, self._f_contiguous,
                           self.base is None)

    property shape:
        """Lengths of axes.

        Setter of this property involves reshaping without copy. If the array
        cannot be reshaped without copy, it raises an exception.

        .. seealso: :attr:`numpy.ndarray.shape`

        """

        def __get__(self):
            return tuple(self._shape)

        def __set__(self, newshape):
            _manipulation._ndarray_shape_setter(self, newshape)

    @property
    def strides(self):
        """Strides of axes in bytes.

        .. seealso:: :attr:`numpy.ndarray.strides`

        """
        return tuple(self._strides)

    @property
    def ndim(self):
        """Number of dimensions.

        ``a.ndim`` is equivalent to ``len(a.shape)``.

        .. seealso:: :attr:`numpy.ndarray.ndim`

        """
        return self._shape.size()

    @property
    def itemsize(self):
        """Size of each element in bytes.

        .. seealso:: :attr:`numpy.ndarray.itemsize`

        """
        return self.dtype.itemsize

    @property
    def nbytes(self):
        """Total size of all elements in bytes.

        It does not count skips between elements.

        .. seealso:: :attr:`numpy.ndarray.nbytes`

        """
        return self.size * self.dtype.itemsize

    # -------------------------------------------------------------------------
    # Other attributes
    # -------------------------------------------------------------------------
    @property
    def T(self):
        """Shape-reversed view of the array.

        If ndim < 2, then this is just a reference to the array itself.

        """
        if self.ndim < 2:
            return self
        else:
            return _manipulation._T(self)

    @property
    def flat(self):
        return cupy.flatiter(self)

    __array_priority__ = 100

    # -------------------------------------------------------------------------
    # Array interface
    # -------------------------------------------------------------------------
    # TODO(beam2d): Implement __array_interface__

    # -------------------------------------------------------------------------
    # foreign function interface
    # -------------------------------------------------------------------------
    @property
    def cstruct(self):
        """C representation of the array.

        This property is used for sending an array to CUDA kernels. The type of
        returned C structure is different for different dtypes and ndims. The
        definition of C type is written in ``cupy/carray.cuh``.

        """
        return _CArray_from_ndarray(self)

    # -------------------------------------------------------------------------
    # Array conversion
    # -------------------------------------------------------------------------
    cpdef item(self):
        """Converts the array with one element to a Python scalar

        Returns:
            int or float or complex: The element of the array.

        .. seealso:: :meth:`numpy.ndarray.item`

        """
        if self.size != 1:
            raise ValueError(
                'can only convert an array of size 1 to a Python scalar')
        return self.get().item()

    cpdef tolist(self):
        """Converts the array to a (possibly nested) Python list.

        Returns:
            list: The possibly nested Python list of array elements.

        .. seealso:: :meth:`numpy.ndarray.tolist`

        """
        return self.get().tolist()

    # TODO(okuta): Implement itemset
    # TODO(okuta): Implement tostring

    cpdef bytes tobytes(self, order='C'):
        """Turns the array into a Python bytes object."""
        return self.get().tobytes(order)

    cpdef tofile(self, fid, sep='', format='%s'):
        """Writes the array to a file.

        .. seealso:: :meth:`numpy.ndarray.tofile`

        """
        self.get().tofile(fid, sep, format)

    cpdef dump(self, file):
        """Dumps a pickle of the array to a file.

        Dumped file can be read back to :class:`cupy.ndarray` by
        :func:`cupy.load`.

        """
        pickle.dump(self, file, -1)

    cpdef bytes dumps(self):
        """Dumps a pickle of the array to a string."""
        return pickle.dumps(self, -1)

    cpdef ndarray astype(
            self, dtype, order='K', casting=None, subok=None, copy=True):
        """Casts the array to given data type.

        Args:
            dtype: Type specifier.
            order ({'C', 'F', 'A', 'K'}): Row-major (C-style) or column-major
                (Fortran-style) order.
                When ``order`` is 'A', it uses 'F' if ``a`` is column-major and
                uses 'C' otherwise.
                And when ``order`` is 'K', it keeps strides as closely as
                possible.
            copy (bool): If it is False and no cast happens, then this method
                returns the array itself. Otherwise, a copy is returned.

        Returns:
            If ``copy`` is False and no cast is required, then the array itself
            is returned. Otherwise, it returns a (possibly casted) copy of the
            array.

        .. note::
           This method currently does not support ``casting``, and ``subok``
           arguments.

        .. seealso:: :meth:`numpy.ndarray.astype`

        """
        cdef strides_t strides
        cdef Py_ssize_t stride

        # TODO(beam2d): Support casting and subok option
        if casting is not None:
            raise TypeError('casting is not supported yet')
        if subok is not None:
            raise TypeError('subok is not supported yet')

        if order is None:
            order = 'K'
        cdef int order_char = internal._normalize_order(order)

        dtype = get_dtype(dtype)
        if dtype == self.dtype:
            if not copy and (
                    order_char == b'K' or
                    order_char == b'A' and (self._c_contiguous or
                                            self._f_contiguous) or
                    order_char == b'C' and self._c_contiguous or
                    order_char == b'F' and self._f_contiguous):
                return self

        order_char = internal._update_order_char(
            self._c_contiguous, self._f_contiguous, order_char)

        if order_char == b'K':
            strides = internal._get_strides_for_order_K(self, dtype)
            newarray = _ndarray_init(self._shape, dtype)
            # TODO(niboshi): Confirm update_x_contiguity flags
            newarray._set_shape_and_strides(self._shape, strides, True, True)
        else:
            newarray = ndarray(self.shape, dtype=dtype, order=chr(order_char))

        if self.size == 0:
            # skip copy
            pass
        elif self.dtype.kind == 'c' and newarray.dtype.kind == 'b':
            cupy.not_equal(self, 0j, out=newarray)
        elif self.dtype.kind == 'c' and newarray.dtype.kind != 'c':
            warnings.warn(
                'Casting complex values to real discards the imaginary part',
                numpy.ComplexWarning)
            elementwise_copy(self.real, newarray)
        else:
            elementwise_copy(self, newarray)
        return newarray

    # TODO(okuta): Implement byteswap

    cpdef ndarray copy(self, order='C'):
        """Returns a copy of the array.

        This method makes a copy of a given array in the current device.
        Even when a given array is located in another device, you can copy it
        to the current device.

        Args:
            order ({'C', 'F', 'A', 'K'}): Row-major (C-style) or column-major
                (Fortran-style) order.
                When ``order`` is 'A', it uses 'F' if ``a`` is column-major and
                uses 'C' otherwise.
                And when `order` is 'K', it keeps strides as closely as
                possible.

        .. seealso::
           :func:`cupy.copy` for full documentation,
           :meth:`numpy.ndarray.copy`

        """
        cdef ndarray x
        if self.size == 0:
            return self.astype(self.dtype, order=order)

        dev_id = device.get_device_id()
        if self.data.device_id == dev_id:
            return self.astype(self.dtype, order=order)

        # It need to make a contiguous copy for copying from another device
        runtime.setDevice(self.data.device_id)
        try:
            x = self.astype(self.dtype, order=order, copy=False)
        finally:
            runtime.setDevice(dev_id)
        newarray = _ndarray_init(x._shape, x.dtype)
        if not x._c_contiguous and not x._f_contiguous:
            raise NotImplementedError(
                'CuPy cannot copy non-contiguous array between devices.')
        # TODO(niboshi): Confirm update_x_contiguity flags
        newarray._strides = x._strides
        newarray._c_contiguous = x._c_contiguous
        newarray._f_contiguous = x._f_contiguous
        if runtime._is_hip_environment:
            # HIP requires changing the active device to the one where
            # src data is before the copy. From the docs:
            # it is recommended to set the current device to the device
            # where the src data is physically located.
            runtime.setDevice(self.data.device_id)
        try:
            newarray.data.copy_from_device_async(x.data, x.nbytes)
        finally:
            if runtime._is_hip_environment:
                runtime.setDevice(dev_id)
        return newarray

    cpdef ndarray view(self, dtype=None):
        """Returns a view of the array.

        Args:
            dtype: If this is different from the data type of the array, the
                returned view reinterpret the memory sequence as an array of
                this type.

        Returns:
            cupy.ndarray: A view of the array. A reference to the original
            array is stored at the :attr:`~ndarray.base` attribute.

        .. seealso:: :meth:`numpy.ndarray.view`

        """
        # Use __new__ instead of __init__ to skip recomputation of contiguity
        cdef Py_ssize_t ndim
        cdef int self_is, v_is
        v = self._view(self._shape, self._strides, False, False)
        if dtype is None:
            return v

        v.dtype, v_is = _dtype.get_dtype_with_itemsize(dtype)
        self_is = self.dtype.itemsize
        if v_is == self_is:
            return v

        ndim = self._shape.size()
        if ndim == 0:
            raise ValueError(
                'Changing the dtype of a 0d array is only supported if '
                'the itemsize is unchanged')
        if not self._c_contiguous:
            raise ValueError(
                'To change to a dtype of a different size, the array must '
                'be C-contiguous')
        v._shape[ndim - 1] = v._shape[ndim - 1] * self_is // v_is
        v._strides[ndim - 1] = v._strides[ndim - 1] * v_is // self_is
        v.size = v.size * self_is // v_is
        return v

    # TODO(okuta): Implement getfield
    # TODO(okuta): Implement setflags

    cpdef fill(self, value):
        """Fills the array with a scalar value.

        Args:
            value: A scalar value to fill the array content.

        .. seealso:: :meth:`numpy.ndarray.fill`

        """
        if isinstance(value, numpy.ndarray):
            if value.shape != ():
                raise ValueError(
                    'non-scalar numpy.ndarray cannot be used for fill')
            value = value.item()

        if value == 0 and self._c_contiguous:
            self.data.memset_async(0, self.nbytes)
        else:
            fill_kernel(value, self)

    # -------------------------------------------------------------------------
    # Shape manipulation
    # -------------------------------------------------------------------------
    def reshape(self, *shape, order='C'):
        """Returns an array of a different shape and the same content.

        .. seealso::
           :func:`cupy.reshape` for full documentation,
           :meth:`numpy.ndarray.reshape`

        """
        return _manipulation._ndarray_reshape(self, shape, order)

    # TODO(okuta): Implement resize

    def transpose(self, *axes):
        """Returns a view of the array with axes permuted.

        .. seealso::
           :func:`cupy.transpose` for full documentation,
           :meth:`numpy.ndarray.reshape`

        """
        return _manipulation._ndarray_transpose(self, axes)

    cpdef ndarray swapaxes(self, Py_ssize_t axis1, Py_ssize_t axis2):
        """Returns a view of the array with two axes swapped.

        .. seealso::
           :func:`cupy.swapaxes` for full documentation,
           :meth:`numpy.ndarray.swapaxes`

        """
        return _manipulation._ndarray_swapaxes(self, axis1, axis2)

    cpdef ndarray flatten(self):
        """Returns a copy of the array flatten into one dimension.

        It currently supports C-order only.

        Returns:
            cupy.ndarray: A copy of the array with one dimension.

        .. seealso:: :meth:`numpy.ndarray.flatten`

        """
        # TODO(beam2d): Support ordering option
        return _manipulation._ndarray_flatten(self)

    cpdef ndarray ravel(self, order='C'):
        """Returns an array flattened into one dimension.

        .. seealso::
           :func:`cupy.ravel` for full documentation,
           :meth:`numpy.ndarray.ravel`

        """
        return _manipulation._ndarray_ravel(self, order)

    cpdef ndarray squeeze(self, axis=None):
        """Returns a view with size-one axes removed.

        .. seealso::
           :func:`cupy.squeeze` for full documentation,
           :meth:`numpy.ndarray.squeeze`

        """
        return _manipulation._ndarray_squeeze(self, axis)

    # -------------------------------------------------------------------------
    # Item selection and manipulation
    # -------------------------------------------------------------------------
    cpdef ndarray take(self, indices, axis=None, out=None):
        """Returns an array of elements at given indices along the axis.

        .. seealso::
           :func:`cupy.take` for full documentation,
           :meth:`numpy.ndarray.take`

        """
        return _indexing._ndarray_take(self, indices, axis, out)

    cpdef put(self, indices, values, mode='wrap'):
        """Replaces specified elements of an array with given values.

        .. seealso::
           :func:`cupy.put` for full documentation,
           :meth:`numpy.ndarray.put`
        """
        return _indexing._ndarray_put(self, indices, values, mode)

    cpdef repeat(self, repeats, axis=None):
        """Returns an array with repeated arrays along an axis.

        .. seealso::
            :func:`cupy.repeat` for full documentation,
            :meth:`numpy.ndarray.repeat`

        """
        return _manipulation._ndarray_repeat(self, repeats, axis)

    cpdef choose(self, choices, out=None, mode='raise'):
        # TODO(niboshi): Write docstring
        return _indexing._ndarray_choose(self, choices, out, mode)

    cpdef sort(self, int axis=-1):
        """Sort an array, in-place with a stable sorting algorithm.

        Args:
            axis (int): Axis along which to sort. Default is -1, which means
                sort along the last axis.

        .. note::
           For its implementation reason, ``ndarray.sort`` currently supports
           only arrays with their own data, and does not support ``kind`` and
           ``order`` parameters that ``numpy.ndarray.sort`` does support.

        .. seealso::
            :func:`cupy.sort` for full documentation,
            :meth:`numpy.ndarray.sort`

        """
        # TODO(takagi): Support kind argument.
        _sorting._ndarray_sort(self, axis)

    cpdef ndarray argsort(self, axis=-1):
        """Returns the indices that would sort an array with stable sorting

        Args:
            axis (int or None): Axis along which to sort. Default is -1, which
                means sort along the last axis. If None is supplied, the array
                is flattened before sorting.

        Returns:
            cupy.ndarray: Array of indices that sort the array.

        .. seealso::
            :func:`cupy.argsort` for full documentation,
            :meth:`numpy.ndarray.argsort`

        """
        # TODO(takagi): Support kind argument.
        return _sorting._ndarray_argsort(self, axis)

    cpdef partition(self, kth, int axis=-1):
        """Partitions an array.

        Args:
            kth (int or sequence of ints): Element index to partition by. If
                supplied with a sequence of k-th it will partition all elements
                indexed by k-th of them into their sorted position at once.

            axis (int): Axis along which to sort. Default is -1, which means
                sort along the last axis.

        .. seealso::
            :func:`cupy.partition` for full documentation,
            :meth:`numpy.ndarray.partition`

        """
        _sorting._ndarray_partition(self, kth, axis)

    cpdef ndarray argpartition(self, kth, axis=-1):
        """Returns the indices that would partially sort an array.

        Args:
            kth (int or sequence of ints): Element index to partition by. If
                supplied with a sequence of k-th it will partition all elements
                indexed by k-th of them into their sorted position at once.
            axis (int or None): Axis along which to sort. Default is -1, which
                means sort along the last axis. If None is supplied, the array
                is flattened before sorting.

        Returns:
            cupy.ndarray: Array of the same type and shape as ``a``.

        .. seealso::
            :func:`cupy.argpartition` for full documentation,
            :meth:`numpy.ndarray.argpartition`

        """
        return _sorting._ndarray_argpartition(self, kth, axis)

    cpdef tuple nonzero(self):
        """Return the indices of the elements that are non-zero.

        Returned Array is containing the indices of the non-zero elements
        in that dimension.

        Returns:
            tuple of arrays: Indices of elements that are non-zero.

        .. warning::

            This function may synchronize the device.

        .. seealso::
            :func:`numpy.nonzero`

        """
        return _indexing._ndarray_nonzero(self)

    cpdef ndarray compress(self, condition, axis=None, out=None):
        """Returns selected slices of this array along given axis.

        .. warning::

            This function may synchronize the device.

        .. seealso::
           :func:`cupy.compress` for full documentation,
           :meth:`numpy.ndarray.compress`

        """
        return _indexing._ndarray_compress(self, condition, axis, out)

    cpdef ndarray diagonal(self, offset=0, axis1=0, axis2=1):
        """Returns a view of the specified diagonals.

        .. seealso::
           :func:`cupy.diagonal` for full documentation,
           :meth:`numpy.ndarray.diagonal`

        """
        return _indexing._ndarray_diagonal(self, offset, axis1, axis2)

    # -------------------------------------------------------------------------
    # Calculation
    # -------------------------------------------------------------------------
    cpdef ndarray max(self, axis=None, out=None, keepdims=False):
        """Returns the maximum along a given axis.

        .. seealso::
           :func:`cupy.amax` for full documentation,
           :meth:`numpy.ndarray.max`

        """
        return _statistics._ndarray_max(self, axis, out, None, keepdims)

    cpdef ndarray argmax(self, axis=None, out=None, dtype=None,
                         keepdims=False):
        """Returns the indices of the maximum along a given axis.

        .. note::
           ``dtype`` and ``keepdim`` arguments are specific to CuPy. They are
           not in NumPy.

        .. note::
           ``axis`` argument accepts a tuple of ints, but this is specific to
           CuPy. NumPy does not support it.

        .. seealso::
           :func:`cupy.argmax` for full documentation,
           :meth:`numpy.ndarray.argmax`

        """
        return _statistics._ndarray_argmax(self, axis, out, dtype, keepdims)

    cpdef ndarray min(self, axis=None, out=None, keepdims=False):
        """Returns the minimum along a given axis.

        .. seealso::
           :func:`cupy.amin` for full documentation,
           :meth:`numpy.ndarray.min`

        """
        return _statistics._ndarray_min(self, axis, out, None, keepdims)

    cpdef ndarray argmin(self, axis=None, out=None, dtype=None,
                         keepdims=False):
        """Returns the indices of the minimum along a given axis.

        .. note::
           ``dtype`` and ``keepdim`` arguments are specific to CuPy. They are
           not in NumPy.

        .. note::
           ``axis`` argument accepts a tuple of ints, but this is specific to
           CuPy. NumPy does not support it.

        .. seealso::
           :func:`cupy.argmin` for full documentation,
           :meth:`numpy.ndarray.argmin`

        """
        return _statistics._ndarray_argmin(self, axis, out, dtype, keepdims)

    cpdef ndarray ptp(self, axis=None, out=None, keepdims=False):
        """Returns (maximum - minimum) along a given axis.

        .. seealso::
           :func:`cupy.ptp` for full documentation,
           :meth:`numpy.ndarray.ptp`

        """
        return _statistics._ndarray_ptp(self, axis, out, keepdims)

    cpdef ndarray clip(self, a_min=None, a_max=None, out=None):
        """Returns an array with values limited to [a_min, a_max].

        .. seealso::
           :func:`cupy.clip` for full documentation,
           :meth:`numpy.ndarray.clip`

        """
        return _math._ndarray_clip(self, a_min, a_max, out)

    cpdef ndarray round(self, decimals=0, out=None):
        """Returns an array with values rounded to the given number of decimals.

        .. seealso::
           :func:`cupy.around` for full documentation,
           :meth:`numpy.ndarray.round`

        """
        return _round_ufunc(self, decimals, out=out)

    cpdef ndarray trace(self, offset=0, axis1=0, axis2=1, dtype=None,
                        out=None):
        """Returns the sum along diagonals of the array.

        .. seealso::
           :func:`cupy.trace` for full documentation,
           :meth:`numpy.ndarray.trace`

        """
        d = self.diagonal(offset, axis1, axis2)
        return d.sum(-1, dtype, out, False)

    cpdef ndarray sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """Returns the sum along a given axis.

        .. seealso::
           :func:`cupy.sum` for full documentation,
           :meth:`numpy.ndarray.sum`

        """
        return _math._ndarray_sum(self, axis, dtype, out, keepdims)

    cpdef ndarray cumsum(self, axis=None, dtype=None, out=None):
        """Returns the cumulative sum of an array along a given axis.

        .. seealso::
           :func:`cupy.cumsum` for full documentation,
           :meth:`numpy.ndarray.cumsum`

        """
        return _math._ndarray_cumsum(self, axis, dtype, out)

    cpdef ndarray mean(self, axis=None, dtype=None, out=None, keepdims=False):
        """Returns the mean along a given axis.

        .. seealso::
           :func:`cupy.mean` for full documentation,
           :meth:`numpy.ndarray.mean`

        """
        return _statistics._ndarray_mean(self, axis, dtype, out, keepdims)

    cpdef ndarray var(self, axis=None, dtype=None, out=None, ddof=0,
                      keepdims=False):
        """Returns the variance along a given axis.

        .. seealso::
           :func:`cupy.var` for full documentation,
           :meth:`numpy.ndarray.var`

        """
        return _statistics._ndarray_var(
            self, axis, dtype, out, ddof, keepdims)

    cpdef ndarray std(self, axis=None, dtype=None, out=None, ddof=0,
                      keepdims=False):
        """Returns the standard deviation along a given axis.

        .. seealso::
           :func:`cupy.std` for full documentation,
           :meth:`numpy.ndarray.std`

        """
        return _statistics._ndarray_std(self, axis, dtype, out, ddof, keepdims)

    cpdef ndarray prod(self, axis=None, dtype=None, out=None, keepdims=None):
        """Returns the product along a given axis.

        .. seealso::
           :func:`cupy.prod` for full documentation,
           :meth:`numpy.ndarray.prod`

        """
        return _math._ndarray_prod(self, axis, dtype, out, keepdims)

    cpdef ndarray cumprod(self, axis=None, dtype=None, out=None):
        """Returns the cumulative product of an array along a given axis.

        .. seealso::
           :func:`cupy.cumprod` for full documentation,
           :meth:`numpy.ndarray.cumprod`

        """
        return _math._ndarray_cumprod(self, axis, dtype, out)

    cpdef ndarray all(self, axis=None, out=None, keepdims=False):
        # TODO(niboshi): Write docstring
        return _logic._ndarray_all(self, axis, out, keepdims)

    cpdef ndarray any(self, axis=None, out=None, keepdims=False):
        # TODO(niboshi): Write docstring
        return _logic._ndarray_any(self, axis, out, keepdims)

    # -------------------------------------------------------------------------
    # Arithmetic and comparison operations
    # -------------------------------------------------------------------------
    # Comparison operators:

    def __richcmp__(object self, object other, int op):
        if isinstance(other, ndarray):
            if op == 0:
                return _logic._ndarray_less(self, other)
            if op == 1:
                return _logic._ndarray_less_equal(self, other)
            if op == 2:
                return _logic._ndarray_equal(self, other)
            if op == 3:
                return _logic._ndarray_not_equal(self, other)
            if op == 4:
                return _logic._ndarray_greater(self, other)
            if op == 5:
                return _logic._ndarray_greater_equal(self, other)
        elif not _should_use_rop(self, other):
            if isinstance(other, numpy.ndarray) and other.ndim == 0:
                other = other.item()  # Workaround for numpy<1.13
            if op == 0:
                return numpy.less(self, other)
            if op == 1:
                return numpy.less_equal(self, other)
            if op == 2:
                return numpy.equal(self, other)
            if op == 3:
                return numpy.not_equal(self, other)
            if op == 4:
                return numpy.greater(self, other)
            if op == 5:
                return numpy.greater_equal(self, other)
        return NotImplemented

    # Truth value of an array (bool):

    def __nonzero__(self):
        if self.size == 0:
            msg = ('The truth value of an empty array is ambiguous. Returning '
                   'False, but in future this will result in an error. Use '
                   '`array.size > 0` to check that an array is not empty.')
            warnings.warn(msg, DeprecationWarning)
            return False
        elif self.size == 1:
            return bool(self.get())
        else:
            msg = ('The truth value of an array with more than one element is '
                   'ambiguous. Use a.any() or a.all()')
            raise ValueError(msg)

    # Unary operations:

    def __neg__(self):
        return _math._negative(self)

    def __pos__(self):
        return self

    def __abs__(self):
        return _math._absolute(self)

    def __invert__(self):
        return _binary._invert(self)

    # Arithmetic:

    def __add__(x, y):
        if isinstance(y, ndarray):
            return _math._add(x, y)
        elif _should_use_rop(x, y):
            return NotImplemented
        else:
            return numpy.add(x, y)

    def __sub__(x, y):
        if isinstance(y, ndarray):
            return _math._subtract(x, y)
        elif _should_use_rop(x, y):
            return NotImplemented
        else:
            return numpy.subtract(x, y)

    def __mul__(x, y):
        if isinstance(y, ndarray):
            return _math._multiply(x, y)
        elif _should_use_rop(x, y):
            return NotImplemented
        else:
            return numpy.multiply(x, y)

    def __matmul__(x, y):
        if not isinstance(y, ndarray) and _should_use_rop(x, y):
            return NotImplemented
        else:
            return cupy.linalg._product.matmul(x, y)

    def __div__(x, y):
        if isinstance(y, ndarray):
            return _math._divide(x, y)
        elif _should_use_rop(x, y):
            return NotImplemented
        else:
            return numpy.divide(x, y)

    def __truediv__(x, y):
        if isinstance(y, ndarray):
            return _math._true_divide(x, y)
        elif _should_use_rop(x, y):
            return NotImplemented
        else:
            return numpy.true_divide(x, y)

    def __floordiv__(x, y):
        if isinstance(y, ndarray):
            return _math._floor_divide(x, y)
        elif _should_use_rop(x, y):
            return NotImplemented
        else:
            return numpy.floor_divide(x, y)

    def __mod__(x, y):
        if isinstance(y, ndarray):
            return _math._remainder(x, y)
        elif _should_use_rop(x, y):
            return NotImplemented
        else:
            return numpy.remainder(x, y)

    def __divmod__(x, y):
        if isinstance(y, ndarray):
            return divmod(x, y)
        elif _should_use_rop(x, y):
            return NotImplemented
        else:
            return numpy.divmod(x, y)

    def __pow__(x, y, modulo):
        # Note that we ignore the modulo argument as well as NumPy.
        if isinstance(y, ndarray):
            return _math._power(x, y)
        elif _should_use_rop(x, y):
            return NotImplemented
        else:
            return numpy.power(x, y)

    def __lshift__(x, y):
        if isinstance(y, ndarray):
            return _binary._left_shift(x, y)
        elif _should_use_rop(x, y):
            return NotImplemented
        else:
            return numpy.left_shift(x, y)

    def __rshift__(x, y):
        if isinstance(y, ndarray):
            return _binary._right_shift(x, y)
        elif _should_use_rop(x, y):
            return NotImplemented
        else:
            return numpy.right_shift(x, y)

    def __and__(x, y):
        if isinstance(y, ndarray):
            return _binary._bitwise_and(x, y)
        elif _should_use_rop(x, y):
            return NotImplemented
        else:
            return numpy.bitwise_and(x, y)

    def __or__(x, y):
        if isinstance(y, ndarray):
            return _binary._bitwise_or(x, y)
        elif _should_use_rop(x, y):
            return NotImplemented
        else:
            return numpy.bitwise_or(x, y)

    def __xor__(x, y):
        if isinstance(y, ndarray):
            return _binary._bitwise_xor(x, y)
        elif _should_use_rop(x, y):
            return NotImplemented
        else:
            return numpy.bitwise_xor(x, y)

    # Arithmetic, in-place:

    def __iadd__(self, other):
        return _math._add(self, other, self)

    def __isub__(self, other):
        return _math._subtract(self, other, self)

    def __imul__(self, other):
        return _math._multiply(self, other, self)

    def __idiv__(self, other):
        return _math._divide(self, other, self)

    def __itruediv__(self, other):
        return _math._true_divide(self, other, self)

    def __ifloordiv__(self, other):
        return _math._floor_divide(self, other, self)

    def __imod__(self, other):
        return _math._remainder(self, other, self)

    def __ipow__(self, other):
        return _math._power(self, other, self)

    def __ilshift__(self, other):
        return _binary._left_shift(self, other, self)

    def __irshift__(self, other):
        return _binary._right_shift(self, other, self)

    def __iand__(self, other):
        return _binary._bitwise_and(self, other, self)

    def __ior__(self, other):
        return _binary._bitwise_or(self, other, self)

    def __ixor__(self, other):
        return _binary._bitwise_xor(self, other, self)

    cpdef ndarray conj(self):
        return _math._ndarray_conj(self)

    cpdef ndarray conjugate(self):
        return _math._ndarray_conj(self)

    @property
    def real(self):
        return _math._ndarray_real_getter(self)

    @real.setter
    def real(self, value):
        _math._ndarray_real_setter(self, value)

    @property
    def imag(self):
        return _math._ndarray_imag_getter(self)

    @imag.setter
    def imag(self, value):
        _math._ndarray_imag_setter(self, value)

    # -------------------------------------------------------------------------
    # Special methods
    # -------------------------------------------------------------------------
    # For standard library functions:

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        with self.device:
            return self.copy()

    def __reduce__(self):
        return array, (self.get(),)

    # Basic customization:

    # cupy.ndarray does not define __new__

    def __array__(self, dtype=None):
        # TODO(imanishi): Support an environment variable or a global
        # configure flag that allows implicit conversions to NumPy array.
        # (See https://github.com/cupy/cupy/issues/589 for the detail.)
        raise TypeError(
            'Implicit conversion to a NumPy array is not allowed. '
            'Please use `.get()` to construct a NumPy array explicitly.')

    # TODO(okuta): Implement __array_wrap__

    # Container customization:

    def __iter__(self):
        if self._shape.size() == 0:
            raise TypeError('iteration over a 0-d array')
        return (self[i] for i in range(self._shape[0]))

    def __len__(self):
        if self._shape.size() == 0:
            raise TypeError('len() of unsized object')
        return self._shape[0]

    def __getitem__(self, slices):
        """x.__getitem__(y) <==> x[y]

        Supports both basic and advanced indexing.

        .. note::

            Currently, it does not support ``slices`` that consists of more
            than one boolean arrays

        .. note::

           CuPy handles out-of-bounds indices differently from NumPy.
           NumPy handles them by raising an error, but CuPy wraps around them.

        Example:

            >>> a = cupy.arange(3)
            >>> a[[1, 3]]
            array([1, 0])

        """
        return _indexing._ndarray_getitem(self, slices)

    def __setitem__(self, slices, value):
        """x.__setitem__(slices, y) <==> x[slices] = y

        Supports both basic and advanced indexing.

        .. note::

            Currently, it does not support ``slices`` that consists of more
            than one boolean arrays

        .. note::

            CuPy handles out-of-bounds indices differently from NumPy when
            using integer array indexing.
            NumPy handles them by raising an error, but CuPy wraps around them.

            >>> import cupy
            >>> x = cupy.arange(3)
            >>> x[[1, 3]] = 10
            >>> x
            array([10, 10,  2])

        .. note::

            The behavior differs from NumPy when integer arrays in ``slices``
            reference the same location multiple times.
            In that case, the value that is actually stored is undefined.

            >>> import cupy
            >>> a = cupy.zeros((2,))
            >>> i = cupy.arange(10000) % 2
            >>> v = cupy.arange(10000).astype(cupy.float_)
            >>> a[i] = v
            >>> a  # doctest: +SKIP
            array([9150., 9151.])

            On the other hand, NumPy stores the value corresponding to the
            last index among the indices referencing duplicate locations.

            >>> import numpy
            >>> a_cpu = numpy.zeros((2,))
            >>> i_cpu = numpy.arange(10000) % 2
            >>> v_cpu = numpy.arange(10000).astype(numpy.float_)
            >>> a_cpu[i_cpu] = v_cpu
            >>> a_cpu
            array([9998., 9999.])

        """
        if _util.ENABLE_SLICE_COPY and (
                type(slices) is slice
                and slices == slice(None, None, None)
                and isinstance(value, numpy.ndarray)
        ):
            if (self.dtype == value.dtype
                    and self.shape == value.shape
                    and (self._f_contiguous or self._c_contiguous)):
                order = 'F' if self._f_contiguous else 'C'
                tmp = value.ravel(order)
                ptr = tmp.ctypes.data
                stream_ptr = stream_module.get_current_stream_ptr()
                if stream_ptr == 0:
                    self.data.copy_from_host(ptr, self.nbytes)
                else:
                    self.data.copy_from_host_async(ptr, self.nbytes)
            else:
                raise ValueError(
                    'copying a numpy.ndarray to a cupy.ndarray by empty slice '
                    'assignment must ensure arrays have same shape and dtype')
        else:
            _indexing._ndarray_setitem(self, slices, value)

    def scatter_add(self, slices, value):
        """Adds given values to specified elements of an array.

        .. seealso::
            :func:`cupyx.scatter_add` for full documentation.

        """
        _indexing._ndarray_scatter_add(self, slices, value)

    def scatter_max(self, slices, value):
        """Stores a maximum value of elements specified by indices to an array.

        .. seealso::
            :func:`cupyx.scatter_max` for full documentation.

        """
        _indexing._ndarray_scatter_max(self, slices, value)

    def scatter_min(self, slices, value):
        """Stores a minimum value of elements specified by indices to an array.

        .. seealso::
            :func:`cupyx.scatter_min` for full documentation.

        """
        _indexing._ndarray_scatter_min(self, slices, value)

    # TODO(okuta): Implement __getslice__
    # TODO(okuta): Implement __setslice__
    # TODO(okuta): Implement __contains__

    # numpy/ufunc compat
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        """Apply unary or binary ufunc to this array

        If binary, only allow if second argument is another cupy ndarray or
        a number, i.e., raise ValueError instead of silently converting a
        numpy array.
        """
        import cupy  # top-level ufuncs
        inout = inputs
        if 'out' in kwargs:
            # need to unfold tuple argument in kwargs
            # TODO(ecastill) GUFuncs support more than one output
            out = kwargs['out']
            if len(out) != 1:
                raise ValueError('The \'out\' parameter must have exactly one '
                                 'array value')
            inout += out
            kwargs['out'] = out[0]

        if method == '__call__':
            name = ufunc.__name__
            try:
                cp_ufunc = getattr(cupy, name)
            except AttributeError:
                return NotImplemented
            for x in inout:
                # numpy.ndarray is handled and then TypeError is raised due to
                # implicit host-to-device conversion.
                # Except for numpy.ndarray, types should be supported by
                # `_kernel._preprocess_args`.
                check = hasattr(x, '__cuda_array_interface__')
                if runtime._is_hip_environment and isinstance(x, ndarray):
                    check = True
                if (not check
                        and not type(x) in _scalar.scalar_type_set
                        and not isinstance(x, numpy.ndarray)):
                    return NotImplemented
            if name in [
                    'greater', 'greater_equal', 'less', 'less_equal',
                    'equal', 'not_equal']:
                # workaround for numpy/numpy#12142
                inputs = tuple([
                    x.item()
                    if isinstance(x, numpy.ndarray) and x.ndim == 0
                    else x
                    for x in inputs
                ])
            return cp_ufunc(*inputs, **kwargs)
        # Don't use for now, interface uncertain
        # elif method =='at' and name == 'add':
            # the only ufunc attribute currently
            # http://docs.cupy.dev/en/stable/reference/ufunc.html#ufunc-at
            # self.scatter_add(*inputs, **kwargs)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        try:
            module = functools.reduce(
                getattr, func.__module__.split('.')[1:], cupy)
            cupy_func = getattr(module, func.__name__)
        except AttributeError:
            return NotImplemented
        if cupy_func is func:
            # avoid NumPy func
            return NotImplemented
        for t in types:
            if t not in _HANDLED_TYPES:
                return NotImplemented
        return cupy_func(*args, **kwargs)

    # Conversion:

    def __int__(self):
        return int(self.get())

    def __float__(self):
        return float(self.get())

    def __complex__(self):
        return complex(self.get())

    def __oct__(self):
        return oct(self.get())

    def __hex__(self):
        return hex(self.get())

    def __bytes__(self):
        return bytes(self.get())

    # String representations:

    def __repr__(self):
        return repr(self.get())

    def __str__(self):
        return str(self.get())

    def __format__(self, format_spec):
        return format(self.get(), format_spec)

    # -------------------------------------------------------------------------
    # Methods outside of the ndarray main documentation
    # -------------------------------------------------------------------------
    def dot(self, ndarray b, ndarray out=None):
        """Returns the dot product with given array.

        .. seealso::
           :func:`cupy.dot` for full documentation,
           :meth:`numpy.ndarray.dot`

        """
        return _linalg.dot(self, b, out)

    # -------------------------------------------------------------------------
    # Cupy specific attributes and methods
    # -------------------------------------------------------------------------
    @property
    def device(self):
        """CUDA device on which this array resides."""
        return self.data.device

    cpdef get(self, stream=None, order='C', out=None):
        """Returns a copy of the array on host memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.
                The default uses CUDA stream object of the current context.
            order ({'C', 'F', 'A'}): The desired memory layout of the host
                array. When ``order`` is 'A', it uses 'F' if the array is
                fortran-contiguous and 'C' otherwise. The ``order`` will be
                ignored if ``out`` is specified.
            out (numpy.ndarray): Output array. In order to enable asynchronous
                copy, the underlying memory should be a pinned memory.

        Returns:
            numpy.ndarray: Copy of the array on host memory.

        """
        if out is not None:
            if not isinstance(out, numpy.ndarray):
                raise TypeError('Only numpy.ndarray can be obtained from'
                                'cupy.ndarray')
            if self.dtype != out.dtype:
                raise TypeError(
                    '{} array cannot be obtained from {} array'.format(
                        out.dtype, self.dtype))
            if self.shape != out.shape:
                raise ValueError(
                    'Shape mismatch. Expected shape: {}, '
                    'actual shape: {}'.format(self.shape, out.shape))
            if not (out.flags.c_contiguous and self._c_contiguous or
                    out.flags.f_contiguous and self._f_contiguous):
                with self.device:
                    if out.flags.c_contiguous:
                        a_gpu = _internal_ascontiguousarray(self)
                    elif out.flags.f_contiguous:
                        a_gpu = _internal_asfortranarray(self)
                    else:
                        raise RuntimeError(
                            '`out` cannot be specified when copying to '
                            'non-contiguous ndarray')
            else:
                a_gpu = self
            a_cpu = out
        else:
            if self.size == 0:
                return numpy.ndarray(self._shape, dtype=self.dtype)

            order = order.upper()
            if order == 'A':
                if self._f_contiguous:
                    order = 'F'
                else:
                    order = 'C'
            if not (order == 'C' and self._c_contiguous or
                    order == 'F' and self._f_contiguous):
                with self.device:
                    if order == 'C':
                        a_gpu = _internal_ascontiguousarray(self)
                    elif order == 'F':
                        a_gpu = _internal_asfortranarray(self)
                    else:
                        raise ValueError('unsupported order: {}'.format(order))
            else:
                a_gpu = self
            a_cpu = numpy.empty(self._shape, dtype=self.dtype, order=order)

        syncdetect._declare_synchronize()
        ptr = a_cpu.ctypes.data
        with self.device:
            if stream is not None:
                a_gpu.data.copy_to_host_async(ptr, a_gpu.nbytes, stream)
            else:
                stream_ptr = stream_module.get_current_stream_ptr()
                if stream_ptr == 0:
                    a_gpu.data.copy_to_host(ptr, a_gpu.nbytes)
                else:
                    a_gpu.data.copy_to_host_async(ptr, a_gpu.nbytes)
        return a_cpu

    cpdef set(self, arr, stream=None):
        """Copies an array on the host memory to :class:`cupy.ndarray`.

        Args:
            arr (numpy.ndarray): The source array on the host memory.
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.
                The default uses CUDA stream object of the current context.

        """
        if not isinstance(arr, numpy.ndarray):
            raise TypeError('Only numpy.ndarray can be set to cupy.ndarray')
        if self.dtype != arr.dtype:
            raise TypeError('{} array cannot be set to {} array'.format(
                arr.dtype, self.dtype))
        if self.shape != arr.shape:
            raise ValueError(
                'Shape mismatch. Old shape: {}, new shape: {}'.format(
                    self.shape, arr.shape))
        if self._c_contiguous:
            arr = numpy.ascontiguousarray(arr)
        elif self._f_contiguous:
            arr = numpy.asfortranarray(arr)
        else:
            raise RuntimeError('Cannot set to non-contiguous array')

        ptr = arr.ctypes.data
        with self.device:
            if stream is not None:
                self.data.copy_from_host_async(ptr, self.nbytes, stream)
            else:
                stream_ptr = stream_module.get_current_stream_ptr()
                if stream_ptr == 0:
                    self.data.copy_from_host(ptr, self.nbytes)
                else:
                    self.data.copy_from_host_async(ptr, self.nbytes)

    cpdef ndarray reduced_view(self, dtype=None):
        """Returns a view of the array with minimum number of dimensions.

        Args:
            dtype: (Deprecated) Data type specifier.
                If it is given, then the memory
                sequence is reinterpreted as the new type.

        Returns:
            cupy.ndarray: A view of the array with reduced dimensions.

        """
        cdef shape_t shape
        cdef strides_t strides
        cdef Py_ssize_t ndim
        cdef ndarray view
        if dtype is not None:
            warnings.warn(
                'calling reduced_view with dtype is deprecated',
                DeprecationWarning)
            return self.reduced_view().view(dtype)

        ndim = self._shape.size()
        if ndim <= 1:
            return self
        if self._c_contiguous:
            view = self.view()
            view._shape.assign(1, self.size)
            view._strides.assign(1, self.dtype.itemsize)
            view._update_f_contiguity()
            return view

        internal.get_reduced_dims(
            self._shape, self._strides, self.dtype.itemsize, shape, strides)
        if ndim == <Py_ssize_t>shape.size():
            return self

        # TODO(niboshi): Confirm update_x_contiguity flags
        return self._view(shape, strides, False, True)

    cpdef _update_c_contiguity(self):
        if self.size == 0:
            self._c_contiguous = True
            return
        self._c_contiguous = internal.get_c_contiguity(
            self._shape, self._strides, self.dtype.itemsize)

    cpdef _update_f_contiguity(self):
        cdef Py_ssize_t i, count
        cdef shape_t rev_shape
        cdef strides_t rev_strides
        if self.size == 0:
            self._f_contiguous = True
            return
        if self._c_contiguous:
            count = 0
            for i in self._shape:
                if i == 1:
                    count += 1
            self._f_contiguous = (<Py_ssize_t>self._shape.size()) - count <= 1
            return
        rev_shape.assign(self._shape.rbegin(), self._shape.rend())
        rev_strides.assign(self._strides.rbegin(), self._strides.rend())
        self._f_contiguous = internal.get_c_contiguity(
            rev_shape, rev_strides, self.dtype.itemsize)

    cpdef _update_contiguity(self):
        self._update_c_contiguity()
        self._update_f_contiguity()

    cpdef _set_shape_and_strides(self, const shape_t& shape,
                                 const strides_t& strides,
                                 bint update_c_contiguity,
                                 bint update_f_contiguity):
        if shape.size() != strides.size():
            raise ValueError('len(shape) != len(strides)')
        self._shape = shape
        self._strides = strides
        self.size = internal.prod(shape)
        if update_c_contiguity:
            self._update_c_contiguity()
        if update_f_contiguity:
            self._update_f_contiguity()

    cdef ndarray _view(self, const shape_t& shape,
                       const strides_t& strides,
                       bint update_c_contiguity,
                       bint update_f_contiguity):
        cdef ndarray v
        v = ndarray.__new__(ndarray)
        v.data = self.data
        v.base = self.base if self.base is not None else self
        v.dtype = self.dtype
        v._c_contiguous = self._c_contiguous
        v._f_contiguous = self._f_contiguous
        v._index_32_bits = self._index_32_bits
        v._set_shape_and_strides(
            shape, strides, update_c_contiguity, update_f_contiguity)
        return v

    cpdef _set_contiguous_strides(
            self, Py_ssize_t itemsize, bint is_c_contiguous):
        self.size = internal.get_contiguous_strides_inplace(
            self._shape, self._strides, itemsize, is_c_contiguous)
        if is_c_contiguous:
            self._c_contiguous = True
            self._update_f_contiguity()
        else:
            self._f_contiguous = True
            self._update_c_contiguity()

    cdef function.CPointer get_pointer(self):
        return _CArray_from_ndarray(self)

    cpdef object toDlpack(self):
        """Zero-copy conversion to a DLPack tensor.

        DLPack is a open in memory tensor structure proposed in this
        repository: `dmlc/dlpack <https://github.com/dmlc/dlpack>`_.

        This function returns a :class:`PyCapsule` object which contains a
        pointer to a DLPack tensor converted from the own ndarray. This
        function does not copy the own data to the output DLpack tensor
        but it shares the pointer which is pointing to the same memory region
        for the data.

        Returns:
            dltensor (:class:`PyCapsule`): Output DLPack tensor which is
                encapsulated in a :class:`PyCapsule` object.

        .. seealso::

            :meth:`~cupy.fromDlpack` is a method for zero-copy conversion from
            a DLPack tensor (which is encapsulated in a :class:`PyCapsule`
            object) to a :class:`ndarray`

        .. warning::

            As of the DLPack v0.3 specification, it is (implicitly) assumed
            that the user is responsible to ensure the Producer and the
            Consumer are operating on the same stream. This requirement might
            be relaxed/changed in a future DLPack version.

        .. admonition:: Example

            >>> import cupy
            >>> array1 = cupy.array([0, 1, 2], dtype=cupy.float32)
            >>> dltensor = array1.toDlpack()
            >>> array2 = cupy.fromDlpack(dltensor)
            >>> cupy.testing.assert_array_equal(array1, array2)

        """
        return dlpack.toDlpack(self)


cdef inline _carray.CArray _CArray_from_ndarray(ndarray arr):
    # Creates CArray from ndarray.
    # Note that this function cannot be defined in _carray.pxd because that
    # would cause cyclic cimport dependencies.
    cdef _carray.CArray carr = _carray.CArray.__new__(_carray.CArray)
    carr.init(<void*>arr.data.ptr, arr.size, arr._shape, arr._strides)
    return carr


_HANDLED_TYPES = (ndarray, numpy.ndarray)


# =============================================================================
# compile_with_cache
# =============================================================================
# TODO(niboshi): Move it out of core.pyx

cdef bint _is_hip = runtime._is_hip_environment
cdef int _cuda_runtime_version = -1
cdef str _cuda_path = ''  # '' for uninitialized, None for non-existing

cdef list cupy_header_list = [
    'cupy/complex.cuh',
    'cupy/carray.cuh',
    'cupy/atomics.cuh',
]
if _is_hip:
    cupy_header_list.append('cupy/math_constants.h')

# expose to Python for unit testing
_cupy_header_list = cupy_header_list

cdef str _cupy_header = ''.join(
    ['#include <%s>\n' % i for i in cupy_header_list])

# This is indirect include header list.
# These header files are subject to a hash key.
cdef list _cupy_extra_header_list = [
    'cupy/complex/complex.h',
    'cupy/complex/math_private.h',
    'cupy/complex/complex_inl.h',
    'cupy/complex/arithmetic.h',
    'cupy/complex/cproj.h',
    'cupy/complex/cexp.h',
    'cupy/complex/cexpf.h',
    'cupy/complex/clog.h',
    'cupy/complex/clogf.h',
    'cupy/complex/cpow.h',
    'cupy/complex/ccosh.h',
    'cupy/complex/ccoshf.h',
    'cupy/complex/csinh.h',
    'cupy/complex/csinhf.h',
    'cupy/complex/ctanh.h',
    'cupy/complex/ctanhf.h',
    'cupy/complex/csqrt.h',
    'cupy/complex/csqrtf.h',
    'cupy/complex/catrig.h',
    'cupy/complex/catrigf.h',
]

cdef str _header_path_cache = None
cdef str _header_source = None
cdef dict _header_source_map = {}


cpdef str _get_header_dir_path():
    global _header_path_cache
    if _header_path_cache is None:
        # Cython cannot use __file__ in global scope
        _header_path_cache = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'include'))
    return _header_path_cache


cpdef str _get_header_source():
    global _header_source
    global _header_source_map
    cdef str header_path, base_path, file_path, header
    cdef list source

    if _header_source is None or not _header_source_map:
        source = []
        base_path = _get_header_dir_path()
        for file_path in _cupy_extra_header_list + cupy_header_list:
            header_path = os.path.join(base_path, file_path)
            with open(header_path) as header_file:
                header = header_file.read()
            source.append(header)
            _header_source_map[file_path.encode()] = header.encode()
        _header_source = '\n'.join(source)
    return _header_source


cpdef dict _get_header_source_map():
    global _header_source_map
    if not _header_source_map:
        _get_header_source()
    return _header_source_map


# added at the module level for precompiling the regex
_cucomplex_include_tokens = ['', '#', 'include', '<', r'cuComplex\.h', '>']
_cucomplex_include_pattern = re.compile(r'\s*'.join(_cucomplex_include_tokens))


cdef inline str _translate_cucomplex_to_thrust(str source):
    lines = []
    for line in source.splitlines(keepends=True):
        if _cucomplex_include_pattern.match(line):
            lines += '#include <cupy/cuComplex_bridge.h>  '\
                     '// translate_cucomplex\n'
        else:
            lines += line
    return ''.join(lines)


cpdef function.Module compile_with_cache(
        str source, tuple options=(), arch=None, cachd_dir=None,
        prepend_cupy_headers=True, backend='nvrtc', translate_cucomplex=False,
        enable_cooperative_groups=False, name_expressions=None,
        log_stream=None, bint jitify=False):
    if translate_cucomplex:
        source = _translate_cucomplex_to_thrust(source)
        cupy_header_list.append('cupy/cuComplex_bridge.h')
        prepend_cupy_headers = True

    if prepend_cupy_headers:
        source = _cupy_header + source
    extra_source = _get_header_source()
    options += ('-I%s' % _get_header_dir_path(),)

    # The variable _cuda_runtime_version is declared in cupy/_core/core.pyx,
    # but it might not have been set appropriately before coming here.
    global _cuda_runtime_version
    if _cuda_runtime_version < 0:
        _cuda_runtime_version = runtime.runtimeGetVersion()

    global _cuda_path
    if _cuda_path == '':
        if not _is_hip:
            _cuda_path = cuda.get_cuda_path()
        else:
            _cuda_path = cuda.get_rocm_path()

    if not _is_hip:
        if 9020 <= _cuda_runtime_version < 9030:
            bundled_include = 'cuda-9.2'
        elif 10000 <= _cuda_runtime_version < 10010:
            bundled_include = 'cuda-10.0'
        elif 10010 <= _cuda_runtime_version < 10020:
            bundled_include = 'cuda-10.1'
        elif 10020 <= _cuda_runtime_version < 10030:
            bundled_include = 'cuda-10.2'
        elif 11000 <= _cuda_runtime_version < 11010:
            bundled_include = 'cuda-11.0'
        elif 11010 <= _cuda_runtime_version < 11020:
            bundled_include = 'cuda-11.1'
        elif 11020 <= _cuda_runtime_version < 11030:
            bundled_include = 'cuda-11.2'
        else:
            # CUDA versions not yet supported.
            bundled_include = None

        if bundled_include is None and _cuda_path is None:
            raise RuntimeError(
                'Failed to auto-detect CUDA root directory. '
                'Please specify `CUDA_PATH` environment variable if you '
                'are using CUDA versions not yet supported by CuPy.')

        if bundled_include is not None:
            options += ('-I' + os.path.join(
                _get_header_dir_path(), 'cupy', '_cuda', bundled_include),)
    elif _is_hip:
        if _cuda_path is None:
            raise RuntimeError(
                'Failed to auto-detect ROCm root directory. '
                'Please specify `ROCM_HOME` environment variable.')

    if _cuda_path is not None:
        options += ('-I' + os.path.join(_cuda_path, 'include'),)

    return cuda.compile_with_cache(
        source, options, arch, cachd_dir, extra_source, backend,
        enable_cooperative_groups=enable_cooperative_groups,
        name_expressions=name_expressions, log_stream=log_stream,
        jitify=jitify)


# =============================================================================
# Routines
# =============================================================================

cdef str _id = 'out0 = in0'

cdef fill_kernel = ElementwiseKernel('T x', 'T y', 'y = x', 'fill')

cdef str _divmod_float = '''
    out0_type a = _floor_divide(in0, in1);
    out0 = a;
    out1 = in0 - a * in1'''


divmod = create_ufunc(
    'cupy_divmod',
    ('bb->bb', 'BB->BB', 'hh->hh', 'HH->HH', 'ii->ii', 'II->II', 'll->ll',
     'LL->LL', 'qq->qq', 'QQ->QQ',
     ('ee->ee', _divmod_float),
     ('ff->ff', _divmod_float),
     ('dd->dd', _divmod_float)),
    '''
    if (in1 == 0) {
        out0 = 0;
        out1 = 0;
    } else {
        out0_type a = _floor_divide(in0, in1);
        out0 = a;
        out1 = in0 - a * in1;
    }''')


cdef _round_preamble = '''
#ifdef __HIP_DEVICE_COMPILE__
#define round_float llrintf
#else
#define round_float __float2ll_rn
#endif

template<typename T> __device__ T pow10(long long n){
  T x = 1, a = 10;
  while (n) {
    if (n & 1) x *= a;
    a *= a;
    n >>= 1;
  }
  return x;
};
'''


cdef _round_float = '''
if (in1 == 0) {
    out0 = rint(in0);
} else {
    double x;
    x = pow10<double>(abs(in1));  // TODO(okuta): Move before loop
    out0 = in1 < 0 ? rint(in0 / x) * x : rint(in0 * x) / x;
}'''

cdef _round_complex = '''
if (in1 == 0) {
    out0 = in0_type(rint(in0.real()), rint(in0.imag()));
} else {
    double x = pow10<double>(abs(in1));  // TODO(okuta): Move before loop
    if (in1 < 0) {
        out0 = in0_type(rint(in0.real() / x) * x,
                        rint(in0.imag() / x) * x);
    } else {
        out0 = in0_type(rint(in0.real() * x) / x,
                        rint(in0.imag() * x) / x);
    }
}'''


# There is a known incompatibility with NumPy (as of 1.16.4) such as
# `numpy.around(2**63, -1) == cupy.around(2**63, -1)` gives `False`.
#
# NumPy seems to round integral values via double.  As double has
# only 53 bit precision, last few bits of (u)int64 value may be lost.
# As a consequence, `numpy.around(2**63, -1)` does NOT round up the
# last digit (9223372036854775808 instead of ...810).
#
# The following code fixes the problem, so `cupy.around(2**63, -1)`
# gives `...810`, which (may correct but) is incompatible with NumPy.
_round_ufunc = create_ufunc(
    'cupy_round',
    ('?q->e',
     'bq->b', 'Bq->B', 'hq->h', 'Hq->H', 'iq->i', 'Iq->I', 'lq->l', 'Lq->L',
     'qq->q', 'Qq->Q',
     ('eq->e', _round_float),
     ('fq->f', _round_float),
     ('dq->d', _round_float),
     ('Fq->F', _round_complex),
     ('Dq->D', _round_complex)),
    '''
    if (in1 >= 0) {
        out0 = in0;
    } else {
        // TODO(okuta): Move before loop
        long long x = pow10<long long>(-in1 - 1);

        // TODO(okuta): Check Numpy
        // `cupy.around(-123456789, -4)` works as follows:
        // (1) scale by `x` above: -123456.789
        // (2) split at the last 2 digits: -123400 + (-5.6789 * 10)
        // (3) round the latter by `rint()`: -123400 + (-6.0 * 10)
        // (4) unscale by `x` above: -123460000
        long long q = in0 / x / 100;
        int r = in0 - q*x*100;
        out0 = (q*100 + round_float(r/(x*10.0f))*10) * x;
    }''', preamble=_round_preamble)


# -----------------------------------------------------------------------------
# Array creation routines
# -----------------------------------------------------------------------------

cpdef ndarray array(obj, dtype=None, bint copy=True, order='K',
                    bint subok=False, Py_ssize_t ndmin=0):
    # TODO(beam2d): Support subok options
    cdef Py_ssize_t ndim
    cdef ndarray a, src
    cdef size_t nbytes

    if subok:
        raise NotImplementedError
    if order is None:
        order = 'K'

    if isinstance(obj, ndarray):
        src = obj
        if dtype is None:
            dtype = src.dtype
        if src.data.device_id == device.get_device_id():
            a = src.astype(dtype, order=order, copy=copy)
        else:
            a = src.copy(order=order).astype(dtype, copy=False)

        ndim = a._shape.size()
        if ndmin > ndim:
            if a is obj:
                # When `copy` is False, `a` is same as `obj`.
                a = a.view()
            a.shape = (1,) * (ndmin - ndim) + a.shape
        return a

    if hasattr(obj, '__cuda_array_interface__'):
        return array(_convert_object_with_cuda_array_interface(obj),
                     dtype, copy, order, subok, ndmin)

    # obj is sequence, numpy array, scalar or the other type of object
    shape, elem_type, elem_dtype = _get_concat_shape(obj)
    if shape is not None and shape[-1] != 0:
        # obj is a non-empty sequence of ndarrays which share same shape
        # and dtype

        # resulting array is C order unless 'F' is explicitly specified
        # (i.e., it ignores order of element arrays in the sequence)
        order = (
            'F'
            if order is not None and len(order) >= 1 and order[0] in 'Ff'
            else 'C')
        ndim = len(shape)
        if ndmin > ndim:
            shape = (1,) * (ndmin - ndim) + shape

        if dtype is None:
            dtype = elem_dtype
        # Note: dtype might not be numpy.dtype in this place

        if issubclass(elem_type, numpy.ndarray):
            # obj is Seq[numpy.ndarray]
            return _send_numpy_array_list_to_gpu(
                obj, elem_dtype, dtype, shape, order, ndmin)

        # obj is Seq[cupy.ndarray]
        assert issubclass(elem_type, ndarray), elem_type
        lst = _flatten_list(obj)

        # convert each scalar (0-dim) ndarray to 1-dim
        lst = [cupy.expand_dims(x, 0) if x.ndim == 0 else x for x in lst]

        a =_manipulation.concatenate_method(lst, 0)
        a = a.reshape(shape)
        a = a.astype(dtype, order=order, copy=False)
        return a

    # obj is:
    # - numpy array
    # - scalar or sequence of scalar
    # - empty sequence or sequence with elements whose shapes or
    #   dtypes are unmatched
    # - other types

    # fallback to numpy array and send it to GPU
    # Note: dtype might not be numpy.dtype in this place
    return _send_object_to_gpu(obj, dtype, order, ndmin)


cdef tuple _get_concat_shape(object obj):
    # Returns a tuple of the following:
    # 1. concatenated shape if it can be converted to a single CuPy array by
    #    just concatenating it (i.e., the object is a (nested) sequence only
    #    which contains NumPy/CuPy array(s) with same shape and dtype).
    #    Returns None otherwise.
    # 2. type of the first item in the object
    # 3. dtype if the object is an array
    if isinstance(obj, (list, tuple)):
        return _get_concat_shape_impl(obj)
    return (None, None, None)


cdef tuple _get_concat_shape_impl(object obj):
    cdef obj_type = type(obj)
    if issubclass(obj_type, (numpy.ndarray, ndarray)):
        # obj.shape is () when obj.ndim == 0
        return (obj.shape, obj_type, obj.dtype)
    if isinstance(obj, (list, tuple)):
        shape = None
        typ = None
        dtype = None
        for elem in obj:
            # Find the head recursively if obj is a nested built-in list
            elem_shape, elem_type, elem_dtype = _get_concat_shape_impl(elem)

            # Use shape of the first element as the common shape.
            if shape is None:
                shape = elem_shape
                typ = elem_type
                dtype = elem_dtype

            # `elem` is not concatable or the shape and dtype does not match
            # with siblings.
            if elem_shape is None or shape != elem_shape:
                return (None, obj_type, None)

            if dtype != elem_dtype:
                dtype = numpy.promote_types(dtype, elem_dtype)

        if shape is None:
            shape = ()
        return (
            (len(obj),) + shape,
            typ,
            dtype)
    # scalar or object
    return (None, obj_type, None)


cdef list _flatten_list(object obj):
    ret = []
    if isinstance(obj, (list, tuple)):
        for elem in obj:
            ret += _flatten_list(elem)
        return ret
    return [obj]


cdef ndarray _send_object_to_gpu(obj, dtype, order, Py_ssize_t ndmin):
    if order is not None and len(order) >= 1 and order[0] in 'KAka':
        if isinstance(obj, numpy.ndarray) and obj.flags.f_contiguous:
            order = 'F'
        else:
            order = 'C'
    a_cpu = numpy.array(obj, dtype=dtype, copy=False, order=order,
                        ndmin=ndmin)
    a_dtype = a_cpu.dtype  # converted to numpy.dtype
    if a_dtype.char not in '?bhilqBHILQefdFD':
        raise ValueError('Unsupported dtype %s' % a_dtype)
    cdef shape_t a_shape = a_cpu.shape
    cdef ndarray a = ndarray(a_shape, dtype=a_dtype, order=order)
    if a_cpu.ndim == 0:
        a.fill(a_cpu)
        return a
    cdef Py_ssize_t nbytes = a.nbytes

    stream = stream_module.get_current_stream()
    cdef pinned_memory.PinnedMemoryPointer mem = (
        _alloc_async_transfer_buffer(nbytes))
    if mem is not None:
        src_cpu = numpy.frombuffer(mem, a_dtype, a_cpu.size)
        src_cpu[:] = a_cpu.ravel(order)
        a.data.copy_from_host_async(mem.ptr, nbytes)
        pinned_memory._add_to_watch_list(stream.record(), mem)
    else:
        a.data.copy_from_host(a_cpu.ctypes.data, nbytes)

    return a


cdef ndarray _send_numpy_array_list_to_gpu(
        list arrays, src_dtype, dst_dtype,
        const shape_t& shape,
        order, Py_ssize_t ndmin):

    a_dtype = get_dtype(dst_dtype)  # convert to numpy.dtype
    if a_dtype.char not in '?bhilqBHILQefdFD':
        raise ValueError('Unsupported dtype %s' % a_dtype)
    cdef ndarray a  # allocate it after pinned memory is secured
    cdef size_t itemcount = internal.prod(shape)
    cdef size_t nbytes = itemcount * a_dtype.itemsize

    stream = stream_module.get_current_stream()
    cdef pinned_memory.PinnedMemoryPointer mem = (
        _alloc_async_transfer_buffer(nbytes))
    cdef size_t offset, length
    if mem is not None:
        # write concatenated arrays to the pinned memory directly
        src_cpu = (
            numpy.frombuffer(mem, a_dtype, itemcount)
            .reshape(shape, order=order))
        _concatenate_numpy_array(
            [numpy.expand_dims(e, 0) for e in arrays],
            0,
            get_dtype(src_dtype),
            a_dtype,
            src_cpu)
        a = ndarray(shape, dtype=a_dtype, order=order)
        a.data.copy_from_host_async(mem.ptr, nbytes)
        pinned_memory._add_to_watch_list(stream.record(), mem)
    else:
        # fallback to numpy array and send it to GPU
        # Note: a_cpu.ndim is always >= 1
        a_cpu = numpy.array(arrays, dtype=a_dtype, copy=False, order=order,
                            ndmin=ndmin)
        a = ndarray(shape, dtype=a_dtype, order=order)
        a.data.copy_from_host(a_cpu.ctypes.data, nbytes)

    return a


cdef bint _numpy_concatenate_has_out_argument = (
    numpy.lib.NumpyVersion(numpy.__version__) >= '1.14.0')


cdef inline _concatenate_numpy_array(arrays, axis, src_dtype, dst_dtype, out):
    # type(*_dtype) must be numpy.dtype

    if (_numpy_concatenate_has_out_argument
            and src_dtype.kind == dst_dtype.kind):
        # concatenate only accepts same_kind casting
        numpy.concatenate(arrays, axis, out)
    else:
        out[:] = numpy.concatenate(arrays, axis)


cdef inline _alloc_async_transfer_buffer(Py_ssize_t nbytes):
    try:
        return pinned_memory.alloc_pinned_memory(nbytes)
    except CUDARuntimeError as e:
        if e.status != runtime.errorMemoryAllocation:
            raise
        warnings.warn(
            'Using synchronous transfer as pinned memory ({} bytes) '
            'could not be allocated. '
            'This generally occurs because of insufficient host memory. '
            'The original error was: {}'.format(nbytes, e),
            _util.PerformanceWarning)

    return None


cpdef ndarray _internal_ascontiguousarray(ndarray a):
    if a._c_contiguous:
        return a
    newarray = _ndarray_init(a._shape, a.dtype)
    elementwise_copy(a, newarray)
    return newarray


cpdef ndarray _internal_asfortranarray(ndarray a):
    cdef ndarray newarray
    cdef int m, n
    cdef intptr_t handle

    if a._f_contiguous:
        return a

    newarray = ndarray(a.shape, a.dtype, order='F')
    if (a._c_contiguous and a._shape.size() == 2 and
            (a.dtype == numpy.float32 or a.dtype == numpy.float64)):
        m, n = a.shape
        handle = device.get_cublas_handle()
        one = numpy.array(1, dtype=a.dtype)
        zero = numpy.array(0, dtype=a.dtype)
        if a.dtype == numpy.float32:
            cublas.sgeam(
                handle,
                1,  # transpose a
                1,  # transpose newarray
                m, n, one.ctypes.data, a.data.ptr, n,
                zero.ctypes.data, a.data.ptr, n, newarray.data.ptr, m)
        elif a.dtype == numpy.float64:
            cublas.dgeam(
                handle,
                1,  # transpose a
                1,  # transpose newarray
                m, n, one.ctypes.data, a.data.ptr, n,
                zero.ctypes.data, a.data.ptr, n, newarray.data.ptr, m)
    else:
        elementwise_copy(a, newarray)
    return newarray


cpdef ndarray ascontiguousarray(ndarray a, dtype=None):
    cdef bint same_dtype = False
    zero_dim = a._shape.size() == 0
    if dtype is None:
        same_dtype = True
        dtype = a.dtype
    else:
        dtype = get_dtype(dtype)
        same_dtype = dtype == a.dtype

    if same_dtype and a._c_contiguous:
        if zero_dim:
            return _manipulation._ndarray_ravel(a, 'C')
        return a

    shape = (1,) if zero_dim else a.shape
    newarray = ndarray(shape, dtype)
    elementwise_copy(a, newarray)
    return newarray


cpdef ndarray asfortranarray(ndarray a, dtype=None):
    cdef ndarray newarray
    cdef int m, n
    cdef bint same_dtype = False
    zero_dim = a._shape.size() == 0

    if dtype is None:
        dtype = a.dtype
        same_dtype = True
    else:
        dtype = get_dtype(dtype)
        same_dtype = dtype == a.dtype

    if same_dtype and a._f_contiguous:
        if zero_dim:
            return _manipulation._ndarray_ravel(a, 'F')
        return a

    if same_dtype and not zero_dim:
        return _internal_asfortranarray(a)

    newarray = ndarray((1,) if zero_dim else a.shape, dtype, order='F')
    elementwise_copy(a, newarray)
    return newarray


cpdef ndarray _convert_object_with_cuda_array_interface(a):
    if runtime._is_hip_environment:
        raise RuntimeError(
            'HIP/ROCm does not support cuda array interface')

    cdef Py_ssize_t sh, st
    cdef dict desc = a.__cuda_array_interface__
    cdef tuple shape = desc['shape']
    cdef int dev_id = -1
    cdef size_t nbytes

    ptr = desc['data'][0]
    dtype = numpy.dtype(desc['typestr'])
    mask = desc.get('mask')
    if mask is not None:
        raise ValueError('CuPy currently does not support masked arrays.')
    strides = desc.get('strides')
    if strides is not None:
        nbytes = 0
        for sh, st in zip(shape, strides):
            nbytes = max(nbytes, abs(sh * st))
    else:
        nbytes = internal.prod_sequence(shape) * dtype.itemsize
    # the v2 protocol sets ptr=0 for 0-size arrays, so we can't look up
    # the pointer attributes and must use the current device
    if nbytes == 0:
        dev_id = device.get_device_id()
    mem = memory_module.UnownedMemory(ptr, nbytes, a, dev_id)
    memptr = memory.MemoryPointer(mem, 0)
    # the v3 protocol requires an immediate synchronization, unless
    # 1. the stream is not set (ex: from v0 ~ v2) or is None
    # 2. users explicitly overwrite this requirement
    stream_ptr = desc.get('stream')
    if stream_ptr is not None:
        if _util.CUDA_ARRAY_INTERFACE_SYNC:
            runtime.streamSynchronize(stream_ptr)
    return ndarray(shape, dtype, memptr, strides)


cdef ndarray _ndarray_init(const shape_t& shape, dtype):
    cdef ndarray ret = ndarray.__new__(ndarray)
    ret._init_fast(shape, dtype, True)
    return ret


cdef ndarray _create_ndarray_from_shape_strides(
        const shape_t& shape, const strides_t& strides, dtype):
    cdef int ndim = shape.size()
    cdef int64_t begin = 0, end = dtype.itemsize
    cdef memory.MemoryPointer ptr
    for i in range(ndim):
        if strides[i] > 0:
            end += strides[i] * (shape[i] - 1)
        elif strides[i] < 0:
            begin += strides[i] * (shape[i] - 1)
    ptr = memory.alloc(end - begin) + begin
    return ndarray(shape, dtype, memptr=ptr, strides=strides)
