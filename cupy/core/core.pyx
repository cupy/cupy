# distutils: language = c++

from __future__ import division
import sys

import ctypes
import numpy
import six

import cupy
from cupy.core import _errors
from cupy.core._kernel import create_ufunc
from cupy.core._kernel import ElementwiseKernel
from cupy.core._kernel import ReductionKernel
from cupy.core._kernel import ufunc  # NOQA
from cupy.core._ufuncs import elementwise_copy
from cupy.core._ufuncs import elementwise_copy_where
from cupy.core import flags
from cupy.cuda import device
from cupy.cuda import memory as memory_module


from cupy import util
from cupy.cuda.runtime import CUDARuntimeError

cimport cpython  # NOQA
cimport cython  # NOQA
from libcpp cimport vector

from cupy.core cimport _dtype
from cupy.core._dtype cimport get_dtype
from cupy.core._kernel cimport create_ufunc
from cupy.core cimport _routines_indexing as _indexing
from cupy.core cimport _routines_logic as _logic
from cupy.core cimport _routines_manipulation as _manipulation
from cupy.core cimport _routines_math as _math
from cupy.core cimport _routines_sorting as _sorting
from cupy.core cimport _routines_statistics as _statistics
from cupy.core cimport dlpack
from cupy.core cimport internal
from cupy.cuda cimport cublas
from cupy.cuda cimport function
from cupy.cuda cimport pinned_memory
from cupy.cuda cimport runtime
from cupy.cuda cimport memory
from cupy.cuda cimport stream as stream_module


DEF MAX_NDIM = 25


@cython.profile(False)
cdef inline _should_use_rop(x, y):
    xp = getattr(x, '__array_priority__', 0)
    yp = getattr(y, '__array_priority__', 0)
    return xp < yp and not isinstance(y, ndarray)


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
            raise TypeError('order not understood. order=%s' % order)

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
        else:
            self.data = memptr

    @property
    def __cuda_array_interface__(self):
        desc = {
            'shape': self.shape,
            'typestr': self.dtype.str,
            'descr': self.dtype.descr,
            'data': (self.data.ptr, False),
            'version': 0,
        }
        if not self._c_contiguous:
            desc['strides'] = self.strides

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
        return CArray(self)

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
    # TODO(okuta): Implement tobytes

    cpdef tofile(self, fid, sep='', format='%s'):
        """Writes the array to a file.

        .. seealso:: :meth:`numpy.ndarray.tolist`

        """
        self.get().tofile(fid, sep, format)

    cpdef dump(self, file):
        """Dumps a pickle of the array to a file.

        Dumped file can be read back to :class:`cupy.ndarray` by
        :func:`cupy.load`.

        """
        six.moves.cPickle.dump(self, file, -1)

    cpdef dumps(self):
        """Dumps a pickle of the array to a string."""
        return six.moves.cPickle.dumps(self, -1)

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
        cdef vector.vector[Py_ssize_t] strides
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

        order_char = _update_order_char(self, order_char)

        if order_char == b'K':
            strides = _get_strides_for_order_K(self, dtype)
            newarray = ndarray(self.shape, dtype=dtype)
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
        newarray = ndarray(x.shape, dtype=x.dtype)
        if not x._c_contiguous and not x._f_contiguous:
            raise NotImplementedError(
                'CuPy cannot copy non-contiguous array between devices.')
        # TODO(niboshi): Confirm update_x_contiguity flags
        newarray._strides = x._strides
        newarray._c_contiguous = x._c_contiguous
        newarray._f_contiguous = x._f_contiguous
        newarray.data.copy_from_device(x.data, x.nbytes)
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

    # TODO(okuta): Implement searchsorted

    cpdef tuple nonzero(self):
        """Return the indices of the elements that are non-zero.

        Returned Array is containing the indices of the non-zero elements
        in that dimension.

        Returns:
            tuple of arrays: Indices of elements that are non-zero.

        .. seealso::
            :func:`numpy.nonzero`

        """
        return _indexing._ndarray_nonzero(self)

    # TODO(okuta): Implement compress

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
    cpdef ndarray max(self, axis=None, out=None, dtype=None, keepdims=False):
        """Returns the maximum along a given axis.

        .. seealso::
           :func:`cupy.amax` for full documentation,
           :meth:`numpy.ndarray.max`

        """
        return _statistics._ndarray_max(self, axis, out, dtype, keepdims)

    cpdef ndarray argmax(self, axis=None, out=None, dtype=None,
                         keepdims=False):
        """Returns the indices of the maximum along a given axis.

        .. seealso::
           :func:`cupy.argmax` for full documentation,
           :meth:`numpy.ndarray.argmax`

        """
        return _statistics._ndarray_argmax(self, axis, out, dtype, keepdims)

    cpdef ndarray _nanargmax(self, axis=None, out=None, dtype=None,
                             keepdims=False):
        """Returns the indices of the maximum with nan along a given axis.

        .. seealso::
           :func:`cupy.nanargmax` for full documentation,
           :meth:`numpy.ndarray.nanargmax`

        """
        return _statistics._ndarray_nanargmax(self, axis, out, dtype, keepdims)

    cpdef ndarray min(self, axis=None, out=None, dtype=None, keepdims=False):
        """Returns the minimum along a given axis.

        .. seealso::
           :func:`cupy.amin` for full documentation,
           :meth:`numpy.ndarray.min`

        """
        return _statistics._ndarray_min(self, axis, out, dtype, keepdims)

    cpdef ndarray argmin(self, axis=None, out=None, dtype=None,
                         keepdims=False):
        """Returns the indices of the minimum along a given axis.

        .. seealso::
           :func:`cupy.argmin` for full documentation,
           :meth:`numpy.ndarray.argmin`

        """
        return _statistics._ndarray_argmin(self, axis, out, dtype, keepdims)

    cpdef ndarray _nanargmin(self, axis=None, out=None, dtype=None,
                             keepdims=False):
        """Returns the indices of the minimum with nan along a given axis.

        .. seealso::
           :func:`cupy.nanargmin` for full documentation,
           :meth:`numpy.ndarray.nanargmin`

        """
        return _statistics._ndarray_nanargmin(self, axis, out, dtype, keepdims)
    # TODO(okuta): Implement ptp

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

    cpdef ndarray _nansum(
            self, axis=None, dtype=None, out=None, keepdims=False):
        """Returns the sum along a given axis treating Not a Numbers (NaNs) as zero.

        .. seealso::
           :func:`cupy.nansum` for full documentation,
           :meth:`numpy.ndarray.nansum`

        """
        return _math._ndarray_nansum(self, axis, dtype, out, keepdims)

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

    cpdef ndarray _nanprod(
            self, axis=None, dtype=None, out=None, keepdims=None):
        """Returns the product along a given axis treating Not a Numbers (NaNs)
        as zero.

        .. seealso::
           :func:`cupy.nanprod` for full documentation,
           :meth:`numpy.ndarray.nanprod`

        """
        return _math._ndarray_nanprod(self, axis, dtype, out, keepdims)

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
        if isinstance(other, numpy.ndarray) and other.ndim == 0:
            other = other.item()  # Workaround for numpy<1.13
        if op == 0:
            return less(self, other)
        if op == 1:
            return less_equal(self, other)
        if op == 2:
            return equal(self, other)
        if op == 3:
            return not_equal(self, other)
        if op == 4:
            return greater(self, other)
        if op == 5:
            return greater_equal(self, other)
        return NotImplemented

    # Truth value of an array (bool):

    def __nonzero__(self):
        if self.size == 0:
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
        return invert(self)

    # Arithmetic:

    def __add__(x, y):
        if _should_use_rop(x, y):
            return y.__radd__(x)
        else:
            return _math._add(x, y)

    def __sub__(x, y):
        if _should_use_rop(x, y):
            return y.__rsub__(x)
        else:
            return _math._subtract(x, y)

    def __mul__(x, y):
        if _should_use_rop(x, y):
            return y.__rmul__(x)
        else:
            return _math._multiply(x, y)

    def __matmul__(x, y):
        if _should_use_rop(x, y):
            return y.__rmatmul__(x)
        else:
            return matmul(x, y)

    def __div__(x, y):
        if _should_use_rop(x, y):
            return y.__rdiv__(x)
        else:
            return _math._divide(x, y)

    def __truediv__(x, y):
        if _should_use_rop(x, y):
            return y.__rtruediv__(x)
        else:
            return _math._true_divide(x, y)

    def __floordiv__(x, y):
        if _should_use_rop(x, y):
            return y.__rfloordiv__(x)
        else:
            return _math._floor_divide(x, y)

    def __mod__(x, y):
        if _should_use_rop(x, y):
            return y.__rmod__(x)
        else:
            return _math._remainder(x, y)

    def __divmod__(x, y):
        if _should_use_rop(x, y):
            return y.__rdivmod__(x)
        else:
            return divmod(x, y)

    def __pow__(x, y, modulo):
        # Note that we ignore the modulo argument as well as NumPy.
        if _should_use_rop(x, y):
            return y.__rpow__(x)
        else:
            return _math._power(x, y)

    def __lshift__(x, y):
        if _should_use_rop(x, y):
            return y.__rlshift__(x)
        else:
            return left_shift(x, y)

    def __rshift__(x, y):
        if _should_use_rop(x, y):
            return y.__rrshift__(x)
        else:
            return right_shift(x, y)

    def __and__(x, y):
        if _should_use_rop(x, y):
            return y.__rand__(x)
        else:
            return bitwise_and(x, y)

    def __or__(x, y):
        if _should_use_rop(x, y):
            return y.__ror__(x)
        else:
            return bitwise_or(x, y)

    def __xor__(x, y):
        if _should_use_rop(x, y):
            return y.__rxor__(x)
        else:
            return bitwise_xor(x, y)

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
        return left_shift(self, other, self)

    def __irshift__(self, other):
        return right_shift(self, other, self)

    def __iand__(self, other):
        return bitwise_and(self, other, self)

    def __ior__(self, other):
        return bitwise_or(self, other, self)

    def __ixor__(self, other):
        return bitwise_xor(self, other, self)

    cpdef ndarray conj(self):
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
        if dtype is None or self.dtype == dtype:
            return self
        else:
            return self.astype(dtype)

    # TODO(okuta): Implement __array_wrap__

    # Container customization:

    def __iter__(self):
        if self._shape.size() == 0:
            raise TypeError('iteration over a 0-d array')
        return (self[i] for i in six.moves.range(self._shape[0]))

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
            >>> v = cupy.arange(10000).astype(cupy.float)
            >>> a[i] = v
            >>> a  # doctest: +SKIP
            array([9150., 9151.])

            On the other hand, NumPy stores the value corresponding to the
            last index among the indices referencing duplicate locations.

            >>> import numpy
            >>> a_cpu = numpy.zeros((2,))
            >>> i_cpu = numpy.arange(10000) % 2
            >>> v_cpu = numpy.arange(10000).astype(numpy.float)
            >>> a_cpu[i_cpu] = v_cpu
            >>> a_cpu
            array([9998., 9999.])

        """
        if (util.ENABLE_SLICE_COPY and slices == slice(None, None, None) and
                isinstance(value, numpy.ndarray)):
            if (self.dtype == value.dtype and
                    self.shape == value.shape):
                if self.strides == value.strides:
                    ptr = ctypes.c_void_p(value.__array_interface__['data'][0])
                else:
                    order = 'F' if self.flags.f_contiguous else 'C'
                    tmp = value.ravel(order)
                    ptr = ctypes.c_void_p(tmp.__array_interface__['data'][0])
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
        if 'out' in kwargs:
            # need to unfold tuple argument in kwargs
            out = kwargs['out']
            if len(out) != 1:
                raise ValueError('The \'out\' parameter must have exactly one '
                                 'array value')
            kwargs['out'] = out[0]

        if method == '__call__':
            if ufunc.signature is not None:
                # we don't support generalised-ufuncs (gufuncs)
                return NotImplemented
            name = ufunc.__name__
            try:
                cp_ufunc = getattr(cupy, name)
            except AttributeError:
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
            # http://docs-cupy.chainer.org/en/stable/reference/ufunc.html#ufunc-at
            # self.scatter_add(*inputs, **kwargs)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        module = cupy
        for submodule in func.__module__.split('.')[1:]:
            try:
                module = getattr(module, submodule)
            except AttributeError:
                return NotImplemented
        if not hasattr(module, func.__name__):
            return NotImplemented
        cupy_func = getattr(module, func.__name__)
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

    if sys.version_info < (3,):
        def __long__(self):
            # Avoid using long() for flake8
            return self.get().__long__()

    def __float__(self):
        return float(self.get())

    def __complex__(self):
        return complex(self.get())

    def __oct__(self):
        return oct(self.get())

    def __hex__(self):
        return hex(self.get())

    # String representations:

    def __repr__(self):
        return repr(self.get())

    def __str__(self):
        return str(self.get())

    # -------------------------------------------------------------------------
    # Methods outside of the ndarray main documentation
    # -------------------------------------------------------------------------
    def dot(self, ndarray b, ndarray out=None):
        """Returns the dot product with given array.

        .. seealso::
           :func:`cupy.dot` for full documentation,
           :meth:`numpy.ndarray.dot`

        """
        return dot(self, b, out)

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
        ptr = ctypes.c_void_p(a_cpu.__array_interface__['data'][0])
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

        ptr = ctypes.c_void_p(arr.__array_interface__['data'][0])
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
            dtype: Data type specifier. If it is given, then the memory
                sequence is reinterpreted as the new type.

        Returns:
            cupy.ndarray: A view of the array with reduced dimensions.

        """
        cdef vector.vector[Py_ssize_t] shape, strides
        cdef Py_ssize_t ndim
        ndim = self._shape.size()
        if ndim <= 1:
            return self
        internal.get_reduced_dims(
            self._shape, self._strides, self.itemsize, shape, strides)
        if ndim == <Py_ssize_t>shape.size():
            return self

        view = self.view(dtype=dtype)
        # TODO(niboshi): Confirm update_x_contiguity flags
        view._set_shape_and_strides(shape, strides, True, True)
        return view

    cpdef _update_c_contiguity(self):
        if self.size == 0:
            self._c_contiguous = True
            return
        self._c_contiguous = internal.get_c_contiguity(
            self._shape, self._strides, self.itemsize)

    cpdef _update_f_contiguity(self):
        cdef Py_ssize_t i, count
        cdef vector.vector[Py_ssize_t] rev_shape, rev_strides
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
            rev_shape, rev_strides, self.itemsize)

    cpdef _update_contiguity(self):
        self._update_c_contiguity()
        self._update_f_contiguity()

    cpdef _set_shape_and_strides(self, const vector.vector[Py_ssize_t]& shape,
                                 const vector.vector[Py_ssize_t]& strides,
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

    cdef ndarray _view(self, const vector.vector[Py_ssize_t]& shape,
                       const vector.vector[Py_ssize_t]& strides,
                       bint update_c_contiguity,
                       bint update_f_contiguity):
        cdef ndarray v
        v = ndarray.__new__(ndarray)
        v.data = self.data
        v.base = self.base if self.base is not None else self
        v.dtype = self.dtype
        v._c_contiguous = self._c_contiguous
        v._f_contiguous = self._f_contiguous
        v._set_shape_and_strides(
            shape, strides, update_c_contiguity, update_f_contiguity)
        return v

    cpdef _set_contiguous_strides(
            self, Py_ssize_t itemsize, bint is_c_contiguous):
        self.size = internal.set_contiguous_strides(
            self._shape, self._strides, itemsize, is_c_contiguous)
        if is_c_contiguous:
            self._c_contiguous = True
            self._update_f_contiguity()
        else:
            self._f_contiguous = True
            self._update_c_contiguity()

    cdef function.CPointer get_pointer(self):
        return CArray(self)

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

        .. admonition:: Example

            >>> import cupy
            >>> array1 = cupy.array([0, 1, 2], dtype=cupy.float32)
            >>> dltensor = array1.toDlpack()
            >>> array2 = cupy.fromDlpack(dltensor)
            >>> cupy.testing.assert_array_equal(array1, array2)

        """
        return dlpack.toDlpack(self)


cpdef int _update_order_char(ndarray x, int order_char):
    # update order_char based on array contiguity
    if order_char == b'A':
        if x._f_contiguous:
            order_char = b'F'
        else:
            order_char = b'C'
    elif order_char == b'K':
        if x._f_contiguous:
            order_char = b'F'
        elif x._c_contiguous:
            order_char = b'C'
    return order_char


cpdef vector.vector[Py_ssize_t] _get_strides_for_order_K(ndarray x, dtype):
    cdef vector.vector[Py_ssize_t] strides
    # strides used when order='K' for astype, empty_like, etc.
    stride_and_index = [
        (abs(s), -i) for i, s in enumerate(x.strides)]
    stride_and_index.sort()
    strides.resize(x.ndim)
    stride = dtype.itemsize
    for s, i in stride_and_index:
        strides[-i] = stride
        stride *= x.shape[-i]
    return strides


_HANDLED_TYPES = (ndarray, numpy.ndarray)


include 'carray.pxi'


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
    out0 = round(in0);
} else {
    double x;
    x = pow10<double>(abs(in1));  // TODO(okuta): Move before loop
    out0 = in1 < 0 ? round(in0 / x) * x : round(in0 * x) / x;
}'''

cdef _round_complex = '''
double x, inv_x;
if (in1 == 0) {
    x = inv_x = 1;
} else {
    x = pow10<double>(abs(in1));  // TODO(okuta): Move before loop
    inv_x = 1.0 / x;
    if (in1 < 0) {
        double y = x;
        x = inv_x;
        inv_x = y;
    }
}
out0 = in0_type(round(in0.real() * x) * inv_x,
                round(in0.imag() * x) * inv_x);'''


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
    if (in1 < 0) {
        // TODO(okuta): Move before loop
        long long x = pow10<long long>(-in1 - 1);
        // TODO(okuta): Check Numpy
        out0 = ((in0 / x + (in0 > 0 ? 5 : -5)) / 10) * x * 10;
    } else {
        out0 = in0;
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
    elif hasattr(obj, '__cuda_array_interface__'):
        return array(_convert_object_with_cuda_array_interface(obj),
                     dtype, copy, order, subok, ndmin)
    else:  # obj is sequence, numpy array, scalar or the other type of object
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
                a = _send_numpy_array_list_to_gpu(
                    obj, elem_dtype, dtype, shape, order, ndmin)
            elif issubclass(elem_type, ndarray):
                # obj is Seq[cupy.ndarray]
                lst = _flatten_list(obj)
                if len(shape) == 1:
                    # convert each scalar (0-dim) ndarray to 1-dim
                    lst = [cupy.expand_dims(x, 0) for x in lst]

                a = (_manipulation.concatenate_method(lst, 0)
                                  .reshape(shape)
                                  .astype(dtype, order=order, copy=False))
            else:
                # should not be reached here
                assert issubclass(elem_type, (numpy.ndarray, ndarray))
        else:
            # obj is:
            # - numpy array
            # - scalar or sequence of scalar
            # - empty sequence or sequence with elements whose shapes or
            #   dtypes are unmatched
            # - other types

            # fallback to numpy array and send it to GPU
            # Note: dtype might not be numpy.dtype in this place
            a = _send_object_to_gpu(obj, dtype, order, ndmin)

    return a


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
            if (elem_shape is None
                    or shape != elem_shape
                    or dtype != elem_dtype):
                return (None, obj_type, None)

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
    cdef vector.vector[Py_ssize_t] a_shape = a_cpu.shape
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
        a.data.copy_from_host_async(ctypes.c_void_p(mem.ptr), nbytes)
        pinned_memory._add_to_watch_list(stream.record(), mem)
    else:
        a.data.copy_from_host(
            ctypes.c_void_p(a_cpu.__array_interface__['data'][0]),
            nbytes)

    return a


cdef ndarray _send_numpy_array_list_to_gpu(
        list arrays, src_dtype, dst_dtype,
        const vector.vector[Py_ssize_t]& shape,
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
        a.data.copy_from_host_async(ctypes.c_void_p(mem.ptr), nbytes)
        pinned_memory._add_to_watch_list(stream.record(), mem)
    else:
        # fallback to numpy array and send it to GPU
        # Note: a_cpu.ndim is always >= 1
        a_cpu = numpy.array(arrays, dtype=a_dtype, copy=False, order=order,
                            ndmin=ndmin)
        a = ndarray(shape, dtype=a_dtype, order=order)
        a.data.copy_from_host(
            ctypes.c_void_p(a_cpu.__array_interface__['data'][0]),
            nbytes)

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
        if e.status != runtime.cudaErrorMemoryAllocation:
            raise
        warnings.warn(
            'Using synchronous transfer as pinned memory ({} bytes) '
            'could not be allocated. '
            'This generally occurs because of insufficient host memory. '
            'The original error was: {}'.format(nbytes, e),
            util.PerformanceWarning)

    return None


cpdef ndarray _internal_ascontiguousarray(ndarray a):
    if a._c_contiguous:
        return a
    newarray = ndarray(a.shape, a.dtype)
    elementwise_copy(a, newarray)
    return newarray


cpdef ndarray _internal_asfortranarray(ndarray a):
    cdef ndarray newarray
    cdef int m, n

    if a._f_contiguous:
        return a

    newarray = ndarray(a.shape, a.dtype, order='F')
    if (a._c_contiguous and a._shape.size() == 2 and
            (a.dtype == numpy.float32 or a.dtype == numpy.float64)):
        m, n = a.shape
        handle = device.get_cublas_handle()
        if a.dtype == numpy.float32:
            cublas.sgeam(
                handle,
                1,  # transpose a
                1,  # transpose newarray
                m, n, 1., a.data.ptr, n, 0., a.data.ptr, n,
                newarray.data.ptr, m)
        elif a.dtype == numpy.float64:
            cublas.dgeam(
                handle,
                1,  # transpose a
                1,  # transpose newarray
                m, n, 1., a.data.ptr, n, 0., a.data.ptr, n,
                newarray.data.ptr, m)
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


# -----------------------------------------------------------------------------
# Binary operations
# -----------------------------------------------------------------------------

cpdef _create_bit_op(name, op, no_bool, doc=''):
    types = () if no_bool else ('??->?',)
    return create_ufunc(
        'cupy_' + name,
        types + ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
                 'LL->L', 'qq->q', 'QQ->Q'),
        'out0 = in0 %s in1' % op,
        doc=doc)


bitwise_and = _create_bit_op(
    'bitwise_and', '&', False,
    '''Computes the bitwise AND of two arrays elementwise.

    Only integer and boolean arrays are handled.

    .. seealso:: :data:`numpy.bitwise_and`

    ''')


bitwise_or = _create_bit_op(
    'bitwise_or', '|', False,
    '''Computes the bitwise OR of two arrays elementwise.

    Only integer and boolean arrays are handled.

    .. seealso:: :data:`numpy.bitwise_or`

    ''')


bitwise_xor = _create_bit_op(
    'bitwise_xor', '^', False,
    '''Computes the bitwise XOR of two arrays elementwise.

    Only integer and boolean arrays are handled.

    .. seealso:: :data:`numpy.bitwise_xor`

    ''')


invert = create_ufunc(
    'cupy_invert',
    (('?->?', 'out0 = !in0'), 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I',
     'l->l', 'L->L', 'q->q', 'Q->Q'),
    'out0 = ~in0',
    doc='''Computes the bitwise NOT of an array elementwise.

    Only integer and boolean arrays are handled.

    .. seealso:: :data:`numpy.invert`

    ''')


left_shift = _create_bit_op(
    'left_shift', '<<', True,
    '''Shifts the bits of each integer element to the left.

    Only integer arrays are handled.

    .. seealso:: :data:`numpy.left_shift`

    ''')


right_shift = _create_bit_op(
    'right_shift', '>>', True,
    '''Shifts the bits of each integer element to the right.

    Only integer arrays are handled

    .. seealso:: :data:`numpy.right_shift`

    ''')


# -----------------------------------------------------------------------------
# Linear algebra
# -----------------------------------------------------------------------------

cpdef ndarray dot(ndarray a, ndarray b, ndarray out=None):
    cdef Py_ssize_t a_ndim, b_ndim, a_axis, b_axis, n, m, k
    cdef bint input_a_is_vec, input_b_is_vec
    cdef vector.vector[Py_ssize_t] ret_shape
    cdef vector.vector[Py_ssize_t] shape

    a_ndim = a._shape.size()
    b_ndim = b._shape.size()

    if out is not None and numpy.result_type(a.dtype, b.dtype) != out.dtype:
        raise ValueError('Not supported dtype combination.')

    if a_ndim == 0 or b_ndim == 0:
        return _math._multiply(a, b, out=out)

    input_a_is_vec = a_ndim == 1
    input_b_is_vec = b_ndim == 1
    if input_a_is_vec:
        shape.clear()
        shape.push_back(1)
        shape.push_back(a.size)
        a = _manipulation._reshape(a, shape)
        a_ndim = 2
    if input_b_is_vec:
        shape.clear()
        shape.push_back(b.size)
        shape.push_back(1)
        b = _manipulation._reshape(b, shape)
        b_ndim = 2

    a_axis = a_ndim - 1
    b_axis = b_ndim - 2

    if a._shape[a_axis] != b._shape[b_axis]:
        raise ValueError('Axis dimension mismatch')

    if a_axis:
        a = _manipulation.rollaxis(a, a_axis, 0)
    if b_axis:
        b = _manipulation.rollaxis(b, b_axis, 0)

    k = a._shape[0]
    if k != 0:
        m = b.size // k
        n = a.size // k
    else:
        # When k==0, the function must return a matrix filled with zero
        # like NumPy.
        m = 0
        n = 0

    if not input_a_is_vec:
        ret_shape.insert(ret_shape.end(), a._shape.begin() + 1, a._shape.end())
    if not input_b_is_vec:
        ret_shape.insert(ret_shape.end(), b._shape.begin() + 1, b._shape.end())
    if out is not None:
        if k != 0 and out.size != n * m:
            raise ValueError('Output array has an invalid size')
        if not out._c_contiguous:
            raise ValueError('Output array must be C-contiguous')

    return tensordot_core(a, b, out, n, m, k, ret_shape)


cdef _mat_ptrs_kernel = ElementwiseKernel(
    'T base, T stride', 'T out',
    'out = base + _ind.get()[_ind.ndim - 1] * stride', 'mat_ptrs',
    reduce_dims=False)


cdef ndarray _mat_ptrs(ndarray a):
    """Creates an array of pointers to matrices
    Args:
        a: A batch of matrices on GPU.
           shape: (A, B, C) -> A ptrs to mat o size (B, C)
           shape: (A_1, ..., A_N, B, C) -> A_1*...*A_N ptrs to mat of
                  size (B, C)
    Returns:
        GPU array of pointers to matrices.
    """
    cdef int ndim = a._shape.size()
    assert ndim > 2
    cdef Py_ssize_t sh_, st_
    cdef ndarray idx
    idx = _mat_ptrs_kernel(
        a.data.ptr, a._strides[0],
        cupy.ndarray((a._shape[0],), dtype=numpy.uintp))

    for i in range(1, ndim - 2):
        idx = _mat_ptrs_kernel(
            idx[:, None], a._strides[i],
            cupy.ndarray((idx.size, a._shape[i]), dtype=numpy.uintp))
        idx = idx.ravel()
    return idx


cdef Py_ssize_t _get_stride_for_strided_batched_gemm(ndarray a) except?0:
    cdef int ndim = a._shape.size()
    assert ndim > 2
    return a._strides[ndim - 3] // <Py_ssize_t>a.itemsize


cpdef ndarray matmul(ndarray a, ndarray b, ndarray out=None):
    """ Returns the matrix product of two arrays and is the implementation of
    the `@` operator introduced in Python 3.5 following PEP465.

    The main difference against cupy.dot are the handling of arrays with more
    than 2 dimensions. For more information see :func:`numpy.matmul`.

    .. note::
        The out array as input is currently not supported.

    Args:
        a (cupy.ndarray): The left argument.
        b (cupy.ndarray): The right argument.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: Output array.

    .. seealso:: :func:`numpy.matmul`

    """

    if out is not None:
        raise NotImplementedError('The out array as input is currently not '
                                  'supported')

    cdef Py_ssize_t i, n, m, ka, kb, a_sh, b_sh, c_sh
    cdef Py_ssize_t batchCount, a_part_outshape, b_part_outshape
    cdef int orig_a_ndim, orig_b_ndim, a_ndim, b_ndim, ndim
    cdef ndarray ap, bp, outp, out_view
    cdef bint use_broadcast

    orig_a_ndim = a._shape.size()
    orig_b_ndim = b._shape.size()
    if orig_a_ndim == 0 or orig_b_ndim == 0:
        raise ValueError('Scalar operands are not allowed, use \'*\' instead')

    ndim = max(orig_a_ndim, orig_b_ndim)
    if ndim <= 2:
        return dot(a, b, out)

    orig_a = a
    orig_b = b
    a_part_outshape = b_part_outshape = 0
    if orig_a_ndim == 1:
        a = _manipulation._reshape(a, (1, a.size))
    else:
        a = a.view()
        a_part_outshape = a._shape[orig_a_ndim - 2]
    if orig_b_ndim == 1:
        b = _manipulation._reshape(b, (b.size, 1))
        ldout = 1
    else:
        b = b.view()
        b_part_outshape = ldout = b._shape[orig_b_ndim - 1]

    # expand dims
    a_ndim = a._shape.size()
    b_ndim = b._shape.size()
    if a_ndim < ndim:
        # TODO(niboshi): Confirm update_x_contiguity flags
        a._set_shape_and_strides(
            (1,) * (ndim - a_ndim) + a.shape,
            (0,) * (ndim - a_ndim) + a.strides,
            True, True)
    if b_ndim < ndim:
        # TODO(niboshi): Confirm update_x_contiguity flags
        b._set_shape_and_strides(
            (1,) * (ndim - b_ndim) + b.shape,
            (0,) * (ndim - b_ndim) + b.strides,
            True, True)

    ret_dtype = numpy.result_type(a.dtype, b.dtype)
    dtype = numpy.find_common_type((ret_dtype, 'f'), ())

    a = ascontiguousarray(a, dtype)
    b = ascontiguousarray(b, dtype)

    # broadcast
    batchCount = 1  # batchCount = numpy.prod(out_shape[:-2])
    out_shape = []
    use_broadcast = False
    for i in range(0, ndim - 2):
        a_sh = a._shape[i]
        b_sh = b._shape[i]
        if a_sh != b_sh and a_sh != 1 and b_sh != 1:
            raise ValueError(
                'operands could not be broadcast together with '
                'remapped shapes')

        if a_sh == 0 or b_sh == 0:
            c_sh = 0
        else:
            c_sh = max(a_sh, b_sh)
        batchCount *= c_sh
        out_shape.append(c_sh)
        if a_sh == 1 and c_sh > 1:
            a._strides[i] = 0
            a._shape[i] = c_sh
            a._c_contiguous = a._f_contiguous = False
            use_broadcast = True

        if b_sh == 1 and c_sh > 1:
            b._strides[i] = 0
            b._shape[i] = c_sh
            b._c_contiguous = b._f_contiguous = False
            use_broadcast = True

    if orig_a_ndim != 1:
        out_shape.append(a_part_outshape)
    if orig_b_ndim != 1:
        out_shape.append(b_part_outshape)

    # (A B)^T = B^T A^T
    a, b = b, a

    ka = a._shape[ndim - 2]
    lda = n = a._shape[ndim - 1]
    m = b._shape[ndim - 2]
    ldb = kb = b._shape[ndim - 1]

    if ka != kb:
        raise ValueError(
            'shapes ({}) and ({}) not aligned'.format(
                ','.join([str(_) for _ in orig_a.shape]),
                ','.join([str(_) for _ in orig_b.shape])))

    if a.size == 0 or b.size == 0:
        return cupy.zeros(out_shape, ret_dtype)

    out = ndarray(out_shape, dtype=dtype)

    if orig_a_ndim == 1 or orig_b_ndim == 1:
        out_view = out.view()
        if orig_b_ndim == 1:
            out_view._shape.push_back(1)
            out_view._strides.push_back(0)
        if orig_a_ndim == 1:
            out_view._shape.insert(out_view._shape.end() - 1, 1)
            out_view._strides.insert(out_view._strides.end() - 1, 0)
        assert out_view._c_contiguous
        out_view._update_f_contiguity()
    else:
        out_view = out

    global _cuda_runtime_version
    if _cuda_runtime_version < 0:
        _cuda_runtime_version = runtime.runtimeGetVersion()

    handle = device.get_cublas_handle()

    # TODO(anaruse) use cublasGemmStridedBatchedEx() when cuda version >= 9.1
    if not use_broadcast:
        strideA = _get_stride_for_strided_batched_gemm(a)
        strideB = _get_stride_for_strided_batched_gemm(b)
        strideC = _get_stride_for_strided_batched_gemm(out_view)
        if dtype == numpy.float32:
            cublas.sgemmStridedBatched(
                handle,
                0,  # transa
                0,  # transb
                n, m, ka, 1.0,
                a.data.ptr, lda, strideA,
                b.data.ptr, ldb, strideB,
                0.0, out_view.data.ptr, ldout, strideC,
                batchCount)
        elif dtype == numpy.float64:
            cublas.dgemmStridedBatched(
                handle,
                0,  # transa
                0,  # transb
                n, m, ka, 1.0,
                a.data.ptr, lda, strideA,
                b.data.ptr, ldb, strideB,
                0.0, out_view.data.ptr, ldout, strideC,
                batchCount)
        elif dtype == numpy.complex64:
            cublas.cgemmStridedBatched(
                handle,
                0,  # transa
                0,  # transb
                n, m, ka, 1,
                a.data.ptr, lda, strideA,
                b.data.ptr, ldb, strideB,
                0, out_view.data.ptr, ldout, strideC,
                batchCount)
        elif dtype == numpy.complex128:
            cublas.zgemmStridedBatched(
                handle,
                0,  # transa
                0,  # transb
                n, m, ka, 1,
                a.data.ptr, lda, strideA,
                b.data.ptr, ldb, strideB,
                0, out_view.data.ptr, ldout, strideC,
                batchCount)
        else:
            raise TypeError(dtype, a.dtype, b.dtype)
    else:
        ap = _mat_ptrs(a)
        bp = _mat_ptrs(b)
        outp = _mat_ptrs(out_view)
        if dtype == numpy.float32:
            cublas.sgemmBatched(
                handle,
                0,  # transa
                0,  # transb
                n, m, ka, 1.0,
                ap.data.ptr, lda,
                bp.data.ptr, ldb,
                0.0, outp.data.ptr, ldout, batchCount)
        elif dtype == numpy.float64:
            cublas.dgemmBatched(
                handle,
                0,  # transa
                0,  # transb
                n, m, ka, 1.0,
                ap.data.ptr, lda,
                bp.data.ptr, ldb,
                0.0, outp.data.ptr, ldout, batchCount)
        elif dtype == numpy.complex64:
            cublas.cgemmBatched(
                handle,
                0,  # transa
                0,  # transb
                n, m, ka, 1,
                ap.data.ptr, lda,
                bp.data.ptr, ldb,
                0, outp.data.ptr, ldout, batchCount)
        elif dtype == numpy.complex128:
            cublas.zgemmBatched(
                handle,
                0,  # transa
                0,  # transb
                n, m, ka, 1,
                ap.data.ptr, lda,
                bp.data.ptr, ldb,
                0, outp.data.ptr, ldout, batchCount)
        else:
            raise TypeError(dtype, a.dtype, b.dtype)

    if dtype == ret_dtype:
        return out
    else:
        ret = ndarray(out_shape, ret_dtype)
        elementwise_copy(out, ret)
        return ret


cdef int _cuda_runtime_version = -1
cdef _tensordot_core_mul_sum = ReductionKernel(
    'S x, T y', 'U out',
    'static_cast<U>(x) * static_cast<U>(y)',
    'a + b', 'out = a', '0', '_tensordot_core_mul_sum')


cpdef ndarray tensordot_core(
        ndarray a, ndarray b, ndarray out, Py_ssize_t n, Py_ssize_t m,
        Py_ssize_t k, const vector.vector[Py_ssize_t]& ret_shape):
    cdef vector.vector[Py_ssize_t] shape
    cdef Py_ssize_t inca, incb, transa, transb, lda, ldb
    cdef Py_ssize_t mode, handle
    cdef bint use_sgemmEx
    cdef float one_fp32, zero_fp32
    ret_dtype = a.dtype.char
    if ret_dtype != b.dtype.char:
        ret_dtype = numpy.find_common_type((ret_dtype, b.dtype), ()).char

    if not a.size or not b.size:
        if out is None:
            out = ndarray(ret_shape, dtype=ret_dtype)
        out.fill(0)
        return out

    global _cuda_runtime_version
    if _cuda_runtime_version < 0:
        _cuda_runtime_version = runtime.runtimeGetVersion()

    use_sgemmEx = (a.dtype == 'e' and b.dtype == 'e' and
                   (ret_dtype == 'e' or ret_dtype == 'f'))
    use_tensor_core = (use_sgemmEx and
                       _cuda_runtime_version >= 9000 and
                       int(device.get_compute_capability()) == 70)

    if use_sgemmEx or ret_dtype in 'fdFD':
        dtype = ret_dtype
    else:
        dtype = numpy.find_common_type((ret_dtype, 'f'), ()).char

    if out is None:
        out = ndarray(ret_shape, dtype)
        if dtype == ret_dtype:
            ret = out
        else:
            ret = ndarray(ret_shape, ret_dtype)
    else:
        ret = out
        if out.dtype != dtype:
            out = ndarray(ret_shape, dtype)

    if m == 1 and n == 1:
        _tensordot_core_mul_sum(
            a.ravel(), b.ravel(), _manipulation._reshape(out, ()))
        if out is not ret:
            elementwise_copy(out, ret)
        return ret

    # It copies the operands if needed
    if a._shape.size() != 2 or a._shape[0] != k or a._shape[1] != n:
        shape.clear()
        shape.push_back(k)
        shape.push_back(n)
        a = _manipulation._reshape(a, shape)
    if b._shape.size() != 2 or b._shape[0] != k or b._shape[1] != m:
        shape.clear()
        shape.push_back(k)
        shape.push_back(m)
        b = _manipulation._reshape(b, shape)
    c = out
    if c._shape.size() != 2 or c._shape[0] != n or c._shape[1] != m:
        c = c.view()
        c.shape = (n, m)

    if not use_sgemmEx:
        a = a.astype(dtype, copy=False)
        b = b.astype(dtype, copy=False)

    # Be careful that cuBLAS uses the FORTRAN-order matrix representation.
    handle = device.get_cublas_handle()
    # Matrix-Matrix product A^T * B
    # c is C-contiguous while cuBLAS assumes F-contiguous inputs, so we
    # compute C^T = B^T * A here.
    a, transa, lda = _mat_to_cublas_contiguous(a, 0)
    b, transb, ldb = _mat_to_cublas_contiguous(b, 1)
    if use_sgemmEx:
        Ctype = runtime.CUDA_R_16F if c.dtype == 'e' else runtime.CUDA_R_32F
        if use_tensor_core:
            one_fp32 = 1
            zero_fp32 = 0
            cublas.setMathMode(handle, cublas.CUBLAS_TENSOR_OP_MATH)
            cublas.gemmEx(
                handle, <int>transb, <int> transa, <int>m, <int>n, <int>k,
                <size_t>&one_fp32,
                b.data.ptr, runtime.CUDA_R_16F, <int>ldb,
                a.data.ptr, runtime.CUDA_R_16F, <int>lda,
                <size_t>&zero_fp32,
                c.data.ptr, Ctype, <int>m,
                runtime.CUDA_R_32F, cublas.CUBLAS_GEMM_DEFAULT_TENSOR_OP)
            cublas.setMathMode(handle, cublas.CUBLAS_DEFAULT_MATH)
        else:
            cublas.sgemmEx(
                handle, <int>transb, <int> transa, <int>m, <int>n, <int>k, 1,
                b.data.ptr, runtime.CUDA_R_16F, <int>ldb, a.data.ptr,
                runtime.CUDA_R_16F, <int>lda, 0, c.data.ptr, Ctype, <int>m)
    elif dtype == 'f':
        cublas.sgemmEx(
            handle, <int>transb, <int> transa, <int>m, <int>n, <int>k, 1,
            b.data.ptr, runtime.CUDA_R_32F, <int>ldb,
            a.data.ptr, runtime.CUDA_R_32F, <int>lda, 0,
            c.data.ptr, runtime.CUDA_R_32F, <int>m)
    elif dtype == 'd':
        cublas.dgemm(
            handle, <int>transb, <int>transa, <int>m, <int>n, <int>k, 1,
            b.data.ptr, <int>ldb, a.data.ptr, <int>lda, 0, c.data.ptr, <int>m)
    elif dtype == 'F':
        cublas.cgemm(
            handle, <int>transb, <int>transa, <int>m, <int>n, <int>k, 1,
            b.data.ptr, <int>ldb, a.data.ptr, <int>lda, 0, c.data.ptr, <int>m)
    elif dtype == 'D':
        cublas.zgemm(
            handle, <int>transb, <int>transa, <int>m, <int>n, <int>k, 1,
            b.data.ptr, <int>ldb, a.data.ptr, <int>lda, 0, c.data.ptr, <int>m)
    else:
        raise ValueError('Invalid dtype: %s' % str(dtype))

    if out is not ret:
        elementwise_copy(out, ret)
    return ret


@cython.profile(False)
cpdef inline tuple _mat_to_cublas_contiguous(ndarray a, Py_ssize_t trans):
    assert a.ndim == 2
    if a._f_contiguous:
        # builtin max function is not used for Cython 0.23
        lda = a._strides[1] // a.itemsize
        if lda < a._shape[0]:
            lda = a._shape[0]
        return a, trans, lda
    if not a._c_contiguous:
        a = a.copy()
    return a, 1 - trans, a._strides[0] // a.itemsize


@cython.profile(False)
cpdef inline tuple _to_cublas_vector(ndarray a, Py_ssize_t rundim):
    if a._strides[rundim] < 0:
        return a.copy(), 1
    else:
        return a, a._strides[rundim] // a.itemsize

# -----------------------------------------------------------------------------
# Logic functions
# -----------------------------------------------------------------------------

cpdef create_comparison(name, op, doc='', no_complex_dtype=True):

    if no_complex_dtype:
        ops = ('??->?', 'bb->?', 'BB->?', 'hh->?', 'HH->?', 'ii->?', 'II->?',
               'll->?', 'LL->?', 'qq->?', 'QQ->?', 'ee->?', 'ff->?', 'dd->?')
    else:
        ops = ('??->?', 'bb->?', 'BB->?', 'hh->?', 'HH->?', 'ii->?', 'II->?',
               'll->?', 'LL->?', 'qq->?', 'QQ->?', 'ee->?', 'ff->?', 'dd->?',
               'FF->?', 'DD->?')

    return create_ufunc(
        'cupy_' + name,
        ops,
        'out0 = in0 %s in1' % op,
        doc=doc)


greater = create_comparison(
    'greater', '>',
    '''Tests elementwise if ``x1 > x2``.

    .. seealso:: :data:`numpy.greater`

    ''',
    no_complex_dtype=False)


greater_equal = create_comparison(
    'greater_equal', '>=',
    '''Tests elementwise if ``x1 >= x2``.

    .. seealso:: :data:`numpy.greater_equal`

    ''',
    no_complex_dtype=False)


less = create_comparison(
    'less', '<',
    '''Tests elementwise if ``x1 < x2``.

    .. seealso:: :data:`numpy.less`

    ''',
    no_complex_dtype=False)


less_equal = create_comparison(
    'less_equal', '<=',
    '''Tests elementwise if ``x1 <= x2``.

    .. seealso:: :data:`numpy.less_equal`

    ''',
    no_complex_dtype=False)


equal = create_comparison(
    'equal', '==',
    '''Tests elementwise if ``x1 == x2``.

    .. seealso:: :data:`numpy.equal`

    ''',
    no_complex_dtype=False)


not_equal = create_comparison(
    'not_equal', '!=',
    '''Tests elementwise if ``x1 != x2``.

    .. seealso:: :data:`numpy.equal`

    ''',
    no_complex_dtype=False)


cpdef ndarray _convert_object_with_cuda_array_interface(a):
    cdef Py_ssize_t sh, st
    desc = a.__cuda_array_interface__
    shape = desc['shape']
    dtype = numpy.dtype(desc['typestr'])
    if 'strides' in desc:
        strides = desc['strides']
        nbytes = 0
        for sh, st in zip(shape, strides):
            nbytes = max(nbytes, abs(sh * st))
    else:
        strides = None
        nbytes = internal.prod(shape) * dtype.itemsize
    mem = memory_module.UnownedMemory(desc['data'][0], nbytes, a)
    memptr = memory.MemoryPointer(mem, 0)
    return ndarray(shape, dtype, memptr, strides)
