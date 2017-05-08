# distutils: language = c++

from __future__ import division
import ctypes
import sys

import numpy
import six

from cupy.core import flags
from cupy.cuda import stream
from cupy import util

cimport cpython
cimport cython
from libcpp cimport vector

from cupy.core cimport internal
from cupy.cuda cimport cublas
from cupy.cuda cimport function
from cupy.cuda cimport runtime
from cupy.cuda cimport memory

DEF MAX_NDIM = 25


@cython.profile(False)
cdef inline _should_use_rop(x, y):
    xp = getattr(x, '__array_priority__', 0)
    yp = getattr(y, '__array_priority__', 0)
    return xp < yp and not isinstance(y, ndarray)


cdef class ndarray:

    """Multi-dimensional array on a CUDA device.

    This class implements a subset of methods of :class:`numpy.ndarray`.
    The difference is that this class allocates the array content on the
    current GPU device.

    Args:
        shape (tuple of ints): Length of axes.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        memptr (cupy.cuda.MemoryPointer): Pointer to the array content head.
        strides (tuple of ints): The strides for axes.
        order ({'C', 'F'}): Row-major (C-style) or column-major
            (Fortran-style) order.

    Attributes:
        base (None or cupy.ndarray): Base array from which this array is
            created as a view.
        data (cupy.cuda.MemoryPointer): Pointer to the array content head.
        dtype(numpy.dtype): Dtype object of element type.

            .. seealso::
               `Data type objects (dtype) \
               <http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html>`_
        size (int): Number of elements this array holds.

            This is equivalent to product over the shape tuple.

            .. seealso:: :attr:`numpy.ndarray.size`


    """

    def __init__(self, shape, dtype=float, memptr=None, order='C'):
        cdef Py_ssize_t x
        self._shape = internal.get_size(shape)
        for x in self._shape:
            if x < 0:
                raise ValueError('Negative dimensions are not allowed')
        self.dtype = numpy.dtype(dtype)
        self.size = internal.prod_ssize_t(self._shape)

        if memptr is None:
            self.data = memory.alloc(self.size * self.dtype.itemsize)
        else:
            self.data = memptr
        self.base = None

        if order == 'C':
            self._strides = internal.get_contiguous_strides(
                self._shape, self.itemsize, is_c_contiguous=True)
            self._c_contiguous = True
            self._update_f_contiguity()
        elif order == 'F':
            self._strides = internal.get_contiguous_strides(
                self._shape, self.itemsize, is_c_contiguous=False)
            self._f_contiguous = True
            self._update_c_contiguity()
        else:
            raise TypeError('order not understood')

    # The definition order of attributes and methods are borrowed from the
    # order of documentation at the following NumPy document.
    # http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html

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
            cdef vector.vector[Py_ssize_t] shape, strides
            if not cpython.PySequence_Check(newshape):
                newshape = (newshape,)
            shape = internal.infer_unknown_dimension(newshape, self.size)
            strides = _get_strides_for_nocopy_reshape(self, shape)
            if strides.size() != shape.size():
                raise AttributeError('incompatible shape')
            self._shape = shape
            self._strides = strides
            self._update_f_contiguity()

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
        """Size of whole elements in bytes.

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
            return self._transpose(vector.vector[Py_ssize_t]())

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
    # TODO(okuta): Implement item

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

    cpdef ndarray astype(self, dtype, copy=True):
        """Casts the array to given data type.

        Args:
            dtype: Type specifier.
            copy (bool): If it is False and no cast happens, then this method
                returns the array itself. Otherwise, a copy is returned.

        Returns:
            If ``copy`` is False and no cast is required, then the array itself
            is returned. Otherwise, it returns a (possibly casted) copy of the
            array.

        .. note::
           This method currently does not support ``order``, ``casting``, and
           ``subok`` arguments.

        .. seealso:: :meth:`numpy.ndarray.astype`

        """
        # TODO(beam2d): Support ordering, casting, and subok option
        dtype = numpy.dtype(dtype)
        if dtype.type == self.dtype.type:
            if copy:
                return self.copy()
            else:
                return self
        else:
            newarray = ndarray(self.shape, dtype=dtype)
            elementwise_copy(self, newarray)
            return newarray

    # TODO(okuta): Implement byteswap

    cpdef ndarray copy(self, order='C'):
        """Returns a copy of the array.

        Args:
            order ({'C', 'F'}): Row-major (C-style) or column-major
                (Fortran-style) order. This function currently does not
                support order 'A' and 'K'.

        .. seealso::
           :func:`cupy.copy` for full documentation,
           :meth:`numpy.ndarray.copy`

        """
        cdef ndarray a, newarray
        # TODO(beam2d): Support ordering option 'A' and 'K'
        if order not in ['C', 'F']:
            raise TypeError('order not understood')

        if self.size == 0:
            return ndarray(self.shape, self.dtype, order=order)

        a = self
        if order == 'C' and not self._c_contiguous:
            with self.device:
                a = ascontiguousarray(self)
            if a.data.device.id == device.get_device_id():
                return a
        elif order == 'F' and not self._f_contiguous:
            with self.device:
                a = asfortranarray(self)
            if a.data.device.id == device.get_device_id():
                return a

        newarray = ndarray(a.shape, a.dtype, order=order)
        newarray.data.copy_from_device(a.data, a.nbytes)
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
        cdef ndarray v
        v = ndarray.__new__(ndarray)
        v.size = self.size
        v._shape = self._shape
        v._strides = self._strides
        v._c_contiguous = self._c_contiguous
        v._f_contiguous = self._f_contiguous
        v.dtype = self.dtype if dtype is None else numpy.dtype(dtype)
        v.data = self.data
        v.base = self.base if self.base is not None else self
        return v

    # TODO(okuta): Implement getfield
    # TODO(okuta): Implement setflags

    cpdef fill(self, value):
        """Fills the array with a scalar value.

        Args:
            value: A scalar value to fill the array content.

        .. seealso:: :meth:`numpy.ndarray.fill`

        """
        if value == 0 and self._c_contiguous:
            self.data.memset_async(0, self.nbytes, stream.Stream(True))
        else:
            elementwise_copy(value, self, dtype=self.dtype)

    # -------------------------------------------------------------------------
    # Shape manipulation
    # -------------------------------------------------------------------------
    cpdef ndarray _reshape(self, vector.vector[Py_ssize_t] shape):
        cdef vector.vector[Py_ssize_t] strides
        cdef ndarray newarray
        shape = internal.infer_unknown_dimension(shape, self.size)
        if internal.vector_equal(shape, self._shape):
            return self.view()

        strides = _get_strides_for_nocopy_reshape(self, shape)
        if strides.size() == shape.size():
            newarray = self.view()
        else:
            newarray = self.copy()
            strides = _get_strides_for_nocopy_reshape(newarray, shape)

        if shape.size() != strides.size():
            raise ValueError('total size of new array must be unchanged')
        newarray._set_shape_and_strides(shape, strides, False)
        return newarray

    def reshape(self, *shape):
        """Returns an array of a different shape and the same content.

        .. seealso::
           :func:`cupy.reshape` for full documentation,
           :meth:`numpy.ndarray.reshape`

        """
        # TODO(beam2d): Support ordering option
        if len(shape) == 1 and cpython.PySequence_Check(shape[0]):
            shape = shape[0]
        return self._reshape(shape)

    # TODO(okuta): Implement resize
    cpdef ndarray _transpose(self, vector.vector[Py_ssize_t] axes):
        cdef ndarray ret
        cdef vector.vector[Py_ssize_t] a_axes, rev_axes
        cdef Py_ssize_t ndim, axis

        ndim = self._shape.size()
        ret = self.view()
        if axes.size() == 0:
            ret._shape.assign(self._shape.rbegin(), self._shape.rend())
            ret._strides.assign(self._strides.rbegin(), self._strides.rend())
            ret._c_contiguous = self._f_contiguous
            ret._f_contiguous = self._c_contiguous
            return ret

        if <Py_ssize_t>axes.size() != ndim:
            raise ValueError('Invalid axes value: %s' % str(axes))

        for i in range(ndim):
            a_axes.push_back(i)
            axis = axes[i]
            if axis < -ndim or axis >= ndim:
                raise IndexError('Axes overrun')
            axes[i] = axis % ndim

        if internal.vector_equal(a_axes, axes):
            return ret
        rev_axes.assign(axes.rbegin(), axes.rend())
        if internal.vector_equal(a_axes, rev_axes):
            ret._shape.assign(self._shape.rbegin(), self._shape.rend())
            ret._strides.assign(self._strides.rbegin(), self._strides.rend())
            ret._c_contiguous = self._f_contiguous
            ret._f_contiguous = self._c_contiguous
            return ret

        if ndim != len({i for i in axes}):
            raise ValueError('Invalid axes value: %s' % str(axes))

        ret._shape.clear()
        ret._strides.clear()
        for axis in axes:
            ret._shape.push_back(self._shape[axis])
            ret._strides.push_back(self._strides[axis])
        ret._update_contiguity()
        return ret

    def transpose(self, *axes):
        """Returns a view of the array with axes permuted.

        .. seealso::
           :func:`cupy.transpose` for full documentation,
           :meth:`numpy.ndarray.reshape`

        """
        cdef ndarray ret
        cdef vector.vector[Py_ssize_t] vec_axes, a_axes, temp_axes
        cdef Py_ssize_t ndim, axis
        if len(axes) == 1:
            a = axes[0]
            if a is None:
                axes = ()
            elif cpython.PySequence_Check(a):
                axes = a
        return self._transpose(axes)

    cpdef ndarray swapaxes(self, Py_ssize_t axis1, Py_ssize_t axis2):
        """Returns a view of the array with two axes swapped.

        .. seealso::
           :func:`cupy.swapaxes` for full documentation,
           :meth:`numpy.ndarray.swapaxes`

        """
        cdef Py_ssize_t ndim=self.ndim
        cdef vector.vector[Py_ssize_t] axes
        if axis1 < -ndim or axis1 >= ndim or axis2 < -ndim or axis2 >= ndim:
            raise ValueError('Axis out of range')
        axis1 %= ndim
        axis2 %= ndim
        for i in range(ndim):
            axes.push_back(i)
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
        return self._transpose(axes)

    cpdef ndarray flatten(self):
        """Returns a copy of the array flatten into one dimension.

        It currently supports C-order only.

        Returns:
            cupy.ndarray: A copy of the array with one dimension.

        .. seealso:: :meth:`numpy.ndarray.flatten`

        """
        # TODO(beam2d): Support ordering option
        if self._c_contiguous:
            newarray = self.copy()
        else:
            newarray = ndarray(self.shape, self.dtype)
            elementwise_copy(self, newarray)

        newarray._shape.assign(<Py_ssize_t>1, self.size)
        newarray._strides.assign(<Py_ssize_t>1, <Py_ssize_t>self.itemsize)
        newarray._c_contiguous = True
        newarray._f_contiguous = True
        return newarray

    cpdef ndarray ravel(self):
        """Returns an array flattened into one dimension.

        .. seealso::
           :func:`cupy.ravel` for full documentation,
           :meth:`numpy.ndarray.ravel`

        """
        # TODO(beam2d): Support ordering option
        cdef vector.vector[Py_ssize_t] shape
        shape.push_back(self.size)
        return self._reshape(shape)

    cpdef ndarray squeeze(self, axis=None):
        """Returns a view with size-one axes removed.

        .. seealso::
           :func:`cupy.squeeze` for full documentation,
           :meth:`numpy.ndarray.squeeze`

        """

        cdef vector.vector[char] axis_flags
        cdef vector.vector[Py_ssize_t] newshape, newstrides
        cdef Py_ssize_t ndim, naxes, _axis

        ndim = self._shape.size()
        axis_flags = vector.vector[char](ndim, 0)

        # Convert axis to boolean flag.
        if axis is None:
            for idim in range(ndim):
                if self._shape[idim] == 1:
                    axis_flags[idim] = 1
        elif isinstance(axis, tuple):
            naxes = <Py_ssize_t>len(axis)
            for i in range(naxes):
                _axis = <Py_ssize_t>axis[i]
                axis_orig = _axis
                if _axis < 0:
                    _axis += ndim
                if _axis < 0 or _axis >= ndim:
                    msg = "'axis' entry %d is out of bounds [-%d, %d)"
                    raise ValueError(msg % (axis_orig, ndim, ndim))
                if axis_flags[_axis] == 1:
                    raise ValueError("duplicate value in 'axis'")
                axis_flags[_axis] = 1
        else:
            _axis = <Py_ssize_t>axis
            axis_orig = _axis
            if _axis < 0:
                _axis += ndim
            if ndim == 0 and (_axis == 0 or _axis == -1):
                # Special case letting axis={-1,0} slip through for scalars,
                # for backwards compatibility reasons.
                pass
            else:
                if _axis < 0 or _axis >= ndim:
                    msg = "'axis' entry %d is out of bounds [-%d, %d)"
                    raise ValueError(msg % (axis_orig, ndim, ndim))
                axis_flags[_axis] = 1

        # Verify that the axes requested are all of size one
        any_ones = 0
        for idim in range(ndim):
            if axis_flags[idim] != 0:
                if self._shape[idim] == 1:
                    any_ones = 1
                else:
                    raise ValueError('cannot select an axis to squeeze out '
                                     'which has size not equal to one')

        # If there were no axes to squeeze out, return the same array
        if any_ones == 0:
            return self

        for i in range(ndim):
            if axis_flags[i] == 0:
                newshape.push_back(self._shape[i])
                newstrides.push_back(self._strides[i])

        v = self.view()
        v._set_shape_and_strides(newshape, newstrides, False)
        return v

    # -------------------------------------------------------------------------
    # Item selection and manipulation
    # -------------------------------------------------------------------------
    cpdef ndarray take(self, indices, axis=None, out=None):
        """Returns an array of elements at given indices along the axis.

        .. seealso::
           :func:`cupy.take` for full documentation,
           :meth:`numpy.ndarray.take`

        """
        return _take(self, indices, li=axis, ri=axis, out=out)

    # TODO(okuta): Implement put

    cpdef repeat(self, repeats, axis=None):
        """Returns an array with repeated arrays along an axis.

        .. seealso::
            :func:`cupy.repeat` for full documentation,
            :meth:`numpy.ndarray.repeat`

        """
        return _repeat(self, repeats, axis)

    cpdef choose(self, choices, out=None, mode='raise'):
        a = self
        n = choices.shape[0]

        # broadcast `a` and `choices[i]` for all i
        if a.ndim < choices.ndim - 1:
            for i in range(choices.ndim - 1 - a.ndim):
                a = a[None, ...]
        elif a.ndim > choices.ndim - 1:
            for i in range(a.ndim + 1 - choices.ndim):
                choices = choices[:, None, ...]
        ba, bcs = broadcast(a, choices).values

        if out is None:
            out = ndarray(ba.shape[1:], choices.dtype)

        n_channel = numpy.prod(bcs[0].shape)
        if mode == 'raise':
            if not ((a < n).all() and (0 <= a).all()):
                raise ValueError('invalid entry in choice array')
            _choose_kernel(ba[0], bcs, n_channel, out)
        elif mode == 'wrap':
            ba = ba[0] % n
            _choose_kernel(ba, bcs, n_channel, out)
        elif mode == 'clip':
            _choose_clip_kernel(ba[0], bcs, n_channel, n, out)
        else:
            raise TypeError('clipmode not understood')

        return out

    # TODO(okuta): Implement sort
    # TODO(okuta): Implement argsort
    # TODO(okuta): Implement partition
    # TODO(okuta): Implement argpartition
    # TODO(okuta): Implement searchsorted

    def nonzero(self):
        """Return the indices of the elements that are non-zero.

        Returned Array is containing the indices of the non-zero elements
        in that dimension.

        Returns:
            tuple of arrays: Indices of elements that are non-zero.

        .. seealso::
            :func:`numpy.nonzero`

        """
        condition = self != 0
        dtype = numpy.int64

        scan_index = scan(condition.astype(dtype).ravel())
        count_nonzero = int(scan_index[-1])

        if self.ndim <= 1:
            dst = ndarray((count_nonzero,), dtype=dtype)

            kern = _nonzero_1d_kernel(self.dtype, dtype)
            kern.linear_launch(self.size, (self.ravel(), scan_index, dst))

            return dst,
        else:
            dst = ndarray((count_nonzero * self.ndim,), dtype=dtype)

            kern = _nonzero_kernel(self.dtype, self.ndim, dtype, dtype)
            kern.linear_launch(self.size,
                               (self.ravel(), Indexer(self.shape),
                                scan_index, dst))
            return tuple([dst[i::self.ndim]
                          for i in six.moves.range(self.ndim)])

    # TODO(okuta): Implement compress

    cpdef ndarray diagonal(self, offset=0, axis1=0, axis2=1):
        """Returns a view of the specified diagonals.

        .. seealso::
           :func:`cupy.diagonal` for full documentation,
           :meth:`numpy.ndarray.diagonal`

        """
        return _diagonal(self, offset, axis1, axis2)

    # -------------------------------------------------------------------------
    # Calculation
    # -------------------------------------------------------------------------
    cpdef ndarray max(self, axis=None, out=None, dtype=None, keepdims=False):
        """Returns the maximum along a given axis.

        .. seealso::
           :func:`cupy.amax` for full documentation,
           :meth:`numpy.ndarray.max`

        """
        return _amax(
            self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)

    cpdef ndarray argmax(self, axis=None, out=None, dtype=None,
                         keepdims=False):
        """Returns the indices of the maximum along a given axis.

        .. seealso::
           :func:`cupy.argmax` for full documentation,
           :meth:`numpy.ndarray.argmax`

        """
        return _argmax(
            self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)

    cpdef ndarray min(self, axis=None, out=None, dtype=None, keepdims=False):
        """Returns the minimum along a given axis.

        .. seealso::
           :func:`cupy.amin` for full documentation,
           :meth:`numpy.ndarray.min`

        """
        return _amin(
            self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)

    cpdef ndarray argmin(self, axis=None, out=None, dtype=None,
                         keepdims=False):
        """Returns the indices of the minimum along a given axis.

        .. seealso::
           :func:`cupy.argmin` for full documentation,
           :meth:`numpy.ndarray.argmin`

        """
        return _argmin(
            self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)

    # TODO(okuta): Implement ptp

    cpdef ndarray clip(self, a_min, a_max, out=None):
        """Returns an array with values limited to [a_min, a_max].

        .. seealso::
           :func:`cupy.clip` for full documentation,
           :meth:`numpy.ndarray.clip`

        """
        return _clip(self, a_min, a_max, out=out)

    # TODO(okuta): Implement round

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
        return _sum(self, axis, dtype, out, keepdims)

    # TODO(okuta): Implement cumsum

    cpdef ndarray mean(self, axis=None, dtype=None, out=None, keepdims=False):
        """Returns the mean along a given axis.

        .. seealso::
           :func:`cupy.mean` for full documentation,
           :meth:`numpy.ndarray.mean`

        """
        return _mean(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    cpdef ndarray var(self, axis=None, dtype=None, out=None, ddof=0,
                      keepdims=False):
        """Returns the variance along a given axis.

        .. seealso::
           :func:`cupy.var` for full documentation,
           :meth:`numpy.ndarray.var`

        """
        return _var(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    cpdef ndarray std(self, axis=None, dtype=None, out=None, ddof=0,
                      keepdims=False):
        """Returns the standard deviation along a given axis.

        .. seealso::
           :func:`cupy.std` for full documentation,
           :meth:`numpy.ndarray.std`

        """
        return _std(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    cpdef ndarray prod(self, axis=None, dtype=None, out=None, keepdims=None):
        """Returns the product along a given axis.

        .. seealso::
           :func:`cupy.prod` for full documentation,
           :meth:`numpy.ndarray.prod`

        """
        return _prod(self, axis, dtype, out, keepdims)

    # TODO(okuta): Implement cumprod

    cpdef ndarray all(self, axis=None, out=None, keepdims=False):
        return _all(self, axis=axis, out=out, keepdims=keepdims)

    cpdef ndarray any(self, axis=None, out=None, keepdims=False):
        return _any(self, axis=axis, out=out, keepdims=keepdims)

    # -------------------------------------------------------------------------
    # Arithmetic and comparison operations
    # -------------------------------------------------------------------------
    # Comparison operators:

    def __richcmp__(object self, object other, int op):
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
        return negative(self)

    def __pos__(self):
        return self

    def __abs__(self):
        return absolute(self)

    def __invert__(self):
        return invert(self)

    # Arithmetic:

    def __add__(x, y):
        if _should_use_rop(x, y):
            return y.__radd__(x)
        else:
            return add(x, y)

    def __sub__(x, y):
        if _should_use_rop(x, y):
            return y.__rsub__(x)
        else:
            return subtract(x, y)

    def __mul__(x, y):
        if _should_use_rop(x, y):
            return y.__rmul__(x)
        else:
            return multiply(x, y)

    def __matmul__(x, y):
        if _should_use_rop(x, y):
            return y.__rmatmul__(x)
        else:
            return matmul(x, y)

    def __div__(x, y):
        if _should_use_rop(x, y):
            return y.__rdiv__(x)
        else:
            return divide(x, y)

    def __truediv__(x, y):
        if _should_use_rop(x, y):
            return y.__rtruediv__(x)
        else:
            return true_divide(x, y)

    def __floordiv__(x, y):
        if _should_use_rop(x, y):
            return y.__rfloordiv__(x)
        else:
            return floor_divide(x, y)

    def __mod__(x, y):
        if _should_use_rop(x, y):
            return y.__rmod__(x)
        else:
            return remainder(x, y)

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
            return power(x, y)

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
        return add(self, other, self)

    def __isub__(self, other):
        return subtract(self, other, self)

    def __imul__(self, other):
        return multiply(self, other, self)

    def __idiv__(self, other):
        return divide(self, other, self)

    def __itruediv__(self, other):
        return true_divide(self, other, self)

    def __ifloordiv__(self, other):
        return floor_divide(self, other, self)

    def __imod__(self, other):
        return remainder(self, other, self)

    def __ipow__(self, other):
        return power(self, other, self)

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

    # -------------------------------------------------------------------------
    # Special methods
    # -------------------------------------------------------------------------
    # For standard library functions:

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
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

           Examples
           --------
           >>> a = cupy.arange(3)
           >>> a[[1, 3]]
           array([1, 0])

        """
        # supports basic indexing (by slices, ints or Ellipsis) and
        # some parts of advanced indexing by integer or boolean arrays.
        # TODO(beam2d): Support the advanced indexing of NumPy.
        cdef Py_ssize_t i, j, offset, ndim, n_newaxes, n_ellipses, ellipsis
        cdef Py_ssize_t ellipsis_sizem, s_start, s_stop, s_step, dim, ind
        cdef vector.vector[Py_ssize_t] shape, strides
        if isinstance(slices, tuple):
            slices = list(slices)
        elif isinstance(slices, list):
            slices = list(slices)  # copy list
            if all([isinstance(s, int) for s in slices]):
                slices = [slices]
        else:
            slices = [slices]

        # Expand ellipsis into empty slices
        ellipsis = -1
        n_newaxes = n_ellipses = 0
        for i, s in enumerate(slices):
            if s is None:
                n_newaxes += 1
            elif s is Ellipsis:
                n_ellipses += 1
                ellipsis = i
        ndim = self._shape.size()
        noneslices = [slice(None)]
        if n_ellipses > 0:
            if n_ellipses > 1:
                raise ValueError('Only one Ellipsis is allowed in index')
            ellipsis_size = ndim - (<Py_ssize_t>len(slices) - n_newaxes - 1)
            slices[ellipsis:ellipsis + 1] = noneslices * ellipsis_size

        slices += noneslices * (ndim - <Py_ssize_t>len(slices) + n_newaxes)

        if len(slices) > self.ndim + n_newaxes:
            raise IndexError('too many indices for array')

        # Check if advanced is true,
        # and convert list/NumPy arrays to cupy.ndarray
        advanced = False
        mask_exists = False
        for i, s in enumerate(slices):
            if isinstance(s, (list, numpy.ndarray)):
                is_list = isinstance(s, list)
                s = array(s)
                # handle the case when s is an empty list
                if is_list and s.size == 0:
                    s = s.astype(numpy.int32)
                slices[i] = s
            if isinstance(s, ndarray):
                if issubclass(s.dtype.type, numpy.integer):
                    advanced = True
                elif issubclass(s.dtype.type, numpy.bool_):
                    mask_exists = True
                else:
                    raise IndexError(
                        'arrays used as indices must be of integer or boolean '
                        'type. (actual: {})'.format(s.dtype.type))

        if mask_exists:
            n_not_slice_none = 0
            for i, s in enumerate(slices):
                if not isinstance(s, slice) or s != slice(None):
                    n_not_slice_none += 1
                    if issubclass(s.dtype.type, numpy.bool_):
                        mask_i = i
            if n_not_slice_none != 1:
                raise ValueError('currently, CuPy only supports slices that '
                                 'consist of one boolean array.')
            return _getitem_mask_single(self, slices[mask_i], mask_i)

        if advanced:
            # split slices that can be handled by basic-indexing
            basic_slices = []
            adv_slices = []
            for i, s in enumerate(slices):
                if type(s) is slice:
                    basic_slices.append(s)
                    adv_slices.append(slice(None))
                elif s is None:
                    basic_slices.append(None)
                    adv_slices.append(slice(None))
                elif (isinstance(s, ndarray) and
                        issubclass(s.dtype.type, numpy.integer)):
                    basic_slices.append(slice(None))
                    adv_slices.append(s)
                elif isinstance(s, int):
                    basic_slices.append(slice(None))
                    scalar_array = ndarray((), dtype=numpy.int64)
                    scalar_array.fill(s)
                    adv_slices.append(scalar_array)
                else:
                    raise IndexError(
                        'only integers, slices (`:`), ellipsis (`...`),'
                        'numpy.newaxis (`None`) and integer or'
                        'boolean arrays are valid indices')

            # check if this is a combination of basic and advanced indexing
            a = self
            for s in basic_slices:
                if s is None or (isinstance(s, slice) and s != slice(None)):
                    a = self[tuple(basic_slices)]
                    break

            arr_slices_mask = [not isinstance(s, slice) for s in adv_slices]
            if sum(arr_slices_mask) == 1:
                axis = arr_slices_mask.index(True)
                return a.take(adv_slices[axis], axis)
            return _getitem_multiple(a, adv_slices)

        # Create new shape and stride
        j = 0
        offset = 0
        for i, s in enumerate(slices):
            if s is None:
                shape.push_back(1)
                if j < ndim:
                    strides.push_back(self._strides[j])
                elif ndim > 0:
                    strides.push_back(self._strides[ndim - 1])
                else:
                    strides.push_back(self.itemsize)
            elif ndim <= j:
                raise IndexError("too many indices for array")
            elif isinstance(s, slice):
                s = internal.complete_slice(s, self._shape[j])
                s_start = s.start
                s_stop = s.stop
                s_step = s.step
                if s_step > 0:
                    dim = (s_stop - s_start - 1) // s_step + 1
                else:
                    dim = (s_stop - s_start + 1) // s_step + 1

                shape.push_back(dim)
                strides.push_back(self._strides[j] * s_step)

                offset += s_start * self._strides[j]
                j += 1
            elif numpy.isscalar(s):
                ind = int(s)
                if ind < 0:
                    ind += self._shape[j]
                if not (0 <= ind < self._shape[j]):
                    msg = ('Index %s is out of bounds for axis %s with '
                           'size %s' % (s, j, self._shape[j]))
                    raise IndexError(msg)
                offset += ind * self._strides[j]
                j += 1
            else:
                raise TypeError('Invalid index type: %s' % type(slices[i]))

        v = self.view()
        v.data = self.data + offset
        v._set_shape_and_strides(shape, strides)
        return v

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

            >>> import cupy; import numpy
            >>> a = cupy.zeros((2,))
            >>> i = cupy.arange(10000) % 2
            >>> v = cupy.arange(10000).astype(numpy.float)
            >>> a[i] = v
            >>> a  # doctest: +SKIP
            array([ 9150.,  9151.])

            On the other hand, NumPy stores the value corresponding to the
            last index among the indices referencing duplicate locations.

            >>> import numpy
            >>> a_cpu = numpy.zeros((2,))
            >>> i_cpu = numpy.arange(10000) % 2
            >>> v_cpu = numpy.arange(10000).astype(numpy.float)
            >>> a_cpu[i_cpu] = v_cpu
            >>> a_cpu
            array([ 9998.,  9999.])

        """
        _scatter_op(self, slices, value, 'update')

    def scatter_add(self, slices, value):
        """Adds given values to specified elements of an array.

        .. seealso::
            :func:`cupy.scatter_add` for full documentation.

        """
        _scatter_op(self, slices, value, 'add')

    # TODO(okuta): Implement __getslice__
    # TODO(okuta): Implement __setslice__
    # TODO(okuta): Implement __contains__

    # Conversion:

    def __int__(self):
        return int(self.get())

    if sys.version_info < (3,):
        def __long__(self):
            # Avoid using long() for flake8
            return self.get().__long__()

    def __float__(self):
        return float(self.get())

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

    cpdef get(self, stream=None):
        """Returns a copy of the array on host memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.

        Returns:
            numpy.ndarray: Copy of the array on host memory.

        """
        if self.size == 0:
            return numpy.ndarray(self.shape, dtype=self.dtype)

        with self.device:
            a_gpu = ascontiguousarray(self)
        a_cpu = numpy.empty(self._shape, dtype=self.dtype)
        ptr = a_cpu.ctypes.get_as_parameter()
        if stream is None:
            a_gpu.data.copy_to_host(ptr, a_gpu.nbytes)
        else:
            a_gpu.data.copy_to_host_async(ptr, a_gpu.nbytes, stream)
        return a_cpu

    cpdef set(self, arr, stream=None):
        """Copies an array on the host memory to :class:`cupy.ndarray`.

        Args:
            arr (numpy.ndarray): The source array on the host memory.
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.

        """
        if not isinstance(arr, numpy.ndarray):
            raise TypeError('Only numpy.ndarray can be set to cupy.ndarray')
        if self.dtype != arr.dtype:
            raise TypeError('{} array cannot be set to {} array'.format(
                arr.dtype, self.dtype))
        if self.shape != arr.shape:
            raise ValueError('Shape mismatch')
        if not self._c_contiguous:
            raise RuntimeError('Cannot set to non-contiguous array')

        arr = numpy.ascontiguousarray(arr)
        ptr = arr.ctypes.get_as_parameter()
        if stream is None:
            self.data.copy_from_host(ptr, self.nbytes)
        else:
            self.data.copy_from_host_async(ptr, self.nbytes, stream)

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
        view._set_shape_and_strides(shape, strides)
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

    cpdef _set_shape_and_strides(self, vector.vector[Py_ssize_t]& shape,
                                 vector.vector[Py_ssize_t]& strides,
                                 bint update_c_contiguity=True):
        if shape.size() != strides.size():
            raise ValueError('len(shape) != len(strides)')
        self._shape = shape
        self._strides = strides
        self.size = internal.prod_ssize_t(shape)
        if update_c_contiguity:
            self._update_contiguity()
        else:
            self._update_f_contiguity()

    cdef function.CPointer get_pointer(self):
        return CArray(self)


cdef object newaxis = numpy.newaxis  # == None


cpdef vector.vector[Py_ssize_t] _get_strides_for_nocopy_reshape(
        ndarray a, vector.vector[Py_ssize_t]& newshape) except *:
    cdef vector.vector[Py_ssize_t] newstrides
    cdef Py_ssize_t size, itemsize, ndim, dim, last_stride
    size = a.size
    if size != internal.prod_ssize_t(newshape):
        return newstrides

    itemsize = a.itemsize
    if size == 1:
        newstrides.assign(<Py_ssize_t>newshape.size(), itemsize)
        return newstrides

    cdef vector.vector[Py_ssize_t] shape, strides
    internal.get_reduced_dims(a._shape, a._strides, itemsize, shape, strides)

    ndim = shape.size()
    dim = 0
    sh = shape[0]
    st = strides[0]
    last_stride = shape[0] * strides[0]
    for size in newshape:
        if size <= 1:
            newstrides.push_back(last_stride)
            continue
        if dim >= ndim or shape[dim] % size != 0:
            newstrides.clear()
            break
        shape[dim] //= size
        last_stride = shape[dim] * strides[dim]
        newstrides.push_back(last_stride)
        if shape[dim] == 1:
            dim += 1
    return newstrides


include "carray.pxi"
include "elementwise.pxi"
include "reduction.pxi"


# =============================================================================
# Routines
# =============================================================================

cdef _id = 'out0 = in0'

_elementwise_copy = create_ufunc(
    'cupy_copy',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'),
    _id)


def elementwise_copy(*args, **kwargs):
    kwargs['casting'] = 'unsafe'
    return _elementwise_copy(*args, **kwargs)


_elementwise_copy_where = create_ufunc(
    'cupy_copy_where',
    ('??->?', 'b?->b', 'B?->B', 'h?->h', 'H?->H', 'i?->i', 'I?->I', 'l?->l',
     'L?->L', 'q?->q', 'Q?->Q', 'e?->e', 'f?->f', 'd?->d'),
    'if (in1) out0 = in0')


def elementwise_copy_where(*args, **kwargs):
    kwargs['casting'] = 'unsafe'
    return _elementwise_copy_where(*args, **kwargs)


cdef _divmod_float = '''
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


cdef _min_max_preamble = '''
template <typename T>
struct min_max_st{
    T value;
    int index;
    __device__ min_max_st() : index(-1) { }
    __device__ min_max_st(T v) : value(v), index(0) { }
    __device__ min_max_st(T v, int i) : value(v), index(i) { }
};

template <typename T>
inline __device__ bool is_nan(T x) {
    return x != x;
}

template <typename T>
__device__ min_max_st<T> my_min(
        const min_max_st<T>& a, const min_max_st<T>& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return min_max_st<T>(min(a.value, b.value));
}
template <typename T>
__device__ min_max_st<T> my_min_float(
        const min_max_st<T>& a, const min_max_st<T>& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    if (is_nan(a.value)) return a;
    if (is_nan(b.value)) return b;
    return min_max_st<T>(min(a.value, b.value));
}

template <typename T>
__device__ min_max_st<T> my_max(
        const min_max_st<T>& a, const min_max_st<T>& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return min_max_st<T>(max(a.value, b.value));
}
template <typename T>
__device__ min_max_st<T> my_max_float(
        const min_max_st<T>& a, const min_max_st<T>& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    if (is_nan(a.value)) return a;
    if (is_nan(b.value)) return b;
    return min_max_st<T>(max(a.value, b.value));
}

template <typename T>
__device__ min_max_st<T> my_argmin(
        const min_max_st<T>& a, const min_max_st<T>& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    if (a.value == b.value)
        return min_max_st<T>(a.value, min(a.index, b.index));
    return (a.value <= b.value) ? a : b;
}
template <typename T>
__device__ min_max_st<T> my_argmin_float(
        const min_max_st<T>& a, const min_max_st<T>& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    if (a.value == b.value)
        return min_max_st<T>(a.value, min(a.index, b.index));
    if (is_nan(a.value)) return a;
    if (is_nan(b.value)) return b;
    return (a.value <= b.value) ? a : b;
}

template <typename T>
__device__ min_max_st<T> my_argmax(
        const min_max_st<T>& a, const min_max_st<T>& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    if (a.value == b.value)
        return min_max_st<T>(a.value, min(a.index, b.index));
    return (a.value >= b.value) ? a : b;
}
template <typename T>
__device__ min_max_st<T> my_argmax_float(
        const min_max_st<T>& a, const min_max_st<T>& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    if (a.value == b.value)
        return min_max_st<T>(a.value, min(a.index, b.index));
    if (is_nan(a.value)) return a;
    if (is_nan(b.value)) return b;
    return (a.value >= b.value) ? a : b;
}
'''


_amin = create_reduction_func(
    'cupy_min',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, 'my_min_float(a, b)', None, None)),
     ('f->f', (None, 'my_min_float(a, b)', None, None)),
     ('d->d', (None, 'my_min_float(a, b)', None, None))),
    ('min_max_st<type_in0_raw>(in0)', 'my_min(a, b)', 'out0 = a.value',
     'min_max_st<type_in0_raw>'),
    None, _min_max_preamble)


_amax = create_reduction_func(
    'cupy_max',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, 'my_max_float(a, b)', None, None)),
     ('f->f', (None, 'my_max_float(a, b)', None, None)),
     ('d->d', (None, 'my_max_float(a, b)', None, None))),
    ('min_max_st<type_in0_raw>(in0)', 'my_max(a, b)', 'out0 = a.value',
     'min_max_st<type_in0_raw>'),
    None, _min_max_preamble)


nanmin = create_reduction_func(
    'cupy_nanmin',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'),
    ('min_max_st<type_in0_raw>(in0)', 'my_min(a, b)', 'out0 = a.value',
     'min_max_st<type_in0_raw>'),
    None, _min_max_preamble)


nanmax = create_reduction_func(
    'cupy_nanmax',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'),
    ('min_max_st<type_in0_raw>(in0)', 'my_max(a, b)', 'out0 = a.value',
     'min_max_st<type_in0_raw>'),
    None, _min_max_preamble)


cdef _argmin = create_reduction_func(
    'cupy_argmin',
    ('?->q', 'B->q', 'h->q', 'H->q', 'i->q', 'I->q', 'l->q', 'L->q',
     'q->q', 'Q->q',
     ('e->q', (None, 'my_argmin_float(a, b)', None, None)),
     ('f->q', (None, 'my_argmin_float(a, b)', None, None)),
     ('d->q', (None, 'my_argmin_float(a, b)', None, None))),
    ('min_max_st<type_in0_raw>(in0, _J)', 'my_argmin(a, b)', 'out0 = a.index',
     'min_max_st<type_in0_raw>'),
    None, _min_max_preamble)


cdef _argmax = create_reduction_func(
    'cupy_argmax',
    ('?->q', 'B->q', 'h->q', 'H->q', 'i->q', 'I->q', 'l->q', 'L->q',
     'q->q', 'Q->q',
     ('e->q', (None, 'my_argmax_float(a, b)', None, None)),
     ('f->q', (None, 'my_argmax_float(a, b)', None, None)),
     ('d->q', (None, 'my_argmax_float(a, b)', None, None))),
    ('min_max_st<type_in0_raw>(in0, _J)', 'my_argmax(a, b)', 'out0 = a.index',
     'min_max_st<type_in0_raw>'),
    None, _min_max_preamble)


# -----------------------------------------------------------------------------
# Array creation routines
# -----------------------------------------------------------------------------

cpdef ndarray array(obj, dtype=None, bint copy=True, Py_ssize_t ndmin=0):
    # TODO(beam2d): Support order and subok options
    cdef Py_ssize_t nvidem
    cdef ndarray a
    if isinstance(obj, ndarray):
        if dtype is None:
            dtype = obj.dtype
        a = obj.astype(dtype, copy)

        ndim = a._shape.size()
        if ndmin > ndim:
            if a is obj:
                # When `copy` is False, `a` is same as `obj`.
                a = a.view()
            a.shape = (1,) * (ndmin - ndim) + a.shape
        return a
    else:
        a_cpu = numpy.array(obj, dtype=dtype, copy=False, ndmin=ndmin)
        if a_cpu.dtype.char not in '?bhilqBHILQefd':
            raise ValueError('Unsupported dtype %s' % a_cpu.dtype)
        if a_cpu.ndim > 0:
            a_cpu = numpy.ascontiguousarray(a_cpu)
        a = ndarray(a_cpu.shape, dtype=a_cpu.dtype)
        a.data.copy_from_host(a_cpu.ctypes.get_as_parameter(), a.nbytes)
        if a_cpu.dtype == a.dtype:
            return a
        else:
            return a.view(dtype=a_cpu.dtype)


cpdef ndarray ascontiguousarray(ndarray a, dtype=None):
    if dtype is None:
        if a._c_contiguous:
            return a
        dtype = a.dtype
    else:
        dtype = numpy.dtype(dtype)
        if a._c_contiguous and dtype == a.dtype:
            return a

    newarray = ndarray(a.shape, dtype)
    elementwise_copy(a, newarray)
    return newarray


cpdef ndarray asfortranarray(ndarray a, dtype=None):
    cdef ndarray newarray
    cdef int m, n

    if dtype is None:
        if a._f_contiguous:
            return a
        dtype = a.dtype
    else:
        dtype = numpy.dtype(dtype)
        if a._f_contiguous and dtype == a.dtype:
            return a

    newarray = ndarray(a.shape, dtype, order='F')
    if (a.flags.c_contiguous and
            (a.dtype == numpy.float32 or a.dtype == numpy.float64) and
            a.ndim == 2 and dtype == a.dtype):
        m, n = a.shape
        if a.dtype == numpy.float32:
            cuda.cublas.sgeam(
                cuda.Device().cublas_handle,
                1,  # transpose a
                1,  # transpose newarray
                m, n, 1., a.data.ptr, n, 0., a.data.ptr, n,
                newarray.data.ptr, m)
        elif a.dtype == numpy.float64:
            cuda.cublas.dgeam(
                cuda.Device().cublas_handle,
                1,  # transpose a
                1,  # transpose newarray
                m, n, 1., a.data.ptr, n, 0., a.data.ptr, n,
                newarray.data.ptr, m)
        return newarray
    else:
        elementwise_copy(a, newarray)
        return newarray


# -----------------------------------------------------------------------------
# Array manipulation routines
# -----------------------------------------------------------------------------

cpdef ndarray rollaxis(ndarray a, Py_ssize_t axis, Py_ssize_t start=0):
    cdef Py_ssize_t i, ndim=a.ndim
    cdef vector.vector[Py_ssize_t] axes
    if axis < 0:
        axis += ndim
    if start < 0:
        start += ndim
    if not (0 <= axis < ndim and 0 <= start <= ndim):
        raise ValueError('Axis out of range')
    if axis < start:
        start -= 1
    if axis == start:
        return a
    if ndim == 2:
        return a._transpose(axes)

    for i in range(ndim):
        axes.push_back(i)
    axes.erase(axes.begin() + axis)
    axes.insert(axes.begin() + start, axis)
    return a._transpose(axes)


def array_split(ndarray ary, indices_or_sections, int axis):

    cdef int i, ndim, size, each_size, index, prev, offset, stride
    cdef vector.vector[Py_ssize_t] shape

    ndim = ary.ndim
    if -ndim > axis or ndim <= axis:
        raise IndexError('Axis exceeds ndim')
    if axis < 0:
        axis += ndim
    size = ary._shape[axis]

    if numpy.isscalar(indices_or_sections):
        each_size = (size - 1) // indices_or_sections + 1
        indices = [i * each_size
                   for i in range(1, indices_or_sections)]
    else:
        indices = indices_or_sections

    if len(indices) == 0:
        return [ary]

    # Make a copy of shape for each view
    shape = ary._shape

    prev = 0
    ret = []
    stride = ary._strides[axis]
    for index in indices:
        shape[axis] = index - prev
        v = ary.view()
        v.data = ary.data + prev * stride
        v._set_shape_and_strides(shape, ary._strides)
        ret.append(v)

        prev = index

    shape[axis] = size - prev
    v = ary.view()
    v.data = ary.data + prev * stride
    v._set_shape_and_strides(shape, ary._strides)
    ret.append(v)

    return ret


cdef class broadcast:
    """Object that performs broadcasting.

    CuPy actually uses this class to support broadcasting in various
    operations. Note that this class does not provide an iterator.

    Args:
        arrays (tuple of arrays): Arrays to be broadcasted.

    Attributes:
        shape (tuple of ints): The broadcasted shape.
        nd (int): Number of dimensions of the broadcasted shape.
        size (int): Total size of the broadcasted shape.
        values (list of arrays): The broadcasted arrays.

    .. seealso:: :class:`numpy.broadcast`

    """

    cdef:
        readonly tuple values
        readonly tuple shape
        readonly Py_ssize_t size
        readonly Py_ssize_t nd

    def __init__(self, *arrays):
        cdef Py_ssize_t i, j, s, ss, a_ndim, a_sh
        cdef vector.vector[Py_ssize_t] shape, strides, r_shape, r_strides
        cdef vector.vector[vector.vector[Py_ssize_t]] shape_arr
        cdef ndarray a, view
        rev = slice(None, None, -1)

        self.nd = 0
        for x in arrays:
            if not isinstance(x, ndarray):
                continue
            a = x
            self.nd = max(self.nd, <Py_ssize_t>a._shape.size())
            r_shape.assign(a._shape.rbegin(), a._shape.rend())
            shape_arr.push_back(r_shape)

        r_shape.clear()
        for i in range(self.nd):
            ss = 0
            for j in range(<Py_ssize_t>shape_arr.size()):
                if i < <Py_ssize_t>shape_arr[j].size():
                    s = shape_arr[j][i]
                    ss = max(ss, s)
            r_shape.push_back(ss)

        shape.assign(r_shape.rbegin(), r_shape.rend())
        self.shape = tuple(shape)
        self.size = internal.prod_ssize_t(shape)

        broadcasted = []
        for x in arrays:
            if not isinstance(x, ndarray):
                broadcasted.append(x)
                continue
            a = x
            if internal.vector_equal(a._shape, shape):
                broadcasted.append(a)
                continue

            r_strides.assign(self.nd, <Py_ssize_t>0)
            a_ndim = a._shape.size()
            for i in range(a_ndim):
                a_sh = a._shape[a_ndim - i - 1]
                if a_sh == r_shape[i]:
                    r_strides[i] = a._strides[a_ndim - i - 1]
                elif a_sh != 1:
                    raise ValueError('Broadcasting failed')

            strides.assign(r_strides.rbegin(), r_strides.rend())
            view = a.view()
            view._set_shape_and_strides(shape, strides)
            broadcasted.append(view)

        self.values = tuple(broadcasted)


cpdef ndarray broadcast_to(ndarray array, shape):
    """Broadcast an array to a given shape.

    .. seealso::
        :func:`cupy.broadcast_to` for full documentation,
        :meth:`numpy.broadcast_to`

    """
    if array.ndim > len(shape):
        raise ValueError(
            'input operand has more dimensions than allowed by the axis '
            'remapping')

    strides = [0] * len(shape)
    for i in range(array.ndim):
        j = -i - 1
        sh = shape[j]
        a_sh = array.shape[j]
        if sh == a_sh:
            strides[j] = array._strides[j % array.ndim]
        elif a_sh != 1:
            raise ValueError('Broadcasting failed')

    view = array.view()
    view._set_shape_and_strides(shape, strides)
    return view


cpdef ndarray _repeat(ndarray a, repeats, axis=None):
    """Repeat arrays along an axis.

    Args:
        a (cupy.ndarray): Array to transform.
        repeats (int, list or tuple): The number of repeats.
        axis (int): The axis to repeat.

    Returns:
        cupy.ndarray: Transformed array with repeats.

    .. seealso:: :func:`numpy.repeat`

    """
    cdef ndarray ret
    if isinstance(repeats, int):
        if repeats < 0:
            raise ValueError(
                "'repeats' should not be negative: {}".format(repeats))
        if axis is None:
            a = a.reshape((-1, 1))
            ret = ndarray((a.size, repeats), dtype=a.dtype)
            if ret.size:
                ret[...] = a
            return ret.ravel()

        repeats = [repeats] * a._shape[axis % a._shape.size()]
    elif cpython.PySequence_Check(repeats):
        for rep in repeats:
            if rep < 0:
                raise ValueError(
                    "all elements of 'repeats' should not be negative: {}"
                    .format(repeats))
        if axis is None:
            raise ValueError(
                "'axis' should be specified if 'repeats' is sequence")
        if a.shape[axis] != len(repeats):
            raise ValueError(
                "'repeats' and 'axis' of 'a' should be same length: {} != {}"
                .format(a.shape[axis], len(repeats)))
    else:
        raise ValueError(
            "'repeats' should be int or sequence: {}".format(repeats))

    if axis < 0:
        axis += a.ndim

    ret_shape = list(a.shape)
    ret_shape[axis] = sum(repeats)
    ret = ndarray(ret_shape, dtype=a.dtype)
    a_index = [slice(None)] * len(ret_shape)
    ret_index = list(a_index)
    offset = 0
    for i in range(a._shape[axis]):
        if repeats[i] == 0:
            continue
        a_index[axis] = slice(i, i + 1)
        ret_index[axis] = slice(offset, offset + repeats[i])
        # convert to tuple because cupy has a indexing bug
        ret[tuple(ret_index)] = a[tuple(a_index)]
        offset += repeats[i]
    return ret


cpdef ndarray concatenate_method(tup, int axis):
    cdef int ndim
    cdef int i
    cdef ndarray a
    cdef bint have_same_types
    cdef vector.vector[Py_ssize_t] shape

    ndim = -1
    dtype = None
    have_same_types = True
    for o in tup:
        if not isinstance(o, ndarray):
            raise TypeError('Only cupy arrays can be concatenated')
        a = o
        if a.ndim == 0:
            raise TypeError('zero-dimensional arrays cannot be concatenated')
        if ndim == -1:
            ndim = a.ndim
            shape = a._shape
            if axis < 0:
                axis += ndim
            if axis < 0 or axis >= ndim:
                raise IndexError(
                    'axis {} out of bounds [0, {})'.format(axis, ndim))
            dtype = a.dtype
            continue

        have_same_types = have_same_types and (a.dtype == dtype)
        if a.ndim != ndim:
            raise ValueError(
                'All arrays to concatenate must have the same ndim')
        for i in range(ndim):
            if i != axis and shape[i] != a._shape[i]:
                raise ValueError(
                    'All arrays must have same shape except the axis to '
                    'concatenate')
        shape[axis] += a._shape[axis]

    if ndim == -1:
        raise ValueError('Cannot concatenate from empty tuple')

    if not have_same_types:
        dtype = numpy.find_common_type([a.dtype for a in tup], [])
    return concatenate(tup, axis, shape, dtype)


cpdef ndarray concatenate(tup, axis, shape, dtype):
    cdef ndarray a, x, ret
    cdef int i, j, base, cum, ndim
    cdef bint all_same_type, all_one_and_contiguous
    cdef Py_ssize_t[:] ptrs
    cdef int[:] cum_sizes
    cdef int[:, :] x_strides

    ret = ndarray(shape, dtype=dtype)

    if len(tup) > 3:
        all_same_type = True
        all_one_and_contiguous = True
        dtype = tup[0].dtype
        for a in tup:
            all_same_type = all_same_type and (a.dtype == dtype)
            all_one_and_contiguous = (
                all_one_and_contiguous and a._c_contiguous and
                a._shape[axis] == 1)

        if all_same_type:
            ptrs = numpy.ndarray(len(tup), numpy.int64)
            for i, a in enumerate(tup):
                ptrs[i] = a.data.ptr
            x = array(ptrs)

            if all_one_and_contiguous:
                base = internal.prod_ssize_t(shape[axis + 1:])
                _concatenate_kernel_one(x, base, ret)
            else:
                ndim = tup[0].ndim
                x_strides = numpy.ndarray((len(tup), ndim), numpy.int32)
                cum_sizes = numpy.ndarray(len(tup), numpy.int32)
                cum = 0
                for i, a in enumerate(tup):
                    for j in range(ndim):
                        x_strides[i, j] = a._strides[j]
                    cum_sizes[i] = cum
                    cum += a._shape[axis]

                _concatenate_kernel(
                    x, axis, len(shape), array(cum_sizes), array(x_strides),
                    ret)
            return ret

    skip = (slice(None),) * axis
    i = 0
    for a in tup:
        aw = a._shape[axis]
        ret[skip + (slice(i, i + aw),)] = a
        i += aw

    return ret

cdef _concatenate_kernel_one = ElementwiseKernel(
    'raw P x, int32 base',
    'T y',
    '''
    int middle = i / base;
    int top = middle / x.size();
    int array_ind = middle - top * x.size();
    int offset = i + (top - middle) * base;
    y = reinterpret_cast<T*>(x[array_ind])[offset];
    ''',
    'cupy_concatenate_one'
)


cdef _concatenate_kernel = ElementwiseKernel(
    '''raw P x, int32 axis, int32 ndim, raw int32 cum_sizes,
    raw int32 x_strides''',
    'T y',
    '''
    int axis_ind = _ind.get()[axis];
    int left = 0;
    int right = cum_sizes.size();

    while (left < right - 1) {
      int m = (left + right) / 2;
      if (axis_ind < cum_sizes[m]) {
        right = m;
      } else {
        left = m;
      }
    }

    int array_ind = left;
    axis_ind -= cum_sizes[left];
    char* ptr = reinterpret_cast<char*>(x[array_ind]);
    for (int j = ndim - 1; j >= 0; --j) {
      int ind[] = {array_ind, j};
      int offset;
      if (j == axis) {
        offset = axis_ind;
      } else {
        offset = _ind.get()[j];
      }
      ptr += x_strides[ind] * offset;
    }

    y = *reinterpret_cast<T*>(ptr);
    ''',
    'cupy_concatenate',
    reduce_dims=False
)


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
# Indexing routines
# -----------------------------------------------------------------------------

cdef _take_kernel = ElementwiseKernel(
    'raw T a, S indices, int32 cdim, int32 rdim, int32 adim, S index_range',
    'T out',
    '''
      S wrap_indices = indices % index_range;
      if (wrap_indices < 0) wrap_indices += index_range;

      int li = i / (rdim * cdim);
      int ri = i % rdim;
      out = a[(li * adim + wrap_indices) * rdim + ri];
    ''',
    'cupy_take')


cdef _take_kernel_0axis = ElementwiseKernel(
    'raw T a, S indices, int32 rdim, S index_range',
    'T out',
    '''
      S wrap_indices = indices % index_range;
      if (wrap_indices < 0) wrap_indices += index_range;

      out = a[wrap_indices * rdim + i % rdim];
    ''',
    'cupy_take_0axis')


cdef _choose_kernel = ElementwiseKernel(
    'S a, raw T choices, int32 n_channel',
    'T y',
    'y = choices[i + n_channel * a]',
    'cupy_choose')


cdef _choose_clip_kernel = ElementwiseKernel(
    'S a, raw T choices, int32 n_channel, int32 n',
    'T y',
    '''
      S x = a;
      if (a < 0) {
        x = 0;
      } else if (a >= n) {
        x = n - 1;
      }
      y = choices[i + n_channel * x];
    ''',
    'cupy_choose_clip')


cdef _scatter_update_kernel = ElementwiseKernel(
    'T v, S indices, int32 cdim, int32 rdim, int32 adim',
    'raw T a',
    '''
      S wrap_indices = indices % adim;
      if (wrap_indices < 0) wrap_indices += adim;
      int li = i / (rdim * cdim);
      int ri = i % rdim;
      a[(li * adim + wrap_indices) * rdim + ri] = v;
    ''',
    'cupy_scatter_update')


cdef _scatter_add_kernel = ElementwiseKernel(
    'raw T v, S indices, int32 cdim, int32 rdim, int32 adim',
    'raw T a',
    '''
      S wrap_indices = indices % adim;
      if (wrap_indices < 0) wrap_indices += adim;
      int li = i / (rdim * cdim);
      int ri = i % rdim;
      atomicAdd(&a[(li * adim + wrap_indices) * rdim + ri], v[i]);
    ''',
    'cupy_scatter_add')


cdef _scatter_update_mask_kernel = ElementwiseKernel(
    'raw T v, bool mask, S mask_scanned',
    'T a',
    'if (mask) a = v[mask_scanned - 1]',
    'cupy_scatter_update_mask')


cdef _scatter_add_mask_kernel = ElementwiseKernel(
    'raw T v, bool mask, S mask_scanned',
    'T a',
    'if (mask) a = a + v[mask_scanned - 1]',
    'cupy_scatter_add_mask')


cdef _getitem_mask_kernel = ElementwiseKernel(
    'T a, bool mask, S mask_scanned',
    'raw T out',
    'if (mask) out[mask_scanned - 1] = a',
    'cupy_getitem_mask')


cpdef _prepare_mask_indexing_single(ndarray a, ndarray mask, int axis):
    cdef ndarray mask_scanned, mask_br, mask_br_scanned
    cdef int n_true
    cdef tuple lshape, rshape, out_shape

    lshape = a.shape[:axis]
    rshape = a.shape[axis + mask.ndim:]

    if mask.size == 0:
        masked_shape = lshape + (0,) + rshape
        mask_br = mask._reshape(masked_shape)
        return mask_br, mask_br, masked_shape

    # Get number of True in the mask to determine the shape of the array
    # after masking.
    if mask.size <= 2 ** 31 - 1:
        mask_type = numpy.int32
    else:
        mask_type = numpy.int64
    mask_scanned = scan(mask.astype(mask_type).ravel())  # starts with 1
    n_true = int(mask_scanned[-1])
    masked_shape = lshape + (n_true,) + rshape

    # When mask covers the entire array, broadcasting is not necessary.
    if mask.ndim == a.ndim and axis == 0:
        return mask, mask_scanned._reshape(mask._shape), masked_shape

    # The scan of the broadcasted array is used to index on kernel.
    mask_br = mask._reshape(
        axis * (1,) + mask.shape + (a.ndim - axis - mask.ndim) * (1,))
    mask_br = broadcast_to(mask_br, a.shape)
    if mask.size <= 2 ** 31 - 1:
        mask_type = numpy.int32
    else:
        mask_type = numpy.int64
    mask_br_scanned = scan(mask_br.astype(mask_type).ravel())
    mask_br_scanned = mask_br_scanned._reshape(mask_br._shape)
    return mask_br, mask_br_scanned, masked_shape


cpdef ndarray _getitem_mask_single(ndarray a, ndarray mask, int axis):
    cdef ndarray mask_scanned
    cdef tuple masked_shape

    mask, mask_scanned, masked_shape = _prepare_mask_indexing_single(
        a, mask, axis)
    out = ndarray(masked_shape, dtype=a.dtype)
    if out.size == 0:
        return out
    return _getitem_mask_kernel(a, mask, mask_scanned, out)


cpdef ndarray _take(ndarray a, indices, li=None, ri=None, ndarray out=None):
    # When li == ri, this function behaves similarly to np.take
    if a.ndim == 0:
        a = a[None]

    if li is None and ri is None:
        a = a.ravel()
        lshape = ()
        rshape = ()
        adim = 1
        index_range = a.size
    else:
        if not (-a.ndim <= li < a.ndim and -a.ndim <= ri < a.ndim):
            raise ValueError('Axis overrun')
        if a.ndim != 0:
            li %= a.ndim
            ri %= a.ndim

        lshape = a.shape[:li]
        rshape = a.shape[ri + 1:]
        adim = internal.prod(a.shape[li:ri + 1])
        index_range = adim

    if numpy.isscalar(indices):
        indices %= index_range
        if li is not None and ri is not None and li == ri:
            a = rollaxis(a, li)
        if out is None:
            return a[indices].copy()
        else:
            if out.dtype != a.dtype:
                raise TypeError('Output dtype mismatch')
            if out.shape != a.shape[1:]:
                raise ValueError('Output shape mismatch')
            out[()] = a[indices]
            return out
    elif not isinstance(indices, ndarray):
        indices = array(indices, dtype=int)

    out_shape = lshape + indices.shape + rshape
    if out is None:
        out = ndarray(out_shape, dtype=a.dtype)
    else:
        if out.dtype != a.dtype:
            raise TypeError('Output dtype mismatch')
        if out.shape != out_shape:
            raise ValueError('Output shape mismatch')

    cdim = indices.size
    rdim = internal.prod(rshape)
    indices = indices.reshape(
        (1,) * len(lshape) + indices.shape + (1,) * len(rshape))
    if (li == 0 and ri == 0) or (li is None and ri is None):
        return _take_kernel_0axis(
            a.reduced_view(), indices, rdim, index_range, out)
    else:
        return _take_kernel(
            a.reduced_view(), indices, cdim, rdim, adim, index_range, out)


cpdef _scatter_op_single(ndarray a, ndarray indices, v,
                         int li=0, int ri=0, op=''):
    # When op == 'update', this function behaves similarly to
    # a code below using NumPy under the condition that a = a._reshape(shape)
    # does not invoke copy.
    #
    # shape = a[:li] +\
    #     (numpy.prod(a[li:ri+1]),) + a[ri+1:]
    # a = a._reshape(shape)
    # slices = (slice(None),) * li + indices +\
    #     (slice(None),) * (a.ndim - indices.ndim - ri)
    # a[slices] = v
    cdef int ndim, adim, cdim, rdim
    cdef tuple a_shape, indices_shape, lshape, rshape, v_shape

    ndim = a._shape.size()

    if ndim == 0:
        raise ValueError("requires a.ndim >= 1")
    if not (-ndim <= li < ndim and -ndim <= ri < ndim):
        raise ValueError('Axis overrun')

    if not isinstance(v, ndarray):
        v = array(v, dtype=a.dtype)
    v = v.astype(a.dtype)

    a_shape = a.shape
    li %= ndim
    ri %= ndim

    lshape = a_shape[:li]
    rshape = a_shape[ri + 1:]
    adim = internal.prod(a_shape[li:ri + 1])

    indices_shape = indices.shape
    v_shape = lshape + indices_shape + rshape
    v = broadcast_to(v, v_shape)

    cdim = indices.size
    rdim = internal.prod(rshape)
    indices = indices._reshape(
        (1,) * len(lshape) + indices_shape + (1,) * len(rshape))
    indices = broadcast_to(indices, v_shape)

    if op == 'update':
        _scatter_update_kernel(
            v, indices, cdim, rdim, adim, a.reduced_view())
    elif op == 'add':
        # There is constraints on types because atomicAdd() in CUDA 7.5
        # only supports int32, uint32, uint64, and float32.
        if not issubclass(v.dtype.type,
                          (numpy.int32, numpy.float32,
                           numpy.uint32, numpy.uint64, numpy.ulonglong)):
            raise TypeError(
                'scatter_add only supports int32, float32, uint32, uint64 as '
                'data type')
        _scatter_add_kernel(
            v, indices, cdim, rdim, adim, a.reduced_view())
    else:
        raise ValueError('provided op is not supported')


cpdef _scatter_op_mask_single(ndarray a, ndarray mask, v, int axis, op):
    cdef ndarray mask_scanned
    cdef tuple masked_shape

    mask, mask_scanned, masked_shape = _prepare_mask_indexing_single(
        a, mask, axis)
    if internal.prod(masked_shape) == 0:
        return

    if not isinstance(v, ndarray):
        v = array(v, dtype=a.dtype)
    v = v.astype(a.dtype)
    # broadcast v to shape determined by the mask
    v = broadcast_to(v, masked_shape)

    if op == 'update':
        _scatter_update_mask_kernel(v, mask, mask_scanned, a)
    elif op == 'add':
        _scatter_add_mask_kernel(v, mask, mask_scanned, a)
    else:
        raise ValueError('provided op is not supported')


cpdef _scatter_op(ndarray a, slices, value, op):
    cdef Py_ssize_t i, ndim, n_newaxes, n_ellipses, ellipsis, axis
    cdef Py_ssize_t n_not_slice_none, mask_i
    cdef Py_ssize_t ellipsis_size
    cdef ndarray v, x, y, a_interm, reduced_idx
    cdef int li, ri

    if isinstance(slices, tuple):
        slices = list(slices)
    elif isinstance(slices, list):
        slices = list(slices)  # copy list
        if all([isinstance(s, int) for s in slices]):
            slices = [slices]
    else:
        slices = [slices]

    # Expand ellipsis into empty slices
    ellipsis = -1
    n_newaxes, n_ellipses = 0, 0
    for i, s in enumerate(slices):
        if s is None:
            n_newaxes += 1
        elif s is Ellipsis:
            n_ellipses += 1
            ellipsis = i
    ndim = a._shape.size()
    noneslices = [slice(None)]
    if n_ellipses > 0:
        if n_ellipses > 1:
            raise ValueError('Only one Ellipsis is allowed in index')
        ellipsis_size = ndim - (<Py_ssize_t>len(slices) - n_newaxes - 1)
        slices[ellipsis:ellipsis + 1] = noneslices * ellipsis_size

    slices += noneslices * (ndim - <Py_ssize_t>len(slices) + n_newaxes)

    if len(slices) > a.ndim + n_newaxes:
        raise IndexError('too many indices for array')

    # Check if advanced is true,
    # and convert list/NumPy arrays to cupy.ndarray
    advanced = False
    mask_exists = False
    for i, s in enumerate(slices):
        if isinstance(s, (list, numpy.ndarray)):
            is_list = isinstance(s, list)
            s = array(s)
            # handle the case when s is an empty list
            if is_list and s.size == 0:
                s = s.astype(numpy.int32)
            slices[i] = s
        if isinstance(s, ndarray):
            if issubclass(s.dtype.type, numpy.integer):
                advanced = True
            elif issubclass(s.dtype.type, numpy.bool_):
                mask_exists = True
            else:
                raise IndexError(
                    'arrays used as indices must be of integer or boolean '
                    'type. (actual: {})'.format(s.dtype.type))

    if mask_exists:
        n_not_slice_none = 0
        for i, s in enumerate(slices):
            if not isinstance(s, slice) or s != slice(None):
                n_not_slice_none += 1
                if issubclass(s.dtype.type, numpy.bool_):
                    mask_i = i
        if n_not_slice_none != 1:
            raise ValueError('currently, CuPy only supports slices that '
                             'consist of one boolean array.')
        _scatter_op_mask_single(a, slices[mask_i], value, mask_i, op)
        return

    if advanced:
        # split slices that can be handled by basic-indexing
        basic_slices = []
        adv_slices = []
        for i, s in enumerate(slices):
            if type(s) is slice:
                basic_slices.append(s)
                adv_slices.append(slice(None))
            elif s is None:
                basic_slices.append(None)
                adv_slices.append(slice(None))
            elif (isinstance(s, ndarray) and
                    issubclass(s.dtype.type, numpy.integer)):
                basic_slices.append(slice(None))
                adv_slices.append(s)
            elif isinstance(s, int):
                basic_slices.append(slice(None))
                scalar_array = ndarray((), dtype=numpy.int64)
                scalar_array.fill(s)
                adv_slices.append(scalar_array)
            else:
                raise IndexError(
                    'only integers, slices (`:`), ellipsis (`...`),'
                    'numpy.newaxis (`None`) and integer or'
                    'boolean arrays are valid indices')

        # check if this is a combination of basic and advanced indexing
        for s in basic_slices:
            if s is None or (isinstance(s, slice) and s != slice(None)):
                # returns a view of a
                a = a[tuple(basic_slices)]
                break

        arr_slices_mask = [not isinstance(s, slice) for s in adv_slices]
        if sum(arr_slices_mask) == 1:
            axis = arr_slices_mask.index(True)
            _scatter_op_single(a, adv_slices[axis], value,
                               li=axis, ri=axis, op=op)
            return

        # scatter_op with multiple integer arrays
        a_interm, reduced_idx, li, ri =\
            _prepare_multiple_array_indexing(a, adv_slices)
        _scatter_op_single(a_interm, reduced_idx, value, li=li, ri=ri, op=op)
        return

    if op == 'update':
        v = a[tuple(slices)]
        if isinstance(value, ndarray):
            y, x = broadcast(v, value).values
            if (internal.vector_equal(y._shape, x._shape) and
                    internal.vector_equal(y._strides, x._strides)):
                if y.data.ptr == x.data.ptr:
                    return  # Skip since x and y are the same array
                elif y._c_contiguous and x.dtype == y.dtype:
                    y.data.copy_from_device_async(x.data, x.nbytes,
                                                  cuda.Stream.null)
                    return
            elementwise_copy(x, y)
        else:
            v.fill(value)
    elif op == 'add':
        v = a[tuple(slices)]
        if not isinstance(value, ndarray):
            value = ndarray(value)
        y, x = broadcast(v, value).values
        elementwise_copy(x + y, y)
    else:
        raise ValueError('this op is not supported')


cpdef ndarray _diagonal(ndarray a, Py_ssize_t offset=0, Py_ssize_t axis1=0,
                        Py_ssize_t axis2=1):
    if axis1 < axis2:
        min_axis, max_axis = axis1, axis2
    else:
        min_axis, max_axis = axis2, axis1

    tr = list(six.moves.range(a.ndim))
    del tr[max_axis]
    del tr[min_axis]
    if offset >= 0:
        a = a.transpose(tr + [axis1, axis2])
    else:
        a = a.transpose(tr + [axis2, axis1])
        offset = -offset

    diag_size = max(0, min(a.shape[-2], a.shape[-1] - offset))
    ret_shape = a.shape[:-2] + (diag_size,)
    if diag_size == 0:
        return ndarray(ret_shape, dtype=a.dtype)

    a = a[..., :diag_size, offset:offset + diag_size]

    ret = a.view()
    ret._set_shape_and_strides(
        a.shape[:-2] + (diag_size,),
        a.strides[:-2] + (a.strides[-1] + a.strides[-2],))
    return ret


cpdef _prepare_multiple_array_indexing(ndarray a, list slices):
    # slices consist of either slice(None) or ndarray
    cdef int i, p, li, ri
    cdef ndarray take_idx, input_flat, out_flat, ret
    cdef tuple a_shape

    br = broadcast(*slices)
    slices = list(br.values)

    # check if transpose is necessasry
    # li:  index of the leftmost array in slices
    # ri:  index of the rightmost array in slices
    do_transpose = False
    prev_arr_i = None
    li = 0
    ri = 0
    for i, s in enumerate(slices):
        if isinstance(s, ndarray):
            if prev_arr_i is None:
                prev_arr_i = i
                li = i
            elif prev_arr_i is not None and i - prev_arr_i > 1:
                do_transpose = True
            else:
                prev_arr_i = i
                ri = i

    if do_transpose:
        transp_a = []
        transp_b = []
        slices_a = []
        slices_b = []

        for i, s in enumerate(slices):
            if isinstance(s, ndarray):
                transp_a.append(i)
                slices_a.append(s)
            else:
                transp_b.append(i)
                slices_b.append(s)
        a = a.transpose(*(transp_a + transp_b))
        slices = slices_a + slices_b
        li = 0
        ri = len(transp_a) - 1

    a_interm_shape = a.shape
    a_interm = a

    # build the strides
    strides = [1]
    for s in a.shape[ri:li:-1]:
        strides.insert(0, s * strides[0])

    # convert all negative indices to wrap_indices
    for i in range(li, ri+1):
        slices[i] %= a_interm_shape[i]

    flattened_indexes = [stride * s
                         for stride, s in zip(strides, slices[li:ri+1])]

    # do stack: flattened_indexes = stack(flattened_indexes, axis=0)
    concat_shape = (len(flattened_indexes),) + br.shape
    flattened_indexes = concatenate(
        [index._reshape((1,) + index.shape) for index in flattened_indexes],
        axis=0, shape=concat_shape, dtype=flattened_indexes[0].dtype)

    reduced_idx = _sum(flattened_indexes, axis=0)

    return a_interm, reduced_idx, li, ri


cpdef ndarray _getitem_multiple(ndarray a, list slices):
    cdef ndarray a_interm, reduced_idx, ret, ret_flat
    cdef tuple a_interm_shape, kern_input_shape, out_shape
    cdef int li, ri

    a_interm, reduced_idx, li, ri = _prepare_multiple_array_indexing(a, slices)

    a_interm_shape = a_interm.shape
    out_shape = a_interm_shape[:li] + reduced_idx.shape + a_interm_shape[ri+1:]
    ret_flat = _take(a_interm, reduced_idx.ravel(), li=li, ri=ri)
    ret = ret_flat._reshape(out_shape)
    return ret


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
        return multiply(a, b, out=out)

    input_a_is_vec = a_ndim == 1
    input_b_is_vec = b_ndim == 1
    if input_a_is_vec:
        shape.clear()
        shape.push_back(1)
        shape.push_back(a.size)
        a = a._reshape(shape)
        a_ndim = 2
    if input_b_is_vec:
        shape.clear()
        shape.push_back(b.size)
        shape.push_back(1)
        b = b._reshape(shape)
        b_ndim = 2

    a_axis = a_ndim - 1
    b_axis = b_ndim - 2

    if a._shape[a_axis] != b._shape[b_axis]:
        raise ValueError('Axis dimension mismatch')

    if a_axis:
        a = rollaxis(a, a_axis, 0)
    if b_axis:
        b = rollaxis(b, b_axis, 0)

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


cpdef ndarray _get_all_addresses(size_t start_adr,
                                 vector.vector[size_t]& shape,
                                 vector.vector[size_t]& strides):
    idx = numpy.array([start_adr])
    for sh_, st_ in zip(shape, strides):
        idx = (idx[:, None] + (numpy.arange(sh_) * st_)[None, :]).ravel()
    idx = idx.astype(numpy.uintp)

    ret = ndarray((idx.size,), dtype=numpy.uintp)
    ret.set(idx)
    return ret


cdef ndarray _mat_ptrs(ndarray a):
    """Creates an array of pointers to matrices
    Args:
        a: A batch of matrices on GPU.
           shape: () -> one ptr
           shape: (A) -> one ptr to mat o size (A)
           shape: (A, B) -> one ptr to mat o size (A, B)
           shape: (A, B, C) -> A ptrs to mat o size (B, C)
           shape: (A_1, ..., A_N, B, C) -> A_1*...*A_N ptrs to mat of
                  size (B, C)
    Returns:
        GPU array of pointers to matrices.
    """
    cdef Py_ssize_t stride, ptr, pointer, i
    cdef ndarray ret
    if a.ndim <= 2:
        ret = ndarray((1,), dtype=numpy.uintp)
        ret.fill(a.data.ptr)
        return ret
    else:
        return _get_all_addresses(a.data.ptr, a.shape[:-2], a.strides[:-2])


cpdef ndarray matmul(ndarray a, ndarray b):
    """ Returns the matrix product of two arrays and is the implementation of
    the `@` operator introduced in Python 3.5 following PEP465.

    The main difference against cupy.dot are the handling of arrays with more
    than 2 dimensions. For more information see :func:`numpy.matmul`.

    .. note::
        Differences to numpy or missing features:

        Currently the output must be real (float16, float32, uint8, ...),
        complex64 and complex128 follow later. This means, that
        numpy.result_type(a.dtype, b.dtype) have to be real.

        The out array as input is currently not supported.

    Args:
        a (cupy.ndarray): The left argument.
        b (cupy.ndarray): The right argument.
        out (cupy.ndarray): Output array.

    .. seealso:: :func:`numpy.matmul`

    """
    # ToDo: Argument out=None is missing
    # ToDo: remove python object .shape
    # ToDo: remove python object .strides
    # ToDo: remove python object out_shape
    # ToDo: remove python object .reshape
    cdef Py_ssize_t i, n, m, ka, kb
    cdef int batchCount
    cdef ndarray out, ap, bp, outp

    ret_dtype = numpy.result_type(a.dtype, b.dtype)
    dtype = numpy.find_common_type((ret_dtype, 'f'), ())

    a = a.astype(dtype, copy=False)
    b = b.astype(dtype, copy=False)

    if a.ndim == 1:
        a = a.reshape(1, len(a))
        a_part_outshape = ()
    else:
        a_part_outshape = (a.shape[-2],)
    if b.ndim == 1:
        b = b.reshape(len(b), 1)
        b_part_outshape = ()
        ldout = 1
    else:
        b_part_outshape = b.shape[-1:]
        ldout = b.shape[-1]

    # expand dims
    if a.ndim < b.ndim:
        view = a.view()
        view._set_shape_and_strides(
            (1,) * (b.ndim - a.ndim) + a.shape,
            (0,) * (b.ndim - a.ndim) + a.strides)
        a = view
    elif a.ndim > b.ndim:
        view = b.view()
        view._set_shape_and_strides(
            (1,) * (a.ndim - b.ndim) + b.shape,
            (0,) * (a.ndim - b.ndim) + b.strides)
        b = view

    broatcast_pre_shape = numpy.maximum(a.shape[:-2], b.shape[:-2])

    out_shape = (*broatcast_pre_shape, *a_part_outshape, *b_part_outshape)

    a = ascontiguousarray(a, dtype=dtype)
    b = ascontiguousarray(b, dtype=dtype)

    # broadcast
    a_strides = list(a.strides)
    a_shape = list(a.shape)
    b_strides = list(b.strides)
    b_shape = list(b.shape)
    for i in range(len(a_strides) - 2):
        if a_shape[i] == 1 and broatcast_pre_shape[i] > 1:
            a_strides[i] = 0
            a_shape[i] = broatcast_pre_shape[i]
    for i in range(len(b_strides) - 2):
        if b_shape[i] == 1 and broatcast_pre_shape[i] > 1:
            b_strides[i] = 0
            b_shape[i] = broatcast_pre_shape[i]

    view = a.view()
    view._set_shape_and_strides(a_shape, a_strides)
    a = view
    view = b.view()
    view._set_shape_and_strides(b_shape, b_strides)
    b = view

    out = ndarray(out_shape, dtype=dtype)
    out.data.memset(0, out.nbytes)

    out_view = out.view()
    out_view_shape = out.shape
    out_view_strides = out.strides
    if a_part_outshape == ():
        out_view_shape += (1,)
        out_view_strides += (0,)
    if b_part_outshape == ():
        out_view_shape += (1,)
        out_view_strides += (0,)

    out_view._set_shape_and_strides(out_view_shape, out_view_strides)

    # (A B)^T = B^T A^T
    a, b = b, a

    lda = a.shape[-1]
    ldb = b.shape[-1]

    *la, ka, n = a.shape
    *lb, m, kb = b.shape

    assert ka == kb
    for la_, lb_ in zip(la, lb):
        assert la_ == lb_ or la_ == 1 or lb_ == 1

    batchCount = 1  # batchCount = numpy.prod(la)
    for i in la:
        batchCount *= i

    ap = _mat_ptrs(a)
    bp = _mat_ptrs(b)
    outp = _mat_ptrs(out_view)

    if dtype == numpy.float32:
        cuda.cublas.sgemmBatched(
            cuda.Device().cublas_handle,
            0,  # transa
            0,  # transb
            n, m, ka, 1.0,
            ap.data.ptr, lda,
            bp.data.ptr, ldb,
            0.0, outp.data.ptr, ldout, batchCount)
    elif dtype == numpy.float64:
        cuda.cublas.dgemmBatched(
            cuda.Device().cublas_handle,
            0,  # transa
            0,  # transb
            n, m, ka, 1.0,
            ap.data.ptr, lda,
            bp.data.ptr, ldb,
            0.0, outp.data.ptr, ldout, batchCount)
    # elif dtype == numpy.complex64:
    #     cuda.cublas.cgemmBatched(
    #         cuda.Device().cublas_handle,
    #         0,  # transa
    #         0,  # transb
    #         n, m, ka, 1,
    #         ap.data.ptr, lda,
    #         bp.data.ptr, ldb,
    #         0, outp.data.ptr, ldout, batchCount)
    # elif dtype == numpy.complex128:
    #     cuda.cublas.zgemmBatched(
    #         cuda.Device().cublas_handle,
    #         0,  # transa
    #         0,  # transb
    #         n, m, ka, 1,
    #         ap.data.ptr, lda,
    #         bp.data.ptr, ldb,
    #         0, outp.data.ptr, ldout, batchCount)
    else:
        raise TypeError(dtype, a.dtype, b.dtype)

    if dtype == ret_dtype:
        return out
    else:
        ret = ndarray(out_shape, ret_dtype)
        elementwise_copy(out, ret)
        return ret


cdef _cuda_runtime_version = None


cpdef ndarray tensordot_core(
        ndarray a, ndarray b, ndarray out, Py_ssize_t n, Py_ssize_t m,
        Py_ssize_t k, vector.vector[Py_ssize_t] ret_shape):
    cdef vector.vector[Py_ssize_t] shape
    cdef int inca, incb, transa, transb, lda, ldb
    cdef Py_ssize_t mode, handle
    cdef str dtype, ret_dtype
    cdef bint use_sgemmEx
    ret_dtype = a.dtype.char
    if ret_dtype != b.dtype.char:
        ret_dtype = numpy.find_common_type((ret_dtype, b.dtype), ()).char

    if not a.size or not b.size:
        if out is None:
            out = ndarray(ret_shape, dtype=ret_dtype)
        out.fill(0)
        return out

    global _cuda_runtime_version
    if _cuda_runtime_version is None:
        _cuda_runtime_version = runtime.runtimeGetVersion()

    use_sgemmEx = (_cuda_runtime_version >= 7500 and
                   a.dtype == 'e' and b.dtype == 'e' and
                   (ret_dtype == 'e' or ret_dtype == 'f'))

    if use_sgemmEx or ret_dtype == 'f' or ret_dtype == 'd':
        dtype = ret_dtype
    else:
        dtype = numpy.find_common_type((ret_dtype, 'f'), ()).char

    if not use_sgemmEx:
        a = a.astype(dtype, copy=False)
        b = b.astype(dtype, copy=False)

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
        (a.ravel() * b.ravel()).sum(out=out.reshape(()))
        if out is not ret:
            elementwise_copy(out, ret)
        return ret

    # It copies the operands if needed
    if a._shape.size() != 2 or a._shape[0] != k or a._shape[1] != n:
        shape.clear()
        shape.push_back(k)
        shape.push_back(n)
        a = a._reshape(shape)
    if b._shape.size() != 2 or b._shape[0] != k or b._shape[1] != m:
        shape.clear()
        shape.push_back(k)
        shape.push_back(m)
        b = b._reshape(shape)
    c = out
    if c._shape.size() != 2 or c._shape[0] != n or c._shape[1] != m:
        c = c.view()
        c.shape = (n, m)

    # Be careful that cuBLAS uses the FORTRAN-order matrix representation.
    handle = device.get_cublas_handle()
    # Matrix-Matrix product A^T * B
    # c is C-contiguous while cuBLAS assumes F-contiguous inputs, so we
    # compute C^T = B^T * A here.
    a, transa, lda = _mat_to_cublas_contiguous(a, 0)
    b, transb, ldb = _mat_to_cublas_contiguous(b, 1)
    if use_sgemmEx:
        Ctype = runtime.CUDA_R_16F if c.dtype == 'e' else runtime.CUDA_R_32F
        cublas.sgemmEx(
            handle, transb, transa, m, n, k, 1, b.data.ptr, runtime.CUDA_R_16F,
            ldb, a.data.ptr, runtime.CUDA_R_16F, lda, 0, c.data.ptr, Ctype, m)
    elif dtype == 'f':
        cublas.sgemm(handle, transb, transa, m, n, k, 1, b.data.ptr, ldb,
                     a.data.ptr, lda, 0, c.data.ptr, m)
    elif dtype == 'd':
        cublas.dgemm(handle, transb, transa, m, n, k, 1, b.data.ptr, ldb,
                     a.data.ptr, lda, 0, c.data.ptr, m)

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

cpdef create_comparison(name, op, doc=''):
    return create_ufunc(
        'cupy_' + name,
        ('??->?', 'bb->?', 'BB->?', 'hh->?', 'HH->?', 'ii->?', 'II->?',
         'll->?', 'LL->?', 'qq->?', 'QQ->?', 'ee->?', 'ff->?', 'dd->?'),
        'out0 = in0 %s in1' % op,
        doc=doc)


greater = create_comparison(
    'greater', '>',
    '''Tests elementwise if ``x1 > x2``.

    .. seealso:: :data:`numpy.greater`

    ''')


greater_equal = create_comparison(
    'greater_equal', '>=',
    '''Tests elementwise if ``x1 >= x2``.

    .. seealso:: :data:`numpy.greater_equal`

    ''')


less = create_comparison(
    'less', '<',
    '''Tests elementwise if ``x1 < x2``.

    .. seealso:: :data:`numpy.less`

    ''')


less_equal = create_comparison(
    'less_equal', '<=',
    '''Tests elementwise if ``x1 <= x2``.

    .. seealso:: :data:`numpy.less_equal`

    ''')


equal = create_comparison(
    'equal', '==',
    '''Tests elementwise if ``x1 == x2``.

    .. seealso:: :data:`numpy.equal`

    ''')


not_equal = create_comparison(
    'not_equal', '!=',
    '''Tests elementwise if ``x1 != x2``.

    .. seealso:: :data:`numpy.equal`

    ''')


_all = create_reduction_func(
    'cupy_all',
    ('?->?', 'B->?', 'h->?', 'H->?', 'i->?', 'I->?', 'l->?', 'L->?',
     'q->?', 'Q->?', 'e->?', 'f->?', 'd->?'),
    ('in0', 'a & b', 'out0 = a', 'bool'),
    'true', '')


_any = create_reduction_func(
    'cupy_any',
    ('?->?', 'B->?', 'h->?', 'H->?', 'i->?', 'I->?', 'l->?', 'L->?',
     'q->?', 'Q->?', 'e->?', 'f->?', 'd->?'),
    ('in0', 'a | b', 'out0 = a', 'bool'),
    'false', '')


# -----------------------------------------------------------------------------
# Mathematical functions
# -----------------------------------------------------------------------------

_sum = create_reduction_func(
    'cupy_sum',
    ('?->l', 'B->L', 'h->l', 'H->L', 'i->l', 'I->L', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d'),
    ('in0', 'a + b', 'out0 = a', None), 0)


_prod = create_reduction_func(
    'cupy_prod',
    ['?->l', 'B->L', 'h->l', 'H->L', 'i->l', 'I->L', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d'],
    ('in0', 'a * b', 'out0 = a', None), 1)


cdef create_arithmetic(name, op, boolop, doc):
    return create_ufunc(
        'cupy_' + name,
        (('??->?', 'out0 = in0 %s in1' % boolop),
         'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
         'LL->L', 'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'),
        'out0 = in0 %s in1' % op,
        doc=doc)


add = create_arithmetic(
    'add', '+', '|',
    '''Adds two arrays elementwise.

    .. seealso:: :data:`numpy.add`

    ''')


negative = create_ufunc(
    'cupy_negative',
    (('?->?', 'out0 = !in0'),
     'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'),
    'out0 = -in0',
    doc='''Takes numerical negative elementwise.

    .. seealso:: :data:`numpy.negative`

    ''')


multiply = create_arithmetic(
    'multiply', '*', '&',
    '''Multiplies two arrays elementwise.

    .. seealso:: :data:`numpy.multiply`

    ''')


divide = create_ufunc(
    'cupy_divide',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q',
     ('ee->e', 'out0 = in0 / in1'),
     ('ff->f', 'out0 = in0 / in1'),
     ('dd->d', 'out0 = in0 / in1')),
    'out0 = in1 == 0 ? 0 : floor((double)in0 / (double)in1)',
    doc='''Divides arguments elementwise.

    .. seealso:: :data:`numpy.divide`

    ''')


power = create_ufunc(
    'cupy_power',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q',
     ('ee->e', 'out0 = powf(in0, in1)'),
     ('ff->f', 'out0 = powf(in0, in1)'),
     ('dd->d', 'out0 = pow(in0, in1)')),
    'out0 = rint(pow((double)in0, (double)in1))',
    doc='''Computes ``x1 ** x2`` elementwise.

    .. seealso:: :data:`numpy.power`

    ''')


subtract = create_arithmetic(
    'subtract', '-', '^',
    '''Subtracts arguments elementwise.

    .. seealso:: :data:`numpy.subtract`

    ''')


true_divide = create_ufunc(
    'cupy_true_divide',
    ('bb->d', 'BB->d', 'hh->d', 'HH->d', 'ii->d', 'II->d', 'll->d', 'LL->d',
     'qq->d', 'QQ->d', 'ee->e', 'ff->f', 'dd->d'),
    'out0 = (out0_type)in0 / (out0_type)in1',
    doc='''Elementwise true division (i.e. division as floating values).

    .. seealso:: :data:`numpy.true_divide`

    ''')


if six.PY3:
    divide = true_divide


floor_divide = create_ufunc(
    'cupy_floor_divide',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'),
    'out0 = _floor_divide(in0, in1)',
    doc='''Elementwise floor division (i.e. integer quotient).

    .. seealso:: :data:`numpy.floor_divide`

    ''')


remainder = create_ufunc(
    'cupy_remainder',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q',
     ('ee->e', 'out0 = in0 - _floor_divide(in0, in1) * in1'),
     ('ff->f', 'out0 = in0 - _floor_divide(in0, in1) * in1'),
     ('dd->d', 'out0 = in0 - _floor_divide(in0, in1) * in1')),
    'out0 = (in0 - _floor_divide(in0, in1) * in1) * (in1 != 0)',
    doc='''Computes the remainder of Python division elementwise.

    .. seealso:: :data:`numpy.remainder`

    ''')


absolute = create_ufunc(
    'cupy_absolute',
    (('?->?', 'out0 = in0'),
     'b->b', ('B->B', 'out0 = in0'), 'h->h', ('H->H', 'out0 = in0'),
     'i->i', ('I->I', 'out0 = in0'), 'l->l', ('L->L', 'out0 = in0'),
     'q->q', ('Q->Q', 'out0 = in0'),
     ('e->e', 'out0 = fabsf(in0)'),
     ('f->f', 'out0 = fabsf(in0)'),
     ('d->d', 'out0 = fabs(in0)')),
    'out0 = in0 > 0 ? in0 : -in0',
    doc='''Elementwise absolute value function.

    .. seealso:: :data:`numpy.absolute`

    ''')


sqrt = create_ufunc(
    'cupy_sqrt',
    ('e->e', 'f->f', 'd->d'),
    'out0 = sqrt(in0)')


_clip = create_ufunc(
    'cupy_clip',
    ('???->?', 'bbb->b', 'BBB->B', 'hhh->h', 'HHH->H', 'iii->i', 'III->I',
     'lll->l', 'LLL->L', 'qqq->q', 'QQQ->Q', 'eee->e', 'fff->f', 'ddd->d'),
    'out0 = in0 < in1 ? in1 : (in0 > in2 ? in2 : in0)')


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

cpdef ndarray _var(ndarray a, axis=None, dtype=None, out=None, ddof=0,
                   keepdims=False):
    if axis is None:
        axis = tuple(range(a.ndim))
    if not isinstance(axis, tuple):
        axis = (axis,)

    if dtype is None and issubclass(a.dtype.type,
                                    (numpy.integer, numpy.bool_)):
        dtype = 'd'

    shape = a.shape
    items = 1
    for ax in axis:
        items *= shape[ax]
    alpha = 1. / max(items - ddof, 0)
    arrmean = a.mean(axis=axis, dtype=dtype, keepdims=True)
    if out is None:
        return _var_core(a, arrmean, alpha, axis=axis, keepdims=keepdims)
    else:
        return _var_core_out(
            a, arrmean, alpha, out, axis=axis, keepdims=keepdims)


cpdef _std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    ret = _var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
    return sqrt(ret, dtype=dtype, out=out)


cdef _var_core = ReductionKernel(
    'S x, T mean, T alpha', 'T out',
    '(x - mean) * (x - mean)',
    'a + b', 'out = alpha * a', '0', '_var_core')

cdef _var_core_out = ReductionKernel(
    'S x, T mean, T alpha', 'U out',
    '(x - mean) * (x - mean)',
    'a + b', 'out = alpha * a', '0', '_var_core')

# TODO(okuta) needs cast
cdef _mean = create_reduction_func(
    'cupy_mean',
    ('?->d', 'B->d', 'h->d', 'H->d', 'i->d', 'I->d', 'l->d', 'L->d',
     'q->d', 'Q->d',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d'),
    ('in0', 'a + b', 'out0 = a / (_in_ind.size() / _out_ind.size())', None))


# -----------------------------------------------------------------------------
# scan
# -----------------------------------------------------------------------------

@util.memoize(for_each_device=True)
def _inclusive_scan_kernel(dtype, block_size):
    """return Prefix Sum(Scan) cuda kernel

    e.g
    if blocksize * 2 >= len(src)
    src [1, 2, 3, 4]
    dst [1, 3, 6, 10]

    if blocksize * 2 < len(src)
    block_size: 2
    src [1, 2, 3, 4, 5, 6]
    dst [1, 3, 6, 10, 5, 11]

    Args:
        dtype: src, dst array type
        block_size: block_size

    Returns:
         cupy.cuda.Function: cuda function
    """

    name = "inclusive_scan_kernel"
    dtype = _get_typename(dtype)
    source = string.Template("""
    extern "C" __global__ void ${name}(const CArray<${dtype}, 1> src,
        CArray<${dtype}, 1> dst){
        long long n = src.size();
        extern __shared__ ${dtype} temp[];
        unsigned int thid = threadIdx.x;
        unsigned int block = 2 * blockIdx.x * blockDim.x;

        unsigned int idx0 = thid + block;
        unsigned int idx1 = thid + blockDim.x + block;

        temp[thid] = (idx0 < n) ? src[idx0] : (${dtype})0;
        temp[thid + blockDim.x] = (idx1 < n) ? src[idx1] : (${dtype})0;
        __syncthreads();

        for(int i = 1; i <= ${block_size}; i <<= 1){
            int index = (threadIdx.x + 1) * i * 2 - 1;
            if (index < (${block_size} << 1)){
                temp[index] = temp[index] + temp[index - i];
            }
            __syncthreads();
        }

        for(int i = ${block_size} >> 1; i > 0; i >>= 1){
            int index = (threadIdx.x + 1) * i * 2 - 1;
            if(index + i < (${block_size} << 1)){
                temp[index + i] = temp[index + i] + temp[index];
            }
            __syncthreads();
        }

        if(idx0 < n){
            dst[idx0] = temp[thid];
        }
        if(idx1 < n){
            dst[idx1] = temp[thid + blockDim.x];
        }
    }
    """).substitute(name=name, dtype=dtype, block_size=block_size)
    module = compile_with_cache(source)
    return module.get_function(name)


@util.memoize(for_each_device=True)
def _add_scan_blocked_sum_kernel(dtype):
    name = "add_scan_blocked_sum_kernel"
    dtype = _get_typename(dtype)
    source = string.Template("""
    extern "C" __global__ void ${name}(CArray<${dtype}, 1> src_dst){
        long long n = src_dst.size();
        unsigned int idxBase = (blockDim.x + 1) * (blockIdx.x + 1);
        unsigned int idxAdded = idxBase + threadIdx.x;
        unsigned int idxAdd = idxBase - 1;

        if(idxAdded < n){
            src_dst[idxAdded] += src_dst[idxAdd];
        }
    }
    """).substitute(name=name, dtype=dtype)
    module = compile_with_cache(source)
    return module.get_function(name)


@util.memoize(for_each_device=True)
def _nonzero_1d_kernel(src_dtype, index_dtype):
    name = "nonzero_1d_kernel"
    src_dtype = _get_typename(src_dtype)
    index_dtype = _get_typename(index_dtype)

    source = string.Template("""
    extern "C" __global__ void ${name}(const CArray<${src_dtype}, 1> src,
        const CArray<${index_dtype}, 1> scaned_index,
        CArray<${index_dtype}, 1> dst){
        int thid = blockIdx.x * blockDim.x + threadIdx.x;
        int n = src.size();
        if (thid < n){
            if (src[thid] != 0){
                dst[scaned_index[thid] - 1] = thid;
            }
        }
    }
    """).substitute(name=name, src_dtype=src_dtype, index_dtype=index_dtype)
    module = compile_with_cache(source)
    return module.get_function(name)


@util.memoize(for_each_device=True)
def _nonzero_kernel(src_dtype, src_ndim, index_dtype, dst_dtype):
    name = "nonzero_kernel"
    src_dtype = _get_typename(src_dtype)
    index_dtype = _get_typename(index_dtype)
    dst_dtype = _get_typename(dst_dtype)

    source = string.Template("""
        extern "C" __global__ void ${name}(const CArray<${src_dtype}, 1> src,
            CIndexer<${src_ndim}> shape,
            const CArray<${index_dtype}, 1> scaned_index,
            CArray<${dst_dtype}, 1> dst){

            int thid = blockIdx.x * blockDim.x + threadIdx.x;

            if (thid < src.size()){
                if (src[thid] != 0){
                    ${index_dtype} idx = scaned_index[thid] - 1;
                    int s = shape.size();

                    shape.set(thid);

                    for(int i = 0; i < ${src_ndim}; i++){
                        dst[idx * ${src_ndim} + i] = shape.get()[i];
                    }
                }
            }
        }
        """).substitute(name=name, src_dtype=src_dtype,
                        src_ndim=src_ndim, index_dtype=index_dtype,
                        dst_dtype=dst_dtype)
    module = compile_with_cache(source)
    return module.get_function(name)


def scan(a, out=None):
    """Return the prefix sum(scan) of the elements.

    Args:
        a (cupy.ndarray): input array.
        out (cupy.ndarray): Alternative output array in which to place
         the result. The same size and same type as the input array(a).

    Returns:
        cupy.ndarray: A new array holding the result is returned.

    """
    if a.ndim != 1:
        raise TypeError("Input array should be 1D array.")

    block_size = 256

    if out is None:
        out = ndarray(a.shape, dtype=a.dtype)
    else:
        if a.size != out.size:
            raise ValueError("Provided out is the wrong size")

    kern_scan = _inclusive_scan_kernel(a.dtype, block_size)
    kern_scan(grid=((a.size - 1) // (2 * block_size) + 1,),
              block=(block_size,),
              args=(a, out),
              shared_mem=a.itemsize * block_size * 2)

    if (a.size - 1) // (block_size * 2) > 0:
        blocked_sum = out[block_size * 2 - 1:None:block_size * 2]
        scan(blocked_sum, blocked_sum)
        kern_add = _add_scan_blocked_sum_kernel(out.dtype)
        kern_add(grid=((a.size - 1) // (2 * block_size),),
                 block=(2 * block_size - 1,),
                 args=(out,))
    return out
