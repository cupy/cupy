from __future__ import division
import collections
import ctypes
import sys

#cimport numpy
import numpy
import six

from cupy.cuda import cublas
from cupy import flags
from cupy import util


DEF MAX_NDIM = 25


def _get_size(size):
    if size is None:
        return ()
    elif isinstance(size, collections.Sequence):
        return tuple(size)
    elif isinstance(size, int):
        return size,
    else:
        raise ValueError('size should be None, collections.Sequence, or int')

cdef _should_use_rop(x, y):
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

    Attributes:
        data (cupy.cuda.MemoryPointer): Pointer to the array content head.
        base (None or cupy.ndarray): Base array from which this array is
            created as a view.

    """

    cdef:
        public Py_ssize_t _size
        public int _c_contiguous
        public int _f_contiguous
        public object _dtype
        public object data
        public ndarray base
        public tuple _shape
        public tuple _strides


    def __init__(self, shape, dtype=float, memptr=None, strides=None):
        self._shape = shape = _get_size(shape)
        self._dtype = dtype = numpy.dtype(dtype)
        size = 1
        for s in shape:
            size *= s
        self._size = size

        if memptr is None:
            self.data = cuda.alloc(size * dtype.itemsize)
        else:
            self.data = memptr

        if strides is None:
            self._strides = get_contiguous_strides(
                shape, dtype.itemsize)
            self._c_contiguous = 1
            self._f_contiguous = int(
                not size or len(shape) - shape.count(1) <= 1)
        else:
            self._strides = strides
            self._c_contiguous = -1
            self._f_contiguous = -1

        self.base = None

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
        if self._c_contiguous == -1:
            self._update_c_contiguity()
        if self._f_contiguous == -1:
            self._update_f_contiguity()
        return flags.Flags(self._c_contiguous, self._f_contiguous,
                           self.base is not None)

    property shape:
        """Lengths of axes.

        Setter of this property involves reshaping without copy. If the array
        cannot be reshaped without copy, it raises an exception.

        .. seealso: :attr:`numpy.ndarray.shape`

        """

        def __get__(self):
            return self._shape

        def __set__(self, newshape):
            newshape = infer_unknown_dimension(newshape, self._size)
            strides = get_strides_for_nocopy_reshape(self, newshape)
            if strides is None:
                raise AttributeError('Incompatible shape')
            self._shape = newshape
            self._strides = strides
            self._f_contiguous = -1

    @property
    def strides(self):
        """Strides of axes in bytes.

        .. seealso:: :attr:`numpy.ndarray.strides`

        """
        return self._strides

    @property
    def ndim(self):
        """Number of dimensions.

        ``a.ndim`` is equivalent to ``len(a.shape)``.

        .. seealso:: :attr:`numpy.ndarray.ndim`

        """
        return len(self._shape)

    @property
    def size(self):
        """Number of elements this array holds.

        This is equivalent to product over the shape tuple.

        .. seealso:: :attr:`numpy.ndarray.size`

        """
        return self._size

    @property
    def itemsize(self):
        """Size of each element in bytes.

        .. seealso:: :attr:`numpy.ndarray.itemsize`

        """
        return self._dtype.itemsize

    @property
    def nbytes(self):
        """Size of whole elements in bytes.

        It does not count skips between elements.

        .. seealso:: :attr:`numpy.ndarray.nbytes`

        """
        return self._size * self.itemsize

    # -------------------------------------------------------------------------
    # Data type
    # -------------------------------------------------------------------------
    @property
    def dtype(self):
        """Dtype object of element type.

        .. seealso::
           `Data type objects (dtype) \
           <http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html>`_

        """
        return self._dtype

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
            return transpose(self, None)

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
        return to_carray(self.data.ptr, self._size, self._shape,
                                self._strides)

    # -------------------------------------------------------------------------
    # Array conversion
    # -------------------------------------------------------------------------
    # TODO(okuta): Implement item

    def tolist(self):
        """Converts the array to a (possibly nested) Python list.

        Returns:
            list: The possibly nested Python list of array elements.

        .. seealso:: :meth:`numpy.ndarray.tolist`

        """
        return self.get().tolist()

    # TODO(okuta): Implement itemset
    # TODO(okuta): Implement tostring
    # TODO(okuta): Implement tobytes

    def tofile(self, fid, sep='', format='%s'):
        """Writes the array to a file.

        .. seealso:: :meth:`numpy.ndarray.tolist`

        """
        self.get().tofile(fid, sep, format)

    def dump(self, file):
        """Dumps a pickle of the array to a file.

        Dumped file can be read back to cupy.ndarray by
        :func:`cupy.load`.

        """
        six.moves.cPickle.dump(self, file, -1)

    def dumps(self):
        """Dumps a pickle of the array to a string."""
        return six.moves.cPickle.dumps(self, -1)

    def astype(self, dtype, copy=True):
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
        if dtype == self._dtype:
            if copy:
                return self.copy()
            else:
                return self
        else:
            newarray = ndarray(self.shape, dtype=dtype)
            elementwise_copy(self, newarray)
            return newarray

    # TODO(okuta): Implement byteswap

    def copy(self):
        """Returns a copy of the array.

        .. seealso::
           :func:`cupy.copy` for full documentation,
           :meth:`numpy.ndarray.copy`

        """
        # TODO(beam2d): Support ordering option
        return copy(self)

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
        v._c_contiguous = self._c_contiguous
        v._f_contiguous = self._f_contiguous
        v._dtype = self._dtype
        v._shape = self._shape
        v._strides = self._strides
        v._size = self._size
        v.data = self.data
        v.base = self.base if self.base is not None else self
        return v

    # TODO(okuta): Implement getfield
    # TODO(okuta): Implement setflags

    def fill(self, value):
        """Fills the array with a scalar value.

        Args:
            value: A scalar value to fill the array content.

        .. seealso:: :meth:`numpy.ndarray.fill`

        """
        elementwise_copy(value, self, dtype=self._dtype)

    # -------------------------------------------------------------------------
    # Shape manipulation
    # -------------------------------------------------------------------------
    def reshape(self, *shape):
        """Returns an array of a different shape and the same content.

        .. seealso::
           :func:`cupy.reshape` for full documentation,
           :meth:`numpy.ndarray.reshape`

        """
        # TODO(beam2d): Support ordering option
        if len(shape) == 1 and isinstance(shape[0], collections.Sequence):
            shape = shape[0]
        return reshape(self, shape)

    # TODO(okuta): Implement resize

    def transpose(self, *axes):
        """Returns a view of the array with axes permuted.

        .. seealso::
           :func:`cupy.transpose` for full documentation,
           :meth:`numpy.ndarray.reshape`

        """
        if len(axes) == 1:
            a = axes[0]
            if a is None or isinstance(a, collections.Sequence):
                axes = a
        return transpose(self, axes)

    def swapaxes(self, axis1, axis2):
        """Returns a view of the array with two axes swapped.

        .. seealso::
           :func:`cupy.swapaxes` for full documentation,
           :meth:`numpy.ndarray.swapaxes`

        """
        return swapaxes(self, axis1, axis2)

    def flatten(self):
        """Returns a copy of the array flatten into one dimension.

        It currently supports C-order only.

        Returns:
            cupy.ndarray: A copy of the array with one dimension.

        .. seealso:: :meth:`numpy.ndarray.flatten`

        """
        # TODO(beam2d): Support ordering option
        if self.flags.c_contiguous:
            newarray = self.copy()
        else:
            newarray = ndarray(self.shape, self.dtype)
            elementwise_copy(self, newarray)

        newarray._shape = self._size,
        newarray._strides = self.itemsize,
        newarray._c_contiguous = 1
        newarray._f_contiguous = 1
        return newarray

    def ravel(self):
        """Returns an array flattend into one dimension.

        .. seealso::
           :func:`cupy.ravel` for full documentation,
           :meth:`numpy.ndarray.ravel`

        """
        # TODO(beam2d): Support ordering option
        return self.reshape(self.size)

    def squeeze(self, axis=None):
        """Returns a view with size-one axes removed.

        .. seealso::
           :func:`cupy.squeeze` for full documentation,
           :meth:`numpy.ndarray.squeeze`

        """
        return squeeze(self, axis)

    # -------------------------------------------------------------------------
    # Item selection and manipulation
    # -------------------------------------------------------------------------
    def take(self, indices, axis=None, out=None):
        """Returns an array of elements at given indices along the axis.

        .. seealso::
           :func:`cupy.take` for full documentation,
           :meth:`numpy.ndarray.take`

        """
        return take(self, indices, axis, out)

    # TODO(okuta): Implement put
    # TODO(okuta): Implement repeat
    # TODO(okuta): Implement choose
    # TODO(okuta): Implement sort
    # TODO(okuta): Implement argsort
    # TODO(okuta): Implement partition
    # TODO(okuta): Implement argpartition
    # TODO(okuta): Implement searchsorted
    # TODO(okuta): Implement nonzero
    # TODO(okuta): Implement compress

    def diagonal(self, offset=0, axis1=0, axis2=1):
        """Returns a view of the specified diagonals.

        .. seealso::
           :func:`cupy.diagonal` for full documentation,
           :meth:`numpy.ndarray.diagonal`

        """
        return diagonal(self, offset, axis1, axis2)

    # -------------------------------------------------------------------------
    # Calculation
    # -------------------------------------------------------------------------
    def max(self, axis=None, out=None, dtype=None, keepdims=False):
        """Returns the maximum along a given axis.

        .. seealso::
           :func:`cupy.amax` for full documentation,
           :meth:`numpy.ndarray.max`

        """
        return amax(
            self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)

    def argmax(self, axis=None, out=None, dtype=None, keepdims=False):
        """Returns the indices of the maximum along a given axis.

        .. seealso::
           :func:`cupy.argmax` for full documentation,
           :meth:`numpy.ndarray.argmax`

        """
        return argmax(
            self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)

    def min(self, axis=None, out=None, dtype=None, keepdims=False):
        """Returns the minimum along a given axis.

        .. seealso::
           :func:`cupy.amin` for full documentation,
           :meth:`numpy.ndarray.min`

        """
        return amin(
            self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)

    def argmin(self, axis=None, out=None, dtype=None, keepdims=False):
        """Returns the indices of the minimum along a given axis.

        .. seealso::
           :func:`cupy.argmin` for full documentation,
           :meth:`numpy.ndarray.argmin`

        """
        return argmin(
            self, axis=axis, out=out, dtype=dtype, keepdims=keepdims)

    # TODO(okuta): Implement ptp

    def clip(self, a_min, a_max, out=None):
        """Returns an array with values limited to [a_min, a_max].

        .. seealso::
           :func:`cupy.clip` for full documentation,
           :meth:`numpy.ndarray.clip`

        """
        return _clip(self, a_min, a_max, out=out)

    # TODO(okuta): Implement round

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        """Returns the sum along diagonals of the array.

        .. seealso::
           :func:`cupy.trace` for full documentation,
           :meth:`numpy.ndarray.trace`

        """
        d = self.diagonal(offset, axis1, axis2)
        return d.sum(-1, dtype, out, False)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """Returns the sum along a given axis.

        .. seealso::
           :func:`cupy.sum` for full documentation,
           :meth:`numpy.ndarray.sum`

        """
        return _sum(self, axis, dtype, out, keepdims)

    # TODO(okuta): Implement cumsum

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        """Returns the mean along a given axis.

        .. seealso::
           :func:`cupy.mean` for full documentation,
           :meth:`numpy.ndarray.mean`

        """
        return _mean(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """Returns the variance along a given axis.

        .. seealso::
           :func:`cupy.var` for full documentation,
           :meth:`numpy.ndarray.var`

        """
        return var(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """Returns the standard deviation along a given axis.

        .. seealso::
           :func:`cupy.std` for full documentation,
           :meth:`numpy.ndarray.std`

        """
        return std(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def prod(self, axis=None, dtype=None, out=None, keepdims=None):
        """Returns the product along a given axis.

        .. seealso::
           :func:`cupy.prod` for full documentation,
           :meth:`numpy.ndarray.prod`

        """
        return _prod(self, axis, dtype, out, keepdims)

    # TODO(okuta): Implement cumprod

    def all(self, axis=None, out=None, keepdims=False):
        return _all(self, axis=axis, out=out, keepdims=keepdims)

    def any(self, axis=None, out=None, keepdims=False):
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
        if self._size == 0:
            return False
        elif self._size == 1:
            return bool(self.get())
        else:
            msg = 'The truth value of an array with more than one element is ' \
                  'ambiguous. Use a.any() or a.all()'
            raise ValueError(msg)

    def __bool__(self):
        return self.__nonzero__()

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

    # Arithmetic __r{op}__ (CuPy specific):
    """

    def __radd__(self, other):
        return add(other, self)

    def __rsub__(self, other):
        return subtract(other, self)

    def __rmul__(self, other):
        return multiply(other, self)

    def __rdiv__(self, other):
        return divide(other, self)

    def __rtruediv__(self, other):
        return true_divide(other, self)

    def __rfloordiv__(self, other):
        return floor_divide(other, self)

    def __rmod__(self, other):
        return remainder(other, self)

    def __rdivmod__(self, other):
        return divmod(other, self)

    def __rpow__(self, other):
        return power(other, self)

    def __rlshift__(self, other):
        return left_shift(other, self)

    def __rrshift__(self, other):
        return right_shift(other, self)

    def __rand__(self, other):
        return bitwise_and(other, self)

    def __ror__(self, other):
        return bitwise_or(other, self)

    def __rxor__(self, other):
        return bitwise_xor(other, self)
    """

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
        if dtype is None or self._dtype == dtype:
            return self
        else:
            return self.astype(dtype)

    # TODO(okuta): Implement __array_wrap__

    # Container customization:

    def __len__(self):
        if not self._shape:
            raise TypeError('len() of unsized object')
        return self._shape[0]

    def __getitem__(self, slices):
        # It supports the basic indexing (by slices, ints or Ellipsis) only.
        # TODO(beam2d): Support the advanced indexing of NumPy.
        if not isinstance(slices, tuple):
            slices = [slices]
        else:
            slices = list(slices)

        if six.moves.builtins.any(isinstance(s, ndarray) for s in slices):
            raise ValueError('Advanced indexing is not supported')

        # Expand ellipsis into empty slices
        n_newaxes = slices.count(newaxis)
        n_ellipses = slices.count(Ellipsis)
        if n_ellipses > 0:
            if n_ellipses > 1:
                raise ValueError('Only one Ellipsis is allowed in index')
            ellipsis = slices.index(Ellipsis)
            ellipsis_size = self.ndim - (len(slices) - n_newaxes - 1)
            slices[ellipsis:ellipsis + 1] = [slice(None)] * ellipsis_size

        slices += [slice(None)] * (self.ndim - len(slices) + n_newaxes)

        # Create new shape and stride
        shape = []
        strides = []

        j = 0
        offset = 0
        for i, s in enumerate(slices):
            if s is newaxis:
                shape.append(1)
                if j < self.ndim:
                    strides.append(self._strides[j])
                elif self.ndim > 0:
                    strides.append(self._strides[-1])
                else:
                    strides.append(self.itemsize)
            elif isinstance(s, slice):
                s = complete_slice(s, self._shape[j])
                if s.step > 0:
                    dim = (s.stop - s.start - 1) // s.step + 1
                else:
                    dim = (s.stop - s.start + 1) // s.step + 1

                shape.append(dim)
                strides.append(self._strides[j] * s.step)

                offset += s.start * self._strides[j]
                j += 1
            elif numpy.isscalar(s):
                s = int(s)
                if s >= self._shape[j]:
                    raise IndexError('Index %s exceeds the size %s at axis %s'
                                     % (s, self._shape[j], j))
                offset += s * self._strides[j]
                j += 1
            else:
                raise TypeError('Invalid index type: %s' % type(slices[i]))

        v = self.view()
        v._shape = tuple(shape)
        v._strides = tuple(strides)
        v._size = internal_prod(shape)
        v.data = self.data + offset
        v._c_contiguous = -1
        v._f_contiguous = -1

        return v

    def __setitem__(self, slices, value):
        v = self[slices]
        if isinstance(value, ndarray):
            y, x = broadcast(v, value).values
            if y._shape == x._shape and y._strides == x._strides:
                if int(y.data) == int(x.data):
                    return  # Skip since x and y are the same array
                elif y.flags.c_contiguous and x.dtype == y.dtype:
                    y.data.copy_from(x.data, x.nbytes)
                    return
            elementwise_copy(x, y)
        else:
            v.fill(value)

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
    def dot(self, b, out=None):
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

    def get(self, stream=None):
        """Returns a copy of the array on host memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.

        Returns:
            numpy.ndarray: Copy of the array on host memory.

        """
        a_gpu = ascontiguousarray(self)
        a_cpu = numpy.empty(self._shape, dtype=self._dtype)
        ptr = a_cpu.ctypes.data_as(ctypes.c_void_p)
        if stream is None:
            a_gpu.data.copy_to_host(ptr, a_gpu.nbytes)
        else:
            a_gpu.data.copy_to_host_async(ptr, a_gpu.nbytes, stream)
        return a_cpu

    def set(self, arr, stream=None):
        """Copies an array on the host memory to cuda.ndarray.

        Args:
            arr (numpy.ndarray): The source array on the host memory.
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.

        """
        if not isinstance(arr, numpy.ndarray):
            raise TypeError('Only numpy.ndarray can be set to cupy.ndarray')
        if self._dtype != arr.dtype:
            raise TypeError('{} array cannot be set to {} array'.format(
                arr.dtype, self._dtype))
        if self._shape != arr.shape:
            raise ValueError('Shape mismatch')
        if not self.flags.c_contiguous:
            raise RuntimeError('Cannot set to non-contiguous array')

        arr = numpy.ascontiguousarray(arr)
        ptr = arr.ctypes.data_as(ctypes.c_void_p)
        if stream is None:
            self.data.copy_from_host(ptr, self.nbytes)
        else:
            self.data.copy_from_host_async(ptr, self.nbytes, stream)

    def reduced_view(self, dtype=None):
        """Returns a view of the array with minimum number of dimensions.

        Args:
            dtype: Data type specifier. If it is given, then the memory
                sequence is reinterpreted as the new type.

        Returns:
            cupy.ndarray: A view of the array with reduced dimensions.

        """
        ndim = self.ndim
        if ndim <= 1:
            return self
        shape, strides = get_reduced_dims(
            self._shape, self._strides, self.itemsize)
        if ndim == len(shape):
            return self

        view = self.view(dtype=dtype)
        view._shape = shape
        view._strides = strides
        if view._c_contiguous == 1:
            view._f_contiguous = int(
                not view.size or len(shape) - shape.count(1) <= 1)
        else:
            view._f_contiguous = -1
        return view

    def _update_c_contiguity(self):
        self._c_contiguous = int(get_c_contiguity(
            self._shape, self._strides, self.itemsize))

    def _update_f_contiguity(self):
        self._f_contiguous = int(get_c_contiguity(
            self._shape[::-1], self._strides[::-1], self.itemsize))

    def _update_contiguity(self):
        self._update_c_contiguity()
        self._update_f_contiguity()


newaxis = numpy.newaxis  # == None

six_range = six.moves.range
six_zip = six.moves.zip


cdef inline long long internal_prod(args, long long init=1):
    cdef long long arg
    for i in range(len(args)):
        arg = args[i]
        init *= arg
    return init


def get_reduced_dims(shape, strides, itemsize):
    if not shape:
        return (), ()
    if 0 in shape:
        return (0,), (itemsize,)

    if len(shape) == 1:
        return shape, strides
    if len(shape) == 2:
        shape0, shape1 = shape
        strides0, strides1 = strides
        if shape0 == 1 or strides0 == shape1 * strides1:
            return (shape0 * shape1,), (strides1,)
        else:
            return shape, strides

    last_shape = shape[0]
    last_stride = strides[0]
    reduced_shape = []
    reduced_strides = []
    reduced_shape_append = reduced_shape.append
    reduced_strides_append = reduced_strides.append

    for sh, st, prev_st in six_zip(shape[1:], strides[1:], strides):
        if last_shape == 1 or prev_st == sh * st:
            last_shape *= sh
            last_stride = st
        else:
            reduced_shape_append(last_shape)
            reduced_strides_append(last_stride)
            last_shape = sh
            last_stride = st
    reduced_shape_append(last_shape)
    reduced_strides_append(last_stride)

    return tuple(reduced_shape), tuple(reduced_strides)


def get_strides_for_nocopy_reshape(a, new_shape):
    a_size = a.size
    size = 1
    for s in new_shape:
        size *= s
    if a_size != size:
        return None

    a_itemsize = a.itemsize
    if a_size == 1:
        return (a_itemsize,) * len(new_shape)

    shape, strides = get_reduced_dims(a.shape, a.strides, a_itemsize)

    ndim = len(shape)
    dim = 0
    sh = shape[0]
    st = strides[0]
    last_stride = sh * st
    new_strides = []
    for size in new_shape:
        if size > 1:
            if sh == 1:
                dim += 1
                if dim >= ndim:
                    return None
                sh = shape[dim]
                st = strides[dim]
            if sh % size != 0:
                return None
            sh //= size
            last_stride = sh * st
        new_strides.append(last_stride)

    return tuple(new_strides)


def get_contiguous_strides(shape, itemsize):
    ndim = len(shape)
    if ndim == 0:
        return ()
    if ndim == 1:
        return itemsize,
    if ndim == 2:
        return shape[1] * itemsize, itemsize

    strides = [0] * ndim
    st = itemsize
    for i in six_range(ndim - 1, -1, -1):
        strides[i] = st
        sh = shape[i]
        if sh > 1:
            st *= sh
    return tuple(strides)


def complete_slice(slc, dim):
    step = 1 if slc.step is None else slc.step
    if step == 0:
        raise ValueError('Slice step must be nonzero.')
    elif step > 0:
        start = 0 if slc.start is None else max(0, min(dim, slc.start))
        stop = dim if slc.stop is None else max(start, min(dim, slc.stop))
    else:
        start = dim - 1 if slc.start is None else max(0, min(dim, slc.start))
        stop = -1 if slc.stop is None else max(0, min(start, slc.stop))
    return slice(start, stop, step)


def get_c_contiguity(shape, strides, itemsize):
    if 0 in shape:
        return True
    _, strides = get_reduced_dims(shape, strides, itemsize)
    ndim = len(strides)
    return ndim == 0 or (ndim == 1 and strides[0] == itemsize)


def infer_unknown_dimension(shape, size):
    cnt = 0
    for dim in shape:
        cnt += dim < 0
    if cnt == 0:
        return shape
    if cnt > 1:
        raise ValueError('can only specify only one unknown dimension')
    p = size
    for dim in shape:
        if dim > 0:
            p //= dim
    return tuple([dim if dim >= 0 else p for dim in shape])



clear_memo = util.clear_memo
memoize = util.memoize

include "carray.pxi"
include "elementwise.pxi"
include "reduction.pxi"


# =============================================================================
# Routines
# =============================================================================

_id = 'out0 = in0'

elementwise_copy = create_ufunc(
    'cupy_copy',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'),
    _id)


elementwise_copy_where = create_ufunc(
    'cupy_copy_where',
    ('??->?', 'b?->b', 'B?->B', 'h?->h', 'H?->H', 'i?->i', 'I?->I', 'l?->l',
     'L?->L', 'q?->q', 'Q?->Q', 'e?->e', 'f?->f', 'd?->d'),
    'if (in1) out0 = in0')


_divmod_float = '''
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


_min_max_preamble = '''
struct min_max_st{
    type_in0_raw value;
    int index;
    __device__ min_max_st() : index(-1) { }
    __device__ min_max_st(type_in0_raw v) : value(v), index(0) { }
    __device__ min_max_st(type_in0_raw v, int i) : value(v), index(i) { }
};
__device__ min_max_st my_min(const min_max_st& a, const min_max_st& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return min_max_st(min(a.value, b.value));
}
__device__ min_max_st my_max(const min_max_st& a, const min_max_st& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return min_max_st(max(a.value, b.value));
}
__device__ min_max_st my_argmin(const min_max_st& a, const min_max_st& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return (a.value <= b.value) ? a : b;
}
__device__ min_max_st my_argmax(const min_max_st& a, const min_max_st& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return (a.value >= b.value) ? a : b;
}'''


amin = create_reduction_func(
    'cupy_min',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'),
    ('min_max_st(in0)', 'my_min(a, b)', 'out0 = a.value', 'min_max_st'),
    None, _min_max_preamble)


amax = create_reduction_func(
    'cupy_max',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'),
    ('min_max_st(in0)', 'my_max(a, b)', 'out0 = a.value', 'min_max_st'),
    None, _min_max_preamble)


argmin = create_reduction_func(
    'cupy_argmin',
    ('?->l', 'B->l', 'h->l', 'H->l', 'i->l', 'I->l', 'l->l', 'L->l',
     'q->l', 'Q->l', 'e->l', 'f->l', 'd->l'),
    ('min_max_st(in0, _J)', 'my_argmin(a, b)', 'out0 = a.index', 'min_max_st'),
    None, _min_max_preamble)


argmax = create_reduction_func(
    'cupy_argmax',
    ('?->l', 'B->l', 'h->l', 'H->l', 'i->l', 'I->l', 'l->l', 'L->l',
     'q->l', 'Q->l', 'e->l', 'f->l', 'd->l'),
    ('min_max_st(in0, _J)', 'my_argmax(a, b)', 'out0 = a.index', 'min_max_st'),
    None, _min_max_preamble)


# -----------------------------------------------------------------------------
# Array creation routines
# -----------------------------------------------------------------------------

def array(obj, dtype=None, copy=True, ndmin=0):
    # TODO(beam2d): Support order and subok options
    if isinstance(obj, ndarray):
        if dtype is None:
            dtype = obj.dtype
        a = obj.astype(dtype, copy)

        ndim = a.ndim
        if ndmin > ndim:
            a.shape = (1,) * (ndmin - ndim) + a.shape
        return a
    else:
        a_cpu = numpy.array(obj, dtype=dtype, copy=False, ndmin=ndmin)
        if a_cpu.ndim > 0:
            a_cpu = numpy.ascontiguousarray(a_cpu)
        a = ndarray(a_cpu.shape, dtype=a_cpu.dtype)
        a.data.copy_from_host(a_cpu.ctypes.data_as(ctypes.c_void_p), a.nbytes)
        if a_cpu.dtype == a.dtype:
            return a
        else:
            return a.view(dtype=a_cpu.dtype)


def ascontiguousarray(a, dtype=None):
    if dtype is None:
        dtype = a.dtype
    else:
        dtype = numpy.dtype(dtype)

    if dtype == a.dtype and a.flags.c_contiguous:
        return a
    else:
        newarray = ndarray(a.shape, dtype)
        elementwise_copy(a, newarray)
        return newarray


def copy(a):
    if a.size == 0:
        return ndarray(a.shape, a.dtype)

    if not a.flags.c_contiguous:
        a = ascontiguousarray(a)
        if a.data.device == cuda.Device():
            return a
    newarray = ndarray(a.shape, a.dtype)
    newarray.data.copy_from(a.data, a.nbytes)
    return newarray

# -----------------------------------------------------------------------------
# Array manipulation routines
# -----------------------------------------------------------------------------

cdef ndarray reshape(ndarray a, newshape):
    cdef ndarray newarray
    # TODO(beam2d): Support ordering option
    if isinstance(newshape, collections.Sequence):
        newshape = tuple(newshape)
    else:
        newshape = newshape,

    shape = a.shape
    if newshape == shape:
        return a.view()

    size = a.size
    newshape = infer_unknown_dimension(newshape, size)
    if newshape == shape:
        return a.view()
    if internal_prod(newshape) != size:
        raise RuntimeError('Total size mismatch on reshape')

    newstrides = get_strides_for_nocopy_reshape(a, newshape)
    if newstrides is not None:
        newarray = a.view()
    else:
        newarray = a.copy()
        newstrides = get_strides_for_nocopy_reshape(
            newarray, newshape)
    newarray._shape = newshape
    newarray._strides = newstrides
    if newarray._c_contiguous == 1:
        newarray._f_contiguous = int(
            not size or len(newshape) - newshape.count(1) <= 1)
    else:
        newarray._f_contiguous = -1
    return newarray


cpdef ndarray rollaxis(ndarray a, axis, start=0):
    ndim = a.ndim
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
        return transpose(a, None)

    axes = list(six.moves.range(ndim))
    del axes[axis]
    axes.insert(start, axis)
    return transpose(a, axes)


cdef ndarray swapaxes(ndarray a, axis1, axis2):
    ndim = a.ndim
    if axis1 >= ndim or axis2 >= ndim:
        raise ValueError('Axis out of range')
    axes = list(six.moves.range(ndim))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
    return transpose(a, axes)


cdef ndarray transpose(ndarray a, axes=None):
    ndim = a.ndim
    a_shape = a._shape
    a_strides = a._strides
    ret = a.view()

    if not axes:
        if ndim > 1:
            ret._shape = a_shape[::-1]
            ret._strides = a_strides[::-1]
            ret._c_contiguous, ret._f_contiguous = \
                a._f_contiguous, a._c_contiguous
        return ret

    if ndim != len(axes):
        raise ValueError('Invalid axes value: %s' % str(axes))

    if ndim <= 2:
        if ndim == 0:
            return ret
        elif ndim == 1:
            if axes[0] == 0:
                return ret
        else:
            axis0, axis1 = axes
            if axis0 == 0 and axis1 == 1:
                return ret
            elif axis0 == 1 and axis1 == 0:
                ret._shape = a_shape[::-1]
                ret._strides = a_strides[::-1]
                ret._c_contiguous, ret._f_contiguous = \
                    a._f_contiguous, a._c_contiguous
                return ret
        raise ValueError('Invalid axes value: %s' % str(axes))

    for axis in axes:
        if axis < -ndim or axis >= ndim:
            raise IndexError('Axes overrun')
    axes = [axis % ndim for axis in axes]

    a_axes = list(six.moves.range(ndim))

    if a_axes == axes:
        return ret

    if a_axes == axes[::-1]:
        ret._shape = a_shape[::-1]
        ret._strides = a_strides[::-1]
        ret._c_contiguous, ret._f_contiguous = \
            a._f_contiguous, a._c_contiguous
        return ret

    if a_axes != sorted(axes):
        raise ValueError('Invalid axes value: %s' % str(axes))

    ret._shape = tuple([a_shape[axis] for axis in axes])
    ret._strides = tuple([a_strides[axis] for axis in axes])
    ret._c_contiguous = -1
    ret._f_contiguous = -1

    return ret


class broadcast(object):
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

    def __init__(self, *arrays):
        rev = slice(None, None, -1)
        shape_arr = [a._shape[rev] for a in arrays
                     if isinstance(a, ndarray)]
        r_shape = [max(ss) for ss
                   in six.moves.zip_longest(*shape_arr, fillvalue=0)]

        self.shape = shape = tuple(r_shape[rev])
        self.size = size = internal_prod(shape)
        self.nd = ndim = len(shape)

        broadcasted = list(arrays)
        for i, a in enumerate(broadcasted):
            if not isinstance(a, ndarray):
                continue

            a_shape = a.shape
            if a_shape == shape:
                continue

            r_strides = [
                a_st if sh == a_sh else (0 if a_sh == 1 else None)
                for sh, a_sh, a_st
                in six_zip(r_shape, a._shape[rev], a._strides[rev])]

            if None in r_strides:
                raise ValueError('Broadcasting failed')

            offset = (0,) * (ndim - len(r_strides))

            broadcasted[i] = view = a.view()
            view._shape = shape
            view._strides = offset + tuple(r_strides[rev])
            view._size = size
            view._c_contiguous = -1
            view._f_contiguous = -1

        self.values = tuple(broadcasted)


cdef ndarray squeeze(ndarray a, axis=None):
    if axis is None:
        axis = tuple(i for i, n in enumerate(a._shape) if n == 1)
    elif isinstance(axis, int):
        axis = axis,

    new_shape = []
    new_strides = []
    j = 0
    for i, n in enumerate(a._shape):
        if j < len(axis) and i == axis[j]:
            if n != 1:
                raise RuntimeError('Cannot squeeze dimension of size > 1')
            j += 1
        else:
            new_shape.append(n)
            new_strides.append(a._strides[i])

    v = a.view()
    v._shape = tuple(new_shape)
    v._strides = tuple(new_strides)
    v._c_contiguous = -1
    v._f_contiguous = -1
    return v

# -----------------------------------------------------------------------------
# Binary operations
# -----------------------------------------------------------------------------

#elementeise
def _create_bit_op(name, op, no_bool, doc=''):
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

_take_kernel = ElementwiseKernel(
    'raw T a, S indices, int64 cdim, int64 rdim',
    'T out',
    '''
      long long li = i / (rdim * cdim);
      long long ri = i % rdim;
      out = a[(li * cdim + indices) * rdim + ri];
    ''',
    'cupy_take')


def take(a, indices, axis=None, out=None):
    if axis is None:
        a = a.ravel()
        lshape = ()
        rshape = ()
    else:
        if axis >= a.ndim:
            raise ValueError('Axis overrun')
        lshape = a.shape[:axis]
        rshape = a.shape[axis + 1:]

    if numpy.isscalar(indices):
        a = rollaxis(a, axis)
        if out is None:
            return a[indices].copy()
        else:
            out[:] = a[indices]
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
    rdim = internal_prod(rshape)
    indices = indices.reshape(
        (1,) * len(lshape) + indices.shape + (1,) * len(rshape))
    return _take_kernel(a, indices, cdim, rdim, out)


def diagonal(a, offset=0, axis1=0, axis2=1):
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
    ret._shape = a.shape[:-2] + (diag_size,)
    ret._strides = a.strides[:-2] + (a.strides[-1] + a.strides[-2],)
    ret._size = internal_prod(ret._shape)
    ret._c_contiguous = -1
    ret._f_contiguous = -1
    return ret


# -----------------------------------------------------------------------------
# Linear algebra
# -----------------------------------------------------------------------------

def dot(a, b, out=None):
    a_ndim = a.ndim
    b_ndim = b.ndim
    assert a_ndim > 0 and b_ndim > 0
    a_is_vec = a_ndim == 1
    b_is_vec = b_ndim == 1

    if a_is_vec:
        a = a.reshape((1, a.size))
        a_ndim = 2
    if b_is_vec:
        b = b.reshape((b.size, 1))
        b_ndim = 2

    a_axis = a_ndim - 1
    b_axis = b_ndim - 2

    if a.shape[a_axis] != b.shape[b_axis]:
        raise ValueError('Axis dimension mismatch')

    if a_axis:
        a = rollaxis(a, a_axis, 0)
    if b_axis:
        b = rollaxis(b, b_axis, 0)

    k = a.shape[0]
    m = b.size // k
    n = a.size // k

    ret_shape = a.shape[1:] + b.shape[1:]
    if out is None:
        if a_is_vec:
            ret_shape = () if b_is_vec else ret_shape[1:]
        elif b_is_vec:
            ret_shape = ret_shape[:-1]
    else:
        if out.size != n * m:
            raise ValueError('Output array has an invalid size')
        if not out.flags.c_contiguous:
            raise ValueError('Output array must be C-contiguous')

    return _tensordot_core(a, b, out, n, m, k, ret_shape)


cpdef ndarray _tensordot_core(
        ndarray a, ndarray b, ndarray out, Py_ssize_t n, Py_ssize_t m,
        Py_ssize_t k, ret_shape):
    ret_dtype = a.dtype.char
    if ret_dtype != b.dtype.char:
        ret_dtype = numpy.find_common_type((ret_dtype, b.dtype), ()).char

    # Cast to float32 or float64
    if ret_dtype == 'f' or ret_dtype == 'd':
        dtype = ret_dtype
    else:
        dtype = numpy.find_common_type((ret_dtype, 'f'), ()).char

    a = a.astype(dtype, copy=False)
    b = b.astype(dtype, copy=False)

    if not a.size or not b.size:
        if a.size or b.size:
            raise ValueError('cannot dot zero-sized and non-zero-sized arrays')
        if out is None:
            out = ndarray(ret_shape, dtype=ret_dtype)
        out.fill(0)
        return out

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

    # It copies the operands if needed
    if a.shape != (k, n):
        a = a.reshape((k, n))
    if b.shape != (k, m):
        b = b.reshape((k, m))
    c = out
    if c.shape != (n, m):
        c = c.view()
        c.shape = (n, m)

    # Be careful that cuBLAS uses the FORTRAN-order matrix representation.
    if k == 1:
        if n == 1:
            # Scalar-vector product
            multiply(a, b, c)
        elif m == 1:
            # Scalar-vector product
            multiply(a.T, b, c)
        else:
            # Outer product A^T * B
            # c is C-contiguous while cuBLAS requires F-contiguous arrays, so
            # we compute C^T = B^T * A here.
            handle = cuda.Device().cublas_handle
            c.fill(0)
            a, inca = _to_cublas_vector(a, 1)
            b, incb = _to_cublas_vector(b, 1)
            if dtype == 'f':
                ger = cublas.sger
            elif dtype == 'd':
                ger = cublas.dger
            ger(handle, m, n, 1, b.data.ptr, incb, a.data.ptr, inca,
                c.data.ptr, m)

        if dtype != ret_dtype:
            elementwise_copy(out, ret)
        return ret

    handle = cuda.Device().cublas_handle
    if n == 1:
        if m == 1:
            # Inner product
            a, inca = _to_cublas_vector(a, 0)
            b, incb = _to_cublas_vector(b, 0)
            mode = cublas.getPointerMode(handle)
            cublas.setPointerMode(handle,
                                  cublas.CUBLAS_POINTER_MODE_DEVICE)
            if dtype == 'f':
                dot = cublas.sdot
            elif dtype == 'd':
                dot = cublas.ddot
            try:
                dot(handle, k, a.data.ptr, inca, b.data.ptr, incb, c.data.ptr)
            finally:
                cublas.setPointerMode(handle, mode)
        else:
            # Matrix-vector product B^T * A
            a, inca = _to_cublas_vector(a, 0)
            b, transb, ldb = _mat_to_cublas_contiguous(b, 1)
            if transb:
                # gemv requires (m, k) as the original matrix dimensions
                # rather than the transposed dimensions.
                m, k = k, m
            if dtype == 'f':
                gemv = cublas.sgemv
            elif dtype == 'd':
                gemv = cublas.dgemv
            gemv(handle, transb, m, k, 1, b.data.ptr, ldb, a.data.ptr, inca,
                 0, c.data.ptr, 1)
    elif m == 1:
        # Matrix-vector product A^T * B
        a, transa, lda = _mat_to_cublas_contiguous(a, 1)
        b, incb = _to_cublas_vector(b, 0)
        if transa:
            # gemv requires (n, k) as the original matrix dimensions rather
            # than the transposed dimensions.
            n, k = k, n
        if dtype == 'f':
            gemv = cublas.sgemv
        elif dtype == 'd':
            gemv = cublas.dgemv
        gemv(handle, transa, n, k, 1, a.data.ptr, lda, b.data.ptr, incb, 0,
             c.data.ptr, 1)
    else:
        # Matrix-Matrix product A^T * B
        # c is C-contiguous while cuBLAS assumes F-contiguous inputs, so we
        # compute C^T = B^T * A here.
        a, transa, lda = _mat_to_cublas_contiguous(a, 0)
        b, transb, ldb = _mat_to_cublas_contiguous(b, 1)
        if dtype == 'f':
            gemm = cublas.sgemm
        elif dtype == 'd':
            gemm = cublas.dgemm
        gemm(handle, transb, transa, m, n, k, 1, b.data.ptr, ldb, a.data.ptr,
             lda, 0, c.data.ptr, m)

    if dtype != ret_dtype:
        elementwise_copy(out, ret)
    return ret


def _move_axes_to_head(a, axes):
    # This function moves the axes of ``s`` to the head of the shape.
    for idx, axis in enumerate(axes):
        if idx != axis:
            break
    else:
        return a

    return a.transpose(
        axes + [i for i in six.moves.range(a.ndim) if i not in axes])


def _mat_to_cublas_contiguous(a, trans):
    assert a.ndim == 2
    f = a.flags
    if f.f_contiguous:
        return a, trans, a.strides[1] // a.itemsize
    if not f.c_contiguous:
        a = a.copy()
    return a, 1 - trans, a.strides[0] // a.itemsize


def _to_cublas_vector(a, rundim):
    if a.strides[rundim] < 0:
        return a.copy(), 1
    else:
        return a, a.strides[rundim] // a.itemsize

# -----------------------------------------------------------------------------
# Logic functions
# -----------------------------------------------------------------------------

def create_comparison(name, op, doc=''):
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


def create_arithmetic(name, op, boolop, doc):
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


# Fixed version of sqrt
sqrt_fixed = create_ufunc(
    'cupy_sqrt',
    ('e->e', 'f->f', 'd->d'),
    'out0 = sqrt(in0)')


_clip = create_ufunc(
    'cupy_clip',
    ('???->?', 'bbb->b', 'BBB->B', 'hhh->h', 'HHH->H', 'iii->i', 'III->I',
     'lll->l', 'LLL->L', 'qqq->q', 'QQQ->Q', 'eee->e', 'fff->f', 'ddd->d'),
    'out0 = min(in2, max(in1, in0))')

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
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


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    ret = var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
    return sqrt_fixed(ret, dtype=dtype, out=out)


_var_core = ReductionKernel(
    'S x, T mean, T alpha', 'T out',
    '(x - mean) * (x - mean)',
    'a + b', 'out = alpha * a', '0', '_var_core')
_var_core_out = ReductionKernel(
    'S x, T mean, T alpha', 'U out',
    '(x - mean) * (x - mean)',
    'a + b', 'out = alpha * a', '0', '_var_core')

# TODO(okuta) needs cast
_mean = create_reduction_func(
    'cupy_mean',
    ('?->d', 'B->d', 'h->d', 'H->d', 'i->d', 'I->d', 'l->d', 'L->d',
     'q->d', 'Q->d',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d'),
    ('in0', 'a + b', 'out0 = a / (_in_ind.size() / _out_ind.size())', None))
