from __future__ import division
import collections
import ctypes
import sys

import numpy
import six

from cupy import binary
from cupy import carray
from cupy import creation
from cupy import cuda
from cupy import elementwise
from cupy import flags
from cupy import indexing
from cupy import internal
from cupy import io
from cupy import linalg
from cupy import logic
from cupy import manipulation
from cupy import math
import cupy.random
from cupy import reduction
from cupy import sorting
from cupy import statistics
from cupy import util

random = cupy.random

# dtype short cut
number = numpy.number
integer = numpy.integer
signedinteger = numpy.signedinteger
unsignedinteger = numpy.unsignedinteger
inexact = numpy.inexact
floating = numpy.floating

bool_ = numpy.bool_
byte = numpy.byte
short = numpy.short
intc = numpy.intc
int_ = numpy.int_
longlong = numpy.longlong
ubyte = numpy.ubyte
ushort = numpy.ushort
uintc = numpy.uintc
uint = numpy.uint
ulonglong = numpy.ulonglong

half = numpy.half
single = numpy.single
float_ = numpy.float_
longfloat = numpy.longfloat

int8 = numpy.int8
int16 = numpy.int16
int32 = numpy.int32
int64 = numpy.int64
uint8 = numpy.uint8
uint16 = numpy.uint16
uint32 = numpy.uint32
uint64 = numpy.uint64

float16 = numpy.float16
float32 = numpy.float32
float64 = numpy.float64


def _get_size(size):
    if size is None:
        return ()
    elif isinstance(size, collections.Sequence):
        return tuple(size)
    elif isinstance(size, int):
        return size,
    else:
        raise ValueError('size should be None, collections.Sequence, or int')


class ndarray(object):

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
            self._strides = internal.get_contiguous_strides(
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

    @property
    def shape(self):
        """Lengths of axes.

        Setter of this property involves reshaping without copy. If the array
        cannot be reshaped without copy, it raises an exception.

        .. seealso: :attr:`numpy.ndarray.shape`

        """
        return self._shape

    @shape.setter
    def shape(self, newshape):
        newshape = internal.infer_unknown_dimension(newshape, self._size)
        strides = internal.get_strides_for_nocopy_reshape(self, newshape)
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
    # ctypes foreign function interface
    # -------------------------------------------------------------------------
    @property
    def ctypes(self):
        """C representation of the array.

        This property is used for sending an array to CUDA kernels. The type of
        returned C structure is different for different dtypes and ndims. The
        definition of C type is written in ``cupy/carray.cuh``.

        .. note::
           The returned value does not have compatibility with
           :attr:`numpy.ndarray.ctypes`.

        """
        return carray.to_carray(self.data.ptr, self._size, self._shape,
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
            newarray = empty_like(self, dtype=dtype)
            elementwise.copy(self, newarray)
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

    def view(self, dtype=None):
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
        elementwise.copy(value, self, dtype=self._dtype)

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
        if len(axes) == 1 and isinstance(axes[0], collections.Sequence):
            axes = axes[0]
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
            newarray = empty_like(self)
            elementwise.copy(self, newarray)

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
        return ravel(self)

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
        return clip(self, a_min, a_max, out=out)

    # TODO(okuta): Implement round

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        """Returns the sum along diagonals of the array.

        .. seealso::
           :func:`cupy.trace` for full documentation,
           :meth:`numpy.ndarray.trace`

        """
        return trace(self, offset, axis1, axis2, dtype, out)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """Returns the sum along a given axis.

        .. seealso::
           :func:`cupy.sum` for full documentation,
           :meth:`numpy.ndarray.sum`

        """
        return sum(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    # TODO(okuta): Implement cumsum

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        """Returns the mean along a given axis.

        .. seealso::
           :func:`cupy.mean` for full documentation,
           :meth:`numpy.ndarray.mean`

        """
        return mean(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

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
        return prod(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    # TODO(okuta): Implement cumprod
    # TODO(okuta): Implement all
    # TODO(okuta): Implement any

    # -------------------------------------------------------------------------
    # Arithmetic and comparison operations
    # -------------------------------------------------------------------------
    # Comparison operators:

    def __lt__(self, other):
        return less(self, other)

    def __le__(self, other):
        return less_equal(self, other)

    def __gt__(self, other):
        return greater(self, other)

    def __ge__(self, other):
        return greater_equal(self, other)

    def __eq__(self, other):
        return equal(self, other)

    def __ne__(self, other):
        return not_equal(self, other)

    # Truth value of an array (bool):

    def __nonzero__(self):
        if self._size == 0:
            return False
        elif self._size == 1:
            return bool(self.get())
        else:
            msg = 'The truth value of an array with more than one element is ' \
                  'ambiguous. Use a.get().any() or a.get().all()'
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

    def __add__(self, other):
        if self._should_use_rop(other):
            return other.__radd__(self)
        else:
            return add(self, other)

    def __sub__(self, other):
        if self._should_use_rop(other):
            return other.__rsub__(self)
        else:
            return subtract(self, other)

    def __mul__(self, other):
        if self._should_use_rop(other):
            return other.__rmul__(self)
        else:
            return multiply(self, other)

    def __div__(self, other):
        if self._should_use_rop(other):
            return other.__rdiv__(self)
        else:
            return divide(self, other)

    def __truediv__(self, other):
        if self._should_use_rop(other):
            return other.__rtruediv__(self)
        else:
            return true_divide(self, other)

    def __floordiv__(self, other):
        if self._should_use_rop(other):
            return other.__rfloordiv__(self)
        else:
            return floor_divide(self, other)

    def __mod__(self, other):
        if self._should_use_rop(other):
            return other.__rmod__(self)
        else:
            return remainder(self, other)

    def __divmod__(self, other):
        if self._should_use_rop(other):
            return other.__rdivmod__(self)
        else:
            return elementwise._divmod(self, other)

    def __pow__(self, other, modulo=None):
        # Note that we ignore the modulo argument as well as NumPy.
        if self._should_use_rop(other):
            return other.__rpow__(self)
        else:
            return power(self, other)

    def __lshift__(self, other):
        if self._should_use_rop(other):
            return other.__rlshift__(self)
        else:
            return left_shift(self, other)

    def __rshift__(self, other):
        if self._should_use_rop(other):
            return other.__rrshift__(self)
        else:
            return right_shift(self, other)

    def __and__(self, other):
        if self._should_use_rop(other):
            return other.__rand__(self)
        else:
            return bitwise_and(self, other)

    def __or__(self, other):
        if self._should_use_rop(other):
            return other.__ror__(self)
        else:
            return bitwise_or(self, other)

    def __xor__(self, other):
        if self._should_use_rop(other):
            return other.__rxor__(self)
        else:
            return bitwise_xor(self, other)

    # Arithmetic __r{op}__ (CuPy specific):

    def __radd__(self, other):
        return add(other, self)

    def __rsub__(self, other):
        return subtract(other, self)

    def __rmul__(self, other):
        if not isinstance(other, ndarray):
            return multiply(other, self)
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
        return elementwise._divmod(other, self)

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

    def __ipow__(self, other, modulo=None):
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

        if any(isinstance(s, ndarray) for s in slices):
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
                s = internal.complete_slice(s, self._shape[j])
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
        v._size = internal.prod(shape)
        v.data = self.data + offset
        v._c_contiguous = -1
        v._f_contiguous = -1

        return v

    def __setitem__(self, slices, value):
        v = self[slices]
        if isinstance(value, ndarray):
            y, x = broadcast_arrays(v, value)
            if y._shape == x._shape and y._strides == x._strides:
                if int(y.data) == int(x.data):
                    return  # Skip since x and y are the same array
                elif y.flags.c_contiguous and x.dtype == y.dtype:
                    y.data.copy_from(x.data, x.nbytes)
                    return
            elementwise.copy(x, y)
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
        shape, strides = internal.get_reduced_dims(
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
        self._c_contiguous = int(internal.get_c_contiguity(
            self._shape, self._strides, self.itemsize))

    def _update_f_contiguity(self):
        self._f_contiguous = int(internal.get_c_contiguity(
            self._shape[::-1], self._strides[::-1], self.itemsize))

    def _update_contiguity(self):
        self._update_c_contiguity()
        self._update_f_contiguity()

    def _should_use_rop(self, a):
        return getattr(a, '__array_priority__', 0) > self.__array_priority__


ufunc = elementwise.ufunc
newaxis = numpy.newaxis  # == None

# =============================================================================
# Routines
#
# The order of these declarations are borrowed from the NumPy document:
# http://docs.scipy.org/doc/numpy/reference/routines.html
# =============================================================================

# -----------------------------------------------------------------------------
# Array creation routines
# -----------------------------------------------------------------------------
empty = creation.basic.empty
empty_like = creation.basic.empty_like
eye = creation.basic.eye
identity = creation.basic.identity
ones = creation.basic.ones
ones_like = creation.basic.ones_like
zeros = creation.basic.zeros
zeros_like = creation.basic.zeros_like
full = creation.basic.full
full_like = creation.basic.full_like

array = creation.from_data.array
asarray = creation.from_data.asarray
asanyarray = creation.from_data.asanyarray
ascontiguousarray = creation.from_data.ascontiguousarray
copy = creation.from_data.copy

arange = creation.ranges.arange
linspace = creation.ranges.linspace

diag = creation.matrix.diag
diagflat = creation.matrix.diagflat

# -----------------------------------------------------------------------------
# Array manipulation routines
# -----------------------------------------------------------------------------
copyto = manipulation.basic.copyto

reshape = manipulation.shape.reshape
ravel = manipulation.shape.ravel

rollaxis = manipulation.transpose.rollaxis
swapaxes = manipulation.transpose.swapaxes
transpose = manipulation.transpose.transpose

atleast_1d = manipulation.dims.atleast_1d
atleast_2d = manipulation.dims.atleast_2d
atleast_3d = manipulation.dims.atleast_3d
broadcast = manipulation.dims.broadcast
broadcast_arrays = manipulation.dims.broadcast_arrays
expand_dims = manipulation.dims.expand_dims
squeeze = manipulation.dims.squeeze

column_stack = manipulation.join.column_stack
concatenate = manipulation.join.concatenate
dstack = manipulation.join.dstack
hstack = manipulation.join.hstack
vstack = manipulation.join.vstack

array_split = manipulation.split.array_split
dsplit = manipulation.split.dsplit
hsplit = manipulation.split.hsplit
split = manipulation.split.split
vsplit = manipulation.split.vsplit

# -----------------------------------------------------------------------------
# Binary operations
# -----------------------------------------------------------------------------
bitwise_and = binary.elementwise.bitwise_and
bitwise_or = binary.elementwise.bitwise_or
bitwise_xor = binary.elementwise.bitwise_xor
invert = binary.elementwise.invert
left_shift = binary.elementwise.left_shift
right_shift = binary.elementwise.right_shift

binary_repr = numpy.binary_repr

# -----------------------------------------------------------------------------
# Data type routines (borrowed from NumPy)
# -----------------------------------------------------------------------------
can_cast = numpy.can_cast
promote_types = numpy.promote_types
min_scalar_type = numpy.min_scalar_type
result_type = numpy.result_type
common_type = numpy.common_type
obj2sctype = numpy.obj2sctype

dtype = numpy.dtype
format_parser = numpy.format_parser

finfo = numpy.finfo
iinfo = numpy.iinfo
MachAr = numpy.MachAr

issctype = numpy.issctype
issubdtype = numpy.issubdtype
issubsctype = numpy.issubsctype
issubclass_ = numpy.issubclass_
find_common_type = numpy.find_common_type

typename = numpy.typename
sctype2char = numpy.sctype2char
mintypecode = numpy.mintypecode

# -----------------------------------------------------------------------------
# Optionally Scipy-accelerated routines
# -----------------------------------------------------------------------------
# TODO(beam2d): Implement it

# -----------------------------------------------------------------------------
# Discrete Fourier Transform
# -----------------------------------------------------------------------------
# TODO(beam2d): Implement it

# -----------------------------------------------------------------------------
# Indexing routines
# -----------------------------------------------------------------------------
take = indexing.indexing.take
diagonal = indexing.indexing.diagonal

# -----------------------------------------------------------------------------
# Input and output
# -----------------------------------------------------------------------------
load = io.npz.load
save = io.npz.save
savez = io.npz.savez
savez_compressed = io.npz.savez_compressed

array_repr = io.formatting.array_repr
array_str = io.formatting.array_str

base_repr = numpy.base_repr

# -----------------------------------------------------------------------------
# Linear algebra
# -----------------------------------------------------------------------------
dot = linalg.product.dot
vdot = linalg.product.vdot
inner = linalg.product.inner
outer = linalg.product.outer
tensordot = linalg.product.tensordot

trace = linalg.norm.trace

# -----------------------------------------------------------------------------
# Logic functions
# -----------------------------------------------------------------------------
isfinite = logic.content.isfinite
isinf = logic.content.isinf
isnan = logic.content.isnan

isscalar = numpy.isscalar

logical_and = logic.ops.logical_and
logical_or = logic.ops.logical_or
logical_not = logic.ops.logical_not
logical_xor = logic.ops.logical_xor

greater = logic.comparison.greater
greater_equal = logic.comparison.greater_equal
less = logic.comparison.less
less_equal = logic.comparison.less_equal
equal = logic.comparison.equal
not_equal = logic.comparison.not_equal

# -----------------------------------------------------------------------------
# Mathematical functions
# -----------------------------------------------------------------------------
sin = math.trigonometric.sin
cos = math.trigonometric.cos
tan = math.trigonometric.tan
arcsin = math.trigonometric.arcsin
arccos = math.trigonometric.arccos
arctan = math.trigonometric.arctan
hypot = math.trigonometric.hypot
arctan2 = math.trigonometric.arctan2
deg2rad = math.trigonometric.deg2rad
rad2deg = math.trigonometric.rad2deg
degrees = math.trigonometric.degrees
radians = math.trigonometric.radians

sinh = math.hyperbolic.sinh
cosh = math.hyperbolic.cosh
tanh = math.hyperbolic.tanh
arcsinh = math.hyperbolic.arcsinh
arccosh = math.hyperbolic.arccosh
arctanh = math.hyperbolic.arctanh

rint = math.rounding.rint
floor = math.rounding.floor
ceil = math.rounding.ceil
trunc = math.rounding.trunc

sum = math.sumprod.sum
prod = math.sumprod.prod

exp = math.explog.exp
expm1 = math.explog.expm1
exp2 = math.explog.exp2
log = math.explog.log
log10 = math.explog.log10
log2 = math.explog.log2
log1p = math.explog.log1p
logaddexp = math.explog.logaddexp
logaddexp2 = math.explog.logaddexp2

signbit = math.floating.signbit
copysign = math.floating.copysign
ldexp = math.floating.ldexp
frexp = math.floating.frexp
nextafter = math.floating.nextafter

add = math.arithmetic.add
reciprocal = math.arithmetic.reciprocal
negative = math.arithmetic.negative
multiply = math.arithmetic.multiply
divide = math.arithmetic.divide
power = math.arithmetic.power
subtract = math.arithmetic.subtract
true_divide = math.arithmetic.true_divide
floor_divide = math.arithmetic.floor_divide
fmod = math.arithmetic.fmod
mod = math.arithmetic.remainder
modf = math.arithmetic.modf
remainder = math.arithmetic.remainder

clip = math.misc.clip
sqrt = math.misc.sqrt
square = math.misc.square
absolute = math.misc.absolute
sign = math.misc.sign
maximum = math.misc.maximum
minimum = math.misc.minimum
fmax = math.misc.fmax
fmin = math.misc.fmin

# -----------------------------------------------------------------------------
# Sorting, searching, and counting
# -----------------------------------------------------------------------------
argmax = sorting.search.argmax
argmin = sorting.search.argmin

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
amin = statistics.order.amin
amax = statistics.order.amax

mean = statistics.meanvar.mean
var = statistics.meanvar.var
std = statistics.meanvar.std


# CuPy specific functions

def asnumpy(a, stream=None):
    """Returns an array on the host memory from an arbitrary source array.

    Args:
        a: Arbitrary object that can be converted to numpy.ndarray.
        stream (cupy.cuda.Stream): CUDA stream object. If it is specified, then
            the device-to-host copy runs asynchronously. Otherwise, the copy is
            synchronous. Note that if ``a`` is not a cupy.ndarray object, then
            this argument has no effect.

    Returns:
        numpy.ndarray: Converted array on the host memory.

    """
    if isinstance(a, ndarray):
        return a.get(stream=stream)
    else:
        return numpy.asarray(a)


_cupy = sys.modules[__name__]


def get_array_module(*args):
    """Returns the array module for arguments.

    This function is used to implement CPU/GPU generic code. If at least one of
    the arguments is a :class:`cupy.ndarray` object, the :mod:`cupy` module is
    returned.

    Args:
        args: Values to determine whether NumPy or CuPy should be used.

    Returns:
        module: :mod:`cupy` or :mod:`numpy` is returned based on the types of
        the arguments.

    .. admonition:: Example

       A NumPy/CuPy generic function can be written as follows::

           def softplus(x):
               xp = cupy.get_array_module(x)
               return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))

    """
    if any(isinstance(arg, ndarray) for arg in args):
        return _cupy
    else:
        return numpy


clear_memo = util.clear_memo
memoize = util.memoize

ElementwiseKernel = elementwise.ElementwiseKernel
ReductionKernel = reduction.ReductionKernel
