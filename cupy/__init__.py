from __future__ import division
import ctypes

import numpy

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
from cupy import sorting
from cupy import statistics


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


class ndarray(object):

    """Multi-dimensional array on a CUDA device."""

    def __init__(self, shape, dtype=numpy.float64, memptr=None, offset=0,
                 strides=None, allocator=cuda.alloc):
        self._shape = tuple(shape)
        self._dtype = numpy.dtype(dtype)
        self._allocator = allocator
        if self.ndim > carray.MAX_NDIM:
            raise RuntimeError('cupy.ndarray does not support ndim > 25.')

        nbytes = self.nbytes
        if memptr is None:
            self.data = allocator(nbytes)
        else:
            self.data = memptr + offset

        if strides is None:
            self._strides = internal.get_contiguous_strides(
                self._shape, self.itemsize)
            self._flags = flags.C_CONTIGUOUS | flags.OWNDATA
            if numpy.sum(dim != 1 for dim in shape) <= 1 or nbytes == 0:
                self._flags |= flags.F_CONTIGUOUS
        else:
            self._strides = strides
            self._flags = flags.OWNDATA
            self._update_contiguity()

        self.base = None

    # The definition order of attributes and methods are borrowed from the
    # order of documentation at the following NumPy document.
    # http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html

    # -------------------------------------------------------------------------
    # Memory layout
    # -------------------------------------------------------------------------
    @property
    def flags(self):
        return flags.Flags(self._flags)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, newshape):
        newshape = internal.infer_unknown_dimension(newshape, self.size)
        strides = internal.get_strides_for_nocopy_reshape(self, newshape)
        if strides is None:
            raise AttributeError('Incompatible shape')
        self._shape = newshape
        self._strides = strides
        self._update_f_contiguity()

    @property
    def strides(self):
        return self._strides

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return numpy.prod(self.shape, dtype=numpy.int64)

    @property
    def itemsize(self):
        return self._dtype.itemsize

    @property
    def nbytes(self):
        return self.size * self.itemsize

    # -------------------------------------------------------------------------
    # Data type
    # -------------------------------------------------------------------------
    @property
    def dtype(self):
        return self._dtype

    # -------------------------------------------------------------------------
    # Other attributes
    # -------------------------------------------------------------------------
    @property
    def T(self):
        if self.ndim < 2:
            return self
        else:
            return self.transpose()

    __array_priority__ = 100

    # -------------------------------------------------------------------------
    # Array interface
    # -------------------------------------------------------------------------
    # TODO(beam2d): Implement it
    # __array_interface__

    # -------------------------------------------------------------------------
    # ctypes foreign function interface
    # -------------------------------------------------------------------------
    @property
    def ctypes(self):
        return carray.to_carray(self.data.ptr, self.size, self.shape,
                                self.strides)

    # -------------------------------------------------------------------------
    # Array conversion
    # -------------------------------------------------------------------------
    # TODO(beam2d): Implement it
    # def item(self, *args):

    def tolist(self):
        return self.get().tolist()

    # TODO(beam2d): Implement these
    # def itemset(self, *args):
    # def tostring(self, order='C'):
    # def tobytes(self, order='C'):

    def tofile(self, fid, sep='', format='%s'):
        self.get().tofile(fid, sep, format)

    # TODO(beam2d): Implement these
    # def dump(self, file):
    # def dumps(self):

    def astype(self, dtype, copy=True, allocator=None):
        # TODO(beam2d): Support ordering, casting, and subok option
        dtype = numpy.dtype(dtype)
        if dtype == self.dtype:
            if copy:
                return self.copy(allocator)
            else:
                return self.view()
        else:
            newarray = empty_like(self, dtype=dtype, allocator=allocator)
            elementwise.copy(self, newarray)
            return newarray

    # TODO(beam2d): Implement it
    # def byteswap(self, inplace=False):

    def copy(self, allocator=None):
        # TODO(beam2d): Support ordering option
        return copy(self, allocator)

    def view(self, dtype=None):
        v = ndarray.__new__(ndarray)
        v._shape = self._shape
        v._dtype = dtype or self._dtype
        v._allocator = self._allocator
        v.data = self.data
        v._strides = self._strides
        v._flags = self._flags & ~flags.OWNDATA
        v.base = self.base if self.base is not None else self
        return v

    # TODO(beam2d): Implement these
    # def getfield(self, dtype, offset=0):
    # def setflags(self, write=None, align=None, uic=None):

    def fill(self, value):
        elementwise.copy(value, self, dtype=self.dtype)

    # -------------------------------------------------------------------------
    # Shape manipulation
    # -------------------------------------------------------------------------
    def reshape(self, *newshape):
        # TODO(beam2d): Support ordering option
        return reshape(self, newshape)

    # TODO(beam2d0: Implement it
    # def resize(self, new_shape, refcheck=True):

    def transpose(self, *axes):
        return transpose(self, axes)

    def swapaxes(self, axis1, axis2):
        return swapaxes(self, axis1, axis2)

    def flatten(self, allocator=None):
        # TODO(beam2d): Support ordering option
        if self.flags.c_contiguous:
            newarray = self.copy()
        else:
            newarray = empty_like(self, allocator=allocator)
            elementwise.copy(self, newarray)

        newarray._shape = self.size,
        newarray._strides = self.itemsize,
        self._flags |= flags.C_CONTIGUOUS | flags.F_CONTIGUOUS
        return newarray

    def ravel(self):
        # TODO(beam2d): Support ordering option
        return ravel(self)

    def squeeze(self, axis=None):
        return squeeze(self, axis)

    # -------------------------------------------------------------------------
    # Item selection and manipulation
    # -------------------------------------------------------------------------
    def take(self, indices, axis=None, out=None, allocator=None):
        return take(self, indices, axis, out, allocator)

    # TODO(beam2d): Implement these
    # def put(self, indices, values, mode='raise'):
    # def repeat(self, repeats, axis=None, allocator=None):
    # def choose(self, choices, out=None, mode='raise', allocator=None):
    # def sort(self, axis=-1, kind='quicksort', order=None):
    # def argsort(self, axis=-1, kind='quicksort', order=None, allocator=None):
    # def partition(self, kth, axis=-1, kind='introselect', order=None):
    # def argpartition(self, kth, axis=-1, kind='introselect', order=None,
    #                  allocator=None):
    # def searchsorted(self, v, side='left', sorter=None, allocator=None):
    # def nonzero(self, allocator=None):
    # def compress(self, condition, axis=None, out=None, allocator=None):

    def diagonal(self, offset=0, axis1=0, axis2=1):
        return diagonal(self, offset, axis1, axis2)

    # -------------------------------------------------------------------------
    # Calculation
    # -------------------------------------------------------------------------
    def max(self, axis=None, out=None, dtype=None, keepdims=False,
            allocator=None):
        return amax(self, axis, out, keepdims, dtype, allocator)

    def argmax(self, axis=None, out=None, dtype=None, keepdims=False,
               allocator=None):
        return argmax(self, axis, dtype, out, keepdims, allocator)

    def min(self, axis=None, out=None, dtype=None, keepdims=False,
            allocator=None):
        return amin(self, axis, out, keepdims, dtype, allocator)

    def argmin(self, axis=None, out=None, dtype=None, keepdims=False,
               allocator=None):
        return argmin(self, axis, dtype, out, keepdims, allocator)

    # TODO(beam2d): Implement it
    # def ptp(self, axis=None, out=None, allocator=None):

    def clip(self, a_min, a_max, out=None, allocator=None):
        return clip(self, a_min, a_max, out=out, allocator=allocator)

    # TODO(beam2d): Implement it
    # def round(self, decimals=0, out=None, allocator=None):

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None,
              allocator=None):
        return trace(self, offset, axis1, axis2, dtype, out, allocator)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False,
            allocator=None):
        return sum(self, axis, dtype, out, keepdims, allocator)

    # TODO(beam2d): Implement it
    # def cumsum(self, axis=None, dtype=None, out=None, allocator=None):

    def mean(self, axis=None, dtype=None, out=None, keepdims=False,
             allocator=None):
        return mean(self, axis, dtype, out, keepdims, allocator)

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False,
            allocator=None):
        return var(self, axis, dtype, out, ddof, keepdims, allocator)

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False,
            allocator=None):
        return std(self, axis, dtype, out, ddof, keepdims, allocator)

    def prod(self, axis=None, dtype=None, out=None, keepdims=None,
             allocator=None):
        return prod(self, axis, dtype, out, keepdims, allocator)

    # TODO(beam2d): Implement these
    # def cumprod(self, axis=None, dtype=None, out=None, allocator=None):
    # def all(self, axis=None, out=None, allocator=None):
    # def any(self, axis=None, out=None, allocator=None):

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
        return self != 0

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
        # TODO(okuta): Support `__array_priority__`
        if not isinstance(other, ndarray):
            return add(self, other)
        x = other.squeeze()
        y = self.squeeze()
        if internal.can_axpy(x, y):
            ret = y.copy()
            internal.axpy(1, x, ret)
            return ret.reshape(self.shape)
        else:
            return add(self, other)

    def __sub__(self, other):
        # TODO(okuta): Support `__array_priority__`
        if not isinstance(other, ndarray):
            return subtract(self, other)
        x = other.squeeze()
        y = self.squeeze()
        if internal.can_axpy(x, y):
            ret = y.copy()
            internal.axpy(-1, x, ret)
            return ret.reshape(self.shape)
        else:
            return subtract(self, other)

    def __mul__(self, other):
        # TODO(okuta): Support `__array_priority__`
        return multiply(self, other)

    def __div__(self, other):
        # TODO(beam2d): Support `__array_priority__`
        return divide(self, other)

    def __truediv__(self, other):
        # TODO(okuta): Support `__array_priority__`
        return true_divide(self, other)

    def __floordiv__(self, other):
        # TODO(beam2d): Support `__array_priority__`
        return floor_divide(self, other)

    def __mod__(self, other):
        # TODO(beam2d): Support `__array_priority__`
        return remainder(self, other)

    def __divmod__(self, other):
        # TODO(beam2d): Support `__array_priority__`
        return elementwise._ivmod(self, other)

    def __pow__(self, other, modulo=None):
        # TODO(okuta): Support `__array_priority__`
        # Note that we ignore the modulo argument as well as NumPy.
        return power(self, other)

    def __lshift__(self, other):
        # TODO(beam2d): Support `__array_priority__`
        return left_shift(self, other)

    def __rshift__(self, other):
        # TODO(beam2d): Support `__array_priority__`
        return right_shift(self, other)

    def __and__(self, other):
        # TODO(beam2d): Support `__array_priority__`
        return bitwise_and(self, other)

    def __or__(self, other):
        # TODO(beam2d): Support `__array_priority__`
        return bitwise_or(self, other)

    def __xor__(self, other):
        # TODO(beam2d): Support `__array_priority__`
        return bitwise_xor(self, other)

    # Arithmetic __r{op}__ (CuPy specific):

    def __radd__(self, other):
        if not isinstance(other, ndarray):
            return add(self, other)
        x = self.squeeze()
        y = other.squeeze()
        if internal.can_axpy(x, y):
            ret = y.copy()
            internal.axpy(1, x, ret)
            return ret.reshape(self.shape)
        else:
            return add(self, other)

    def __rsub__(self, other):
        if not isinstance(other, ndarray):
            return subtract(other, self)
        x = self.squeeze()
        y = other.squeeze()
        if internal.can_axpy(x, y):
            ret = y.copy()
            internal.axpy(-1, x, ret)
            return ret.reshape(self.shape)
        else:
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
        if not isinstance(other, ndarray):
            return add(self, other, self)
        x = other.squeeze()
        y = self.squeeze()
        if internal.can_axpy(x, y):
            internal.axpy(1, x, y)
            return self
        else:
            return add(self, other, self)

    def __isub__(self, other):
        if not isinstance(other, ndarray):
            return subtract(self, other, self)
        x = other.squeeze()
        y = self.squeeze()
        if internal.can_axpy(x, y):
            internal.axpy(-1, x, y)
            return self
        else:
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

    # cupy.ndarray uses __getstate__ instead of __reduce__ for pickling
    def __getstate__(self):
        return (self.get(),)

    def __setstate__(self, arr):
        arr = arr[0]
        self._shape = arr.shape
        self._dtype = arr.dtype
        self._allocator = cuda.alloc
        self.data = self._allocator(self.nbytes)
        self._strides = arr.strides
        self.base = None
        self._flags = flags.OWNDATA
        self._update_contiguity()

        self.set(arr)

    # Basic customization:

    # cupy.ndarray does not define __new__

    def __array__(self, dtype=None):
        if dtype is None or self.dtype == dtype:
            return self
        else:
            return self.astype(dtype)

    # TODO(beam2d): Impleent it
    # def __array_wrap__(self, obj):

    # Container customization:

    def __len__(self):
        if not self.shape:
            raise TypeError('len() of unsized object')
        return self.shape[0]

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
        v.data = self.data + offset
        v._update_contiguity()

        return v

    def __setitem__(self, slices, value):
        v = self[slices]
        if isinstance(value, ndarray):
            y, x = broadcast_arrays(v, value)
            if y._shape == x._shape and y._strides == x._strides:
                if int(y.data) == int(x.data):
                    return  # Skip since x and y are the same array
                elif y.flags.c_contiguous:
                    y.data.copy_from(x.data, x.nbytes)
                    return
            elementwise.copy(x, y)
        else:
            v.fill(value)

    # TODO(beam2d): Implement these
    # def __getslice__(self, i, j):
    # def __setslice__(self, i, j, y):
    # def __contains__(self, y):

    # Conversion:

    def __int__(self):
        return int(self.get())

    def __long__(self):
        return long(self.get())

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
    def dot(self, b, out=None, allocator=None):
        return dot(self, b, out, allocator)

    # -------------------------------------------------------------------------
    # Cupy specific attributes and methods
    # -------------------------------------------------------------------------
    @property
    def allocator(self):
        return self._allocator

    @property
    def _fptr(self):
        if self.dtype.type == numpy.float64:
            return ctypes.cast(self.data.ptr, ctypes.POINTER(ctypes.c_double))
        else:
            return ctypes.cast(self.data.ptr, ctypes.POINTER(ctypes.c_float))

    def get(self, stream=None):
        a_gpu = ascontiguousarray(self)
        a_cpu = numpy.empty(self.shape, dtype=self.dtype)
        ptr = internal.get_ndarray_ptr(a_cpu)
        if stream is None:
            a_gpu.data.copy_to_host(ptr, a_gpu.nbytes)
        else:
            a_gpu.data.copy_to_host_async(ptr, a_gpu.nbytes, stream)
        return a_cpu

    def set(self, arr, stream=None):
        if not isinstance(arr, numpy.ndarray):
            raise TypeError('Only numpy.ndarray can be set to cupy.ndarray')
        if self.dtype != arr.dtype:
            raise TypeError('{} array cannot be set to {} array'.format(
                arr.dtype, self.dtype))
        if self.shape != arr.shape:
            raise ValueError('Shape mismatch')
        if not self.flags.c_contiguous:
            raise RuntimeError('Cannot set to non-contiguous array')

        arr = numpy.ascontiguousarray(arr)
        ptr = internal.get_ndarray_ptr(arr)
        if stream is None:
            self.data.copy_from_host(ptr, self.nbytes)
        else:
            self.data.copy_from_host_async(ptr, self.nbytes, stream)

    def reduced_view(self, dtype=None):
        view = self.view(dtype=dtype)
        shape, strides = internal.get_reduced_dims_from_array(self)
        view._shape = shape
        view._strides = strides
        view._update_f_contiguity()
        return view

    def _update_c_contiguity(self):
        self._flags &= ~flags.C_CONTIGUOUS
        if internal.get_c_contiguity(self.shape, self.strides, self.itemsize):
            self._flags |= flags.C_CONTIGUOUS

    def _update_f_contiguity(self):
        self._flags &= ~flags.F_CONTIGUOUS
        if internal.get_c_contiguity(tuple(reversed(self.shape)),
                                     tuple(reversed(self.strides)),
                                     self.itemsize):
            self._flags |= flags.F_CONTIGUOUS

    def _update_contiguity(self):
        self._update_c_contiguity()
        self._update_f_contiguity()

# -----------------------------------------------------------------------------
# Add comparison of `__array_priority__` to ndarray binary operator
# -----------------------------------------------------------------------------

# TODO(okuta): Fix this


def _wrap_operation(obj, op):
    op_name = '__{}__'.format(op)
    rop_name = '__r{}__'.format(op)
    raw_op = getattr(obj, op_name)

    def new_op(self, other):
        rop = getattr(other, rop_name, None)
        if rop is None:
            return raw_op(self, other)
        self_pri = getattr(self,  '__array_priority__', 0.0)
        other_pri = getattr(other, '__array_priority__', 0.0)
        if self_pri >= other_pri:
            return raw_op(self, other)
        return rop(self)
    setattr(obj, op_name, new_op)


for op in ('add', 'sub', 'mul', 'div', 'pow', 'truediv', 'floordiv',
           'mod', 'divmod', 'lshift', 'rshift', 'and', 'or', 'xor'):
    _wrap_operation(ndarray, op)

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
Broadcast = manipulation.dims.Broadcast
broadcast = manipulation.dims.broadcast
broadcast_arrays = manipulation.dims.broadcast_arrays
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
    if isinstance(a, ndarray):
        return a.get(stream=stream)
    else:
        return numpy.asarray(a)
