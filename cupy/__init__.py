from __future__ import division
import sys

import numpy
import six


from cupy import binary
from cupy import core
from cupy import creation
from cupy import indexing
from cupy import io
from cupy import linalg
from cupy import logic
from cupy import manipulation
from cupy import math
import cupy.random
from cupy import sorting
from cupy import statistics
from cupy import testing  # NOQA
from cupy import util

random = cupy.random

ndarray = core.ndarray

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

ufunc = core.ufunc

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
broadcast_to = manipulation.dims.broadcast_to
expand_dims = manipulation.dims.expand_dims
squeeze = manipulation.dims.squeeze

column_stack = manipulation.join.column_stack
concatenate = manipulation.join.concatenate
dstack = manipulation.join.dstack
hstack = manipulation.join.hstack
vstack = manipulation.join.vstack

asfortranarray = manipulation.kind.asfortranarray

array_split = manipulation.split.array_split
dsplit = manipulation.split.dsplit
hsplit = manipulation.split.hsplit
split = manipulation.split.split
vsplit = manipulation.split.vsplit

tile = manipulation.tiling.tile
repeat = manipulation.tiling.repeat

roll = manipulation.rearrange.roll

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

all = logic.truth.all
any = logic.truth.any

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
abs = math.misc.absolute
sign = math.misc.sign
maximum = math.misc.maximum
minimum = math.misc.minimum
fmax = math.misc.fmax
fmin = math.misc.fmin

# -----------------------------------------------------------------------------
# Sorting, searching, and counting
# -----------------------------------------------------------------------------
count_nonzero = sorting.count.count_nonzero

argmax = sorting.search.argmax
argmin = sorting.search.argmin
where = sorting.search.where

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
amin = statistics.order.amin
min = statistics.order.amin
amax = statistics.order.amax
max = statistics.order.amax

mean = statistics.meanvar.mean
var = statistics.meanvar.var
std = statistics.meanvar.std

bincount = statistics.histogram.bincount


# CuPy specific functions
def asnumpy(a, stream=None):
    """Returns an array on the host memory from an arbitrary source array.

    Args:
        a: Arbitrary object that can be converted to :class:`numpy.ndarray`.
        stream (cupy.cuda.Stream): CUDA stream object. If it is specified, then
            the device-to-host copy runs asynchronously. Otherwise, the copy is
            synchronous. Note that if ``a`` is not a :class:`cupy.ndarray`
            object, then this argument has no effect.

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

       >>> def softplus(x):
       ...     xp = cupy.get_array_module(x)
       ...     return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))

    """
    if six.moves.builtins.any(isinstance(arg, ndarray) for arg in args):
        return _cupy
    else:
        return numpy


clear_memo = util.clear_memo
memoize = util.memoize

ElementwiseKernel = core.ElementwiseKernel
ReductionKernel = core.ReductionKernel
