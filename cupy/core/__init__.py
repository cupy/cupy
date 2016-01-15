from cupy.core import core
from cupy.core import internal

ndarray = core.ndarray

get_size = internal.get_size
complete_slice = internal.complete_slice

ufunc = core.ufunc
create_ufunc = core.create_ufunc
ElementwiseKernel = core.ElementwiseKernel
create_reduction_func = core.create_reduction_func
ReductionKernel = core.ReductionKernel


# =============================================================================
# Routines
# =============================================================================

elementwise_copy = core.elementwise_copy
elementwise_copy_where = core.elementwise_copy_where

divmod = core.divmod


# -----------------------------------------------------------------------------
# Array creation routines
# -----------------------------------------------------------------------------

array = core.array
ascontiguousarray = core.ascontiguousarray


# -----------------------------------------------------------------------------
# Array manipulation routines
# -----------------------------------------------------------------------------

rollaxis = core.rollaxis
broadcast = core.broadcast


# -----------------------------------------------------------------------------
# Binary operations
# -----------------------------------------------------------------------------

bitwise_and = core.bitwise_and
bitwise_or = core.bitwise_or
bitwise_xor = core.bitwise_xor
invert = core.invert
left_shift = core.left_shift
right_shift = core.right_shift


# -----------------------------------------------------------------------------
# Linear algebra
# -----------------------------------------------------------------------------

dot = core.dot
tensordot_core = core.tensordot_core


# -----------------------------------------------------------------------------
# Logic functions
# -----------------------------------------------------------------------------

create_comparison = core.create_comparison

greater = core.greater
greater_equal = core.greater_equal
less = core.less
less_equal = core.less_equal
equal = core.equal
not_equal = core.not_equal


# -----------------------------------------------------------------------------
# Mathematical functions
# -----------------------------------------------------------------------------

add = core.add
negative = core.negative
multiply = core.multiply
divide = core.divide
power = core.power
subtract = core.subtract
true_divide = core.true_divide
floor_divide = core.floor_divide
remainder = core.remainder

sqrt_fixed = core.sqrt_fixed
absolute = core.absolute
