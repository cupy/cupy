import numpy as _numpy

from cupy.typing import _ufunc
from cupy import _core


AxisError = _numpy.AxisError
ComplexWarning = _numpy.ComplexWarning
DataSource = _numpy.DataSource
Inf: float
Infinity: float
ModuleDeprecationWarning = _numpy.ModuleDeprecationWarning
NAN: float
NINF: float
NZERO: float
NaN: float
PINF: float
PZERO: float
RankWarning = _numpy.RankWarning
TooHardError = _numpy.TooHardError
VisibleDeprecationWarning = _numpy.VisibleDeprecationWarning
abs: _ufunc._UFunc_Nin1_Nout1
absolute: _ufunc._UFunc_Nin1_Nout1
add: _ufunc._UFunc_Nin2_Nout1
arccos: _ufunc._UFunc_Nin1_Nout1
arccosh: _ufunc._UFunc_Nin1_Nout1
arcsin: _ufunc._UFunc_Nin1_Nout1
arcsinh: _ufunc._UFunc_Nin1_Nout1
arctan: _ufunc._UFunc_Nin1_Nout1
arctan2: _ufunc._UFunc_Nin2_Nout1
arctanh: _ufunc._UFunc_Nin1_Nout1
bitwise_and: _ufunc._UFunc_Nin2_Nout1
bitwise_not: _ufunc._UFunc_Nin1_Nout1
bitwise_or: _ufunc._UFunc_Nin2_Nout1
bitwise_xor: _ufunc._UFunc_Nin2_Nout1
bool8 = _numpy.bool8
bool_ = _numpy.bool_
broadcast_shapes = _numpy.broadcast_shapes
byte = _numpy.byte
cbrt: _ufunc._UFunc_Nin1_Nout1
cdouble = _numpy.cdouble
ceil: _ufunc._UFunc_Nin1_Nout1
cfloat = _numpy.cfloat
complex128 = _numpy.complex128
complex64 = _numpy.complex64
complex_ = _numpy.complex_
complexfloating = _numpy.complexfloating
conj: _ufunc._UFunc_Nin1_Nout1
conjugate: _ufunc._UFunc_Nin1_Nout1
copysign: _ufunc._UFunc_Nin2_Nout1
cos: _ufunc._UFunc_Nin1_Nout1
cosh: _ufunc._UFunc_Nin1_Nout1
csingle = _numpy.csingle
deg2rad: _ufunc._UFunc_Nin1_Nout1
degrees: _ufunc._UFunc_Nin1_Nout1
disable_experimental_feature_warning: bool
disp = _numpy.disp
divide: _ufunc._UFunc_Nin2_Nout1
divmod: _ufunc._UFunc_Nin2_Nout2
double = _numpy.double
dtype = _numpy.dtype
e: float
equal: _ufunc._UFunc_Nin2_Nout1
euler_gamma: float
exp: _ufunc._UFunc_Nin1_Nout1
exp2: _ufunc._UFunc_Nin1_Nout1
expm1: _ufunc._UFunc_Nin1_Nout1
fabs: _ufunc._UFunc_Nin1_Nout1
find_common_type = _numpy.find_common_type
finfo = _numpy.finfo
fix: _ufunc._UFunc_Nin1_Nout1
float16 = _numpy.float16
float32 = _numpy.float32
float64 = _numpy.float64
float_ = _numpy.float_
float_power: _ufunc._UFunc_Nin2_Nout1
floating = _numpy.floating
floor: _ufunc._UFunc_Nin1_Nout1
floor_divide: _ufunc._UFunc_Nin2_Nout1
fmax: _ufunc._UFunc_Nin2_Nout1
fmin: _ufunc._UFunc_Nin2_Nout1
fmod: _ufunc._UFunc_Nin2_Nout1
format_parser = _numpy.format_parser
frexp: _ufunc._UFunc_Nin1_Nout2
gcd: _ufunc._UFunc_Nin2_Nout1
generic = _numpy.generic
get_array_wrap = _numpy.get_array_wrap
get_printoptions = _numpy.get_printoptions
greater: _ufunc._UFunc_Nin2_Nout1
greater_equal: _ufunc._UFunc_Nin2_Nout1
half = _numpy.half
heaviside: _ufunc._UFunc_Nin2_Nout1
hypot: _ufunc._UFunc_Nin2_Nout1
i0: _ufunc._UFunc_Nin1_Nout1
iinfo = _numpy.iinfo
index_exp = _numpy.index_exp
inexact = _numpy.inexact
inf: float
infty: float
int0 = _numpy.int0
int16 = _numpy.int16
int32 = _numpy.int32
int64 = _numpy.int64
int8 = _numpy.int8
int_ = _numpy.int_
intc = _numpy.intc
integer = _numpy.integer
intp = _numpy.intp
invert: _ufunc._UFunc_Nin1_Nout1
isfinite: _ufunc._UFunc_Nin1_Nout1
isinf: _ufunc._UFunc_Nin1_Nout1
isnan: _ufunc._UFunc_Nin1_Nout1
issctype = _numpy.issctype
issubclass_ = _numpy.issubclass_
issubdtype = _numpy.issubdtype
issubsctype = _numpy.issubsctype
iterable = _numpy.iterable
lcm: _ufunc._UFunc_Nin2_Nout1
ldexp: _ufunc._UFunc_Nin2_Nout1
left_shift: _ufunc._UFunc_Nin2_Nout1
less: _ufunc._UFunc_Nin2_Nout1
less_equal: _ufunc._UFunc_Nin2_Nout1
log: _ufunc._UFunc_Nin1_Nout1
log10: _ufunc._UFunc_Nin1_Nout1
log1p: _ufunc._UFunc_Nin1_Nout1
log2: _ufunc._UFunc_Nin1_Nout1
logaddexp: _ufunc._UFunc_Nin2_Nout1
logaddexp2: _ufunc._UFunc_Nin2_Nout1
logical_and: _ufunc._UFunc_Nin2_Nout1
logical_not: _ufunc._UFunc_Nin1_Nout1
logical_or: _ufunc._UFunc_Nin2_Nout1
logical_xor: _ufunc._UFunc_Nin2_Nout1
longfloat = _numpy.longfloat
longlong = _numpy.longlong
maximum: _ufunc._UFunc_Nin2_Nout1
minimum: _ufunc._UFunc_Nin2_Nout1
mintypecode = _numpy.mintypecode
mod: _ufunc._UFunc_Nin2_Nout1
modf: _ufunc._UFunc_Nin1_Nout2
multiply: _ufunc._UFunc_Nin2_Nout1
nan: float
ndarray = _core.ndarray
ndindex = _numpy.ndindex
negative: _ufunc._UFunc_Nin1_Nout1
nextafter: _ufunc._UFunc_Nin2_Nout1
not_equal: _ufunc._UFunc_Nin2_Nout1
number = _numpy.number
obj2sctype = _numpy.obj2sctype
pi: float
positive: _ufunc._UFunc_Nin1_Nout1
power: _ufunc._UFunc_Nin2_Nout1
printoptions = _numpy.printoptions
promote_types = _numpy.promote_types
rad2deg: _ufunc._UFunc_Nin1_Nout1
radians: _ufunc._UFunc_Nin1_Nout1
reciprocal: _ufunc._UFunc_Nin1_Nout1
remainder: _ufunc._UFunc_Nin2_Nout1
right_shift: _ufunc._UFunc_Nin2_Nout1
rint: _ufunc._UFunc_Nin1_Nout1
s_ = _numpy.s_
safe_eval = _numpy.safe_eval
sctype2char = _numpy.sctype2char
set_printoptions = _numpy.set_printoptions
set_string_function = _numpy.set_string_function
short = _numpy.short
sign: _ufunc._UFunc_Nin1_Nout1
signbit: _ufunc._UFunc_Nin1_Nout1
signedinteger = _numpy.signedinteger
sin: _ufunc._UFunc_Nin1_Nout1
sinc: _ufunc._UFunc_Nin1_Nout1
single = _numpy.single
singlecomplex = _numpy.singlecomplex
sinh: _ufunc._UFunc_Nin1_Nout1
sqrt: _ufunc._UFunc_Nin1_Nout1
square: _ufunc._UFunc_Nin1_Nout1
subtract: _ufunc._UFunc_Nin2_Nout1
tan: _ufunc._UFunc_Nin1_Nout1
tanh: _ufunc._UFunc_Nin1_Nout1
true_divide: _ufunc._UFunc_Nin2_Nout1
trunc: _ufunc._UFunc_Nin1_Nout1
typename = _numpy.typename
ubyte = _numpy.ubyte
ufunc = _core.ufunc
uint = _numpy.uint
uint0 = _numpy.uint0
uint16 = _numpy.uint16
uint32 = _numpy.uint32
uint64 = _numpy.uint64
uint8 = _numpy.uint8
uintc = _numpy.uintc
uintp = _numpy.uintp
ulonglong = _numpy.ulonglong
unsignedinteger = _numpy.unsignedinteger
ushort = _numpy.ushort
