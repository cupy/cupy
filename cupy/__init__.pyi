from typing import Literal as _L

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
abs: _ufunc._UFunc_Nin1_Nout1[_L['cupy_absolute'], _L[16], None]
absolute: _ufunc._UFunc_Nin1_Nout1[_L['cupy_absolute'], _L[16], None]
add: _ufunc._UFunc_Nin2_Nout1[_L['cupy_add'], _L[16], None]
arccos: _ufunc._UFunc_Nin1_Nout1[_L['cupy_arccos'], _L[5], None]
arccosh: _ufunc._UFunc_Nin1_Nout1[_L['cupy_arccosh'], _L[5], None]
arcsin: _ufunc._UFunc_Nin1_Nout1[_L['cupy_arcsin'], _L[5], None]
arcsinh: _ufunc._UFunc_Nin1_Nout1[_L['cupy_arcsinh'], _L[5], None]
arctan: _ufunc._UFunc_Nin1_Nout1[_L['cupy_arctan'], _L[5], None]
arctan2: _ufunc._UFunc_Nin2_Nout1[_L['cupy_arctan2'], _L[5], None]
arctanh: _ufunc._UFunc_Nin1_Nout1[_L['cupy_arctanh'], _L[5], None]
bitwise_and: _ufunc._UFunc_Nin2_Nout1[_L['cupy_bitwise_and'], _L[11], None]
bitwise_not: _ufunc._UFunc_Nin1_Nout1[_L['cupy_invert'], _L[11], None]
bitwise_or: _ufunc._UFunc_Nin2_Nout1[_L['cupy_bitwise_or'], _L[11], None]
bitwise_xor: _ufunc._UFunc_Nin2_Nout1[_L['cupy_bitwise_xor'], _L[11], None]
bool8 = _numpy.bool8
bool_ = _numpy.bool_
broadcast_shapes = _numpy.broadcast_shapes
byte = _numpy.byte
cbrt: _ufunc._UFunc_Nin1_Nout1[_L['cupy_cbrt'], _L[3], None]
cdouble = _numpy.cdouble
ceil: _ufunc._UFunc_Nin1_Nout1[_L['cupy_ceil'], _L[3], None]
cfloat = _numpy.cfloat
complex128 = _numpy.complex128
complex64 = _numpy.complex64
complex_ = _numpy.complex_
complexfloating = _numpy.complexfloating
conj: _ufunc._UFunc_Nin1_Nout1[_L['cupy_conjugate'], _L[15], None]
conjugate: _ufunc._UFunc_Nin1_Nout1[_L['cupy_conjugate'], _L[15], None]
copysign: _ufunc._UFunc_Nin2_Nout1[_L['cupy_copysign'], _L[5], None]
cos: _ufunc._UFunc_Nin1_Nout1[_L['cupy_cos'], _L[5], None]
cosh: _ufunc._UFunc_Nin1_Nout1[_L['cupy_cosh'], _L[5], None]
csingle = _numpy.csingle
deg2rad: _ufunc._UFunc_Nin1_Nout1[_L['cupy_deg2rad'], _L[3], None]
degrees: _ufunc._UFunc_Nin1_Nout1[_L['cupy_rad2deg'], _L[3], None]
disable_experimental_feature_warning: bool
disp = _numpy.disp
divide: _ufunc._UFunc_Nin2_Nout1[_L['cupy_true_divide'], _L[15], None]
divmod: _ufunc._UFunc_Nin2_Nout2[_L['cupy_divmod'], _L[13], None]
double = _numpy.double
dtype = _numpy.dtype
e: float
equal: _ufunc._UFunc_Nin2_Nout1[_L['cupy_equal'], _L[16], None]
euler_gamma: float
exp: _ufunc._UFunc_Nin1_Nout1[_L['cupy_exp'], _L[5], None]
exp2: _ufunc._UFunc_Nin1_Nout1[_L['cupy_exp2'], _L[5], None]
expm1: _ufunc._UFunc_Nin1_Nout1[_L['cupy_expm1'], _L[5], None]
fabs: _ufunc._UFunc_Nin1_Nout1[_L['cupy_fabs'], _L[3], None]
find_common_type = _numpy.find_common_type
finfo = _numpy.finfo
fix: _ufunc._UFunc_Nin1_Nout1[_L['cupy_fix'], _L[3], None]
float16 = _numpy.float16
float32 = _numpy.float32
float64 = _numpy.float64
float_ = _numpy.float_
float_power: _ufunc._UFunc_Nin2_Nout1[_L['cupy_float_power'], _L[3], None]
floating = _numpy.floating
floor: _ufunc._UFunc_Nin1_Nout1[_L['cupy_floor'], _L[3], None]
floor_divide: _ufunc._UFunc_Nin2_Nout1[_L['cupy_floor_divide'], _L[13], None]
fmax: _ufunc._UFunc_Nin2_Nout1[_L['cupy_fmax'], _L[16], None]
fmin: _ufunc._UFunc_Nin2_Nout1[_L['cupy_fmin'], _L[16], None]
fmod: _ufunc._UFunc_Nin2_Nout1[_L['cupy_fmod'], _L[13], None]
format_parser = _numpy.format_parser
frexp: _ufunc._UFunc_Nin1_Nout2[_L['cupy_frexp'], _L[3], None]
gcd: _ufunc._UFunc_Nin2_Nout1[_L['cupy_gcd'], _L[11], None]
generic = _numpy.generic
get_array_wrap = _numpy.get_array_wrap
get_printoptions = _numpy.get_printoptions
greater: _ufunc._UFunc_Nin2_Nout1[_L['cupy_greater'], _L[16], None]
greater_equal: _ufunc._UFunc_Nin2_Nout1[_L['cupy_greater_equal'], _L[16], None]
half = _numpy.half
heaviside: _ufunc._UFunc_Nin2_Nout1[_L['cupy_heaviside'], _L[3], None]
hypot: _ufunc._UFunc_Nin2_Nout1[_L['cupy_hypot'], _L[5], None]
i0: _ufunc._UFunc_Nin1_Nout1[_L['cupy_i0'], _L[5], None]
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
invert: _ufunc._UFunc_Nin1_Nout1[_L['cupy_invert'], _L[11], None]
isfinite: _ufunc._UFunc_Nin1_Nout1[_L['cupy_isfinite'], _L[5], None]
isinf: _ufunc._UFunc_Nin1_Nout1[_L['cupy_isinf'], _L[5], None]
isnan: _ufunc._UFunc_Nin1_Nout1[_L['cupy_isnan'], _L[5], None]
issctype = _numpy.issctype
issubclass_ = _numpy.issubclass_
issubdtype = _numpy.issubdtype
issubsctype = _numpy.issubsctype
iterable = _numpy.iterable
lcm: _ufunc._UFunc_Nin2_Nout1[_L['cupy_lcm'], _L[11], None]
ldexp: _ufunc._UFunc_Nin2_Nout1[_L['cupy_ldexp'], _L[6], None]
left_shift: _ufunc._UFunc_Nin2_Nout1[_L['cupy_left_shift'], _L[10], None]
less: _ufunc._UFunc_Nin2_Nout1[_L['cupy_less'], _L[16], None]
less_equal: _ufunc._UFunc_Nin2_Nout1[_L['cupy_less_equal'], _L[16], None]
log: _ufunc._UFunc_Nin1_Nout1[_L['cupy_log'], _L[5], None]
log10: _ufunc._UFunc_Nin1_Nout1[_L['cupy_log10'], _L[5], None]
log1p: _ufunc._UFunc_Nin1_Nout1[_L['cupy_log1p'], _L[5], None]
log2: _ufunc._UFunc_Nin1_Nout1[_L['cupy_log2'], _L[5], None]
logaddexp: _ufunc._UFunc_Nin2_Nout1[_L['cupy_logaddexp'], _L[3], None]
logaddexp2: _ufunc._UFunc_Nin2_Nout1[_L['cupy_logaddexp2'], _L[3], None]
logical_and: _ufunc._UFunc_Nin2_Nout1[_L['cupy_logical_and'], _L[14], None]
logical_not: _ufunc._UFunc_Nin1_Nout1[_L['cupy_logical_not'], _L[14], None]
logical_or: _ufunc._UFunc_Nin2_Nout1[_L['cupy_logical_or'], _L[14], None]
logical_xor: _ufunc._UFunc_Nin2_Nout1[_L['cupy_logical_xor'], _L[14], None]
longfloat = _numpy.longfloat
longlong = _numpy.longlong
maximum: _ufunc._UFunc_Nin2_Nout1[_L['cupy_maximum'], _L[16], None]
minimum: _ufunc._UFunc_Nin2_Nout1[_L['cupy_minimum'], _L[16], None]
mintypecode = _numpy.mintypecode
mod: _ufunc._UFunc_Nin2_Nout1[_L['cupy_remainder'], _L[13], None]
modf: _ufunc._UFunc_Nin1_Nout2[_L['cupy_modf'], _L[3], None]
multiply: _ufunc._UFunc_Nin2_Nout1[_L['cupy_multiply'], _L[16], None]
nan: float
ndarray = _core.ndarray
ndindex = _numpy.ndindex
negative: _ufunc._UFunc_Nin1_Nout1[_L['cupy_negative'], _L[16], None]
nextafter: _ufunc._UFunc_Nin2_Nout1[_L['cupy_nextafter'], _L[5], None]
not_equal: _ufunc._UFunc_Nin2_Nout1[_L['cupy_not_equal'], _L[16], None]
number = _numpy.number
obj2sctype = _numpy.obj2sctype
pi: float
positive: _ufunc._UFunc_Nin1_Nout1[_L['cupy_positive'], _L[16], None]
power: _ufunc._UFunc_Nin2_Nout1[_L['cupy_power'], _L[16], None]
printoptions = _numpy.printoptions
promote_types = _numpy.promote_types
rad2deg: _ufunc._UFunc_Nin1_Nout1[_L['cupy_rad2deg'], _L[3], None]
radians: _ufunc._UFunc_Nin1_Nout1[_L['cupy_deg2rad'], _L[3], None]
reciprocal: _ufunc._UFunc_Nin1_Nout1[_L['cupy_reciprocal'], _L[15], None]
remainder: _ufunc._UFunc_Nin2_Nout1[_L['cupy_remainder'], _L[13], None]
right_shift: _ufunc._UFunc_Nin2_Nout1[_L['cupy_right_shift'], _L[10], None]
rint: _ufunc._UFunc_Nin1_Nout1[_L['cupy_rint'], _L[5], None]
s_ = _numpy.s_
safe_eval = _numpy.safe_eval
sctype2char = _numpy.sctype2char
set_printoptions = _numpy.set_printoptions
set_string_function = _numpy.set_string_function
short = _numpy.short
sign: _ufunc._UFunc_Nin1_Nout1[_L['cupy_sign'], _L[15], None]
signbit: _ufunc._UFunc_Nin1_Nout1[_L['cupy_signbit'], _L[3], None]
signedinteger = _numpy.signedinteger
sin: _ufunc._UFunc_Nin1_Nout1[_L['cupy_sin'], _L[5], None]
sinc: _ufunc._UFunc_Nin1_Nout1[_L['cupy_sinc'], _L[5], None]
single = _numpy.single
singlecomplex = _numpy.singlecomplex
sinh: _ufunc._UFunc_Nin1_Nout1[_L['cupy_sinh'], _L[5], None]
sqrt: _ufunc._UFunc_Nin1_Nout1[_L['cupy_sqrt'], _L[5], None]
square: _ufunc._UFunc_Nin1_Nout1[_L['cupy_square'], _L[15], None]
subtract: _ufunc._UFunc_Nin2_Nout1[_L['cupy_subtract'], _L[16], None]
tan: _ufunc._UFunc_Nin1_Nout1[_L['cupy_tan'], _L[5], None]
tanh: _ufunc._UFunc_Nin1_Nout1[_L['cupy_tanh'], _L[5], None]
true_divide: _ufunc._UFunc_Nin2_Nout1[_L['cupy_true_divide'], _L[15], None]
trunc: _ufunc._UFunc_Nin1_Nout1[_L['cupy_trunc'], _L[3], None]
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
