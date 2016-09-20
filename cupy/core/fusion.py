import __builtin__
import inspect
import six
import string
import warnings

import numpy

from cupy.core import core
from cupy import logic
from cupy import math
from cupy import sorting
from cupy import statistics


class FusionOp(object):

    def __init__(self, name, operation, param_names,
                 nin, nout, in_vars, types):
        self.name = name
        self.operation = operation
        self.param_names = param_names
        self.nin = nin
        self.nout = nout
        self.in_vars = in_vars
        self.out_nums = [None] * self.nout
        self.types = types
        self.num = None


class _FusionVar(object):

    def __init__(self, op, idx, ty, num=None, const=None):
        self.op = op
        self.idx = idx
        self.num = num
        self.const = const
        self.ty = ty


class _FusionRef(object):

    def __init__(self, var):
        self._var = var
        self.dtype = var.ty

    def __repr__(self):
        return "<_FusionRef, dtype=%s>" % self.dtype

    def __neg__(self):
        return negative(self)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __div__(self, other):
        return divide(self, other)

    def __rdiv__(self, other):
        return divide(other, self)

    def __truediv__(self, other):
        return true_divide(self, other)

    def __rtruediv__(self, other):
        return true_divide(other, self)

    def __floordiv__(self, other):
        return floor_divide(self, other)

    def __rfloordiv__(self, other):
        return floor_divide(other, self)

    def __mod__(self, other):
        return remainder(self, other)

    def __rmod__(self, other):
        return remainder(other, self)

    def __lshift__(self, other):
        return left_shift(self, other)

    def __rlshift__(self, other):
        return left_shift(other, self)

    def __rshift__(self, other):
        return right_shift(self, other)

    def __rrshift__(self, other):
        return right_shift(other, self)

    def __and__(self, other):
        return bitwise_and(self, other)

    def __rand__(self, other):
        return bitwise_and(other, self)

    def __or__(self, other):
        return bitwise_or(self, other)

    def __ror__(self, other):
        return bitwise_or(other, self)

    def __xor__(self, other):
        return bitwise_xor(self, other)

    def __rxor__(self, other):
        return bitwise_xor(other, self)

    def __invert__(self):
        return invert(self)

    def __lt__(self, other):
        return less(self, other)

    def __le__(self, other):
        return less_equal(self, other)

    def __eq__(self, other):
        return equal(self, other)

    def __ne__(self, other):
        return not_equal(self, other)

    def __gt__(self, other):
        return greater(self, other)

    def __ge__(self, other):
        return greater_equal(self, other)

    def __nonzero__(self):
        raise Exception("Can't cast to bool")

    def __bool__(self):
        raise Exception("Can't cast to bool")


_kind_score = {
    'b': 0,
    'u': 1,
    'i': 1,
    'f': 2,
}

_dtype_to_ctype = {
    numpy.dtype('float64'): 'double',
    numpy.dtype('float32'): 'float',
    numpy.dtype('float16'): 'float16',
    numpy.dtype('int64'): 'long long',
    numpy.dtype('int32'): 'int',
    numpy.dtype('int16'): 'short',
    numpy.dtype('int8'): 'signed char',
    numpy.dtype('uint64'): 'unsigned long long',
    numpy.dtype('uint32'): 'unsigned int',
    numpy.dtype('uint16'): 'unsigned short',
    numpy.dtype('uint8'): 'unsigned char',
    numpy.dtype('bool'): 'bool',
}

_dtype_list = map(numpy.dtype, '?bhilqBHILQefd')


def _const_to_str(val):
    return str(val).lower() if type(val) is bool else str(val)


def _normalize_arg(arg):
    arg_type = type(arg)
    if arg_type is _FusionRef:
        return arg._var
    is_scalar = arg_type in [int, float, bool]
    is_ndarray = hasattr(arg, 'dtype') and arg.dtype in _dtype_list
    if is_scalar or is_ndarray:
        t = numpy.dtype(arg_type)
        return _FusionVar(None, None, t, const=arg)
    raise Exception('Unsupported type %s' % arg_type)


def _convert(f):
    if type(f) is core.ufunc:
        return _convert_from_ufunc(f)
    if type(f) is core.ElementwiseKernel:
        return _convert_from_elementwise(f)
    raise Exception("Can't convert from %s to FusionOp" % type(f))


def _should_use_min_scalar(in_args):
    max_array_kind = -2
    max_scalar_kind = -1
    for i in in_args:
        kind = _kind_score[i.ty.kind]
        if i.const is None:
            max_array_kind = max(max_array_kind, kind)
        else:
            max_scalar_kind = max(max_scalar_kind, kind)
    return (max_scalar_kind != -1 and
            max_array_kind >= max_scalar_kind)


def _convert_from_ufunc(ufunc):
    nin = ufunc.nin
    nout = ufunc.nout

    def can_cast1(args, ty_ins):
        for i in six.moves.range(nin):
            if args[i].const is None:
                if not numpy.can_cast(args[i].ty, ty_ins[i]):
                    return False
            else:
                if not numpy.can_cast(args[i].const, ty_ins[i]):
                    return False
        return True

    def can_cast2(args, ty_ins):
        for i in six.moves.range(nin):
            if not numpy.can_cast(args[i].ty, ty_ins[i]):
                return False
        return True

    def res(*args):
        assert nin <= len(args) and len(args) <= nin + nout
        vars = map(_normalize_arg, args)
        can_cast = can_cast1 if _should_use_min_scalar(vars) else can_cast2
        for ty_ins, ty_outs, op in ufunc._ops:
            ty_ins = map(numpy.dtype, ty_ins)
            ty_outs = map(numpy.dtype, ty_outs)
            if can_cast(vars, ty_ins):
                param_names = (['in%d' % i for i in six.moves.range(nin)] +
                               ['out%d' % i for i in six.moves.range(nout)])
                op = FusionOp(ufunc.name, op, param_names, nin, nout,
                              vars, ty_ins + ty_outs)
                ret = [_FusionVar(op, i, ty_outs[i])
                       for i in six.moves.range(nout)]
                for i in six.moves.range(len(args) - nin):
                    args[i + nin]._var = ret[i]
                ret = map(_FusionRef, ret)
                return ret[0] if len(ret) == 1 else tuple(ret)
        raise TypeError('Invalid type cast')
    return res


def _convert_from_elementwise(elem):
    raise Exception('Not Impletmented')


class _Counter(object):

    def __init__(self, n):
        self.n = n

    def __call__(self):
        ret = self.n
        self.n = self.n + 1
        return ret


def _get_var_list(trees, res, counter):
    for tree in trees:
        if tree.num is None:
            tree.num = counter()
            res.append(tree)
            if tree.op is not None:
                _get_var_list(tree.op.in_vars, res, counter)
                tree.op.out_nums[tree.idx] = tree.num


def _get_operation_list(trees, res, counter):
    for tree in trees:
        op = tree.op
        if op is not None and op.num is None:
            _get_operation_list(op.in_vars, res, counter)
            op.num = counter()
            res.append(op)


def _gather_submodules(ops):
    return {(op.name, tuple(op.types)): op for op in ops}


def _get_parameters(var):
    return '%s v%d' % (var.ty, var.num)


def _get_declaration_from_var(var):
    if var.const is None:
        return '%s v%d;\n' % (_dtype_to_ctype[var.ty], var.num)
    else:
        return 'const %s v%d = %s;\n' % (
            _dtype_to_ctype[var.ty],
            var.num,
            _const_to_str(var.const))


def _get_declaration_from_op(op):
    return ''.join('%s v%d_%d;\n' % (_dtype_to_ctype[t], op.num, i)
                   for i, t in enumerate(op.types))


def _get_operation_code(op):
    code = ''.join('v%d_%d = v%d;\n' % (op.num, i, v.num)
                   for i, v in enumerate(op.in_vars))
    params = ['v%d_%d' % (op.num, i)
              for i in six.moves.range(op.nin + op.nout)]
    code += op.name + '(' + ', '.join(params) + ');\n'
    code += ''.join('v%d = v%d_%d;\n' %
                    (v, op.num, i + op.nin)
                    for i, v in enumerate(op.out_nums))
    return code


def _get_submodule_code(op):
    parameters = ', '.join('%s &%s' % (_dtype_to_ctype[t], name)
                           for i, (name, t)
                           in enumerate(zip(op.param_names, op.types)))
    typedecl = ''.join(('typedef %s in%d_type;\n' % (_dtype_to_ctype[t], i))
                       for i, t in enumerate(op.types[:op.nin]))
    typedecl += ''.join(('typedef %s out%d_type;\n' % (_dtype_to_ctype[t], i))
                        for i, t in enumerate(op.types[op.nin:]))
    module_code = string.Template('''
    __device__ void ${name}(${parameters}) {
      ${typedecl}
      ${operation};
    }
    ''').substitute(
        name=op.name,
        parameters=parameters,
        operation=op.operation,
        typedecl=typedecl)
    return module_code


def _get_pre_code(var_list, nin, operation):
    params = ', '.join('%s v%s' % (_dtype_to_ctype[v.ty], v.num)
                       for v in var_list[:nin])
    declaration = '%s v%s;\n' % (_dtype_to_ctype[var_list[nin].ty], nin)
    module_code = string.Template('''
    __device__ ${return_type} _pre_map(${params}) {
      ${declaration}
      ${operation};
      return ${return_var};
    }
    ''').substitute(
        return_type=_dtype_to_ctype[var_list[nin].ty],
        params=params,
        declaration=declaration,
        operation=operation,
        return_var='v%d' % nin)
    return module_code


def _get_reduce_op(ops, dtype):
    for i in ops._ops:
        if numpy.can_cast(dtype.type, i[0][0]):
            return i
    raise TypeError("Type is mismatched. %s(...), %s" % (ops.name, dtype.type))


def _get_post_code(post_vars, operation, post_out):
    module_code = string.Template('''
    __device__ ${return_type} _post_map(${arg_type} v0) {
      ${operation};
      return v${return_var};
    }
    ''').substitute(
        arg_type=_dtype_to_ctype[post_vars[0].ty],
        return_type=_dtype_to_ctype[post_vars[post_out.num].ty],
        operation=operation,
        return_var=post_out.num)
    return module_code


def _get_fix_code(data_type, fixed_type, operation):
    module_code = string.Template('''
    __device__ ${fixed_type} _post_fix(${data_type} a) {
      ${fixed_type} out0;
      ${operation};
      return out0;
    }
    ''').substitute(
        data_type=data_type,
        fixed_type=_dtype_to_ctype[fixed_type],
        operation=operation)
    return module_code


def _get_fusion(func, nin, immutable_num,
                reduce, post_map, identity, input_types):
    if nin is None:
        nin = len(inspect.getargspec(func).args)
    if immutable_num is None:
        immutable_num = nin
    assert nin == len(input_types)
    assert immutable_num <= nin
    input_vars = [_FusionVar(None, None, input_types[i], i)
                  for i in six.moves.range(nin)]
    input_refs = map(_FusionRef, input_vars)
    ret_refs = func(*input_refs)
    ret_refs = list(ret_refs) if type(ret_refs) == tuple else [ret_refs]
    ret_vars = map(_normalize_arg, ret_refs)
    ret_types = [var.ty for var in ret_vars]
    if input_types[nin:] != ret_types[:(immutable_num - nin)]:
        raise TypeError('Type is mismatched')

    output_vars = [_FusionVar(None, None, t, i + immutable_num)
                   for i, t in enumerate(ret_types)]
    nout = len(output_vars)
    nargs = immutable_num + nout

    var_list = input_vars[:immutable_num] + output_vars
    operation_list = []
    _get_var_list(ret_vars, var_list, _Counter(nargs))
    _get_operation_list(ret_vars, operation_list, _Counter(0))
    in_params = ', '.join(map(_get_parameters, var_list[:immutable_num]))
    out_params = ', '.join(map(_get_parameters, var_list[immutable_num:nargs]))
    operation = ''.join(map(_get_declaration_from_var, var_list[nargs:]))
    operation += ''.join(map(_get_declaration_from_op,  operation_list))
    operation += '\n'.join(map(_get_operation_code, operation_list))
    operation += ''.join('v%d = v%d;\n' % (i + immutable_num, v.num)
                         for i, v in enumerate(ret_vars))
    if reduce is None:
        submodules = _gather_submodules(operation_list)
        submodule_code = ''.join(
            [_get_submodule_code(submodules[p]) for p in submodules]) + '\n'
        return core.ElementwiseKernel(in_params, out_params,
                                      operation, preamble=submodule_code)
    else:
        if nin != immutable_num or nout != 1:
            raise Exception("Wrong number of number of arguments")
        # pre-map
        pre_type = ret_vars[0].ty
        pre_code = _get_pre_code(var_list, nin, operation)

        # reduce
        reduce_op = _get_reduce_op(reduce._raw, pre_type)
        reduce_code = reduce_op[2][1]
        reduce_type = numpy.dtype(reduce_op[1][0])
        rtype = reduce_op[2][3]
        post_type = "type_in0_raw" if rtype is None else rtype
        pre_code += "typedef %s type_in0_raw;\n" % _dtype_to_ctype[reduce_type]

        # post-map
        post_in = [_FusionVar(None, None, reduce_type, 0)]
        post_out = _normalize_arg(post_map(*map(_FusionRef, post_in)))
        if type(post_out) == tuple:
            raise Exception("Can't reduce a tuple")
        post_vars = post_in
        post_ops = []
        _get_var_list([post_out], post_vars, _Counter(1))
        _get_operation_list([post_out], post_ops, _Counter(0))
        post_code = ''.join(map(_get_declaration_from_var, post_vars[1:]))
        post_code += ''.join(map(_get_declaration_from_op,  post_ops))
        post_code += '\n'.join(map(_get_operation_code, post_ops))
        post_code = _get_post_code(post_vars, post_code, post_out)
        post_code += _get_fix_code(post_type, reduce_type, reduce_op[2][2])

        submodules = _gather_submodules(operation_list + post_ops)
        submodule_code = ''.join(_get_submodule_code(v)
                                 for v in submodules.values())
        submodule_code += reduce._raw._preamble + pre_code + post_code
        operation_args = ['v' + str(i) for i in six.moves.range(nin)]
        operation = '_pre_map(' + ', '.join(operation_args) + ')'
        out_params = '%s res' % post_out.ty
        return core.ReductionKernel(in_params, out_params, operation,
                                    reduce_code,
                                    'res = _post_map(_post_fix(a))',
                                    identity,
                                    reduce_type=post_type,
                                    preamble=submodule_code)


class Fusion(object):

    """Function class.

    This class can be get by using `fuse` and
    works like ElementwiseKernel or ReductionKernel.

    Attributes:
        func (function): Rhe function before fusing.
        name (str): The name of the function.
    """

    def __init__(self, func, input_num, immutable_num, reduce, post_map):
        self.func = func
        self.name = func.__name__
        self.input_num = input_num
        self.immutable_num = immutable_num
        self.reduce = reduce
        self.post_map = post_map
        self.identity = None if reduce is None else self.reduce._raw.identity
        self._memo = {}

    def __repr__(self):
        return "<Fusion '%s'>" % self.name

    def __call__(self, *args, **kwargs):
        axis = kwargs['axis'] if 'axis' in kwargs else None
        if len(args) == 0:
            raise Exception('number of arguments must be more than 0')
        is_cupy_data = lambda a: isinstance(a, (core.ndarray, numpy.generic))
        if __builtin__.all(map(is_cupy_data, args)):
            types = map(lambda x: x.dtype, args)
            key = tuple(types)
            if key not in self._memo:
                f = _get_fusion(self.func, self.input_num,
                                self.immutable_num, self.reduce,
                                self.post_map, self.identity, types)
                self._memo[key] = f
            f = self._memo[key]
            if self.reduce is None:
                return f(*args)
            else:
                return f(*args, axis=axis)
        else:
            if __builtin__.any(map(lambda a: type(a) is core.ndarray, args)):
                types = '.'.join(map(repr, map(type, args)))
                message = "Can't fuse \n %s(%s)" % (self.name, types)
                warnings.warn(message)
            if self.reduce is None:
                return self.func(*args)
            elif axis is None:
                return self.post_map(self.reduce(self.func(*args)))
            else:
                return self.post_map(self.reduce(self.func(*args), axis=axis))


def fuse(input_num=None, immutable_num=None,
         reduce=None, post_map=lambda x: x):
    """Function fusing decorator.

    This decorator can be used to define an elementwise or reduction kernel
    more easily than ElementwiseKernel class.

    This decorator makes Fusion class from an function.

    Args:
        input_num (int): Number of input arguments of the function.
        immutable_num (int): Number of immutable input arguments of
            the function.
        reduce (function): The reduce function which is applied after
            pre-mapping step.
        post_map (function): post-map function.
    """
    return lambda f: Fusion(f, input_num, immutable_num, reduce, post_map)


class ufunc(core.ufunc):

    def __init__(self, fusion_op, cupy_op, numpy_op):
        self.name = fusion_op.name
        self.nin = fusion_op.nin
        self.nout = fusion_op.nout
        self.nargs = fusion_op.nargs
        self._ops = fusion_op._ops
        self._preamble = fusion_op._preamble
        self.__doc__ = fusion_op.__doc__
        self._params = fusion_op._params
        self._routine_cache = fusion_op._routine_cache

        self._fusion_op = fusion_op
        self._cupy_op = cupy_op
        self._numpy_op = numpy_op

    def __repr__(self):
        return repr(self._cupy_op)

    def __call__(self, *args, **kwargs):
        if __builtin__.any(type(i) is _FusionRef for i in args):
            return _convert(self._fusion_op)(*args, **kwargs)
        elif __builtin__.any(type(i) is numpy.ndarray for i in args):
            return self._numpy_op(*args, **kwargs)
        else:
            return self._cupy_op(*args, **kwargs)

    __doc__ = core.ufunc.__doc__
    __call__.__doc__ = core.ufunc.__call__.__doc__


def _create_ufunc(cupy_ufunc, numpy_ufunc):
    return ufunc(cupy_ufunc, cupy_ufunc, numpy_ufunc)


_where = ufunc(sorting.search._where_ufunc,
               sorting.search.where, numpy.where)


def where(*args, **kwargs):
    return _where(*args, **kwargs)


_clip = ufunc(core._clip, math.misc.clip, numpy.clip)


def clip(*args, **kwargs):
    return _clip(*args, **kwargs)


bitwise_and = _create_ufunc(core.bitwise_and, numpy.bitwise_and)
bitwise_or = _create_ufunc(core.bitwise_or, numpy.bitwise_or)
bitwise_xor = _create_ufunc(core.bitwise_xor, numpy.bitwise_xor)
invert = _create_ufunc(core.invert, numpy.invert)
left_shift = _create_ufunc(core.left_shift, numpy.left_shift)
right_shift = _create_ufunc(core.right_shift, numpy.right_shift)

greater = _create_ufunc(core.greater, numpy.greater)
greater_equal = _create_ufunc(core.greater_equal, numpy.greater_equal)
less = _create_ufunc(core.less, numpy.less)
less_equal = _create_ufunc(core.less_equal, numpy.less_equal)
equal = _create_ufunc(core.equal, numpy.equal)
not_equal = _create_ufunc(core.not_equal, numpy.not_equal)

isfinite = _create_ufunc(logic.content.isfinite, numpy.isfinite)
isinf = _create_ufunc(logic.content.isinf, numpy.isinf)
isnan = _create_ufunc(logic.content.isnan, numpy.isnan)

logical_and = _create_ufunc(logic.ops.logical_and, numpy.logical_and)
logical_or = _create_ufunc(logic.ops.logical_or, numpy.logical_or)
logical_not = _create_ufunc(logic.ops.logical_not, numpy.logical_not)
logical_xor = _create_ufunc(logic.ops.logical_xor, numpy.logical_xor)

sin = _create_ufunc(math.trigonometric.sin, numpy.sin)
cos = _create_ufunc(math.trigonometric.cos, numpy.cos)
tan = _create_ufunc(math.trigonometric.tan, numpy.tan)
arcsin = _create_ufunc(math.trigonometric.arcsin, numpy.arcsin)
arccos = _create_ufunc(math.trigonometric.arccos, numpy.arccos)
arctan = _create_ufunc(math.trigonometric.arctan, numpy.arctan)
arctan2 = _create_ufunc(math.trigonometric.arctan2, numpy.arctan2)
hypot = _create_ufunc(math.trigonometric.hypot, numpy.hypot)
deg2rad = _create_ufunc(math.trigonometric.deg2rad, numpy.deg2rad)
rad2deg = _create_ufunc(math.trigonometric.rad2deg, numpy.rad2deg)
degrees = _create_ufunc(math.trigonometric.degrees, numpy.degrees)
radians = _create_ufunc(math.trigonometric.radians, numpy.radians)

sinh = _create_ufunc(math.hyperbolic.sinh, numpy.sinh)
cosh = _create_ufunc(math.hyperbolic.cosh, numpy.cosh)
tanh = _create_ufunc(math.hyperbolic.tanh, numpy.tanh)
arcsinh = _create_ufunc(math.hyperbolic.arcsinh, numpy.arcsinh)
arccosh = _create_ufunc(math.hyperbolic.arccosh, numpy.arccosh)
arctanh = _create_ufunc(math.hyperbolic.arctanh, numpy.arctanh)

rint = _create_ufunc(math.rounding.rint, numpy.rint)
floor = _create_ufunc(math.rounding.floor, numpy.floor)
ceil = _create_ufunc(math.rounding.ceil, numpy.ceil)
trunc = _create_ufunc(math.rounding.trunc, numpy.trunc)

exp = _create_ufunc(math.explog.exp, numpy.exp)
expm1 = _create_ufunc(math.explog.expm1, numpy.expm1)
exp2 = _create_ufunc(math.explog.exp2, numpy.exp2)
log = _create_ufunc(math.explog.log, numpy.log)
log10 = _create_ufunc(math.explog.log10, numpy.log10)
log2 = _create_ufunc(math.explog.log2, numpy.log2)
log1p = _create_ufunc(math.explog.log1p, numpy.log1p)
logaddexp = _create_ufunc(math.explog.logaddexp, numpy.logaddexp)
logaddexp2 = _create_ufunc(math.explog.logaddexp2, numpy.logaddexp2)

signbit = _create_ufunc(math.floating.signbit, numpy.signbit)
copysign = _create_ufunc(math.floating.copysign, numpy.copysign)
ldexp = _create_ufunc(math.floating.ldexp, numpy.ldexp)
frexp = _create_ufunc(math.floating.frexp, numpy.frexp)
nextafter = _create_ufunc(math.floating.nextafter, numpy.nextafter)

add = _create_ufunc(math.arithmetic.add, numpy.add)
reciprocal = _create_ufunc(math.arithmetic.reciprocal, numpy.reciprocal)
negative = _create_ufunc(math.arithmetic.negative, numpy.negative)
multiply = _create_ufunc(math.arithmetic.multiply, numpy.multiply)
divide = _create_ufunc(math.arithmetic.divide, numpy.divide)
power = _create_ufunc(math.arithmetic.power, numpy.power)
subtract = _create_ufunc(math.arithmetic.subtract, numpy.subtract)
true_divide = _create_ufunc(math.arithmetic.true_divide, numpy.true_divide)
floor_divide = _create_ufunc(math.arithmetic.floor_divide, numpy.floor_divide)
fmod = _create_ufunc(math.arithmetic.fmod, numpy.fmod)
mod = _create_ufunc(math.arithmetic.remainder, numpy.mod)
modf = _create_ufunc(math.arithmetic.modf, numpy.modf)
remainder = _create_ufunc(math.arithmetic.remainder, numpy.remainder)

sqrt = _create_ufunc(math.misc.sqrt, numpy.sqrt)
sqrt_fixed = _create_ufunc(math.misc.sqrt_fixed, numpy.sqrt)
square = _create_ufunc(math.misc.square, numpy.square)
absolute = _create_ufunc(math.misc.absolute, numpy.absolute)
abs = _create_ufunc(math.misc.absolute, numpy.abs)
sign = _create_ufunc(math.misc.sign, numpy.sign)
maximum = _create_ufunc(math.misc.maximum, numpy.maximum)
minimum = _create_ufunc(math.misc.minimum, numpy.minimum)
fmax = _create_ufunc(math.misc.fmax, numpy.fmax)
fmin = _create_ufunc(math.misc.fmin, numpy.fmin)


class reduction(object):

    def __init__(self, cupy_op, numpy_op):
        self._cupy_op = cupy_op
        self._numpy_op = numpy_op

    def __call__(self, *args, **kwargs):
        if __builtin__.any(type(i) == numpy.ndarray for i in args):
            return self._numpy_op(*args, **kwargs)
        else:
            return self._cupy_op(*args, **kwargs)


_all = reduction(logic.truth.all, numpy.all)
_any = reduction(logic.truth.any, numpy.any)
_sum = reduction(math.sumprod.sum, numpy.sum)
_prod = reduction(math.sumprod.prod, numpy.prod)
_amax = reduction(statistics.order.amax, numpy.amax)
_amin = reduction(statistics.order.amin, numpy.amin)


def all(*args, **kwargs):
    return _all(*args, **kwargs)


def any(*args, **kwargs):
    return _any(*args, **kwargs)


def sum(*args, **kwargs):
    return _sum(*args, **kwargs)


def prod(*args, **kwargs):
    return _prod(*args, **kwargs)


def amax(*args, **kwargs):
    return _amax(*args, **kwargs)


def amin(*args, **kwargs):
    return _amin(*args, **kwargs)


all._raw = core._all
any._raw = core._any
sum._raw = core._sum
prod._raw = core._prod
amax._raw = core._amax
amin._raw = core._amin
