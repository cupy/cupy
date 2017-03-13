import inspect
import six
from six.moves import builtins
import string
import warnings

import numpy

from cupy.core import core
from cupy import creation
from cupy import logic
from cupy import math
from cupy import sorting
from cupy import statistics
from cupy import util


class FusionOp(object):

    def __init__(self, name, operation, param_names,
                 nin, nout, in_vars, out_vars, types, num):
        self.name = name
        self.operation = operation
        self.param_names = param_names
        self.nin = nin
        self.nout = nout
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.types = types
        self.num = num

    def __repr__(self):
        return "<FusionOp, name={}, types=[{}]>".format(
            self.name, ', '.join(_.name for _ in self.types))

    def build_kernel_name(self):
        return self.name + '_' + '_'.join([
            'IN_' + '_'.join(build_kernel_name(_) for _ in self.in_vars),
            'OUT_' + '_'.join(build_kernel_name(_) for _ in self.out_vars),
        ])


class _FusionVar(object):

    def __init__(self, num, ty, const=None):
        self.num = num
        self.ty = ty
        self.const = const

    def __repr__(self):
        return "<_FusionVar, num={}, ty={}, const={}>".format(
            self.num, self.ty, self.const)

    def build_kernel_name(self):
        return self.ty.name + '_at' + str(self.num)


class _FusionMem(object):

    def __init__(self, var_list):
        self.op_list = []
        self.var_list = var_list[:]

    def __repr__(self):
        return "<_FusionMem, op_list={}, var_list={}>".format(
            self.op_list,
            self.var_list)

    def get_fresh(self, ty, **kwargs):
        n = len(self.var_list)
        ret = _FusionVar(n, ty, **kwargs)
        self.var_list.append(ret)
        return ret

    def set_op(self, name, operation, param_names,
               nin, nout, in_vars, out_vars, types):
        num = len(self.op_list)
        op = FusionOp(name, operation, param_names,
                      nin, nout, in_vars, out_vars, types, num)
        self.op_list.append(op)


class _FusionRef(object):

    def __init__(self, var, mem):
        self._var = var
        self.dtype = var.ty
        self._mem = mem

    def __repr__(self):
        return "<_FusionRef, dtype=%s>" % self.dtype

    def build_kernel_name(self):
        return build_kernel_name(self._var)

    def __neg__(self):
        return negative(self)

    def __add__(self, other):
        return add(self, other)

    def __iadd__(self, other):
        return add(self, other, self)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return subtract(self, other)

    def __isub__(self, other):
        return subtract(self, other, self)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other):
        return multiply(self, other)

    def __imul__(self, other):
        return multiply(self, other, self)

    def __rmul__(self, other):
        return multiply(other, self)

    def __div__(self, other):
        return divide(self, other)

    def __idiv__(self, other):
        return divide(self, other, self)

    def __rdiv__(self, other):
        return divide(other, self)

    def __truediv__(self, other):
        return true_divide(self, other)

    def __itruediv__(self, other):
        return true_divide(self, other, self)

    def __rtruediv__(self, other):
        return true_divide(other, self)

    def __floordiv__(self, other):
        return floor_divide(self, other)

    def __ifloordiv__(self, other):
        return floor_divide(self, other, self)

    def __rfloordiv__(self, other):
        return floor_divide(other, self)

    def __mod__(self, other):
        return remainder(self, other)

    def __imod__(self, other):
        return remainder(self, other, self)

    def __rmod__(self, other):
        return remainder(other, self)

    def __pow__(x, y):
        return power(x, y)

    def __ipow__(self, other):
        return power(self, other, self)

    def __lshift__(self, other):
        return left_shift(self, other)

    def __ilshift__(self, other):
        return left_shift(self, other, self)

    def __rlshift__(self, other):
        return left_shift(other, self)

    def __rshift__(self, other):
        return right_shift(self, other)

    def __irshift__(self, other):
        return right_shift(self, other, self)

    def __rrshift__(self, other):
        return right_shift(other, self)

    def __and__(self, other):
        return bitwise_and(self, other)

    def __iand__(self, other):
        return bitwise_and(self, other, self)

    def __rand__(self, other):
        return bitwise_and(other, self)

    def __or__(self, other):
        return bitwise_or(self, other)

    def __ior__(self, other):
        return bitwise_or(self, other, self)

    def __ror__(self, other):
        return bitwise_or(other, self)

    def __xor__(self, other):
        return bitwise_xor(self, other)

    def __ixor__(self, other):
        return bitwise_xor(self, other, self)

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

    def copy(self):
        return copy(self)


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

_dtype_list = [numpy.dtype(_) for _ in '?bhilqBHILQefd']


def _const_to_str(val):
    return str(val).lower() if type(val) is bool else str(val)


def _normalize_arg(arg, mem):
    arg_type = type(arg)
    if arg_type is _FusionRef:
        return arg._var
    is_scalar = arg_type in [int, float, bool]
    is_ndarray = hasattr(arg, 'dtype') and arg.dtype in _dtype_list
    if is_scalar or is_ndarray:
        return mem.get_fresh(numpy.dtype(arg_type), const=arg)
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

    def get_mem(args):
        for i in args:
            if type(i) == _FusionRef:
                return i._mem
        raise Exception('number of ndarray arguments must be more than 0')

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

    def res(*args, **kwargs):
        mem = get_mem(args)
        var_list = [_normalize_arg(_, mem) for _ in args]
        if 'out' in kwargs:
            var_list.append(_normalize_arg.pop('out'))
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)
        assert nin <= len(var_list) and len(var_list) <= nin + nout
        in_vars = var_list[:nin]
        out_vars = var_list[nin:]
        can_cast = can_cast1 if _should_use_min_scalar(in_vars) else can_cast2
        for ty_ins, ty_outs, op in ufunc._ops:
            ty_ins = [numpy.dtype(_) for _ in ty_ins]
            ty_outs = [numpy.dtype(_) for _ in ty_outs]
            if can_cast(in_vars, ty_ins):
                param_names = (['in%d' % i for i in six.moves.range(nin)] +
                               ['out%d' % i for i in six.moves.range(nout)])
                ret = []
                for i in six.moves.range(nout):
                    if i >= len(out_vars):
                        v = mem.get_fresh(ty_outs[i])
                        out_vars.append(v)
                        ret.append(_FusionRef(v, mem))
                    elif numpy.can_cast(ty_outs[i], out_vars[i].ty,
                                        "same_kind"):
                        v = out_vars[i]
                        ret.append(_FusionRef(v, mem))
                    else:
                        raise TypeError("Cannot cast from %s to %s"
                                        % (ty_outs[i], out_vars[i].ty)
                                        + " with casting rule 'same_kind'")
                mem.set_op(ufunc.name, op, param_names, nin, nout,
                           in_vars, out_vars, ty_ins + ty_outs)
                return ret[0] if len(ret) == 1 else tuple(ret)
        raise TypeError('Invalid type cast in \'{}\': {} -> {}'.format(
            ufunc.name,
            [_.ty for _ in in_vars],
            [_.ty for _ in out_vars]))
    return res


def _convert_from_elementwise(elem):
    raise Exception('Not Impletmented')


def _gather_submodules(ops):
    return {(op.name, tuple(op.types)): op for op in ops}


def _get_params(var_list):
    return ['%s v%d' % (var.ty, var.num) for var in var_list]


def _get_out_params(var_list):
    return ['%s ret%d' % (var.ty, i) for i, var in enumerate(var_list)]


def _get_declaration_from_var(var):
    if var.const is None:
        return '%s v%d;\n' % (_dtype_to_ctype[var.ty], var.num)
    else:
        return 'const %s v%d = %s;\n' % (
            _dtype_to_ctype[var.ty],
            var.num,
            _const_to_str(var.const))


def _get_declaration_from_op(op):
    return ''.join('%s v%d_%d;\n' % (_dtype_to_ctype[t], op.num, j)
                   for j, t in enumerate(op.types))


def _get_operation_code(op):
    code = ''.join('v%d_%d = v%d;\n' % (op.num, i, v.num)
                   for i, v in enumerate(op.in_vars))
    params = ['v%d_%d' % (op.num, i)
              for i in six.moves.range(op.nin + op.nout)]
    code += op.name + '(' + ', '.join(params) + ');\n'
    code += ''.join('v%d = v%d_%d;\n' %
                    (v.num, op.num, i + op.nin)
                    for i, v in enumerate(op.out_vars))
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
    return module_code + '\n'


def _get_pre_code(in_vars, out_vars, operation):
    in_params = ', '.join('%s v%s' % (_dtype_to_ctype[v.ty], v.num)
                          for v in in_vars)
    out_params = ''.join('%s v%s;\n' % (_dtype_to_ctype[v.ty], v.num)
                         for v in out_vars)
    module_code = string.Template('''
    __device__ ${return_type} _pre_map(${in_params}) {
      ${out_params}
      ${operation};
      return ${return_var};
    }
    ''').substitute(
        return_type=_dtype_to_ctype[out_vars[0].ty],
        in_params=in_params,
        out_params=out_params,
        operation=operation,
        return_var='v%d' % out_vars[0].num)
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


def _get_fusion(func, nin, reduce, post_map, identity, input_types, name=None):
    if nin is None:
        nin = len(inspect.getargspec(func).args)
    in_vars = [_FusionVar(i, t) for i, t in enumerate(input_types)]
    mem = _FusionMem(in_vars)
    in_refs = [_FusionRef(_, mem) for _ in in_vars]
    out_refs = func(*in_refs)
    out_refs = list(out_refs) if type(out_refs) == tuple else [out_refs]
    out_refs = filter(lambda i: i is not None, out_refs)
    out_refs = [_FusionRef(_normalize_arg(_, mem), mem) for _ in out_refs]
    out_vars = [_normalize_arg(copy(_), mem) for _ in out_refs]
    nout = len(out_vars)
    op_list = mem.op_list
    tmpvars = mem.var_list[nin:-nout] if nout > 0 else mem.var_list[nin:]

    in_params = ', '.join(_get_params(in_vars))
    out_params = ', '.join(_get_params(out_vars))
    operation = ''.join(_get_declaration_from_var(_) for _ in tmpvars)
    operation += ''.join(_get_declaration_from_op(_) for _ in op_list)
    operation += '\n'.join(_get_operation_code(_) for _ in op_list)

    if name is None:
        name = 'fusion__' + '__'.join(build_kernel_name(_) for _ in op_list)

    if reduce is None:
        if not out_params:
            in_params = ', '.join(_get_params(in_vars[:-1]))
            out_params = ', '.join(_get_params([in_vars[-1]]))
        submodules = _gather_submodules(op_list)
        submodule_code = ''.join(_get_submodule_code(_)
                                 for _ in submodules.values())
        return core.ElementwiseKernel(in_params, out_params,
                                      operation, preamble=submodule_code,
                                      name=name)
    else:
        if nout != 1:
            raise Exception("Wrong number of number of arguments")
        # pre-map
        pre_type = out_vars[0].ty
        pre_code = _get_pre_code(in_vars, out_vars, operation)

        # reduce
        reduce_op = _get_reduce_op(reduce._raw, pre_type)
        reduce_code = reduce_op[2][1]
        reduce_type = numpy.dtype(reduce_op[1][0])
        rtype = reduce_op[2][3]
        post_type = "type_in0_raw" if rtype is None else rtype
        pre_code += "typedef %s type_in0_raw;\n" % _dtype_to_ctype[reduce_type]

        # post-map
        post_in = [_FusionVar(0, reduce_type)]
        mem = _FusionMem(post_in)
        post_in_ref = [_FusionRef(_, mem) for _ in post_in]
        post_out = _normalize_arg(post_map(*post_in_ref), mem)
        if type(post_out) == tuple:
            raise Exception("Can't reduce a tuple")
        post_vars = mem.var_list
        post_ops = mem.op_list
        post_code = ''.join(_get_declaration_from_var(_)
                            for _ in post_vars[1:])
        post_code += ''.join(_get_declaration_from_op(_) for _ in post_ops)
        post_code += '\n'.join(_get_operation_code(_) for _ in post_ops)
        post_code = _get_post_code(post_vars, post_code, post_out)
        post_code += _get_fix_code(post_type, reduce_type, reduce_op[2][2])

        submodules = _gather_submodules(op_list + post_ops)
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

    This class can be get by using `fuse` function and
    works like `ElementwiseKernel` or `ReductionKernel`.

    Attributes:
        func (function): The function before fusing.
        name (str): The name of the function.
        reduce (ufunc): Reduction ufunc.
        post_map (function): Mapping function for reduced values.
    """

    def __init__(self, func, input_num, reduce, post_map):
        self.func = func
        self.name = func.__name__
        self.input_num = input_num
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
        if builtins.any(
                not isinstance(_, (core.ndarray, numpy.ndarray, numpy.generic))
                for _ in args):
            raise TypeError('Invalid argument type for \'{}\': ({})'.format(
                self.name,
                ', '.join(repr(type(_)) for _ in args)))

        def is_cupy_data(a):
            return isinstance(a, (core.ndarray, numpy.generic))
        if builtins.all(is_cupy_data(_) for _ in args):
            types = [_.dtype for _ in args]
            key = tuple(types)
            if key not in self._memo:
                f = _get_fusion(self.func, self.input_num, self.reduce,
                                self.post_map, self.identity, types)
                self._memo[key] = f
            f = self._memo[key]
            if self.reduce is None:
                return f(*args)
            else:
                return f(*args, axis=axis)
        else:
            if builtins.any(type(_) is core.ndarray for _ in args):
                types = '.'.join(repr(type(_)) for _ in args)
                message = "Can't fuse \n %s(%s)" % (self.name, types)
                warnings.warn(message)
            if self.reduce is None:
                return self.func(*args)
            elif axis is None:
                return self.post_map(self.reduce(self.func(*args)))
            else:
                return self.post_map(self.reduce(self.func(*args), axis=axis))


def fuse(input_num=None, reduce=None, post_map=lambda x: x):
    """Function fusing decorator.

    This decorator can be used to define an elementwise or reduction kernel
    more easily than `ElementwiseKernel` class or `ReductionKernel` class.

    This decorator makes `Fusion` class from the given function.

    Args:
        input_num (int): Number of input arguments of the given function.
        reduce (function): The reduce function which is applied after
            pre-mapping step. If not assigned, reduction step is skipped.
        post_map (function): Mapping function for reduced values.
            If not assigned, post_map step is skipped.
    """
    util.experimental('cupy.core.fusion')
    return lambda f: Fusion(f, input_num, reduce, post_map)


def build_kernel_name(entity):
    if isinstance(entity, FusionOp):
        return entity.build_kernel_name()
    elif isinstance(entity, _FusionVar):
        return entity.build_kernel_name()
    elif isinstance(entity, _FusionRef):
        return entity.build_kernel_name()
    else:
        assert False, type(entity)


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
        if builtins.any(type(_) is _FusionRef for _ in args):
            return _convert(self._fusion_op)(*args, **kwargs)
        elif builtins.any(type(_) is numpy.ndarray for _ in args):
            return self._numpy_op(*args, **kwargs)
        else:
            return self._cupy_op(*args, **kwargs)

    __doc__ = core.ufunc.__doc__
    __call__.__doc__ = core.ufunc.__call__.__doc__


def _create_ufunc(cupy_ufunc, numpy_ufunc):
    return ufunc(cupy_ufunc, cupy_ufunc, numpy_ufunc)


_where = ufunc(sorting.search._where_ufunc,
               sorting.search.where, numpy.where)

_clip = ufunc(core._clip, math.misc.clip, numpy.clip)

_elementwise_copy = ufunc(core._elementwise_copy,
                          creation.from_data.copy, numpy.copy)


def where(*args, **kwargs):
    return _where(*args, **kwargs)


def clip(*args, **kwargs):
    return _clip(*args, **kwargs)


def copy(*args, **kwargs):
    return _elementwise_copy(*args, **kwargs)


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
fix = _create_ufunc(math.rounding.fix, numpy.fix)

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
        if builtins.any(type(_) == numpy.ndarray for _ in args):
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
