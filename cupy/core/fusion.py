import functools
import six
from six.moves import builtins
import string
import threading
import warnings

import numpy

from cupy.core import core
from cupy import creation
from cupy import logic
from cupy import math
from cupy import sorting
from cupy import statistics


_thread_local = threading.local()

_kind_score = {
    'b': 0,
    'u': 1,
    'i': 1,
    'f': 2,
    'c': 3,
}

_dtype_to_ctype = {
    numpy.dtype('float64'): 'double',
    numpy.dtype('float32'): 'float',
    numpy.dtype('float16'): 'float16',
    numpy.dtype('complex128'): 'complex<double>',
    numpy.dtype('complex64'): 'complex<float>',
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

_dtype_list = [numpy.dtype(_) for _ in '?bhilqBHILQefdFD']


class Submodule(object):
    """Submodule class.

    Ufunc or elementwise kernel with types.

    Attributes:
       name (str): The name of submodule
       in_params (list of tuples of dtype and str):
         The tuple of dtype and name of input parameters.
       out_params (list of tuples of dtype and str):
         The tuple of dtype and name of output parameters.
       op (str): The operation code.
       preamble (str): The preamble code.
       type (list of dtypes): The list of types of the parameters.
    """

    def __init__(self, ufunc, in_params, out_params, op):
        self.name = ufunc.name
        self.in_params = in_params
        self.out_params = out_params
        self.op = op
        self.preamble = ufunc._preamble
        self.types = [ty for ty, _ in self.in_params + self.out_params]

    def __repr__(self):
        return "<Submodule {}>".format(self.name)

    def fcall(self, args):
        return self.name + '(' + ', '.join(args) + ');\n'

    def key(self):
        return (self.name, tuple(self.types))

    def code(self):
        params = ', '.join('%s &%s' % (_dtype_to_ctype[t], pname)
                           for t, pname in self.in_params + self.out_params)
        typedef = ''.join('typedef %s %s_type;\n' % (_dtype_to_ctype[t], pname)
                          for t, pname in self.in_params + self.out_params)
        module_code = string.Template('''
        __device__ void ${name}(${parameters}) {
          ${typedef}
          ${operation};
        }
        ''').substitute(
            name=self.name,
            parameters=params,
            operation=self.op,
            typedef=typedef)
        return module_code + '\n'


class _FusionVarCUDA(object):

    """_FusionVarCUDA class.

    Local variable in CUDA program.

    Attributes:
        index (int): The name of the variable.
        ty (dtype): The type of the variable.
        const (any of types): The constant value (or None)
    """

    def __init__(self, index, ty, const=None):
        self.index = index
        self.ty = ty
        self.const = const

    def __repr__(self):
        return "v{}".format(self.index)

    def declaration(self):
        c = self.const
        val = numpy.asscalar(c) if hasattr(c, 'dtype') else c
        ctype = _dtype_to_ctype[self.ty]

        if self.const is None:
            return "{} v{};\n".format(ctype, self.index)

        if isinstance(val, bool):
            init = '= %s' % str(c).lower()
        elif isinstance(val, complex):
            init = '(%s, %s)' % (c.real, c.imag)
        elif isinstance(val, six.integer_types + (float,)):
            init = '= %s' % str(c)
        else:
            raise TypeError('Invalid constant type: {}'.format(type(c)))
        return 'const %s v%d %s;\n' % (ctype, self.index, init)

    def declaration_param(self):
        return "{} v{}".format(self.ty, self.index)


class FusionOp(object):

    """FusionOp class.

    Function call with arguments in CUDA program.

    Attributes:
        index (int): The index of this operation.
        func (submodule): The submodules called in this operation.
        args (list of _FusionVarCUDA): The arguments.
        types (list of dtype): The types of parameters.
    """

    def __init__(self, index, func, args):
        self.index = index
        self.func = func
        self.args = args
        self.types = func.types

    def __repr__(self):
        return "<FusionOp #{}, {} types=[{}]>".format(
            self.index, self.func.name, ', '.join(self.types))

    def declaration_args(self):
        return ' '.join('{} v{}_{};'.format(_dtype_to_ctype[t], self.index, j)
                        for j, t in enumerate(self.types)) + '\n'

    def code(self):
        args_sub = ["v{}_{}".format(self.index, i)
                    for i in six.moves.range(len(self.args))]
        code = "// op  # {}\n".format(self.index)
        code += ''.join("{} = v{};\n".format(s, v.index)
                        for v, s in zip(self.args, args_sub))
        code += self.func.fcall(args_sub)
        code += ''.join("v{} = {};\n".format(v.index, s)
                        for v, s in zip(self.args, args_sub)
                        if v.const is None)
        return code


class _FusionHistory(object):

    """_FusionHistory class.

    History of operation exectuted in the target function of fusion.

    Attributes:
        op_list (list of FusionOp): The list of operations.
        param_list (list of _FusionVarCUDA): The list of parameters.
        local_list (list of _FusionVarCUDA): The list of local variables.
        submodules (dictionary from str to submodule): submodules.
    """

    def __init__(self):
        self.op_list = []
        self.preamble_set = set()
        self.param_list = []
        self.local_list = []
        self.reduce_op = None
        self.reduce_identity = None
        self.reduce_kwargs = None
        self.postmap_op_list = []
        self.premap_ret = None
        self.postmap_param_list = []
        self.postmap_local_list = []
        self.submodules = {}
        self.count = 0

    def __repr__(self):
        return "<_FusionMem, op_list={}, var_list={}>".format(
            self.op_list, self.var_list)

    def fresh_index(self):
        res = self.count
        self.count += 1
        return res

    def fresh_param(self, *args, **kwargs):
        index = self.fresh_index()
        var = _FusionVarCUDA(index, *args, **kwargs)
        self.param_list.append(var)
        return var

    def fresh_local(self, *args, **kwargs):
        index = self.fresh_index()
        var = _FusionVarCUDA(index, *args, **kwargs)
        self.local_list.append(var)
        return var

    def fresh_postmap_param(self, *args, **kwargs):
        index = self.fresh_index()
        var = _FusionVarCUDA(index, *args, **kwargs)
        self.postmap_param_list.append(var)
        return var

    def fresh_postmap_local(self, *args, **kwargs):
        index = self.fresh_index()
        var = _FusionVarCUDA(index, *args, **kwargs)
        self.postmap_local_list.append(var)
        return var

    def set_op(self, *args, **kwargs):
        op = FusionOp(len(self.op_list), *args, **kwargs)
        subm = op.func
        self.submodules[subm.key()] = subm
        self.op_list.append(op)
        self.preamble_set.add(subm.preamble)
        return op

    def set_postmap_op(self, *args, **kwargs):
        op = FusionOp(len(self.postmap_op_list), *args, **kwargs)
        subm = op.func
        self.submodules[subm.key()] = subm
        self.postmap_op_list.append(op)
        self.preamble_set.add(subm.preamble)
        return op

    def get_submodules(self):
        return '\n'.join([_.code() for _ in self.submodules.values()])


class FusionVarPython(object):

    """FusionVarPython class.

    The values of variables in target function of fusion.

    Attributes:
        dtype (dtype): The data type.
    """
    #   shape (tuple of ints): !! not supported !!

    def __init__(self, var, history):
        self._var = var
        self.dtype = var.ty
        self._history = history

    def __repr__(self):
        return "<FusionVarPython, dtype=%s>" % self.dtype

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

    def __setitem__(self, slices, value):
        if slices is Ellipsis or (isinstance(slices, slice) and
                                  slices == slice(None)):
            copy(value, self)
        else:
            raise ValueError('The fusion supports `[...]` or `[:]`.')

    def copy(self):
        return copy(self)


class ReducedVarPython(FusionVarPython):
    """ReducedVarPython class.

    The value of variables after the reduction phase.

    Attributes:
        dtype (dtype): The data type.
    """
    #   shape (tuple of ints): !! not supported !!


def _normalize_arg(arg, mem, is_postmap):
    """This converts `arg` to _FusionVarCUDA data.

    Args:
       arg (FusionVarPython or a primitive type)
       mem (_FusionHistory)
       is_postmap (bool)
    Return value: _FusionVarCUDA
    """
    arg_type = type(arg)
    if is_postmap:
        if arg_type is ReducedVarPython:
            return arg._var
        elif arg_type is FusionVarPython:
            raise Exception("Pre-map after reducing")
    else:
        if arg_type is FusionVarPython:
            return arg._var
        elif arg_type is ReducedVarPython:
            raise Exception("Pre-map after reducing")
    is_scalar = arg_type in six.integer_types + (float, bool, complex)
    is_ndarray = hasattr(arg, 'dtype') and arg.dtype in _dtype_list
    if is_scalar or is_ndarray:
        if is_postmap:
            return mem.fresh_postmap_local(numpy.dtype(arg_type), const=arg)
        else:
            return mem.fresh_local(numpy.dtype(arg_type), const=arg)
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

    def get_history(args):
        for elem in args:
            if isinstance(elem, (FusionVarPython, ReducedVarPython)):
                return elem._history
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

    def func(*args, **kwargs):
        history = get_history(args)
        is_postmap = history.reduce_op is not None
        var_list = [_normalize_arg(_, history, is_postmap) for _ in args]
        if is_postmap:
            var_python, fresh = (ReducedVarPython, history.fresh_postmap_local)
        else:
            var_python, fresh = (FusionVarPython, history.fresh_local)
        if 'out' in kwargs:
            var = _normalize_arg(kwargs.pop('out'), history, is_postmap)
            var_list.append(var)
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)
        assert nin <= len(var_list) <= nin + nout
        in_vars = var_list[:nin]
        out_vars = var_list[nin:]
        can_cast = can_cast1 if _should_use_min_scalar(in_vars) else can_cast2
        for ty_ins, ty_outs, op in ufunc._ops:
            ty_ins = [numpy.dtype(_) for _ in ty_ins]
            ty_outs = [numpy.dtype(_) for _ in ty_outs]
            if can_cast(in_vars, ty_ins):
                ret = []
                for i in six.moves.range(nout):
                    if i >= len(out_vars):
                        v = fresh(ty_outs[i])
                        out_vars.append(v)
                        ret.append(var_python(v, history))
                    elif numpy.can_cast(ty_outs[i], out_vars[i].ty,
                                        "same_kind"):
                        v = out_vars[i]
                        ret.append(var_python(v, history))
                    else:
                        raise TypeError(
                            'output (typecode \'{}\') could not be coerced '
                            'to provided output parameter (typecode \'{}\') '
                            'according to the casting rule '
                            '"same_kind"'.format(
                                ty_outs[i].char, out_vars[i].ty.char))
                in_params = [(ty_ins[i], 'in{}'.format(i))
                             for i, t in enumerate(in_vars)]
                out_params = [(ty_outs[i], 'out{}'.format(i))
                              for i, t in enumerate(out_vars)]
                subm = Submodule(ufunc, in_params, out_params, op)
                if is_postmap:
                    history.set_postmap_op(subm, in_vars + out_vars)
                else:
                    history.set_op(subm, in_vars + out_vars)
                return ret[0] if len(ret) == 1 else tuple(ret)
        raise TypeError('Invalid type cast in \'{}\': {} -> {}'.format(
            ufunc.name,
            [_.ty for _ in in_vars],
            [_.ty for _ in out_vars]))
    return func


def _convert_from_elementwise(elem):
    raise NotImplemented


def _premap_code(in_params, return_var, operation):
    module_code = string.Template('''
    __device__ ${return_type} _pre_map(${in_params}) {
    ${operation};
    return ${return_var};
    }
    ''').substitute(
        return_type=_dtype_to_ctype[return_var.ty],
        in_params=', '.join('{} v{}'.format(_dtype_to_ctype[v.ty], v.index)
                            for v in in_params),
        operation=operation,
        return_var=return_var)
    return module_code


def _postmap_code(in_param, out_params, operation):
    in_type = _dtype_to_ctype[in_param.ty]
    module_code = string.Template('''
    __device__ void _post_map(${in_type} in, ${out_params}) {
    ${in_param} = in;
    ${operation};
    }
    ''').substitute(
        in_type=in_type,
        in_param="{} v{}".format(in_type, in_param.index),
        out_params=', '.join('{} &v{}'.format(_dtype_to_ctype[v.ty], v.index)
                             for v in out_params),
        operation=operation)
    return module_code


def _postfix_code(data_type, fixed_type, operation):
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


def _get_fusion_from_types(func, in_types, name):
    history = _FusionHistory()
    in_params = [history.fresh_param(t) for t in in_types]
    in_pvars = [FusionVarPython(_, history) for _ in in_params]
    out_pvars = func(*in_pvars)
    out_pvars = list(out_pvars) if type(out_pvars) == tuple else [out_pvars]
    out_pvars = [_ for _ in out_pvars if _ is not None]
    is_postmap = history.reduce_op is not None
    out_cvars = [_normalize_arg(_, history, is_postmap) for _ in out_pvars]

    out_types = [_.dtype for _ in out_pvars]
    out_params = [history.fresh_param(t) for t in out_types]

    in_params_code = ', '.join(var.declaration_param() for var in in_params)
    out_params_code = ', '.join(var.declaration_param() for var in out_params)

    operation = '// {} operations\n'.format(len(history.op_list))
    operation += ''.join(v.declaration() for v in history.local_list)
    operation += ''.join(op.declaration_args() for op in history.op_list)
    operation += ''.join(op.code() for op in history.op_list)
    submodules = ''.join(history.preamble_set)

    if history.reduce_op is None:
        operation += ' '.join("{} = {};".format(t, s)
                              for s, t in zip(out_cvars, out_params))
        submodules += history.get_submodules()
        kernel = core.ElementwiseKernel(
            in_params_code, out_params_code, operation,
            preamble=submodules,
            name=name)
        return (kernel, {})
    else:
        reduce_code = history.reduce_op[2][1]
        rtype = history.reduce_op[2][3]
        post_type = 'type_in0_raw' if rtype is None else rtype
        reduce_type = numpy.dtype(history.reduce_op[1][0])
        ctype = _dtype_to_ctype[reduce_type]

        postmap = '// {} operations\n'.format(len(history.postmap_op_list))
        postmap += ''.join(v.declaration() for v in history.postmap_local_list)
        postmap += ''.join(op.declaration_args()
                           for op in history.postmap_op_list)
        postmap += ''.join(op.code() for op in history.postmap_op_list)
        postmap += ' '.join("{} = {};".format(t, s)
                            for s, t in zip(out_cvars, out_params))

        submodules += history.get_submodules()
        submodules += _premap_code(in_params, history.premap_ret, operation)
        submodules += "typedef {} type_in0_raw;\n".format(ctype)
        submodules += "typedef {} type_out0_raw;\n".format(ctype)
        submodules += _postfix_code(post_type,
                                    reduce_type, history.reduce_op[2][2])
        submodules += _postmap_code(history.postmap_param_list[0],
                                    out_params, postmap)

        kernel = core.ReductionKernel(
            in_params_code,
            out_params_code,
            '_pre_map({})'.format(', '.join([repr(_) for _ in in_params])),
            reduce_code,
            '_post_map(_post_fix(a), {})'.format(
                ', '.join([repr(_) for _ in out_params])),
            history.reduce_identity,
            name=name,
            reduce_type=post_type,
            preamble=submodules)
        return (kernel, history.reduce_kwargs)


class Fusion(object):

    """Function class.

    This class can be get by using `fuse` function and
    works like `ElementwiseKernel` or `ReductionKernel`.

    Attributes:
        func (function): The function before fusing.
        name (str): The name of the function.
    """

    def __init__(self, func, name=None):
        self.func = func
        self.name = name or func.__name__
        self._memo = {}

    def __repr__(self):
        return "<Fusion '%s'>" % self.name

    def __call__(self, *args, **kwargs):
        _thread_local.in_fusion = True
        try:
            return self._call(*args, **kwargs)
        finally:
            _thread_local.in_fusion = False

    def compile(self, *args, **kwargs):
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
                self._memo[key] = _get_fusion_from_types(
                    self.func, types, self.name)
            return self._memo[key]
        else:
            if builtins.any(type(_) is core.ndarray for _ in args):
                types = '.'.join(repr(type(_)) for _ in args)
                message = "Can't fuse \n %s(%s)" % (self.name, types)
                warnings.warn(message)
            else:
                return (self.func, {})

    def _call(self, *args, **kwargs):
        func, kw = self.compile(*args, **kwargs)
        kwargs = dict(six.itertools.chain(kwargs.items(), kw.items()))
        return func(*args, **kwargs)


def fuse(*args, **kwargs):
    """Function fusing decorator.

    This decorator can be used to define an elementwise or reduction kernel
    more easily than `ElementwiseKernel` class or `ReductionKernel` class.

    This decorator makes `Fusion` class from the given function.

    Args:
        kernel_name (str): Name of the fused kernel function.
            If omitted, the name of the decorated function is used.

    .. note::
       This API is currently experimental and the interface may be changed in
       the future version.

    """

    def wrapper(f, kernel_name=None):
        return Fusion(f, kernel_name)

    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return functools.update_wrapper(wrapper(args[0]), args[0])
    else:
        return lambda f: functools.update_wrapper(
            wrapper(f, *args, **kwargs), f)


class ufunc(core.ufunc):

    def __init__(self, fusion_op, cupy_op, numpy_op):
        self.name = fusion_op.name
        self.__name__ = self.name
        self.nin = fusion_op.nin
        self.nout = fusion_op.nout
        self.nargs = fusion_op.nargs
        self._ops = fusion_op._ops
        self._preamble = fusion_op._preamble
        self.__doc__ = cupy_op.__doc__
        self._params = fusion_op._params
        self._routine_cache = fusion_op._routine_cache

        self._fusion_op = fusion_op
        self._cupy_op = cupy_op
        self._numpy_op = numpy_op
        self._preamble = fusion_op._preamble

    def __repr__(self):
        return repr(self._cupy_op)

    def __call__(self, *args, **kwargs):
        in_fusion = getattr(_thread_local, 'in_fusion', False)
        if in_fusion:
            if builtins.any(isinstance(_, (FusionVarPython, ReducedVarPython))
                            for _ in args):
                return _convert(self._fusion_op)(*args, **kwargs)
            elif builtins.any(isinstance(_, numpy.ndarray) for _ in args):
                return self._numpy_op(*args, **kwargs)

        return self._cupy_op(*args, **kwargs)

    __doc__ = core.ufunc.__doc__
    __call__.__doc__ = core.ufunc.__call__.__doc__


def _create_ufunc(cupy_ufunc, numpy_ufunc):
    return ufunc(cupy_ufunc, cupy_ufunc, numpy_ufunc)


where = ufunc(sorting.search._where_ufunc,
              sorting.search.where, numpy.where)

clip = ufunc(core._clip, math.misc.clip, numpy.clip)

copy = ufunc(core.elementwise_copy,
             creation.from_data.copy, numpy.copy)

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
angle = _create_ufunc(math.arithmetic.angle, numpy.angle)
conj = _create_ufunc(math.arithmetic.conj, numpy.conj)
real = _create_ufunc(math.arithmetic.real, numpy.real)
imag = _create_ufunc(math.arithmetic.imag, numpy.imag)
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

    def __init__(self, cupy_op, numpy_op, raw):
        self._cupy_op = cupy_op
        self._numpy_op = numpy_op
        self._raw = raw
        self.__doc__ = cupy_op.__doc__
        self.identity = raw.identity
        self._preamble = raw._preamble

    def __call__(self, *args, **kwargs):
        arg = args[0]
        if isinstance(arg, FusionVarPython):
            if len(args) != 1:
                raise Exception("Can't reduce a tuple")
            dtype = arg.dtype
            history = arg._history
            for op in self._raw._ops:
                if numpy.can_cast(dtype.type, op[0][0]):
                    return_type = numpy.dtype(op[1][0])
                    history.premap_ret = _normalize_arg(arg, history, False)
                    return_var = history.fresh_postmap_param(return_type)
                    history.reduce_op = op
                    history.reduce_identity = self.identity
                    history.reduce_kwargs = kwargs
                    history.preamble_set.add(self._preamble)
                    return ReducedVarPython(return_var, history)
            raise TypeError("Type is mismatched. {}(...), {}".format(
                self._raw._ops.name, dtype.type))
        if builtins.any(type(_) == numpy.ndarray for _ in args):
            return self._numpy_op(*args, **kwargs)
        else:
            return self._cupy_op(*args, **kwargs)


all = reduction(logic.truth.all, numpy.all, core._all)
any = reduction(logic.truth.any, numpy.any, core._any)
sum = reduction(math.sumprod.sum, numpy.sum, core._sum_auto_dtype)
prod = reduction(math.sumprod.prod, numpy.prod, core._prod_auto_dtype)
amax = reduction(statistics.order.amax, numpy.amax, core._amax)
amin = reduction(statistics.order.amin, numpy.amin, core._amin)

if hasattr(numpy, "divmod"):
    divmod = _create_ufunc(core.divmod, numpy.divmod)
else:
    divmod = core.divmod
