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
_thread_local.history = None

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
    """Ufunc or elementwise kernel with types.

    Attributes:
       name (str): The name of submodule
       in_params (list of tuples of dtype and str):
         The tuple of dtype and name of input parameters.
       out_params (list of tuples of dtype and str):
         The tuple of dtype and name of output parameters.
       op (str): The operation code.
       preamble (str): The preamble code.
       dtypes (list of dtypes): The list of dtypes of the parameters.
    """

    def __init__(self, ufunc, in_params, out_params, op):
        self.name = ufunc.name
        self.in_params = in_params
        self.out_params = out_params
        self.op = op
        self.preamble = ufunc._preamble
        self.dtypes = [dtype for dtype, _ in self.in_params + self.out_params]

    def __repr__(self):
        return '<Submodule {}>'.format(self.name)

    def fcall(self, args):
        return self.name + '(' + ', '.join(args) + ');\n'

    def key(self):
        return (self.name, tuple(self.dtypes))

    def code(self):
        params = ', '.join('{} &{}'.format(_dtype_to_ctype[t], s)
                           for t, s in self.in_params + self.out_params)
        typedef = ''.join('typedef {} {}_type;\n'.format(_dtype_to_ctype[t], s)
                          for t, s in self.in_params + self.out_params)
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

    """Local variable in CUDA program.

    Attributes:
        index (int): The name of the variable.
        dtype (dtype): The dtype of the variable.
        const (any of primitive types): The constant value (or None)
    """

    def __init__(self, index, dtype, const=None):
        self.index = index
        self.dtype = dtype
        self.const = const

    def __repr__(self):
        return 'v{}'.format(self.index)

    def declaration(self):
        c = self.const
        val = numpy.asscalar(c) if hasattr(c, 'dtype') else c
        ctype = _dtype_to_ctype[self.dtype]

        if self.const is None:
            return '{} v{};\n'.format(ctype, self.index)

        if isinstance(val, bool):
            init = '= {}'.format(str(c).lower())
        elif isinstance(val, complex):
            init = '({}, {})'.format(c.real, c.imag)
        elif isinstance(val, six.integer_types + (float,)):
            init = '= {}'.format(c)
        else:
            raise TypeError('Invalid constant type: {}'.format(type(c)))
        return 'const {} v{} {};\n'.format(ctype, self.index, init)

    def declaration_param(self):
        return '{} v{}'.format(self.dtype, self.index)


class FusionOp(object):

    """Function call with arguments in CUDA program.

    Attributes:
        index (int): The index of this operation.
        submodule (submodule): The submodules called in this operation.
        args (list of _FusionVarCUDA): The arguments.
        types (list of dtype): The types of parameters.
    """

    def __init__(self, index, submodule, args):
        self.index = index
        self.submodule = submodule
        self.args = args
        self.dtypes = submodule.dtypes

    def __repr__(self):
        return '<FusionOp #{}, {} types=[{}]>'.format(
            self.index, self.submodule.name, ', '.join(self.dtypes))

    def declaration_args(self):
        return ' '.join('{} v{}_{};'.format(_dtype_to_ctype[t], self.index, j)
                        for j, t in enumerate(self.dtypes)) + '\n'

    def code(self):
        args_sub = ['v{}_{}'.format(self.index, i)
                    for i in six.moves.range(len(self.args))]
        args_list = list(zip(self.args, args_sub))
        code = '// op  # {}\n'.format(self.index)
        code += ''.join('{} = v{};\n'.format(s, v.index) for v, s in args_list)
        code += self.submodule.fcall(args_sub)
        code += ''.join('v{} = {};\n'.format(v.index, s)
                        for v, s in args_list[len(self.submodule.in_params):])
        return code


class FusionVarPython(object):

    """The values of variables in target function of fusion.

    Args:
        var (_FusionVarCUDA)

    Attributes:
        dtype (dtype): The data type.
    """

    def __init__(self, var, is_postmap):
        self._var = var
        self.dtype = var.dtype
        self._is_postmap = is_postmap

    def __repr__(self):
        return '<FusionVarPython, dtype={}>'.format(self.dtype)

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
        raise Exception('Can\'t cast to bool')

    def __bool__(self):
        raise Exception('Can\'t cast to bool')

    def __setitem__(self, slices, value):
        if slices is Ellipsis or (isinstance(slices, slice) and
                                  slices == slice(None)):
            copy(value, self)
        else:
            raise ValueError('The fusion supports `[...]` or `[:]`.')

    def copy(self):
        return copy(self)


class _FusionHistory(object):

    """History of operation exectuted in the target function of fusion.

    Attributes:
        preamble_set (set of str): The preambles of submodules.
        submodules (dict from str to submodule): The submodules.
        count (int): The number of variables in the fused function.

        op_list (list of FusionOp): The map operations.
        param_list (list of _FusionVarCUDA): The parameters
        local_list (list of _FusionVarCUDA): The local variables.

    Only when fusing the reduction, the following attributes are updated.

        reduce_op (tuple): One of the element of reduction.***._raws._ops.
        reduce_identity (any type): The identity value of the reduction.
        reduce_kwargs (dict or None): kwargs of the reduction.

        premap_ret (_FusionVarCUDA or None): The target of reduction
        postmap_param (_FusionVarCUDA or None): The result of reduction
        postmap_op_list (list of FuisonOp): The post-map operations.
        postmap_local_list (list of _FusionVarCUDA): The local variables which
    appears in the post-map operations
    """

    def __init__(self):
        self.preamble_set = set()
        self.submodules = dict()
        self.count = 0

        self.op_list = []
        self.param_list = []
        self.local_list = []

        self.reduce_op = None
        self.reduce_identity = None
        self.reduce_kwargs = None

        self.postmap_op_list = []
        self.premap_ret = None
        self.postmap_param = None
        self.postmap_local_list = []

    def __repr__(self):
        return '<_FusionMem, op_list={}, var_list={}>'.format(
            self.op_list, self.var_list)

    def _has_reduction(self):
        return self.reduce_op is not None

    def _fresh_index(self):
        res = self.count
        self.count += 1
        return res

    def _fresh_premap_param(self, *args, **kwargs):
        index = self._fresh_index()
        var = _FusionVarCUDA(index, *args, **kwargs)
        self.param_list.append(var)
        return var

    def _fresh_postmap_param(self, *args, **kwargs):
        assert self.postmap_param is None
        index = self._fresh_index()
        var = _FusionVarCUDA(index, *args, **kwargs)
        self.postmap_param = var
        return var

    def _fresh_premap_local(self, *args, **kwargs):
        index = self._fresh_index()
        var = _FusionVarCUDA(index, *args, **kwargs)
        self.local_list.append(var)
        return var

    def _fresh_postmap_local(self, *args, **kwargs):
        index = self._fresh_index()
        var = _FusionVarCUDA(index, *args, **kwargs)
        self.postmap_local_list.append(var)
        return var

    def _fresh_local(self, *args, **kwargs):
        if self._has_reduction():
            return self._fresh_postmap_local(*args, **kwargs)
        else:
            return self._fresh_premap_local(*args, **kwargs)

    def _add_premap_op(self, *args, **kwargs):
        op = FusionOp(len(self.op_list), *args, **kwargs)
        subm = op.submodule
        self.submodules[subm.key()] = subm
        self.op_list.append(op)
        self._add_preamble(subm.preamble)
        return op

    def _add_postmap_op(self, *args, **kwargs):
        op = FusionOp(len(self.postmap_op_list), *args, **kwargs)
        subm = op.submodule
        self.submodules[subm.key()] = subm
        self.postmap_op_list.append(op)
        self._add_preamble(subm.preamble)
        return op

    def add_op(self, *args, **kwargs):
        if self._has_reduction():
            return self._add_postmap_op(*args, **kwargs)
        else:
            return self._add_premap_op(*args, **kwargs)

    def set_reduce_op(self, raw, arg, kwargs):
        assert self.reduce_op is None
        for op in raw._ops:
            (input_type,), (output_type,), _ = op
            if numpy.can_cast(arg.dtype.type, input_type):
                return_dtype = numpy.dtype(output_type)
                self.premap_ret = self._get_cuda_var(arg)
                self.reduce_op = op
                self.reduce_identity = raw.identity
                self.reduce_kwargs = kwargs
                self._add_preamble(raw._preamble)
                return self._fresh_postmap_param(return_dtype)
        raise TypeError('Type is mismatched. {}(...), {}'.format(
            self.raw._ops.name, arg.dtype.type))

    def _add_preamble(self, preamble):
        self.preamble_set.add(preamble)

    def _get_cuda_var(self, arg):
        """This converts `arg` to _FusionVarCUDA data.

        Args:
            arg (FusionVarPython or a primitive type)

        Return value: _FusionVarCUDA
        """
        if isinstance(arg, FusionVarPython):
            if arg._is_postmap == self._has_reduction():
                return arg._var
            else:
                # Map operation between pre-map variable and post-map variable
                raise Exception('Shape mismatch')
        is_scalar = isinstance(arg, six.integer_types + (float, bool, complex))
        is_ndarray = hasattr(arg, 'dtype') and arg.dtype in _dtype_list
        if is_scalar or is_ndarray:
            return self._fresh_local(numpy.dtype(type(arg)), const=arg)
        raise Exception('Unsupported type {}'.format(type(type)))

    def call_ufunc(self, ufunc, args, kwargs):
        nin = ufunc.nin
        nout = ufunc.nout

        # Corresponds to _check_should_use_min_scalar in elementwise.pxi
        # This function decides which typecast rule to use.
        def _should_use_min_scalar(in_args):
            max_array_kind = -2
            max_scalar_kind = -1
            for arg in in_args:
                kind = _kind_score[arg.dtype.kind]
                if arg.const is None:
                    max_array_kind = max(max_array_kind, kind)
                else:
                    max_scalar_kind = max(max_scalar_kind, kind)
            return (max_scalar_kind != -1 and
                    max_array_kind >= max_scalar_kind)

        def can_cast1(args, in_dtypes):
            for i in six.moves.range(nin):
                if args[i].const is None:
                    if not numpy.can_cast(args[i].dtype, in_dtypes[i]):
                        return False
                else:
                    if not numpy.can_cast(args[i].const, in_dtypes[i]):
                        return False
            return True

        def can_cast2(args, in_dtypes):
            for i in six.moves.range(nin):
                if not numpy.can_cast(args[i].dtype, in_dtypes[i]):
                    return False
            return True

        var_list = [self._get_cuda_var(_) for _ in args]
        if 'out' in kwargs:
            var_list.append(self._get_cuda_var(kwargs.pop('out')))
        if kwargs:
            raise TypeError('Wrong arguments {}'.format(kwargs))
        assert nin <= len(var_list) <= nin + nout
        in_vars = var_list[:nin]
        out_vars = var_list[nin:]
        can_cast = can_cast1 if _should_use_min_scalar(in_vars) else can_cast2
        for in_dtypes, out_dtypes, op in ufunc._ops:
            in_dtypes = [numpy.dtype(_) for _ in in_dtypes]
            out_dtypes = [numpy.dtype(_) for _ in out_dtypes]
            if can_cast(in_vars, in_dtypes):
                ret = []
                for i in six.moves.range(nout):
                    if i >= len(out_vars):
                        v = self._fresh_local(out_dtypes[i])
                        out_vars.append(v)
                        ret.append(FusionVarPython(v, self._has_reduction()))
                    elif numpy.can_cast(out_dtypes[i], out_vars[i].dtype,
                                        'same_kind'):
                        v = out_vars[i]
                        ret.append(FusionVarPython(v, self._has_reduction()))
                    else:
                        raise TypeError(
                            'output (typecode \'{}\') could not be coerced '
                            'to provided output parameter (typecode \'{}\') '
                            'according to the casting rule '
                            '"same_kind"'.format(
                                out_dtypes[i].char, out_vars[i].dtype.char))
                in_params = [(in_dtypes[i], 'in{}'.format(i))
                             for i, t in enumerate(in_vars)]
                out_params = [(out_dtypes[i], 'out{}'.format(i))
                              for i, t in enumerate(out_vars)]
                subm = Submodule(ufunc, in_params, out_params, op)
                self.add_op(subm, in_vars + out_vars)
                return ret[0] if len(ret) == 1 else tuple(ret)
        in_dtypes = [_.dtype for _ in in_vars]
        out_dtypes = [_.dtype for _ in out_vars]
        raise TypeError('Invalid type cast in \'{}\': {} -> {}'.format(
            ufunc.name, in_dtypes, out_dtypes))

    def call_elementwise(self, f, args, kwargs):
        raise NotImplementedError(
            'Fusion for elementwise-kernel is not implemented yet')

    def _emit_submodules_code(self):
        res = ''.join(self.preamble_set)
        res += '\n'.join([_.code() for _ in self.submodules.values()])
        return res

    def _emit_operation_code(self):
        res = '// {} operations\n'.format(len(self.op_list))
        res += ''.join(v.declaration() for v in self.local_list)
        res += ''.join(op.declaration_args() for op in self.op_list)
        res += ''.join(op.code() for op in self.op_list)
        return res

    def _emit_premap_code(self, in_params, operation):
        return_var = self.premap_ret
        module_code = string.Template('''
        __device__ ${return_ctype} _pre_map(${in_params}) {
        ${operation};
        return ${return_var};
        }
        ''').substitute(
            return_ctype=_dtype_to_ctype[return_var.dtype],
            in_params=', '.join('{} v{}'.format(_dtype_to_ctype[v.dtype],
                                                v.index)
                                for v in in_params),
            operation=operation,
            return_var=return_var)
        return module_code

    def _emit_postmap_code(self, out_params, operation):
        in_param = self.postmap_param
        in_ctype = _dtype_to_ctype[in_param.dtype]
        module_code = string.Template('''
        __device__ void _post_map(${in_ctype} in, ${out_params}) {
        ${in_param} = in;
        ${operation};
        }
        ''').substitute(
            in_ctype=in_ctype,
            in_param='{} v{}'.format(in_ctype, in_param.index),
            out_params=', '.join('{} &v{}'.format(_dtype_to_ctype[v.dtype],
                                                  v.index)
                                 for v in out_params),
            operation=operation)
        return module_code

    def _emit_postmap_cast_code(self, reduce_ctype, postmap_dtype, operation):
        module_code = string.Template('''
        __device__ ${postmap_ctype} _postmap_cast(${reduce_ctype} a) {
        ${postmap_ctype} out0;
        ${operation};
        return out0;
        }
        ''').substitute(
            reduce_ctype=reduce_ctype,
            postmap_ctype=_dtype_to_ctype[postmap_dtype],
            operation=operation)
        return module_code

    def get_fusion(self, func, in_dtypes, name):
        """This generates CUDA kernel from the given function and dtypes.

        This function generates ElementwiseKernel or ReductioKernel from the
        given function and the list of dtypes of parameters.

        Args:
            func (function): The function to be fused.
            in_types (list of dtypes): The list of dtypes of input parameters.
            name (str): The name of the kernel.

        Return value (tuple of ElementwiseKernel/ReductionKernel and dict):
            The second element of return values is kwargs that will give into
            the elementwise kernel or reduction kernel.
        """
        in_params = [self._fresh_premap_param(t) for t in in_dtypes]
        in_pvars = [FusionVarPython(_, False) for _ in in_params]
        return_value = func(*in_pvars)

        if isinstance(return_value, tuple):
            return_tuple = True
            no_return = False
            out_pvars = return_value
        elif isinstance(return_value, FusionVarPython):
            return_tuple = False
            no_return = False
            out_pvars = [return_value]
        elif return_value is None:
            return_tuple = False
            no_return = True
            out_pvars = []
        else:
            raise TypeError(
                'Fusion function can\'t return {}'.format(type(return_value)))

        out_pvars = [_ for _ in out_pvars if _ is not None]
        out_cvars = [self._get_cuda_var(_) for _ in out_pvars]

        out_dtypes = [_.dtype for _ in out_pvars]
        out_params = [self._fresh_premap_param(t) for t in out_dtypes]

        in_params_code = ', '.join(var.declaration_param()
                                   for var in in_params)
        out_params_code = ', '.join(var.declaration_param()
                                    for var in out_params)

        operation = self._emit_operation_code()
        submodule_code = self._emit_submodules_code()

        if self.reduce_op is None:
            operation += ' '.join('{} = {};'.format(t, s)
                                  for s, t in zip(out_cvars, out_params))
            kernel = core.ElementwiseKernel(
                in_params_code, out_params_code, operation,
                preamble=submodule_code,
                return_tuple=return_tuple,
                no_return=no_return,
                name=name)
            return kernel, {}
        else:
            _, (postmap_type,), (_, reduce_code, postmap_cast_code,
                                 reduce_ctype) = self.reduce_op
            if reduce_ctype is None:
                reduce_ctype = 'type_in0_raw'

            postmap_dtype = numpy.dtype(postmap_type)
            postmap_ctype = _dtype_to_ctype[postmap_dtype]

            postmap_code = '// {} operations\n'.format(
                len(self.postmap_op_list))
            postmap_code += ''.join(v.declaration()
                                    for v in self.postmap_local_list)
            postmap_code += ''.join(op.declaration_args()
                                    for op in self.postmap_op_list)
            postmap_code += ''.join(op.code() for op in self.postmap_op_list)
            postmap_code += ' '.join('{} = {};'.format(t, s)
                                     for s, t in zip(out_cvars, out_params))

            submodule_code += self._emit_premap_code(in_params, operation)
            submodule_code += 'typedef {} type_in0_raw;\n'.format(
                postmap_ctype)
            submodule_code += 'typedef {} type_out0_raw;\n'.format(
                postmap_ctype)
            submodule_code += self._emit_postmap_cast_code(
                reduce_ctype, postmap_dtype, postmap_cast_code)
            submodule_code += self._emit_postmap_code(out_params, postmap_code)

            kernel = core.ReductionKernel(
                in_params_code,
                out_params_code,
                '_pre_map({})'.format(', '.join([repr(p) for p in in_params])),
                reduce_code,
                '_post_map(_postmap_cast(a), {})'.format(
                    ', '.join([repr(p) for p in out_params])),
                self.reduce_identity,
                name=name,
                reduce_type=reduce_ctype,
                preamble=submodule_code)
            return kernel, self.reduce_kwargs


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
        return '<Fusion \'{}\'>'.format(self.name)

    def __call__(self, *args, **kwargs):
        if _thread_local.history is None:
            _thread_local.history = _FusionHistory()
            try:
                return self._call(*args, **kwargs)
            finally:
                _thread_local.history = None
        else:
            return self.func(*args, **kwargs)

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
            dtypes = [_.dtype for _ in args]
            key = tuple(dtypes)
            if key not in self._memo:
                self._memo[key] = _thread_local.history.get_fusion(
                    self.func, dtypes, self.name)
            return self._memo[key]
        else:
            if builtins.any(type(_) is core.ndarray for _ in args):
                types_str = '.'.join(repr(type(_)) for _ in args)
                message = 'Can\'t fuse \n {}({})'.format(self.name, types_str)
                warnings.warn(message)
            else:
                return self.func, {}

    def _call(self, *args, **kwargs):
        func, kw = self.compile(*args, **kwargs)
        kwargs = dict(kwargs, **kw)
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
    # Args:
    #   fusion_op (cupy.ufunc)
    #   cupy_op (cupyu.ufunc or function)
    #   numpy_op (numpy.ufunc or function)

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
        if _thread_local.history is not None:
            if builtins.any(isinstance(_, FusionVarPython) for _ in args):
                return _thread_local.history.call_ufunc(
                    self._fusion_op, args, kwargs)
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

i0 = _create_ufunc(math.special.i0, numpy.i0)
sinc = _create_ufunc(math.special.sinc, numpy.sinc)

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
    # Args:
    #   cupy_op (function): The CuPy reduction function.
    #   numpy_op (function): The NumPy reduction function.
    #   raw (core.simple_reduction_function object):
    #     This object must have identity, _preamble and _ops attributes
    def __init__(self, cupy_op, numpy_op, raw):
        self._cupy_op = cupy_op
        self._numpy_op = numpy_op
        self._raw = raw
        self.__doc__ = cupy_op.__doc__

    def __call__(self, *args, **kwargs):
        arg = args[0]
        if isinstance(arg, FusionVarPython):
            if arg._is_postmap:
                # Multiple reduction
                raise NotImplementedError(
                    'Multiple reduction is not implemented yet')
            if len(args) != 1:
                mes = '{}() takes 1 positional argument but {} were given'
                raise TypeError(mes.format(self._raw._ops.name, len(args)))
            return FusionVarPython(
                _thread_local.history.set_reduce_op(self._raw, arg, kwargs),
                True)
        elif builtins.any(type(_) == numpy.ndarray for _ in args):
            return self._numpy_op(*args, **kwargs)
        else:
            return self._cupy_op(*args, **kwargs)


def _create_reduction(cupy_op, numpy_op, raw):
    op = reduction(cupy_op, numpy_op, raw)

    def wrapper(*args, **kwargs):
        return op(*args, **kwargs)

    return functools.update_wrapper(wrapper, cupy_op)


all = _create_reduction(logic.truth.all, numpy.all, core._all)
any = _create_reduction(logic.truth.any, numpy.any, core._any)
sum = _create_reduction(math.sumprod.sum, numpy.sum, core._sum_auto_dtype)
prod = _create_reduction(math.sumprod.prod, numpy.prod, core._prod_auto_dtype)
amax = _create_reduction(statistics.order.amax, numpy.amax, core._amax)
amin = _create_reduction(statistics.order.amin, numpy.amin, core._amin)

if hasattr(numpy, "divmod"):
    divmod = _create_ufunc(core.divmod, numpy.divmod)
else:
    divmod = core.divmod
