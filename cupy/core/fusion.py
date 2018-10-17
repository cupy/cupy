import functools
import six
from six.moves import builtins
import string
import threading
import warnings

import numpy

import cupy
from cupy.core._dtype import get_dtype
from cupy.core import core


_thread_local = threading.local()

_kind_score = {
    'b': 0,
    'u': 1,
    'i': 1,
    'f': 2,
    'c': 2,
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
        val = c.item() if hasattr(c, 'dtype') else c
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
        ctypes = [_dtype_to_ctype[t] for t in self.dtypes]
        args_list = list(zip(self.args, args_sub, ctypes))
        code = '// op  # {}\n'.format(self.index)
        code += ''.join('{} = static_cast< {} >(v{});\n'.format(s, t, v.index)
                        for v, s, t in args_list)
        code += self.submodule.fcall(args_sub)
        code += ''.join('v{} = static_cast< {} >({});\n'.format(
            v.index, _dtype_to_ctype[v.dtype], s)
            for v, s, _ in
            args_list[len(self.submodule.in_params):])
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
        return cupy.negative(self)

    def __add__(self, other):
        return cupy.add(self, other)

    def __iadd__(self, other):
        return cupy.add(self, other, self)

    def __radd__(self, other):
        return cupy.add(other, self)

    def __sub__(self, other):
        return cupy.subtract(self, other)

    def __isub__(self, other):
        return cupy.subtract(self, other, self)

    def __rsub__(self, other):
        return cupy.subtract(other, self)

    def __mul__(self, other):
        return cupy.multiply(self, other)

    def __imul__(self, other):
        return cupy.multiply(self, other, self)

    def __rmul__(self, other):
        return cupy.multiply(other, self)

    def __div__(self, other):
        return cupy.divide(self, other)

    def __idiv__(self, other):
        return cupy.divide(self, other, self)

    def __rdiv__(self, other):
        return cupy.divide(other, self)

    def __truediv__(self, other):
        return cupy.true_divide(self, other)

    def __itruediv__(self, other):
        return cupy.true_divide(self, other, self)

    def __rtruediv__(self, other):
        return cupy.true_divide(other, self)

    def __floordiv__(self, other):
        return cupy.floor_divide(self, other)

    def __ifloordiv__(self, other):
        return cupy.floor_divide(self, other, self)

    def __rfloordiv__(self, other):
        return cupy.floor_divide(other, self)

    def __mod__(self, other):
        return cupy.remainder(self, other)

    def __imod__(self, other):
        return cupy.remainder(self, other, self)

    def __rmod__(self, other):
        return cupy.remainder(other, self)

    def __pow__(x, y):
        return cupy.power(x, y)

    def __ipow__(self, other):
        return cupy.power(self, other, self)

    def __lshift__(self, other):
        return cupy.left_shift(self, other)

    def __ilshift__(self, other):
        return cupy.left_shift(self, other, self)

    def __rlshift__(self, other):
        return cupy.left_shift(other, self)

    def __rshift__(self, other):
        return cupy.right_shift(self, other)

    def __irshift__(self, other):
        return cupy.right_shift(self, other, self)

    def __rrshift__(self, other):
        return cupy.right_shift(other, self)

    def __and__(self, other):
        return cupy.bitwise_and(self, other)

    def __iand__(self, other):
        return cupy.bitwise_and(self, other, self)

    def __rand__(self, other):
        return cupy.bitwise_and(other, self)

    def __or__(self, other):
        return cupy.bitwise_or(self, other)

    def __ior__(self, other):
        return cupy.bitwise_or(self, other, self)

    def __ror__(self, other):
        return cupy.bitwise_or(other, self)

    def __xor__(self, other):
        return cupy.bitwise_xor(self, other)

    def __ixor__(self, other):
        return cupy.bitwise_xor(self, other, self)

    def __rxor__(self, other):
        return cupy.bitwise_xor(other, self)

    def __invert__(self):
        return cupy.invert(self)

    def __lt__(self, other):
        return cupy.less(self, other)

    def __le__(self, other):
        return cupy.less_equal(self, other)

    def __eq__(self, other):
        return cupy.equal(self, other)

    def __ne__(self, other):
        return cupy.not_equal(self, other)

    def __gt__(self, other):
        return cupy.greater(self, other)

    def __ge__(self, other):
        return cupy.greater_equal(self, other)

    def __nonzero__(self):
        raise Exception('Can\'t cast to bool')

    def __bool__(self):
        raise Exception('Can\'t cast to bool')

    def __setitem__(self, slices, value):
        if slices is Ellipsis or (isinstance(slices, slice) and
                                  slices == slice(None)):
            cupy.copy(value, self)
        else:
            raise ValueError('The fusion supports `[...]` or `[:]`.')

    def copy(self):
        return cupy.copy(self)

    def astype(self, dtype, order=None, casting=None, subok=None, copy=True):
        dtype = get_dtype(dtype)
        if order is not None:
            raise TypeError('order is not supported yet')
        if casting is not None:
            raise TypeError('casting is not supported yet')
        if subok is not None:
            raise TypeError('subok is not supported yet')
        if not copy and self.dtype == dtype:
            return self
        return _dtype_to_astype(dtype)(self)


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

        in_params_code = ', '.join('_non_const ' + var.declaration_param()
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
        if not hasattr(_thread_local, 'history'):
            func, kw = self._compile(*args, **kwargs)
            kwargs = dict(kwargs, **kw)
            return func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)

    def _compile_from_dtypes(self, *dtypes):
        assert not hasattr(_thread_local, 'history')
        _thread_local.history = _FusionHistory()
        try:
            key = tuple(dtypes)
            if key not in self._memo:
                self._memo[key] = _thread_local.history.get_fusion(
                    self.func, dtypes, self.name)
            return self._memo[key]
        finally:
            del _thread_local.history

    def _compile(self, *args, **kwargs):
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
            return self._compile_from_dtypes(*dtypes)
        else:
            if builtins.any(type(_) is core.ndarray for _ in args):
                types_str = '.'.join(repr(type(_)) for _ in args)
                message = 'Can\'t fuse \n {}({})'.format(self.name, types_str)
                warnings.warn(message)
            else:
                return self.func, {}

    def clear_cache(self):
        self._memo = {}


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


def _ufunc_wrapper(fusion_op):
    def func(f):
        def call(*args, **kwargs):
            if not hasattr(_thread_local, 'history'):
                return f(*args, **kwargs)
            return _thread_local.history.call_ufunc(fusion_op, args, kwargs)
        return functools.update_wrapper(call, f)
    return func


def _reduction_wrapper(fusion_op):
    def func(f):
        def call(*args, **kwargs):
            if not hasattr(_thread_local, 'history'):
                return f(*args, **kwargs)
            arg = args[0]
            if arg._is_postmap:
                # Multiple reduction
                raise NotImplementedError(
                    'Multiple reduction is not implemented yet')
            if len(args) != 1:
                mes = '{}() takes 1 positional argument but {} were given'
                raise TypeError(mes.format(fusion_op._ops.name, len(args)))
            return FusionVarPython(
                _thread_local.history.set_reduce_op(fusion_op, arg, kwargs),
                True)
        return functools.update_wrapper(call, f)
    return func


def _create_astype_ufunc(dtype):
    name = 'astype_{}'.format(dtype)
    rules = tuple(['{}->{}'.format(cast_from.char, dtype.char)
                   for cast_from in _dtype_list])
    command = 'out0 = static_cast<{}>(in0)'.format(_dtype_to_ctype[dtype])
    return core.create_ufunc(name, rules, command)


_dtype_to_astype_dict = None


def _dtype_to_astype(dtype):
    global _dtype_to_astype_dict
    if _dtype_to_astype_dict is None:
        _dtype_to_astype_dict = dict([
            (dt, _create_astype_ufunc(dt))
            for dt in _dtype_list])
    return _dtype_to_astype_dict[dtype]
