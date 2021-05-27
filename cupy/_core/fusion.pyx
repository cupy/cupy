from cupy._core cimport _accelerator
from cupy._core._accelerator cimport ACCELERATOR_CUB
from cupy._core._scalar cimport get_typename

import functools
import string

import numpy

import cupy
from cupy._core._dtype import get_dtype
from cupy._core import _kernel
from cupy._core import _fusion_thread_local
from cupy._core import _reduction
from cupy._core import core
from cupy._core import new_fusion



_is_fusing = _fusion_thread_local.is_fusing  # NOQA
_thread_local = _fusion_thread_local.thread_local

cdef dict _kind_score = {
    'b': 0,
    'u': 1,
    'i': 1,
    'f': 2,
    'c': 2,
}

cdef list _dtype_list = [numpy.dtype(_) for _ in '?bhilqBHILQefdFD']

cdef tuple _acceptable_types = (
    core.ndarray, numpy.ndarray, numpy.generic,
    int, float, complex, bool, type(None))


class _Submodule(object):
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
        return '<_Submodule {}>'.format(self.name)

    def fcall(self, args):
        return self.name + '(' + ', '.join(args) + ');\n'

    def key(self):
        return (self.name, tuple(self.dtypes))

    def code(self):
        params = ', '.join('{} &{}'.format(get_typename(t), s)
                           for t, s in self.in_params + self.out_params)
        typedef = ''.join('typedef {} {}_type;\n'.format(get_typename(t), s)
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
        const_value (any of primitive types): The constant value (or None)
    """

    def __init__(self, index, dtype, const_value=None):
        self.index = index
        self.dtype = dtype
        self.const_value = const_value
        self.mutable = False

    def __repr__(self):
        return 'v{}'.format(self.index)

    def mutate(self):
        self.mutable = True

    def declaration(self):
        c = self.const_value
        val = c.item() if hasattr(c, 'dtype') else c
        ctype = get_typename(self.dtype)

        if self.const_value is None:
            return '{} v{};\n'.format(ctype, self.index)

        if isinstance(val, bool):
            init = '= {}'.format(str(c).lower())
        elif isinstance(val, complex):
            init = '({}, {})'.format(c.real, c.imag)
        elif isinstance(val, (int, float)):
            init = '= {}'.format(c)
        else:
            raise TypeError('Invalid constant type: {}'.format(type(c)))
        return 'const {} v{} {};\n'.format(ctype, self.index, init)

    def declaration_in_param(self):
        non_const = '_non_const ' if self.mutable else ''
        return '{}{} v{}'.format(non_const, self.dtype, self.index)

    def declaration_out_param(self):
        return '{} v{}'.format(self.dtype, self.index)


class _FusionOp(object):

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
        return '<_FusionOp #{}, {} types=[{}]>'.format(
            self.index, self.submodule.name, ', '.join(map(str, self.dtypes)))

    def declaration_args(self):
        return ' '.join('{} v{}_{};'.format(get_typename(t), self.index, j)
                        for j, t in enumerate(self.dtypes)) + '\n'

    def code(self):
        args_sub = ['v{}_{}'.format(self.index, i)
                    for i in range(len(self.args))]
        ctypes = [get_typename(t) for t in self.dtypes]
        args_list = list(zip(self.args, args_sub, ctypes))
        code = '// op  # {}\n'.format(self.index)
        code += ''.join('{} = static_cast< {} >(v{});\n'.format(s, t, v.index)
                        for v, s, t in args_list)
        code += self.submodule.fcall(args_sub)
        code += ''.join('v{} = static_cast< {} >({});\n'.format(
            v.index, get_typename(v.dtype), s)
            for v, s, _ in
            args_list[len(self.submodule.in_params):])
        return code


class _FusionVarScalar(object):

    """The values of variables in target function of fusion.

    Args:
        var (_FusionVarCUDA)
        ndim (int)
        is_postmap (bool)

    Attributes:
        dtype (dtype): The data type.
    """

    def __init__(self, var, ndim, is_postmap):
        self._var = var
        self.dtype = var.dtype
        self.ndim = ndim
        self._is_postmap = is_postmap
        assert ndim == -1

    def __repr__(self):
        return '<_FusionVar {} scalar>'.format(self.dtype)

    def __neg__(self):
        return cupy.negative(self)

    def __add__(self, other):
        return cupy.add(self, other)

    def __radd__(self, other):
        return cupy.add(other, self)

    def __sub__(self, other):
        return cupy.subtract(self, other)

    def __rsub__(self, other):
        return cupy.subtract(other, self)

    def __mul__(self, other):
        return cupy.multiply(self, other)

    def __rmul__(self, other):
        return cupy.multiply(other, self)

    def __div__(self, other):
        return cupy.divide(self, other)

    def __rdiv__(self, other):
        return cupy.divide(other, self)

    def __truediv__(self, other):
        return cupy.true_divide(self, other)

    def __rtruediv__(self, other):
        return cupy.true_divide(other, self)

    def __floordiv__(self, other):
        return cupy.floor_divide(self, other)

    def __rfloordiv__(self, other):
        return cupy.floor_divide(other, self)

    def __mod__(self, other):
        return cupy.remainder(self, other)

    def __rmod__(self, other):
        return cupy.remainder(other, self)

    def __pow__(x, y):
        return cupy.power(x, y)

    def __lshift__(self, other):
        return cupy.left_shift(self, other)

    def __rlshift__(self, other):
        return cupy.left_shift(other, self)

    def __rshift__(self, other):
        return cupy.right_shift(self, other)

    def __rrshift__(self, other):
        return cupy.right_shift(other, self)

    def __and__(self, other):
        return cupy.bitwise_and(self, other)

    def __rand__(self, other):
        return cupy.bitwise_and(other, self)

    def __or__(self, other):
        return cupy.bitwise_or(self, other)

    def __ror__(self, other):
        return cupy.bitwise_or(other, self)

    def __xor__(self, other):
        return cupy.bitwise_xor(self, other)

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
            _call_ufunc(core.elementwise_copy, value, out=self)
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


class _FusionVarArray(_FusionVarScalar):

    def __init__(self, var, ndim, is_postmap):
        self._var = var
        self.dtype = var.dtype
        self.ndim = ndim
        self._is_postmap = is_postmap
        assert ndim >= 0

    def __repr__(self):
        return '<_FusionVar {} {}-dim array>'.format(self.dtype, self.ndim)

    def __iadd__(self, other):
        return cupy.add(self, other, self)

    def __isub__(self, other):
        return cupy.subtract(self, other, self)

    def __imul__(self, other):
        return cupy.multiply(self, other, self)

    def __idiv__(self, other):
        return cupy.divide(self, other, self)

    def __itruediv__(self, other):
        return cupy.true_divide(self, other, self)

    def __ifloordiv__(self, other):
        return cupy.floor_divide(self, other, self)

    def __imod__(self, other):
        return cupy.remainder(self, other, self)

    def __ipow__(self, other):
        return cupy.power(self, other, self)

    def __ilshift__(self, other):
        return cupy.left_shift(self, other, self)

    def __irshift__(self, other):
        return cupy.right_shift(self, other, self)

    def __iand__(self, other):
        return cupy.bitwise_and(self, other, self)

    def __ior__(self, other):
        return cupy.bitwise_or(self, other, self)

    def __ixor__(self, other):
        return cupy.bitwise_xor(self, other, self)


class _FusionHistory(object):

    """History of operation exectuted in the target function of fusion.

    Attributes:
        preamble_set (set of str): The preambles of submodules.
        submodules (dict from str to submodule): The submodules.
        count (int): The number of variables in the fused function.

        op_list (list of _FusionOp): The map operations.
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
        return '<_FusionMem, op_list={}, param_list={}, local_list={}>'.format(
            self.op_list, self.param_list, self.local_list)

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
        op = _FusionOp(len(self.op_list), *args, **kwargs)
        subm = op.submodule
        self.submodules[subm.key()] = subm
        self.op_list.append(op)
        self._add_preamble(subm.preamble)
        return op

    def _add_postmap_op(self, *args, **kwargs):
        op = _FusionOp(len(self.postmap_op_list), *args, **kwargs)
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
        for op in raw._ops.ops:
            input_type, = op.in_types
            output_type, = op.out_types
            if numpy.can_cast(arg.dtype.type, input_type):
                return_dtype = numpy.dtype(output_type)
                self.premap_ret = self._get_fusion_var(arg)._var
                self.reduce_op = op
                self.reduce_identity = raw.identity
                self.reduce_kwargs = kwargs
                self._add_preamble(raw.preamble)
                return self._fresh_postmap_param(return_dtype)
        raise TypeError('Type is mismatched. {}(...), {}'.format(
            self.raw._ops.name, arg.dtype.type))

    def _add_preamble(self, preamble):
        self.preamble_set.add(preamble)

    def _get_fusion_var(self, arg):
        """This converts `arg` to _FusionVarScalar or _FusionVarArray data.

        Args:
            arg (_FusionVarScalar, _FusionVarArray or a primitive type)

        Return value:
            _FusionVarScalar or _FusionVarArray
        """
        if isinstance(arg, (_FusionVarScalar, _FusionVarArray)):
            if arg._is_postmap == self._has_reduction():
                return arg
            else:
                # Map operation between pre-map variable and post-map variable
                raise Exception('Shape mismatch')
        if isinstance(arg, (int, float, bool, complex, numpy.generic)):
            var = self._fresh_local(numpy.dtype(type(arg)), const_value=arg)
            return _FusionVarScalar(var, -1, self._has_reduction())
        raise TypeError('Unsupported type {}'.format(type(arg)))

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
                if isinstance(arg, _FusionVarArray):
                    max_array_kind = max(max_array_kind, kind)
                elif isinstance(arg, _FusionVarScalar):
                    max_scalar_kind = max(max_scalar_kind, kind)
                else:
                    assert False
            return (max_scalar_kind != -1 and
                    max_array_kind >= max_scalar_kind)

        def can_cast1(args, in_dtypes):
            for i in range(nin):
                arg = args[i]
                if isinstance(arg, _FusionVarArray):
                    if not numpy.can_cast(arg.dtype, in_dtypes[i]):
                        return False
                elif isinstance(arg, _FusionVarScalar):
                    scalar_value = arg._var.const_value
                    if scalar_value is None:
                        # This typecast is not safe.
                        # The result of a typecast of an element-wise operation
                        # between a numpy ndarray and a numpy scalar is not
                        # decidable statically, because it depends on the value
                        # of the scalar variable.
                        scalar_value = arg.dtype.type(0)
                    if not numpy.can_cast(scalar_value, in_dtypes[i]):
                        return False
                else:
                    assert False
            return True

        def can_cast2(args, in_dtypes):
            for i in range(nin):
                if not numpy.can_cast(args[i].dtype, in_dtypes[i]):
                    return False
            return True

        def make_fusion_var(var, ndim):
            if ndim == -1:
                return _FusionVarScalar(var, ndim, self._has_reduction())
            else:
                return _FusionVarArray(var, ndim, self._has_reduction())

        # Make FusionVar list
        var_list = [self._get_fusion_var(arg) for arg in args]
        in_vars = var_list[:nin]
        out_vars = var_list[nin:]
        if 'out' in kwargs:
            out = kwargs.pop('out')
            if out_vars:
                raise ValueError('cannot specify \'out\' as both a positional '
                                 'and keyword argument')
            if isinstance(out, _FusionVarArray):
                out_vars.append(self._get_fusion_var(out))
            elif out is not None:
                raise ValueError('The \'out\' tuple must have exactly one '
                                 'entry per ufunc output')
        if kwargs:
            raise TypeError('Wrong arguments {}'.format(kwargs))
        if len(in_vars) != nin or len(out_vars) > nout:
            raise ValueError('invalid number of arguments')
        if not all([isinstance(v, _FusionVarArray) for v in out_vars]):
            raise TypeError('return arrays must be of ArrayType')
        var_list = in_vars + out_vars

        # Broadcast
        ndim = max([v.ndim for v in in_vars])
        if any([v.ndim < ndim for v in out_vars]):
            raise ValueError('non-broadcastable output operand')

        # Typecast and add an operation
        can_cast = can_cast1 if _should_use_min_scalar(var_list) else can_cast2
        # TODO(asi1024): Fix to use ``guess_routine``.
        for op in ufunc._ops.ops:
            in_dtypes = [numpy.dtype(t) for t in op.in_types]
            out_dtypes = [numpy.dtype(t) for t in op.out_types]
            if can_cast(var_list, in_dtypes):
                ret = []
                for i in range(nout):
                    if i >= len(out_vars):
                        out_var = self._fresh_local(out_dtypes[i])
                        out_var = make_fusion_var(out_var, ndim)
                        out_vars.append(out_var)
                        ret.append(out_var)
                    elif numpy.can_cast(out_dtypes[i], out_vars[i].dtype,
                                        'same_kind'):
                        out_var = out_vars[i]
                        ret.append(out_var)
                    else:
                        raise TypeError(
                            'output (typecode \'{}\') could not be coerced '
                            'to provided output parameter (typecode \'{}\') '
                            'according to the casting rule '
                            '"same_kind"'.format(
                                out_dtypes[i].char, out_vars[i].dtype.char))
                    out_var._var.mutate()
                in_params = [(in_dtypes[i], 'in{}'.format(i))
                             for i, _ in enumerate(in_vars)]
                out_params = [(out_dtypes[i], 'out{}'.format(i))
                              for i, _ in enumerate(out_vars)]
                subm = _Submodule(ufunc, in_params, out_params, op.routine)
                self.add_op(subm, [v._var for v in in_vars + out_vars])
                return ret[0] if len(ret) == 1 else tuple(ret)
        in_dtypes = [v.dtype for v in in_vars]
        out_dtypes = [v.dtype for v in out_vars]
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
            return_ctype=get_typename(return_var.dtype),
            in_params=', '.join('{} v{}'.format(get_typename(v.dtype),
                                                v.index)
                                for v in in_params),
            operation=operation,
            return_var=return_var)
        return module_code

    def _emit_postmap_code(self, out_params, operation):
        in_param = self.postmap_param
        in_ctype = get_typename(in_param.dtype)
        module_code = string.Template('''
        __device__ void _post_map(${in_ctype} in, ${out_params}) {
        ${in_param} = in;
        ${operation};
        }
        ''').substitute(
            in_ctype=in_ctype,
            in_param='{} v{}'.format(in_ctype, in_param.index),
            out_params=', '.join('{} &v{}'.format(get_typename(v.dtype),
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
            postmap_ctype=get_typename(postmap_dtype),
            operation=operation)
        return module_code

    def _gen_abstracted_args(self, a):
        if isinstance(a, core.ndarray):
            cuda_var = self._fresh_premap_param(a.dtype)
            python_var = _FusionVarArray(cuda_var, a.ndim, False)
        elif a is None:
            cuda_var = None
            python_var = None
        else:
            cuda_var = self._fresh_premap_param(numpy.dtype(type(a)))
            python_var = _FusionVarScalar(cuda_var, -1, False)
        return cuda_var, python_var

    def get_fusion(self, func, args, name):
        """This generates CUDA kernel from the given function and dtypes.

        This function generates ElementwiseKernel or ReductioKernel from the
        given function and the list of dtypes of parameters.

        Args:
            func (function): The function to be fused.
            args (tuple): The tuple of arguments.
            name (str): The name of the kernel.

        Return value (tuple of ElementwiseKernel/ReductionKernel and dict):
            The second element of return values is kwargs that will give into
            the elementwise kernel or reduction kernel.
        """
        self.ndim = max([a.ndim for a in args if isinstance(a, core.ndarray)])

        in_params = []
        function_args = []
        for a in args:
            cuda_var, python_var = self._gen_abstracted_args(a)
            if cuda_var is not None:
                in_params.append(cuda_var)
            function_args.append(python_var)

        return_value = func(*function_args)

        if isinstance(return_value, tuple):
            return_tuple = True
            no_return = False
            out_pvars = return_value
        elif isinstance(return_value, (_FusionVarScalar, _FusionVarArray)):
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
        out_cvars = [self._get_fusion_var(_)._var for _ in out_pvars]

        out_dtypes = [_.dtype for _ in out_pvars]
        out_params = [self._fresh_premap_param(t) for t in out_dtypes]

        in_params_code = ', '.join(var.declaration_in_param()
                                   for var in in_params)
        out_params_code = ', '.join(var.declaration_out_param()
                                    for var in out_params)

        operation = self._emit_operation_code()
        submodule_code = self._emit_submodules_code()

        if self.reduce_op is None:
            operation += ' '.join('{} = {};'.format(t, s)
                                  for s, t in zip(out_cvars, out_params))
            kernel = _kernel.ElementwiseKernel(
                in_params_code, out_params_code, operation,
                preamble=submodule_code,
                return_tuple=return_tuple,
                no_return=no_return,
                name=name)
            return kernel, {}
        else:
            _, reduce_expr, postmap_expr, reduce_ctype = self.reduce_op.routine
            if reduce_ctype is None:
                reduce_ctype = 'type_out0_raw'

            postmap_type, = self.reduce_op.out_types
            postmap_dtype = numpy.dtype(postmap_type)
            postmap_ctype = get_typename(postmap_dtype)

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
            use_cub = ACCELERATOR_CUB in _accelerator._reduction_accelerators
            if not use_cub:
                submodule_code += 'typedef {} type_in0_raw;\n'.format(
                    postmap_ctype)
            submodule_code += 'typedef {} type_out0_raw;\n'.format(
                postmap_ctype)
            submodule_code += self._emit_postmap_cast_code(
                reduce_ctype, postmap_dtype, postmap_expr)
            submodule_code += self._emit_postmap_code(out_params, postmap_code)

            kernel = _reduction.ReductionKernel(
                in_params_code,
                out_params_code,
                '_pre_map({})'.format(', '.join([repr(p) for p in in_params])),
                reduce_expr,
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
        self.new_fusion = None

    def __repr__(self):
        return '<Fusion \'{}\'>'.format(self.name)

    def __call__(self, *args):
        if self.new_fusion is not None:
            return self.new_fusion(*args)

        # Inner function of composition of multiple fused functions.
        if _fusion_thread_local.is_old_fusing():
            return self.func(*args)

        exec_cupy = False
        for a in args:
            if isinstance(a, core.ndarray):
                exec_cupy = True
                break
        if not exec_cupy:
            # No cupy ndarray exists in the arguments
            return self.func(*args)

        # Invalid argument types
        for arg in args:
            if not isinstance(arg, _acceptable_types):
                mes = 'Invalid argument type for \'{}\': ({})'
                arg_types = ', '.join(repr(type(a)) for a in args)
                raise TypeError(mes.format(self.name, arg_types))

        # Cache the result of execution path analysis
        cdef list params_info = []
        for arg in args:
            if isinstance(arg, core.ndarray):
                params_info.append(arg.dtype.char)
                params_info.append(arg.ndim)
            elif isinstance(arg, numpy.generic):
                params_info.append(arg.dtype.char)
            elif arg is None:
                params_info.append(None)
            elif isinstance(arg, float):
                params_info.append('d')
            elif isinstance(arg, int):
                params_info.append('l')
            elif isinstance(arg, bool):
                params_info.append('?')
            elif isinstance(arg, complex):
                params_info.append('D')
            else:
                assert False

        cdef tuple key = tuple(params_info)
        if key not in self._memo:
            try:
                history = _FusionHistory()
                _thread_local.history = history
                _thread_local.is_old_fusing = True
                try:
                    self._memo[key] = history.get_fusion(
                        self.func, args, self.name)
                except Exception:
                    self.new_fusion = new_fusion.Fusion(self.func, self.name)
                    _thread_local.history = None
                    _thread_local.is_old_fusing = False
                    return self.new_fusion(*args)
            finally:
                _thread_local.history = None
                _thread_local.is_old_fusing = False
        kernel, kwargs = self._memo[key]

        return kernel(
            *[a for a in args if a is not None],
            **kwargs)

    def clear_cache(self):
        self._memo = {}


def fuse(*args, **kwargs):
    """Decorator that fuses a function.

    This decorator can be used to define an elementwise or reduction kernel
    more easily than :class:`~cupy.ElementwiseKernel` or
    :class:`~cupy.ReductionKernel`.

    Since the fused kernels are cached and reused, it is recommended to reuse
    the same decorated functions instead of e.g. decorating local functions
    that are defined multiple times.

    Args:
        kernel_name (str): Name of the fused kernel function.
            If omitted, the name of the decorated function is used.

    Example:

        >>> @cupy.fuse(kernel_name='squared_diff')
        ... def squared_diff(x, y):
        ...     return (x - y) * (x - y)
        ...
        >>> x = cupy.arange(10)
        >>> y = cupy.arange(10)[::-1]
        >>> squared_diff(x, y)
        array([81, 49, 25,  9,  1,  1,  9, 25, 49, 81])
    """

    def wrapper(f, kernel_name=None):
        return Fusion(f, kernel_name)

    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return functools.update_wrapper(wrapper(args[0]), args[0])
    else:
        return lambda f: functools.update_wrapper(
            wrapper(f, *args, **kwargs), f)


def _call_ufunc(fusion_op, *args, **kwargs):
    return _thread_local.history.call_ufunc(fusion_op, args, kwargs)


def _call_reduction(fusion_op, *args, **kwargs):
    if len(args) != 1:
        mes = '{}() takes 1 positional argument but {} were given'
        raise TypeError(mes.format(fusion_op._ops.name, len(args)))

    arg = args[0]
    kwargs = dict([(key, value) for key, value in kwargs.items()
                   if (key in ('axis', 'out') and value is not None)])

    if arg._is_postmap:
        # Multiple reduction
        raise NotImplementedError(
            'Multiple reduction is not implemented yet')

    var = _thread_local.history.set_reduce_op(fusion_op, arg, kwargs)

    src_ndim = max(0, arg.ndim)
    if 'axis' in kwargs:
        axis = kwargs['axis']
        if isinstance(axis, (tuple, list)):
            ndim = src_ndim - len(axis)
        else:
            ndim = src_ndim - 1
    else:
        ndim = 0
    if ndim < 0:
        raise numpy.AxisError(axis, src_ndim)

    _thread_local.history.ndim = ndim
    if ndim >= 1:
        return _FusionVarArray(var, ndim, True)
    else:
        return _FusionVarScalar(var, -1, True)


def _create_astype_ufunc(dtype):
    name = 'astype_{}'.format(dtype)
    rules = tuple(['{}->{}'.format(cast_from.char, dtype.char)
                   for cast_from in _dtype_list])
    command = 'out0 = static_cast< {} >(in0)'.format(get_typename(dtype))
    return core.create_ufunc(name, rules, command)


_dtype_to_astype_dict = None


def _dtype_to_astype(dtype):
    global _dtype_to_astype_dict
    if _dtype_to_astype_dict is None:
        _dtype_to_astype_dict = dict([
            (dt, _create_astype_ufunc(dt))
            for dt in _dtype_list])
    return _dtype_to_astype_dict[dtype]
