import functools
import six
import string

import copy
import cupy
from libcpp cimport vector
from cupy.core.core cimport ndarray, Indexer
from cupy.core._routines_manipulation cimport broadcast_to
from cupy.core import _kernel
from cupy.core cimport internal
from cupy.core._dtype import get_dtype
from cupy.core._kernel import _is_fusingx
from cupy.core.fusion cimport _acceptable_types, _kind_score, _dtype_to_ctype
from cupy.core.fusion import _Submodule
import numpy
from cupy.core.core cimport compile_with_cache

_thread_local = _kernel._thread_local

def _call_ufunc(fusion_op, *args, **kwargs):
    return _thread_local.historyx.call_ufunc(fusion_op, args, kwargs)

def _call_reduction(fusion_op, *args, **kwargs):
    return _thread_local.historyx.call_reduction(fusion_op, args, kwargs)

# function args
def _get_pvar_param_name(pvar):
    if isinstance(pvar, _FusionXVarArray):
        return 'a{}_{}'.format(pvar.index, pvar.dup)
    elif isinstance(pvar, _FusionXVarScalar):
        return 'a{}'.format(pvar.index)
    else:
        raise TypeError('Unknown type {}.'.format(type(pvar)))

# inside function
def _get_pvar_decl_name(pvar):
    if isinstance(pvar, _FusionXVarArray):
        return 'v{}_{}'.format(pvar.index, pvar.dup)
    elif isinstance(pvar, _FusionXVarScalar):
        return 'v{}'.format(pvar.index)
    else:
        raise TypeError('Unknown type {}.'.format(type(pvar)))

def _get_indexer_name(pvar):
    return 'ind{}_{}'.format(pvar.index, pvar.dup)

class _FusionXVarScalar(object):
    def __init__(self, index, dtype, const_value):
        self.index = index
        self.dtype = dtype
        self.ndim = -1

        # resettable
        self.const_value = const_value

    def reset(self):
        pass

    def __eq__(self, other):
        if isinstance(other, _FusionXVarScalar):
            if isinstance(other, _FusionXVarArray):
                return False
            return self.index == other.index
        return False

    def __hash__(self):
        return self.index

    def __repr__(self):
        return '<_FusionXVarScalar name={}, dtype={}>'.format(_get_pvar_param_name(self), self.dtype)

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

    def copy(self):
        return cupy.copy(self)

class _FusionXVarArray(_FusionXVarScalar):
    def __init__(self, index, ndim, dtype, block_index, init_abstracted_shape=False):
        self.index = index
        self.dup = None
        self.ndim = ndim
        self.dtype = dtype
        self.prev_block_index = 999
        self.block_index = block_index
        self.is_input = False
        self.input_order = None
        self.is_output = False
        self.output_order = None
        self.updated_from = None
        self.broadcasted_from = None
        self.abstracted_shape = tuple('v{}.{}'.format(self.index, i) for i in range(self.ndim)) if init_abstracted_shape else None

        # resettable
        self.size = None
        self.real_shape = None
        self.ndarray = None

    def reset(self):
        self.size = None
        self.real_shape = None
        self.ndarray = None

    def set_size(self):
        self.size = internal.prod(self.real_shape)

    def set_updated(self):
        if self.updated_from is not None:
            self.updated_from.set_updated()
            self.ndarray = copy.copy(self.updated_from.ndarray)

    def __repr__(self):
        return '<_FusionXVarArray name={}, dtype={}, ndim={}, abstracted_shape={}>'.format(_get_pvar_param_name(self), self.dtype, self.ndim, self.abstracted_shape)

    def __eq__(self, other):
        if isinstance(other, _FusionXVarArray):
            return self.index == other.index and self.block_index == other.block_index and self.abstracted_shape == other.abstracted_shape
        return False

    def __hash__(self):
        return hash((self.index, self.block_index, self.abstracted_shape))

    def __iadd__(self, other):
        if self.broadcasted_from is not None:
            raise ValueError('output array is read-only')
        return cupy.add(self, other, self)

    def __isub__(self, other):
        if self.broadcasted_from is not None:
            raise ValueError('output array is read-only')
        return cupy.subtract(self, other, self)

    def __imul__(self, other):
        if self.broadcasted_from is not None:
            raise ValueError('output array is read-only')
        return cupy.multiply(self, other, self)

    def __idiv__(self, other):
        if self.broadcasted_from is not None:
            raise ValueError('output array is read-only')
        return cupy.divide(self, other, self)

    def __itruediv__(self, other):
        if self.broadcasted_from is not None:
            raise ValueError('output array is read-only')
        return cupy.true_divide(self, other, self)

    def __ifloordiv__(self, other):
        if self.broadcasted_from is not None:
            raise ValueError('output array is read-only')
        return cupy.floor_divide(self, other, self)

    def __imod__(self, other):
        if self.broadcasted_from is not None:
            raise ValueError('output array is read-only')
        return cupy.remainder(self, other, self)

    def __ipow__(self, other):
        if self.broadcasted_from is not None:
            raise ValueError('output array is read-only')
        return cupy.power(self, other, self)

    def __ilshift__(self, other):
        if self.broadcasted_from is not None:
            raise ValueError('output array is read-only')
        return cupy.left_shift(self, other, self)

    def __irshift__(self, other):
        if self.broadcasted_from is not None:
            raise ValueError('output array is read-only')
        return cupy.right_shift(self, other, self)

    def __iand__(self, other):
        if self.broadcasted_from is not None:
            raise ValueError('output array is read-only')
        return cupy.bitwise_and(self, other, self)

    def __ior__(self, other):
        if self.broadcasted_from is not None:
            raise ValueError('output array is read-only')
        return cupy.bitwise_or(self, other, self)

    def __ixor__(self, other):
        if self.broadcasted_from is not None:
            raise ValueError('output array is read-only')
        return cupy.bitwise_xor(self, other, self)

class _FusionXOp(object):
    def __init__(self, ufunc, in_pvars, out_pvars, in_dtypes, out_dtypes):
        self.ufunc = ufunc
        self.in_pvars = in_pvars
        self.out_pvars = out_pvars
        self.in_dtypes = in_dtypes
        self.out_dtypes = out_dtypes

    def __repr__(self):
        in_pvars = ', '.join([_get_pvar_param_name(a) for a in self.in_pvars])
        out_pvars = ', '.join([_get_pvar_param_name(a) for a in self.out_pvars])
        return '<_FusionXOp name={}, in_pvars={}, out_pvars={}>'.format(self.ufunc.name, in_pvars, out_pvars)

    def _get_indexer_setup(self):
        ret = ''
        for a in self.in_pvars + self.out_pvars:
            ret += '        {}.set(i);\n'.format(_get_indexer_name(a))
        return ret

    def _get_declaration(self):
        ret = ''
        used = set()
        for a in self.in_pvars + self.out_pvars:
            if a in used:
                continue
            used.add(a)
            if isinstance(a, _FusionXVarArray):
                ret += '        {} {}_ = {}[{}.get()];\n'.format(
                    _dtype_to_ctype[a.dtype], _get_pvar_decl_name(a), _get_pvar_param_name(a), _get_indexer_name(a))
            elif isinstance(a, _FusionXVarScalar):
                if a.const_value is None:
                    ret += '        {} {}_ = {};\n'.format(
                        _dtype_to_ctype[a.dtype], _get_pvar_decl_name(a), _get_pvar_param_name(a))
                else:
                    ret += '        {} {}_ = {};\n'.format(
                        _dtype_to_ctype[a.dtype], _get_pvar_decl_name(a), a.const_value)
            else:
                raise TypeError('Unknown type {}.'.format(type(a)))
        return ret

    def _get_operation(self):
        pre_conversion = ''
        operation = ''
        post_conversion = ''
        used = set()
        for a, d in zip(self.in_pvars + self.out_pvars, self.in_dtypes + self.out_dtypes):
            if a in used:
                continue
            used.add(a)
            pre_conversion += '        {type} {name} = ({type}) {name}_;\n'.format(
                type=_dtype_to_ctype[d], name=_get_pvar_decl_name(a))
        operation += '        {}({});\n'.format(self.ufunc.name, ', '.join([_get_pvar_decl_name(a) for a in self.in_pvars + self.out_pvars]))
        for a in self.out_pvars:
            post_conversion += '        {name}_ = ({type}) {name};\n'.format(
                type=_dtype_to_ctype[a.dtype], name=_get_pvar_decl_name(a))
        return pre_conversion + operation + post_conversion

    def _get_after_operation(self):
        ret = ''
        for a in self.out_pvars:
            if isinstance(a, _FusionXVarArray):
                ret += '        {}[{}.get()] = {}_;\n'.format(
                    _get_pvar_param_name(a), _get_indexer_name(a), _get_pvar_decl_name(a))
            elif isinstance(a, _FusionXVarScalar):
                ret += '        {} = {}_;\n'.format(
                    _get_pvar_param_name(a), _get_pvar_decl_name(a))
            else:
                raise TypeError('Unknown type {}.'.format(type(a)))
        return ret

    def code(self):
        return string.Template('''
    CUPY_FOR(i, ${indexer_name}.size()) {
${indexer_setup}
${declaration}
${operation}
${after_operation}
    }
''').substitute(
            indexer_name=_get_indexer_name(self.out_pvars[0]),
            declaration=self._get_declaration(),
            indexer_setup=self._get_indexer_setup(),
            operation=self._get_operation(),
            after_operation=self._get_after_operation())

class _FusionXReductionOp(object):
    def __init__(self, func, in_pvar, out_pvar, axis, op, reduction_index):
        self.func = func
        self.in_pvar = in_pvar
        self.out_pvar = out_pvar
        # for negative axis
        self.axis = axis % in_pvar.ndim
        self.reduce_identity = func.identity
        _, _, (_, self.reduce_expr, _, _) = op
        self.reduction_index = reduction_index

    def __repr__(self):
        return '<_FusionXReductionOp name={}, in_pvar={}, out_pvar={}>'.format(self.func.name, self.in_pvar, self.out_pvar)

    def _get_func_name_suffix(self):
        def get_type(pvar):
            ret = pvar.dtype.char
            if ret == '?':
                ret = '_'
            return ret
        return '_{}{}{}{}{}'.format(self.func.name, get_type(self.in_pvar), self.in_pvar.ndim, get_type(self.out_pvar), self.out_pvar.ndim)

    def _add_to_reduction_submodules(self, target):
        name = self._get_func_name_suffix()
        if name not in target.reduction_submodules:
            submodule_code = reduction_code_template(name, self.in_pvar, self.out_pvar, self.reduce_expr, self.reduce_identity)
            target.reduction_submodules[name] = submodule_code

    def code(self):
        return string.Template('''
    reduce${name}(${in_arr}, ${out_arr}, ${in_ind}, ${out_ind}, block_stride${reduction_index});
''').substitute(
            name=self._get_func_name_suffix(),
            in_arr=_get_pvar_param_name(self.in_pvar),
            out_arr=_get_pvar_param_name(self.out_pvar),
            in_ind=_get_indexer_name(self.in_pvar),
            out_ind=_get_indexer_name(self.out_pvar),
            reduction_index = self.reduction_index)

class _ShapeConstraints(object):
    def __init__(self):
        self.eq_constraints = set()
        self.one_constraints = set()

    def add_eq_constraint(self, a, b):
        if a == b:
            return
        if a == '1' or b == '1':
            self.add_one_constraint(a)
            self.add_one_constraint(b)
            return
        if a > b:
            a, b = b, a
        self.eq_constraints.add((a, b))

    def add_one_constraint(self, a):
        self.one_constraints.add(a)

    def __repr__(self):
        return '<_ShapeConstraints eq_constraints={}, one_constraints={}>'.format(self.eq_constraints, self.one_constraints)

class _FusionXHistory(object):
    def __init__(self):
        self.count = 0
        self.dup_count = dict()
        self.block_count = 0
        # TODO: reuse dead variables
        self.param_list_base_used = set()
        self.param_list_base = list()
        # contains broadcasted params
        self.param_list_used = set()
        self.param_list = list()
        self.shape_constraints = _ShapeConstraints()
        self.op_list = list()
        self.reduction_op_list = list()

        self.ufunc_submodules = dict()
        self.reduction_submodules = dict()
        self.block_strides = list()

        self.cuda_body = None

    def __repr__(self):
        return '<_FusionXHistory> op_list={}, param_list={}, shape_constraints={}'.format(self.op_list, self.param_list, self.shape_constraints)

    def _fresh_index(self):
        ret = self.count
        self.count += 1
        return ret

    def _append_param(self, pvar, need_malloc=True):
        if pvar in self.param_list_used:
            return
        if isinstance(pvar, _FusionXVarArray):
            dup = self.dup_count.get(pvar.index, 0)
            pvar.dup = dup
            self.dup_count[pvar.index] = dup + 1
        if need_malloc and pvar not in self.param_list_base_used:
            self.param_list_base_used.add(pvar)
            self.param_list_base.append(pvar)
        self.param_list_used.add(pvar)
        self.param_list.append(pvar)

    def _get_pvar(self, arg):
        if isinstance(arg, _FusionXVarScalar):
            return arg
        if isinstance(arg, six.integer_types +
                      (float, bool, complex, numpy.generic)):
            index = self._fresh_index()
            dtype = numpy.dtype(type(arg))
            pvar = _FusionXVarScalar(index, dtype, arg)
            self._append_param(pvar)
            return pvar
        raise TypeError('Unsupported type {}'.format(type(arg)))

    def _make_input_param(self, arg, idx):
        if isinstance(arg, cupy.core.ndarray):
            index = self._fresh_index()
            ret = _FusionXVarArray(index, arg.ndim, arg.dtype, 0, True)
            ret.real_shape = arg.shape
            ret.is_input = True
            ret.input_order = idx
            self._append_param(ret)
        else:
            index = self._fresh_index()
            dtype = numpy.dtype(type(arg))
            ret = _FusionXVarScalar(index, dtype)
            self._append_param(ret)
        return ret

    def _broadcast_to(self, pvar, abstracted_shape, real_shape):
        ret = copy.copy(pvar)
        ret.ndim = len(abstracted_shape)
        ret.abstracted_shape = abstracted_shape
        ret.real_shape = real_shape
        ret.updated_from = None
        ret.broadcasted_from = pvar
        ret.is_input = False
        self._append_param(ret, False)
        return ret

    def _update_block_index(self, pvar):
        ret = copy.copy(pvar)
        ret.prev_block_index = pvar.block_index
        ret.updated_from = pvar
        ret.block_index = self.block_count
        self._append_param(ret, False)
        return ret

    def _make_new_param(self, ndim, dtype, abstracted_shape, real_shape, next_block=False):
        index = self._fresh_index()
        block = self.block_count
        if next_block:
            block += 1
        ret = _FusionXVarArray(index, ndim, dtype, block)
        ret.abstracted_shape = abstracted_shape
        ret.real_shape = real_shape
        self._append_param(ret, True)
        return ret

    def _add_ufunc_op(self, ufunc, in_pvars, out_pvars, in_dtypes, out_dtypes):
        op = _FusionXOp(ufunc, in_pvars, out_pvars, in_dtypes, out_dtypes)
        self.op_list.append(op)

    def _add_reduction_op(self, func, in_pvar, out_pvar, axis, op):
        op = _FusionXReductionOp(func, in_pvar, out_pvar, axis, op, self.block_count)
        self.block_count += 1
        self.op_list.append(op)
        self.reduction_op_list.append(op)

    def call_ufunc(self, ufunc, args, kwargs):
        nin = ufunc.nin
        nout = ufunc.nout

        # Corresponds to _checkshould_use_min_scalar in elementwise.pxi
        # This function decides which typecast rule to use.
        def _should_use_min_scalar(in_args):
            max_array_kind = -2
            max_scalar_kind = -1
            for arg in in_args:
                kind = _kind_score[arg.dtype.kind]
                if isinstance(arg, _FusionXVarArray):
                    max_array_kind = max(max_array_kind, kind)
                elif isinstance(arg, _FusionXVarScalar):
                    max_scalar_kind = max(max_scalar_kind, kind)
                else:
                    assert False
            return (max_scalar_kind != -1 and
                    max_array_kind >= max_scalar_kind)

        def can_cast1(args, in_dtypes):
            for i in six.moves.range(nin):
                arg = args[i]
                if isinstance(arg, _FusionXVarArray):
                    if not numpy.can_cast(arg.dtype, in_dtypes[i]):
                        return False
                elif isinstance(arg, _FusionXVarScalar):
                    scalar_value = arg.const_value
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
            for i in six.moves.range(nin):
                if not numpy.can_cast(args[i].dtype, in_dtypes[i]):
                    return False
            return True

        pvar_list = [self._get_pvar(arg) for arg in args]
        in_pvars = pvar_list[:nin]
        out_pvars = pvar_list[nin:]

        for pvar in in_pvars:
            if isinstance(pvar, _FusionXVarArray):
                if pvar.block_index != self.block_count:
                    pvar = self._update_block_index(pvar)

        if 'out' in kwargs:
            out = kwargs.pop('out')
            if out_pvars:
                raise ValueError('cannot specify \'out\' as both a positional and keyword argument')
            if isinstance(out, _FusionXVarArray):
                out_pvars.append(out)
            elif out is not None:
                raise ValueError('The \'out\' tuple must have exactly one entry per ufunc output')
        if kwargs:
            raise TypeError('Wrong arguments {}'.format(kwargs))
        if len(in_pvars) != nin or len(out_pvars) > nout:
            raise ValueError('Invalid number of arguments')
        if not all([isinstance(v, _FusionXVarArray) for v in out_pvars]):
            raise TypeError('Return arrays must be of ArrayType')

        # Broadcast
        ndim = max([v.ndim for v in in_pvars])
        if ndim < 0:
            raise ValueError('Invalid ndim {} in {}'.format(ndim, ufunc.name))

        if ndim == 0:
            out_abstracted_shape = ()
            out_real_shape = ()
        else:
            out_abstracted_shape = [None for _ in range(ndim)]
            out_real_shape = [0 for _ in range(ndim)]

            # Determine shape
            for in_pvar in in_pvars:
                cur_ndim = in_pvar.ndim
                for j in range(cur_ndim):
                    if in_pvar.real_shape[cur_ndim - 1 - j] > out_real_shape[ndim - 1 - j]:
                        out_real_shape[ndim - 1 - j] = in_pvar.real_shape[cur_ndim - 1 - j]
                        out_abstracted_shape[ndim - 1 - j] = in_pvar.abstracted_shape[cur_ndim - 1 - j]

            out_abstracted_shape = tuple(out_abstracted_shape)
            out_real_shape = tuple(out_real_shape)

            for idx, in_pvar in enumerate(in_pvars):
                cur_ndim = in_pvar.ndim
                if cur_ndim == -1:
                    continue
                is_necessary = cur_ndim != ndim
                for j in range(cur_ndim):
                    if in_pvar.real_shape[cur_ndim - 1 - j] < out_real_shape[ndim - 1 - j]:
                        if in_pvar.real_shape[cur_ndim - 1 - j] == 1:
                            self.shape_constraints.add_one_constraint(in_pvar.abstracted_shape[cur_ndim - 1 - j])
                        else:
                            raise ValueError('Cannot broadcast shape {} to {}.'.format(in_pvar.real_shape, out_real_shape))
                        is_necessary = True
                    else:
                        self.shape_constraints.add_eq_constraint(in_pvar.abstracted_shape[cur_ndim - 1 - j], out_abstracted_shape[ndim - 1 - j])
                if is_necessary:
                    in_pvars[idx] = self._broadcast_to(in_pvar, out_abstracted_shape, out_real_shape)

            op_index = len(self.op_list)
            # TODO: life analysis

            can_cast = can_cast1 if _should_use_min_scalar(in_pvars) else can_cast2
            for in_dtypes, out_dtypes, op in ufunc._ops:
                in_dtypes = [numpy.dtype(t) for t in in_dtypes]
                out_dtypes = [numpy.dtype(t) for t in out_dtypes]
                if can_cast(in_pvars, in_dtypes):
                    ret = list()
                    for i in range(nout):
                        if i >= len(out_pvars):
                            out_pvar = self._make_new_param(ndim, out_dtypes[i], out_abstracted_shape, out_real_shape)
                            out_pvars.append(out_pvar)
                            ret.append(out_pvar)
                        elif numpy.can_cast(out_dtypes[i], out_pvars[i].dtype, 'same_kind'):
                            out_pvar = out_pvars[i]
                            ret.append(out_pvar)
                        else:
                            raise TypeError(
                                'output (typecode \'{}\') could not be coerced '
                                'to provided output parameter (typecode \'{}\') '
                                'according to the casting rule '
                                '"same_kind"'.format(
                                    out_dtypes[i].char, out_pvars[i].dtype.char))
                    in_params = [(in_dtypes[i], 'in{}'.format(i)) for i, _ in enumerate(in_pvars)]
                    out_params = [(out_dtypes[i], 'out{}'.format(i)) for i, _ in enumerate(out_pvars)]
                    subm = _Submodule(ufunc, in_params, out_params, op)
                    self.ufunc_submodules[subm.key()] = subm
                    self._add_ufunc_op(ufunc, in_pvars, out_pvars, in_dtypes, out_dtypes)
                    return ret[0] if len(ret) == 1 else tuple(ret)
            raise TypeError('Invalid type cast in \'{}\' in_dtypes={}'.format(
                ufunc.name, [v.dtype for v in in_pvars]))

    def call_reduction(self, fusion_op, args, kwargs):
        if len(args) != 1:
            raise NotImplementedError('Reduction for multiple arguments is not implemented. args: {}'.format(args))
        arg = args[0]
        if not isinstance(arg, _FusionXVarArray):
            raise ValueError('Cannot reduce {}.'.format(type(arg)))
        axis = kwargs.pop('axis', None)
        dtype = kwargs.pop('dtype', None)
        keepdims = kwargs.pop('keepdims', False)

        if axis is None and not keepdims:
            axis = 0
            out_abstracted_shape = ()
            out_real_shape = ()
        else:
            out_abstracted_shape = list(arg.abstracted_shape)
            out_real_shape = list(arg.real_shape)
            if keepdims:
                reduce_axis = [axis] if axis else list(range(arg.ndim))
                for i in reduce_axis:
                    out_abstracted_shape[i] = '1'
                    out_real_shape[i] = 1
                if axis is None:
                    axis = 0
            else:
                out_abstracted_shape.pop(axis)
                out_real_shape.pop(axis)


            out_abstracted_shape = tuple(out_abstracted_shape)
            out_real_shape = tuple(out_real_shape)

        if dtype is not None:
            dtype = get_dtype(dtype).type

        op_index = len(self.op_list)
        # TODO: life analysis

        out_ndim = len(out_abstracted_shape)

        for op in fusion_op._ops:
            (in_dtype,), (out_dtype,), _ = op
            if numpy.can_cast(arg.dtype.type, in_dtype):
                if dtype is not None:
                    if numpy.can_cast(out_dtype, dtype):
                        out_dtype = dtype
                    else:
                        continue
                out_dtype = numpy.dtype(out_dtype)
                out_pvar = self._make_new_param(out_ndim, out_dtype, out_abstracted_shape, out_real_shape, True)
                self._add_reduction_op(fusion_op, arg, out_pvar, axis, op)
                return out_pvar
        raise TypeError('No viable type cast of {}, arg_type={}'.format(
            fusion_op.name, arg.dtype))

    def prepare_reduction_submodules(self):
        for op in self.reduction_op_list:
            op._add_to_reduction_submodules(self)

    def _get_output(self):
        out = None if self.no_return or self.return_size < 0 else list(None for _ in range(self.return_size))
        for a in self.param_list:
            if a.is_output:
                if self.return_size < 0:
                    out = a.ndarray
                else:
                    assert out[a.output_order] is None
                    out[a.output_order] = a.ndarray

        if isinstance(out, list):
            out = tuple(out)
        return out

    def _get_inout_args(self):
        params = list()
        indexers = list()
        block_strides = list()
        kern_size = 1

        for a in self.param_list:
            kern_size = max(kern_size, a.size)
            if isinstance(a, _FusionXVarArray):
                params.append(a.ndarray)
                indexers.append(Indexer(a.ndarray.shape))
            elif isinstance(a, _FusionXVarScalar):
                if a.const_value is not None:
                    continue
                params.append(0)
            else:
                raise TypeError('Unknown type {}.'.format(type(a)))
        for op in self.reduction_op_list:
            in_pvar = op.in_pvar
            out_pvar = op.out_pvar
            block_size = 512
            reduce_block_size = max(1, in_pvar.size // out_pvar.size)
            contiguous_size = min(in_pvar.ndarray.strides[-1], 32)
            block_stride = max(contiguous_size, block_size // reduce_block_size)
            block_stride = internal.clp2(block_stride // 2 + 1)  # floor
            block_strides.append(block_stride)
            kern_size = max(kern_size, (out_pvar.size + block_stride - 1) // block_stride * block_size)

        return params + indexers + block_strides, kern_size

    def _get_cuda_params(self):
        params = list()
        indexers = list()
        block_strides = list()

        for a in self.param_list:
            if isinstance(a, _FusionXVarArray):
                if a.ndarray is None:
                    raise ValueError('ndarray must be malloced before launching kernel.')
                # a.ndim can differ from a.ndarray.ndim due to reduce_dims
                ndim = a.ndarray.ndim
                params.append('CArray<{}, {}> {}'.format(_dtype_to_ctype[a.dtype], ndim, _get_pvar_param_name(a)))
                indexers.append('CIndexer<{}> {}'.format(ndim, _get_indexer_name(a)))
            elif isinstance(a, _FusionXVarScalar):
                if a.const_value is not None:
                    continue
                params.append('{} {}'.format(_dtype_to_ctype[a.dtype], _get_pvar_param_name(a)))
            else:
                raise TypeError('Unknown type {}.'.format(type(a)))

        for i in range(self.block_count):
            block_strides.append('int block_stride{}'.format(i))

        return params + indexers + block_strides

    def _get_fusionx_info(self, func, args, name):
        self.name = name
        in_pvars = [self._make_input_param(arg, idx) for idx, arg in enumerate(args)]
        return_value = func(*in_pvars)

        # size >= 0 for tuples
        return_size = -1
        no_return = False
        if isinstance(return_value, tuple):
            return_size = len(return_value)
            for idx, pvar in enumerate(return_value):
                pvar.is_output = True
                pvar.output_order = idx
        elif isinstance(return_value, _FusionXVarArray):
            return_value.is_output = True
            return_value.output_order = 0
        elif return_value is None:
            no_return = True
        else:
            raise TypeError('Invalid return type {}.'.format(type(return_value)))

        self.return_size = -1
        self.no_return = no_return
        self.cuda_body = '    __syncthreads();'.join([op.code() for op in self.op_list])

        return self, self.shape_constraints

    def _reset_vals(self):
        for pvar in self.param_list:
            if isinstance(pvar, _FusionXVarArray):
                pvar.ndarray = None

    def _set_real_shape(self, shape_map):
        for pvar in self.param_list:
            if isinstance(pvar, _FusionXVarArray):
                ndim = pvar.ndim
                real_shape = [None for _ in range(ndim)]
                for i in range(ndim):
                    assert pvar.abstracted_shape[i] in shape_map
                    real_shape[i] = shape_map[pvar.abstracted_shape[i]]
                pvar.real_shape = tuple(real_shape)
                pvar.set_size()

    def _set_ndarray(self, args):
        for pvar in self.param_list_base:
            if isinstance(pvar, _FusionXVarArray):
                if pvar.is_input:
                    pvar.ndarray = args[pvar.input_order]
                else:
                    pvar.ndarray = ndarray(pvar.real_shape, pvar.dtype)

    def _set_updated(self):
        for pvar in self.param_list:
            pvar.set_updated()

    def _broadcast(self):
        for pvar in self.param_list:
            if isinstance(pvar, _FusionXVarArray):
                if pvar.broadcasted_from is not None:
                    assert pvar.broadcasted_from.ndarray is not None
                    pvar.ndarray = broadcast_to(pvar.broadcasted_from.ndarray, pvar.real_shape)

    def _rotate(self):
        pass

    def _reduce_dims(self):
        for pvar in self.param_list:
            if pvar.ndim < 2:
                continue
            pvar.ndarray = _reduce_dims_core(pvar.ndarray)

    def _get_kernel(self):
        code = ''
        for submodule in self.ufunc_submodules.values():
            code += submodule.code()
        for submodule_code in self.reduction_submodules.values():
            code += submodule_code
        code += string.Template('''
extern "C" __global__ void ${name}(${params}) {
${body}
}''').substitute(
            name=self.name,
            params=', '.join(self.cuda_params),
            body=self.cuda_body)

        module = compile_with_cache(code)
        return module.get_function(self.name)

    def exec(self, shape_map, *args):
        self._reset_vals()
        self._set_real_shape(shape_map)
        self._set_ndarray(args)
        self._set_updated()
        self._broadcast()
        ret = self._get_output()
        self._rotate()
        self._reduce_dims()
        self.prepare_reduction_submodules()
        inout_args, kern_size = self._get_inout_args()

        self.cuda_params = self._get_cuda_params()
        if len(inout_args) != len(self.cuda_params):
            raise AssertionError('''Length of inout_args is not consistent with length of cuda_params.
len(inout_args): {}, len(cuda_params): {}
inout_args: {}, cuda_params: {}'''.format(len(inout_args), len(self.cuda_params), inout_args, self.cuda_params))

        kern = self._get_kernel()
        kern.linear_launch(kern_size, inout_args, 0, 512)
        return ret

cpdef ndarray _reduce_dims_core(ndarray array):
    cdef vector.vector[Py_ssize_t] shape, strides, new_shape, new_strides
    cdef Py_ssize_t back
    shape = array._shape
    strides = array._strides
    new_shape.push_back(shape.back())
    shape.pop_back()
    new_strides.push_back(strides.back())
    strides.pop_back()
    while len(shape) > 0:
        if new_shape.back() * new_strides.back() == strides.back():
            new_shape[len(new_shape) - 1] *= shape.back()
        else:
            new_shape.push_back(shape.back())
            new_strides.push_back(strides.back())
        shape.pop_back()
        strides.pop_back()
    new_shape = new_shape[::-1]
    new_strides = new_strides[::-1]
    return ndarray(new_shape, array.dtype, array.data, new_strides)

class FusionX(object):
    def __init__(self, func, name=None):
        self.func = func
        self.name = name or func.__name__
        # self._memo1[param_info][constraints] = fusionx_info
        # self._memo2[param_info][param_shape] = fusionx_info
        self._memo1 = dict()
        self._memo2 = dict()

    def __repr__(self):
        return '<FusionX name={}>'.format(self.name)

    def __call__(self, *args):
        if _is_fusingx():
            # Inner function of composition of multiple fused functions.
            return self.func(*args)

        exec_cupy = False
        for a in args:
            if isinstance(a, cupy.core.ndarray):
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
        cdef list shape_info = []
        for arg in args:
            if isinstance(arg, cupy.core.ndarray):
                params_info.append(arg.dtype.char)
                params_info.append(arg.ndim)
                shape_info.append(arg.shape)
            elif isinstance(arg, numpy.generic):
                params_info.append(arg.dtype.char)
            elif arg is None:
                params_info.append(None)
            elif isinstance(arg, float):
                params_info.append('d')
            elif isinstance(arg, six.integer_types):
                params_info.append('l')
            elif isinstance(arg, bool):
                params_info.append('?')
            elif isinstance(arg, complex):
                params_info.append('D')
            else:
                raise TypeError('Unsupported input type {}.'.format(type(arg)))

        args = [arg for arg in args if arg is not None]

        cdef tuple param_key = tuple(params_info)
        cdef tuple shape_key = tuple(shape_info)

        map = self._get_shape_map(args)
        fusionx_info = None
        if param_key in self._memo1:
            if shape_key in self._memo2[param_key]:
                fusionx_info = self._memo2[param_key][shape_key]
            else:
                for constraints in self._memo1[param_key]:
                    if self._satisfy(constraints, map):
                        fusionx_info = self._memo1[param_key][constraints]
                        self._memo2[param_key][shape_key] = fusionx_info
                        break
        else:
            self._memo1[param_key] = dict()
            self._memo2[param_key] = dict()
        if fusionx_info is None:
            kernel, constraints = self._setup_fusionx_info(args)
            self._memo1[param_key][constraints] = kernel
            self._memo2[param_key][shape_key] = kernel
            fusionx_info = kernel

        return fusionx_info.exec(map, *args)

    def _setup_fusionx_info(self, args):
        history = _FusionXHistory()
        # avoid conflict with _thread_local.history
        _thread_local.historyx = history
        ret = history._get_fusionx_info(self.func, args, self.name)
        _thread_local.historyx = None
        return ret

    def _get_shape_map(self, args):
        map = dict()
        for idx, arg in enumerate(args):
            if not isinstance(arg, cupy.core.ndarray):
                continue
            ndim = arg.ndim
            shape = arg.shape
            for i in range(ndim):
                map['v{}.{}'.format(idx, i)] = shape[i]
        return map

    def _satisfy(self, constraints, map):
        for a, b in constraints.eq_constraints:
            if map[a] != map[b]:
                return False
        for a in constraints.one_constraints:
            if map[a] != 1:
                return False
        return True

def fusex(*args, **kwargs):
    def wrapper(f, kernel_name=None):
        return FusionX(f, kernel_name)

    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return functools.update_wrapper(wrapper(args[0]), args[0])
    else:
        return lambda f: functools.update_wrapper(
            wrapper(f, *args, **kwargs), f)

def reduction_code_template(name, in_pvar, out_pvar, reduce_expr, identity):
    """
    statically determined params
        name: name of reduce_func
        in_ndim: ndim of input CArray
        out_ndim: ndim of output CArray
        block_size: needed because sdata must be fixed-size
        type:
        reduce_expr
        identity:
    """
    if in_pvar.ndarray is None or out_pvar.ndarray is None:
        raise ValueError('ndarray must be malloced before launching kernel.')
    return string.Template('''
#define REDUCE${name}(a, b) (${reduce_expr})
#define _REDUCE${name}(offset) sdata[tid] = REDUCE${name}(sdata[tid], sdata[tid + offset]);

__device__ void reduce${name}(CArray<${in_type}, ${in_ndim}> in_arr, CArray<${out_type}, ${out_ndim}> out_arr,
        CIndexer<${in_ndim}> in_ind, CIndexer<${out_ndim}> out_ind, int block_stride) {
    extern __shared__ ${out_type} sdata[${block_size}];
    unsigned int tid = threadIdx.x;
    int _J = tid >> __popc(block_stride - 1);
    ptrdiff_t _j = (ptrdiff_t)_J * out_ind.size();
    int J_stride = ${block_size} >> __popc(block_stride - 1);
    ptrdiff_t j_stride = (ptrdiff_t)J_stride * out_ind.size();

    for (ptrdiff_t _i = (ptrdiff_t)blockIdx.x * block_stride; _i < out_ind.size(); _i += (ptrdiff_t)gridDim.x * block_stride) {
        ${out_type} s = (${out_type})${identity};
        ptrdiff_t i = _i + (tid & (block_stride - 1));
        for (ptrdiff_t j = i + _j; j < in_ind.size(); j += j_stride) {
            in_ind.set(j);
            ${in_type} &a = in_arr[in_ind.get()];
            s = REDUCE${name}(s, (${out_type})a);
        }
        sdata[tid] = s;
        __syncthreads();
        for (unsigned int block = ${block_size} / 2; block >= block_stride; block >>= 1) {
            if (tid < block) {
                _REDUCE${name}(block);
            }
            __syncthreads();
        }
        if (tid < block_stride) {
            s = sdata[tid];
        }
        if (tid < block_stride && i < out_ind.size()) {
            out_ind.set(i);
            ${out_type} &a = out_arr[out_ind.get()];
            a = s;
        }
        __syncthreads();
    }
}
''').substitute(
        name=name,
        in_ndim=in_pvar.ndarray.ndim,
        in_type=_dtype_to_ctype[in_pvar.dtype],
        out_ndim=out_pvar.ndarray.ndim,
        out_type=_dtype_to_ctype[out_pvar.dtype],
        block_size=512,
        reduce_expr=reduce_expr,
        identity=identity)
