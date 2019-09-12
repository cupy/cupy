# -*- coding: utf-8 -*-
import functools
import six
import string

import copy
import cupy
from libcpp cimport vector
from cupy.core.core cimport ndarray, Indexer
from cupy.core._routines_manipulation cimport broadcast_to
from cupy.core cimport _routines_manipulation as _manipulation
from cupy.core import _kernel
from cupy.core cimport internal
from cupy.core._dtype import get_dtype
from cupy.core._kernel import _is_fusingx
from cupy.core.fusion cimport _acceptable_types, _kind_score, _dtype_to_ctype
from cupy.core.fusion import _Submodule
import numpy
from cupy.core.core cimport compile_with_cache
from cupy import util

_thread_local = _kernel._thread_local

def _call_ufunc(fusion_op, *args, **kwargs):
    return _thread_local.historyx.call_ufunc(fusion_op, args, kwargs)

def _call_reduction(fusion_op, *args, **kwargs):
    return _thread_local.historyx.call_reduction(fusion_op, args, kwargs)

# 関数の引数の宣言で使う変数名。__repr__でも使われる
def _get_pvar_param_name(pvar):
    if isinstance(pvar, _FusionXVarArray):
        return 'a{}_{}'.format(pvar.index, pvar.dup)
    elif isinstance(pvar, _FusionXVarScalar):
        return 'a{}'.format(pvar.index)
    else:
        raise TypeError('Unknown type {}.'.format(type(pvar)))

# 関数内で各スレッドが扱うローカルな変数に使う変数名
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
        self.is_input = False
        # outputにはできない
        self.input_order = None
        # 実はconst_value is Noneとis_inputは同じ
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
        # indexが同じなら_FusionXVarScalarは区別可能
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
    def __init__(self, index, ndim, dtype, init_abstracted_shape=False):
        # broadcast, rotate, subscriptしてできる変数間は同じindexを共有する
        self.index = index
        # _compress_pvarsのときにindexがnew_indexで上書きされる
        self.new_index = index
        # _compress_pvarsのときに設定される
        # broadcast, rotate, subscriptでできる変数用のidentifier
        self.dup = None
        self.ndim = ndim
        self.dtype = dtype
        self.is_input = False
        # 入力の何番目か。input_order is not Noneとis_inputは同じ
        self.input_order = None
        self.is_output = False
        self.output_order = None
        self.broadcasted_from = None
        self.rotated_from = None
        # どの軸でreduceしたか
        self.axis = None
        self.indexed_from = None
        # どのkeyでsubscriptしたか。int, tuple, sliceのみ対応
        self.index_key = None
        self.abstracted_shape = tuple('v{}.{}'.format(self.index, i) for i in range(self.ndim)) if init_abstracted_shape else None

        # resettable
        self.size = None
        self.real_shape = None
        self.ndarray = None

    def reset(self):
        self.size = None
        self.real_shape = None
        self.ndarray = None

    # _compress_pvarsのときに、その変数のindexが入力に関わるかを判定。そのままindexで判定したほうが楽か
    def _is_input_recursive(self):
        ret = self.is_input
        if self.indexed_from is not None:
            ret |= self.indexed_from._is_input_recursive()
        return ret

    # exec内_set_real_shapeのときに呼ばれる
    def set_size(self):
        self.size = internal.prod(self.real_shape)

    # これもexec内
    def rotate(self):
        axis_permutes = list(self.axis)
        for i in range(self.ndim):
            if i not in self.axis:
                axis_permutes.append(i)
        axis_permutes = tuple(axis_permutes)
        self.ndarray = _manipulation._transpose(self.rotated_from.ndarray, axis_permutes)

    # これもexec内
    def subscript(self):
        if self.ndarray is not None:
            return
        if self.indexed_from is not None:
            self.indexed_from.subscript()
            self.ndarray = self.indexed_from.ndarray[self.index_key]

    def _data_base(self):
        if self.indexed_from is not None:
            return self.indexed_from._data_base()
        return self

    def _is_invalid_inplace(self, other):
        if not isinstance(other, _FusionXVarArray):
            return False
        if self.indexed_from is None and other.indexed_from is None:
            return False
        return self._data_base() == other._data_base()

    def __repr__(self):
        return '<_FusionXVarArray name={}, dtype={}, ndim={}, abstracted_shape={}>'.format(_get_pvar_param_name(self), self.dtype, self.ndim, self.abstracted_shape)

    def __eq__(self, other):
        if isinstance(other, _FusionXVarArray):
            return self.index == other.index and self.abstracted_shape == other.abstracted_shape and \
                self.broadcasted_from == other.broadcasted_from and self.rotated_from == other.rotated_from and \
                self.indexed_from == other.indexed_from and self.index_key == other.index_key
        return False

    def __hash__(self):
        return hash((self.index, self.abstracted_shape, self.broadcasted_from, self.rotated_from, self.indexed_from, self.index_key))

    def _is_readonly(self):
        return self.broadcasted_from is not None or self.rotated_from is not None

    def __iadd__(self, other):
        if self._is_readonly():
            raise ValueError('output array is read-only')
        if self._is_invalid_inplace(other):
            raise ValueError('This kind of inplace operation is not supported.')
        return cupy.add(self, other, self)

    def __isub__(self, other):
        if self._is_readonly():
            raise ValueError('output array is read-only')
        if self._is_invalid_inplace(other):
            raise ValueError('This kind of inplace operation is not supported.')
        return cupy.subtract(self, other, self)

    def __imul__(self, other):
        if self._is_readonly():
            raise ValueError('output array is read-only')
        if self._is_invalid_inplace(other):
            raise ValueError('This kind of inplace operation is not supported.')
        return cupy.multiply(self, other, self)

    def __idiv__(self, other):
        if self._is_readonly():
            raise ValueError('output array is read-only')
        if self._is_invalid_inplace(other):
            raise ValueError('This kind of inplace operation is not supported.')
        return cupy.divide(self, other, self)

    def __itruediv__(self, other):
        if self._is_readonly():
            raise ValueError('output array is read-only')
        if self._is_invalid_inplace(other):
            raise ValueError('This kind of inplace operation is not supported.')
        return cupy.true_divide(self, other, self)

    def __ifloordiv__(self, other):
        if self._is_readonly():
            raise ValueError('output array is read-only')
        if self._is_invalid_inplace(other):
            raise ValueError('This kind of inplace operation is not supported.')
        return cupy.floor_divide(self, other, self)

    def __imod__(self, other):
        if self._is_readonly():
            raise ValueError('output array is read-only')
        if self._is_invalid_inplace(other):
            raise ValueError('This kind of inplace operation is not supported.')
        return cupy.remainder(self, other, self)

    def __ipow__(self, other):
        if self._is_readonly():
            raise ValueError('output array is read-only')
        if self._is_invalid_inplace(other):
            raise ValueError('This kind of inplace operation is not supported.')
        return cupy.power(self, other, self)

    def __ilshift__(self, other):
        if self._is_readonly():
            raise ValueError('output array is read-only')
        if self._is_invalid_inplace(other):
            raise ValueError('This kind of inplace operation is not supported.')
        return cupy.left_shift(self, other, self)

    def __irshift__(self, other):
        if self._is_readonly():
            raise ValueError('output array is read-only')
        if self._is_invalid_inplace(other):
            raise ValueError('This kind of inplace operation is not supported.')
        return cupy.right_shift(self, other, self)

    def __iand__(self, other):
        if self._is_readonly():
            raise ValueError('output array is read-only')
        if self._is_invalid_inplace(other):
            raise ValueError('This kind of inplace operation is not supported.')
        return cupy.bitwise_and(self, other, self)

    def __ior__(self, other):
        if self._is_readonly():
            raise ValueError('output array is read-only')
        if self._is_invalid_inplace(other):
            raise ValueError('This kind of inplace operation is not supported.')
        return cupy.bitwise_or(self, other, self)

    def __ixor__(self, other):
        if self._is_readonly():
            raise ValueError('output array is read-only')
        if self._is_invalid_inplace(other):
            raise ValueError('This kind of inplace operation is not supported.')
        return cupy.bitwise_xor(self, other, self)

    def __getitem__(self, key):
        return _thread_local.historyx._make_indexed_param(self, key)

class _FusionXOp(object):
    def __init__(self, ufunc, in_pvars, out_pvars, in_dtypes, out_dtypes):
        self.ufunc = ufunc
        self.in_pvars = in_pvars
        self.out_pvars = out_pvars
        self.in_dtypes = in_dtypes
        self.out_dtypes = out_dtypes
        # in_pvars + out_pvarsの全変数はこのabst_shapeになっているはず
        self.abstracted_shape = out_pvars[0].abstracted_shape

        for pvar in self.in_pvars + self.out_pvars:
            if isinstance(pvar, _FusionXVarArray):
                _thread_local.historyx.abstracted_shape_uf.unite(pvar.abstracted_shape, self.abstracted_shape)

    def __repr__(self):
        in_pvars = ', '.join([_get_pvar_param_name(a) for a in self.in_pvars])
        out_pvars = ', '.join([_get_pvar_param_name(a) for a in self.out_pvars])
        return '<_FusionXOp name={}, in_pvars={}, out_pvars={}>'.format(self.ufunc.name, in_pvars, out_pvars)

    def _get_indexer_setup(self):
        ret = ''
        used = set()
        for a in self.in_pvars + self.out_pvars:
            if a in used:
                continue
            used.add(a)
            if isinstance(a, _FusionXVarArray):
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
            pre_conversion += '            {type} {name} = ({type}) {name}_;\n'.format(
                type=_dtype_to_ctype[d], name=_get_pvar_decl_name(a))
        operation += '            {}({});\n'.format(self.ufunc.name, ', '.join([_get_pvar_decl_name(a) for a in self.in_pvars + self.out_pvars]))
        for a in self.out_pvars:
            post_conversion += '            {name}_ = ({type}) {name};\n'.format(
                type=_dtype_to_ctype[a.dtype], name=_get_pvar_decl_name(a))
        return '        {\n' + pre_conversion + operation + post_conversion + '        }\n'

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
    def __init__(self, func, in_pvar, out_pvar, op, reduction_index):
        self.func = func
        self.in_pvar = in_pvar
        self.out_pvar = out_pvar
        self.reduce_identity = func.identity
        _, _, (_, self.reduce_expr, _, _) = op
        self.reduction_index = reduction_index

    def __repr__(self):
        return '<_FusionXReductionOp name={}, in_pvar={}, out_pvar={}>'.format(self.func.name, self.in_pvar, self.out_pvar)

    # reductionのコードのみ実行時に追加される(reduce-dims後のndarrayのshapeに依存するので)
    def _get_reduction_code(self):
        return reduction_code_template(self.reduction_index, self.in_pvar, self.out_pvar, self.reduce_expr, self.reduce_identity)

    def code(self):
        return string.Template('''
    reduce${reduction_index}(${in_arr}, ${out_arr}, ${in_ind}, ${out_ind}, block_stride${reduction_index});
''').substitute(
            in_arr=_get_pvar_param_name(self.in_pvar),
            out_arr=_get_pvar_param_name(self.out_pvar),
            in_ind=_get_indexer_name(self.in_pvar),
            out_ind=_get_indexer_name(self.out_pvar),
            reduction_index = self.reduction_index)

# 同じabst_shapeのelementwise演算をmergeする
class _FusionXMergedOp(_FusionXOp):
    def __init__(self):
        self.in_pvars = list()
        self.in_pvars_used = set()
        self.out_pvars = list()
        self.out_pvars_used = set()
        self.in_dtypes = list()
        self.out_dtypes = list()
        self.ops = list()

    def add_op(self, fusionx_op):
        for in_pvar, in_dtype in zip(fusionx_op.in_pvars, fusionx_op.in_dtypes):
            if in_pvar in self.in_pvars_used:
                continue
            self.in_pvars_used.add(in_pvar)
            self.in_pvars.append(in_pvar)
            self.in_dtypes.append(in_dtype)
        for out_pvar, out_dtype in zip(fusionx_op.out_pvars, fusionx_op.out_dtypes):
            if out_pvar in self.out_pvars_used:
                continue
            self.out_pvars_used.add(out_pvar)
            self.out_pvars.append(out_pvar)
            self.out_dtypes.append(out_dtype)
        self.ops.append(fusionx_op)

    def size(self):
        return len(self.ops)

    def _get_operation(self):
        ret = ''
        for op in self.ops:
            ret += op._get_operation()
        return ret

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

class _UnionFind:
    def __init__(self):
        self.node = set()
        self.parent = dict()

    def add_node(self, v):
        if v not in self.node:
            self.node.add(v)
            self.parent[v] = v

    def root(self, v):
        if v == self.parent[v]:
            return v
        ret = self.root(self.parent[v])
        self.parent[v] = ret
        return ret

    def same(self, u, v):
        return self.root(u) == self.root(v)

    def unite(self, u, v):
        self.add_node(u)
        self.add_node(v)
        u = self.root(u)
        v = self.root(v)
        if u != v:
            self.parent[u] = v

class _FusionXHistory(object):
    def __init__(self):
        self.count = 0
        # base_abst_shape[index] = broadcast, rotate, subscriptする前のshape
        self.base_abstracted_shape = dict()
        # input argsの割り当て、mallocが必要な変数のリスト
        self.param_list_base = list()
        # param_list_used[pvar] = pvarの形で登録する。同一とみなせるpvarを一つで代用する
        # param_listに含まれるpvarの集合とparam_list_usedに含まれるpvarの集合は同一
        self.param_list_used = dict()
        self.param_list = list()
        self.shape_constraints = _ShapeConstraints()
        self.op_list = list()
        self.reduction_op_list = list()
        self.last_op = dict()

        self.abstracted_shape_uf = _UnionFind()

        self.ufunc_submodules = dict()
        # reduce_dims後のshapeがkey
        self.full_submodules = dict()
        # reduce_dims後のshapeがkey
        self.cuda_params_memo = dict()
        # reduction時のblock_stridesはコードに埋め込むのではなく関数の引数として渡す
        self.block_strides = list()

    def __repr__(self):
        return '<_FusionXHistory> op_list={}, param_list={}, shape_constraints={}'.format(self.op_list, self.param_list, self.shape_constraints)

    def _fresh_index(self):
        ret = self.count
        self.count += 1
        return ret

    # 新しくparamを生成する場合は必ずこの関数が介される。pvarがすでに生成されているものの場合、それを代用
    def _append_param(self, pvar, need_malloc=True):
        if pvar in self.param_list_used:
            return self.param_list_used[pvar]
        if isinstance(pvar, _FusionXVarArray):
            if need_malloc:
                self.base_abstracted_shape[pvar.index] = pvar.abstracted_shape
        if need_malloc:
            self.param_list_base.append(pvar)
        self.param_list_used[pvar] = pvar
        self.param_list.append(pvar)
        return pvar

    def _get_new_len(self, step, abstracted_shape0, real_shape0):
        raise NotImplementedError('TODO: implement')

    def _get_pvar(self, arg):
        if isinstance(arg, _FusionXVarScalar):
            return arg
        if isinstance(arg, six.integer_types +
                      (float, bool, complex, numpy.generic)):
            index = self._fresh_index()
            dtype = numpy.dtype(type(arg))
            pvar = _FusionXVarScalar(index, dtype, arg)
            return self._append_param(pvar)
        raise TypeError('Unsupported type {}'.format(type(arg)))

    def _make_input_param(self, arg, idx):
        if isinstance(arg, cupy.core.ndarray):
            index = self._fresh_index()
            ret = _FusionXVarArray(index, arg.ndim, arg.dtype, True)
            ret.real_shape = arg.shape
            ret.is_input = True
            ret.input_order = idx
            ret = self._append_param(ret)
        else:
            index = self._fresh_index()
            dtype = numpy.dtype(type(arg))
            ret = _FusionXVarScalar(index, dtype, None)
            ret.is_input = True
            ret.input_order = idx
            ret = self._append_param(ret)
        return ret

    def _broadcast_to(self, pvar, abstracted_shape, real_shape):
        ret = copy.copy(pvar)
        ret.ndim = len(abstracted_shape)
        ret.abstracted_shape = abstracted_shape
        ret.real_shape = real_shape
        ret.broadcasted_from = pvar
        # is_inputであるかどうか、broadcasted(rotated/indexed)_fromがNoneかどうかによってndarrayの割当方法が異なる
        ret.is_input = False
        ret.indexed_from = None
        return self._append_param(ret, False)

    def _make_rotated_param(self, pvar, axis):
        ret = copy.copy(pvar)
        ret.rotated_from = pvar
        ret.axis = axis
        ret.is_input = False
        ret.indexed_from = None
        return self._append_param(ret, False)

    def _make_new_param(self, ndim, dtype, abstracted_shape, real_shape):
        index = self._fresh_index()
        ret = _FusionXVarArray(index, ndim, dtype)
        ret.abstracted_shape = abstracted_shape
        ret.real_shape = real_shape
        return self._append_param(ret)

    def _add_ufunc_op(self, ufunc, in_pvars, out_pvars, in_dtypes, out_dtypes):
        op = _FusionXOp(ufunc, in_pvars, out_pvars, in_dtypes, out_dtypes)
        self.op_list.append(op)

    def _add_reduction_op(self, func, in_pvar, out_pvar, axis, op):
        # この時点でaxisを正規化。範囲外チェックはやっていない
        if axis is None or axis == tuple(range(len(axis))):
            pass
        else:
            # in_pvar.ndimであまりをとるべきだけどやってない
            in_pvar = self._make_rotated_param(in_pvar, axis)
        op = _FusionXReductionOp(func, in_pvar, out_pvar, op, len(self.reduction_op_list))
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

        # ここまででout_abst_shape, out_real_shapeが決定したので、各変数にbroadcastが必要かを次に調べる
        # broadcastが必要な場合、新たにpvarを作成
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

        # 生存解析用
        op_index = len(self.op_list)
        for in_pvar in in_pvars:
            self.last_op[in_pvar.index] = op_index

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
                # in_params, out_paramsは完全に_Submoduleのための変数
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
        out_pvar = kwargs.pop('out', None)
        if kwargs:
            raise ValueError('Unknown kwargs: {}.'.format(kwargs))
        if keepdims:
            raise NotImplementedError('keepdims is not supported.')

        if axis is None and not keepdims:
            out_abstracted_shape = ()
            out_real_shape = ()
        else:
            out_abstracted_shape = list(arg.abstracted_shape)
            out_real_shape = list(arg.real_shape)
            if isinstance(axis, int):
                axis = (axis,)
            for a in sorted(axis)[::-1]:
                out_abstracted_shape.pop(a)
                out_real_shape.pop(a)

            out_abstracted_shape = tuple(out_abstracted_shape)
            out_real_shape = tuple(out_real_shape)

        if dtype is not None:
            dtype = get_dtype(dtype).type

        # 生存解析用
        op_index = len(self.op_list)
        self.last_op[arg.index] = op_index

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
                if out_pvar is not None:
                    if out_pvar.real_shape != out_real_shape:
                        raise ValueError('Shape of specified output variable is not consistent with reduced shape.')
                else:
                    out_pvar = self._make_new_param(out_ndim, out_dtype, out_abstracted_shape, out_real_shape)
                self._add_reduction_op(fusion_op, arg, out_pvar, axis, op)
                return out_pvar
        raise TypeError('No viable type cast of {}, arg_type={}'.format(
            fusion_op.name, arg.dtype))

    def _append_reduction_submodules(self, key):
        if key in self.full_submodules:
            return
        code = self.ufunc_submodule_code
        for op in self.reduction_op_list:
            code += op._get_reduction_code()
        self.full_submodules[key] = code

    def _get_output(self):
        out = None if self.no_return or self.return_size < 0 else list(None for _ in range(self.return_size))
        for a in self.param_list:
            if isinstance(a, _FusionXVarArray) and a.is_output:
                if self.return_size < 0:
                    out = a.ndarray
                else:
                    assert out[a.output_order] is None
                    out[a.output_order] = a.ndarray

        if isinstance(out, list):
            out = tuple(out)
        return out

    # 関数の引数はこの戻り値の配列で決定される(順序もこの順)
    def _get_inout_args(self, args):
        params = list()
        indexers = list()
        block_strides = list()
        kern_size = 1

        for a in self.param_list:
            if isinstance(a, _FusionXVarArray):
                params.append(a.ndarray)
                indexers.append(Indexer(a.ndarray.shape))
                kern_size = max(kern_size, a.size)
            elif isinstance(a, _FusionXVarScalar):
                if a.is_input:
                    params.append(args[a.input_order])
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

    # 関数の引数の文字列
    def _get_cuda_params(self, key):
        if key in self.cuda_params_memo:
            return self.cuda_params_memo[key]
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

        for i in range(len(self.reduction_op_list)):
            block_strides.append('int block_stride{}'.format(i))

        ret = params + indexers + block_strides
        self.cuda_params_memo[key] = ret
        return ret

    # このkeyが同じなら同じメモリを割り当てて良い
    def _get_pvar_meminfo(self, pvar):
        return (self.base_abstracted_shape[pvar.index], pvar.dtype)

    def _compress_pvars(self):
        def get_last_op(pvar):
            return self.last_op.get(pvar.index, -1)

        # dead[pvar_meminfo] = [indexes]
        dead = dict()
        alive = set()
        # index_map[index] = new_index
        # broadcast/rotate/subscriptした後/する前の変数のnew_indexが変わったら他もまとめて変える必要があるため
        index_map = dict()
        for op_index, op in enumerate(self.op_list):
            if isinstance(op, _FusionXOp):
                in_pvars = op.in_pvars
                out_pvars = op.out_pvars
            elif isinstance(op, _FusionXReductionOp):
                in_pvars = [op.in_pvar]
                out_pvars = [op.out_pvar]
            else:
                raise TypeError('Unknown type {}.'.format(type(op)))

            for pvar in in_pvars:
                if not isinstance(pvar, _FusionXVarArray):
                    continue
                if pvar.index in index_map:
                    pvar.new_index = index_map[pvar.index]
                is_input = pvar.is_input
                # 入力の変数は死んだらダメ
                # 出力の変数もダメだけど、last_opの値的に絶対に死なない
                if pvar.broadcasted_from is not None:
                    is_input |= pvar.broadcasted_from.is_input
                if pvar.rotated_from is not None:
                    is_input |= pvar.rotated_from.is_input
                is_input |= pvar._is_input_recursive()
                if not is_input and get_last_op(pvar) <= op_index:
                    # pvar becomes dead
                    key = self._get_pvar_meminfo(pvar)
                    if key in dead:
                        dead[key].append(pvar.new_index)
                    else:
                        dead[key] = [pvar.new_index]
                    if pvar.new_index in alive:
                        alive.remove(pvar.new_index)
                else:
                    alive.add(pvar.new_index)
            for pvar in out_pvars:
                if not isinstance(pvar, _FusionXVarArray):
                    continue
                if pvar.index in index_map:
                    pvar.new_index = index_map[pvar.index]
                if pvar.new_index not in alive:
                    # assign new index
                    key = self._get_pvar_meminfo(pvar)
                    if key in dead and len(dead[key]) > 0:
                        pvar.new_index = dead[key].pop(0)
                    index_map[pvar.index] = pvar.new_index
                    alive.add(pvar.new_index)
        # update index
        for pvar in self.param_list:
            if isinstance(pvar, _FusionXVarArray):
                pvar.index = pvar.new_index
        param_list = list()
        param_list_base = list()
        param_list_used = dict()
        dup_count = dict()
        # set dup
        for pvar in self.param_list:
            if not isinstance(pvar, _FusionXVarArray):
                if pvar in param_list_used:
                    continue
                param_list_used[pvar] = pvar
                param_list.append(pvar)
                param_list_base.append(pvar)
                continue
            if pvar in param_list_used:
                if pvar.is_output:
                    param_list_used[pvar].is_output = True
                    param_list_used[pvar].output_order = pvar.output_order
                continue
            dup = dup_count.get(pvar.index, 0)
            dup_count[pvar.index] = dup + 1
            pvar.dup = dup
            param_list_used[pvar] = pvar
            param_list.append(pvar)
            if not pvar._is_readonly() and pvar.indexed_from is None:
                param_list_base.append(pvar)
        self.param_list = param_list
        self.param_list_base = param_list_base

        def update(pvar):
            if not isinstance(pvar, _FusionXVarArray):
                return pvar
            if pvar.rotated_from is not None:
                pvar.rotated_from = update(pvar.rotated_from)
            if pvar.broadcasted_from is not None:
                pvar.broadcasted_from = update(pvar.broadcasted_from)
            if pvar.indexed_from is not None:
                pvar.indexed_from = update(pvar.indexed_from)
            return param_list_used[pvar]

        # param_listのdupを全部更新しても、op内のpvarのdupが更新されているとは限らない
        # 実際、pvar BのindexをAのindexに揃えた場合、B.dupはNoneのままである。これを解消させるのが以下
        for op in self.op_list:
            if isinstance(op, _FusionXOp):
                for idx, pvar in enumerate(op.in_pvars):
                    op.in_pvars[idx] = update(pvar)
                for idx, pvar in enumerate(op.out_pvars):
                    op.out_pvars[idx] = update(pvar)
            elif isinstance(op, _FusionXReductionOp):
                op.in_pvar = update(op.in_pvar)
                op.out_pvar = update(op.out_pvar)
            else:
                assert False

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
                # outputを死なせないために
                self.last_op[pvar.index] = len(self.op_list)
        elif isinstance(return_value, _FusionXVarArray):
            return_value.is_output = True
            return_value.output_order = 0
            self.last_op[return_value.index] = len(self.op_list)
        elif return_value is None:
            no_return = True
        else:
            raise TypeError('Invalid return type {}.'.format(type(return_value)))

        self.return_size = return_size
        self.no_return = no_return

        self._compress_pvars()

        merged_op_list = list()
        prev = None
        merged = _FusionXMergedOp()

        for op in self.op_list:
            if isinstance(op, _FusionXReductionOp):
                if merged.size() > 0:
                    merged_op_list.append(merged)
                prev = None
                merged = _FusionXMergedOp()
                merged_op_list.append(op)
            elif isinstance(op, _FusionXOp):
                if prev is not None and not self.abstracted_shape_uf.same(prev.abstracted_shape, op.abstracted_shape):
                    if merged.size() > 0:
                        merged_op_list.append(merged)
                    merged = _FusionXMergedOp()
                merged.add_op(op)
                prev = op
            else:
                raise TypeError('Unknown op type {}.'.format(type(op)))
        if merged.size() > 0:
            merged_op_list.append(merged)

        # ufuncのsubmoduleは実行ごとに変化しないためこの時点で生成
        ufunc_submodule_code = ''
        for submodule in self.ufunc_submodules.values():
            ufunc_submodule_code += submodule.code()

        self.ufunc_submodule_code = ufunc_submodule_code
        self.cuda_body = '    __syncthreads();'.join([op.code() for op in merged_op_list])
        # print(self.cuda_body)
        return self, self.shape_constraints

    def _reset_vals(self):
        for pvar in self.param_list:
            if isinstance(pvar, _FusionXVarArray):
                if pvar.ndarray is not None:
                    if pvar.ndarray.shape != pvar.real_shape or pvar.ndarray.dtype != pvar.dtype:
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
                elif pvar.ndarray is None:
                    pvar.ndarray = ndarray(pvar.real_shape, pvar.dtype)

    def _broadcast(self):
        for pvar in self.param_list:
            if isinstance(pvar, _FusionXVarArray):
                if pvar.broadcasted_from is not None:
                    assert pvar.broadcasted_from.ndarray is not None
                    pvar.ndarray = broadcast_to(pvar.broadcasted_from.ndarray, pvar.real_shape)

    def _rotate(self):
        for pvar in self.param_list:
            if isinstance(pvar, _FusionXVarArray) and pvar.rotated_from is not None:
                pvar.rotate()

    def _subscript(self):
        for pvar in self.param_list:
            if isinstance(pvar, _FusionXVarArray):
                pvar.subscript()

    def _reduce_dims(self):
        shapes = []
        for pvar in self.param_list:
            if pvar.ndim < 2:
                continue
            pvar.ndarray = _reduce_dims_core(pvar.ndarray)
            shapes.append(pvar.ndarray.shape)
        # keyでreduction_submodulesのコードを管理
        key = tuple(shapes)
        return key

    def _get_kernel(self, cuda_params, key):
        return _cuda_compile(self.full_submodules[key], self.name, cuda_params, self.cuda_body)

    def exec(self, shape_map, *args):
        # TODO: put everything into single loop
        self._set_real_shape(shape_map)
        # 以下から_rotate()までndarrayの割当。順序はめちゃくちゃ重要
        self._reset_vals()
        self._set_ndarray(args)
        self._subscript()
        self._broadcast()
        ret = self._get_output()
        self._rotate()
        key = self._reduce_dims()
        self._append_reduction_submodules(key)
        inout_args, kern_size = self._get_inout_args(args)

        cuda_params = self._get_cuda_params(key)
        if len(inout_args) != len(cuda_params):
            raise AssertionError('''Length of inout_args is not consistent with length of cuda_params.
len(inout_args): {}, len(cuda_params): {}
inout_args: {}, cuda_params: {}'''.format(len(inout_args), len(cuda_params), inout_args, cuda_params))

        kern = self._get_kernel(', '.join(cuda_params), key)
        kern.linear_launch(kern_size, inout_args, 0, 512)
        return ret

    # Basic indexing
    def _make_indexed_param(self, pvar, key):
        assert not pvar._is_readonly()
        ret = copy.copy(pvar)
        ret.indexed_from = pvar
        ret.index_key = key
        # 再掲だがis_inputかどうかによってndarrayの割当方法が異なる。これはsubscriptによってndarrayを割り当てるのでFalse
        # indexed_fromがbroadcasted/rotatedとなることはないので、ret.broadcasted_from = Noneなどは不要
        ret.in_input = False
        abstracted_shape = list(pvar.abstracted_shape)
        real_shape = list(pvar.real_shape)
        if isinstance(key, int):
            if pvar.ndim == 0:
                raise IndexError('too many indices for array.')
            ret.ndim -= 1
            abstracted_shape.pop(0)
            real_shape.pop(0)
            ret.abstracted_shape = tuple(abstracted_shape)
            ret.real_shape = tuple(real_shape)
            return self._append_param(ret, False)
        elif isinstance(key, tuple):
            # TODO: support indexing when slice and int are mixed
            for k in key:
                if not isinstance(k, int):
                    raise IndexError('Cannot subscript by type {}.'.format(type(k)))
            if len(key) > pvar.ndim:
                raise IndexError('too many indices for array.')
            ret.ndim -= len(key)
            for _ in range(len(key)):
                abstracted_shape.pop(0)
                real_shape.pop(0)
            ret.abstracted_shape = tuple(abstracted_shape)
            ret.real_shape = tuple(real_shape)
            return self._append_param(ret, False)
        elif isinstance(key, slice):
            raise NotImplementedError('Indexing by slice is not supported. Here is the alternative: \
                make an indexed array by that slice first, and then include that array into the input.')
        else:
            raise TypeError('Indexing by unknown type {}.'.format(type(key)))

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

@util.memoize()
def _cuda_compile(code, name, cuda_params, cuda_body):
    code += 'extern "C" __global__ void {}({})'.format(
        name, cuda_params) + '{\n' + cuda_body + '}\n'
    # print(code)
    module = compile_with_cache(code)
    return module.get_function(name)

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

def reduction_code_template(reduction_index, in_pvar, out_pvar, reduce_expr, identity):
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
#define REDUCE${reduction_index}(a, b) (${reduce_expr})
#define _REDUCE${reduction_index}(offset) sdata_${out_type_char}${reduction_index}[tid] = REDUCE${reduction_index}(sdata_${out_type_char}${reduction_index}[tid], sdata_${out_type_char}${reduction_index}[tid + offset]);

__device__ void reduce${reduction_index}(CArray<${in_type}, ${in_ndim}> in_arr, CArray<${out_type}, ${out_ndim}> out_arr,
        CIndexer<${in_ndim}> in_ind, CIndexer<${out_ndim}> out_ind, int block_stride) {
    __shared__ ${out_type} sdata_${out_type_char}${reduction_index}[${block_size}];
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
            s = REDUCE${reduction_index}(s, (${out_type})a);
        }
        sdata_${out_type_char}${reduction_index}[tid] = s;
        __syncthreads();
        for (unsigned int block = ${block_size} / 2; block >= block_stride; block >>= 1) {
            if (tid < block) {
                _REDUCE${reduction_index}(block);
            }
            __syncthreads();
        }
        if (tid < block_stride) {
            s = sdata_${out_type_char}${reduction_index}[tid];
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
        reduction_index=reduction_index,
        in_ndim=in_pvar.ndarray.ndim,
        in_type=_dtype_to_ctype[in_pvar.dtype],
        out_ndim=out_pvar.ndarray.ndim,
        out_type=_dtype_to_ctype[out_pvar.dtype],
        out_type_char=out_pvar.dtype.char,
        block_size=512,
        reduce_expr=reduce_expr,
        identity=identity)
