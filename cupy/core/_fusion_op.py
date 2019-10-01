from cupy.core._fusion_variable import _FusionCudaArray
from cupy.core._fusion_variable import _FusionVariableSet
from cupy.core import _fusion_thread_local
from cupy.core._fusion_emit_code import _CodeBlock
from cupy.core._fusion_device_func import _SubmoduleUfunc
from cupy.core._fusion_device_func import _SubmoduleReduction


# Returns a CUDA code: setting a raw index to indexers.
def _emit_set_index_code(indexed_params, tid):
    _fusion_thread_local.check_not_runtime()
    assert isinstance(indexed_params, _FusionVariableSet)

    return _CodeBlock(*[
        p.format('${indexer}.set(${tid});', tid=tid)
        for p in indexed_params
    ])


# Returns a tuple of size 2.
# 1. CUDA code: declaring local variables.
# 2. The set of arrays which require indexer.
def _emit_declaration_code(params, in_params):
    _fusion_thread_local.check_not_runtime()

    indexed_arrays = _FusionVariableSet()
    code = []
    for var in params:
        if var in in_params:
            if isinstance(var, _FusionCudaArray):
                indexed_arrays.add(var)
                f = '${type} ${lvar} = ${var}[${indexer}.get()];'
            else:
                f = '${type} ${lvar} = ${var};'
        else:
            f = '${type} ${lvar};'
        code.append(var.format(f))

    return _CodeBlock(*code), indexed_arrays


# Returns a tuple of size 2.
# 1. CUDA code: writing the results of operations back to global memory.
# 2. The set of arrays which require indexer.
def _emit_after_operation(out_params):
    _fusion_thread_local.check_not_runtime()

    indexed_arrays = _FusionVariableSet()
    codes = []
    for var in out_params:
        if isinstance(var, _FusionCudaArray):
            indexed_arrays.add(var)
            f = '${var}[${indexer}.get()] = ${lvar};'
        else:
            f = '${var} = ${lvar};'
        codes.append(var.format(f))

    return _CodeBlock(*codes), indexed_arrays


class _FusionElementwiseOp(object):
    """Ufunc or elementwise kernel with types.

    Attributes:
        ops (list of _SubmoduleUfunc): The submodules.
        params (_FusionVariableSet): The set of parameters the operation uses.
        in_params (_FusionVariableSet): The paramters the operation looks up.
        out_params (_FusionVariableSet): The parameters the operation mutates.
        forget_params (_FusionVariableSet): The parameters which is not used
            after this operation. This variable is set to empty set initially,
            and will be updated in optimization phase.
        ashape (tuple of _AbstractDim): The shape.
    """

    def __init__(self, op, in_params, out_params, ashape):
        # The `in_params` and `out_params` should be already broadcasted to
        # `ashape`, but they don't guarantee to be exactly same as
        # `param.ashape`.

        _fusion_thread_local.check_not_runtime()
        assert isinstance(op, _SubmoduleUfunc)
        assert isinstance(in_params, list)
        assert isinstance(out_params, list)
        assert isinstance(ashape, tuple)

        self.ops = [op]
        self.in_params = _FusionVariableSet(*in_params)
        self.out_params = _FusionVariableSet(*out_params)
        self.ashape = ashape

    @property
    def params(self):
        """Returns the set of all variable the loop uses.
        """
        res = _FusionVariableSet()
        for op in self.ops:
            res += _FusionVariableSet(*op.in_params)
            res += _FusionVariableSet(*op.out_params)
        return res

    def emit_code(self):
        _fusion_thread_local.check_not_runtime()

        declaration, s1 = _emit_declaration_code(self.params, self.in_params)
        operation = _CodeBlock(*[op.emit_call_code() for op in self.ops])
        after_operation, s2 = _emit_after_operation(self.out_params)
        index_name = 'i'
        indexed_array = s1 + s2
        indexer_name = next(iter(indexed_array)).indexer_name
        indexer_setup = _emit_set_index_code(indexed_array, index_name)

        return _CodeBlock(
            'CUPY_FOR(${index_name}, ${indexer}.size()) {', [
                *indexer_setup.codes,
                *declaration.codes,
                *operation.codes,
                *after_operation.codes],
            '}',
            index_name=index_name,
            indexer=indexer_name)


class _FusionReductionOp(object):
    def __init__(self, reduce_op, in_param, out_param, axis):
        """Reduction operation.

        Attributes:
            reduce_op(_SubmodleReduction): The submodule.
            in_param(_FusionCudaArray): The input variable.
            out_param(_FusionCudaArray): The output variable.
            reduce_identity(int or float): The identity value of the reduction.
            axis(tuple of int): The axis to be reduced.
        """
        _fusion_thread_local.check_not_runtime()
        assert isinstance(reduce_op, _SubmoduleReduction)
        assert isinstance(in_param, _FusionCudaArray)
        assert isinstance(out_param, _FusionCudaArray)
        assert isinstance(axis, tuple)
        assert all([0 <= x < in_param.ndim for x in axis])

        self.reduce_op = reduce_op
        self.in_params = _FusionVariableSet(in_param)
        self.out_params = _FusionVariableSet(out_param)
        self.block_stride_name = 'block_stride_' + reduce_op.name
        self.axis = axis

        self.premap_op = None
        self.postmap_op = None

    @property
    def params(self):
        return self.in_params + self.out_params

    def emit_code(self):
        _fusion_thread_local.check_not_runtime()
        assert len(self.in_params) == 1
        assert len(self.out_params) == 1
        in_param = list(self.in_params)[0]
        out_param = list(self.out_params)[0]
        params = ', '.join([
            in_param.var_name,
            out_param.var_name,
            in_param.indexer_name,
            out_param.indexer_name,
        ])
        return _CodeBlock(
            '${name}(${params}, ${block_stride});',
            name=self.reduce_op.name,
            params=params,
            block_stride=self.block_stride_name)
