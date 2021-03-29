from cupy._core import _fusion_variable
from cupy._core import _fusion_op


def _reduce_memory_access(ops):
    required_memories = set()

    for op in ops:
        for p in op.in_params + op.out_params:
            if p.memory.is_inout:
                required_memories.add(p.memory)

    for op in ops[::-1]:
        in_memories = set([p.memory for p in op.in_params])

        new_out_params = []
        for p in op.out_params:
            if p.memory in required_memories:
                new_out_params.append(p)
        op.out_params = _fusion_variable._VariableSet(*new_out_params)

        # TODO(asi1024): The following improvement can be applicable only
        # when the memory space is used at most once.
        # `required_memories -= out_memories`
        required_memories |= in_memories

    return [op for op in ops if len(op.out_params) > 0]


def _normalize_ashapes(ops, variables, shape_constraints):
    def normalize(shape):
        return tuple([shape_constraints.evaluate(d) for d in shape])

    for var in variables:
        var.ashape = normalize(var.ashape)

    for op in ops:
        if isinstance(op, _fusion_op._ElementwiseTraceOp):
            op.ashape = normalize(op.ashape)


def _fuse_two_ops(op1, op2):
    """Returns a fused Op if the two ops can be fused, and ``None`` otherwise.
    """
    # TODO(asi1024): Supoort reduction postmap.
    if not isinstance(op1, _fusion_op._ElementwiseTraceOp):
        return None

    # TODO(asi1024): Supoort reduction premap.
    if not isinstance(op2, _fusion_op._ElementwiseTraceOp):
        return None

    if op1.ashape != op2.ashape:
        return None

    new_in_params = op1.in_params + (op2.in_params - op1.out_params)
    new_out_params = op1.out_params + op2.out_params
    for in_param in new_in_params:
        for out_param in new_out_params:
            # Checks if two arrays may share the same memory space.
            if in_param.memory == out_param.memory and in_param != out_param:
                return None

    op1.ops.extend(op2.ops)
    op1.in_params = new_in_params
    op1.out_params = new_out_params
    return op1


def _fuse_consecutive_ops(ops, shape_constraints):
    res = []
    for op in ops:
        if len(res) == 0:
            res.append(op)
        else:
            prev_op = res.pop(-1)
            new_op = _fuse_two_ops(prev_op, op)
            if new_op is None:
                res.extend([prev_op, op])
            else:
                res.append(new_op)
    return res


def optimize(ops, variables, shape_constraints):
    _normalize_ashapes(ops, variables, shape_constraints)
    ops = _reduce_memory_access(ops)
    ops = _fuse_consecutive_ops(ops, shape_constraints)
    ops = _reduce_memory_access(ops)
    return ops
