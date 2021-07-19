import ast

import numpy
import operator

from cupy._logic import ops
from cupy._math import arithmetic
from cupy._logic import comparison
from cupy._binary import elementwise
from cupy import _core

from cupyx.jit import _cuda_types


_numpy_scalar_true_divide = _core.create_ufunc(
    'numpy_scalar_true_divide',
    ('??->d', '?i->d', 'i?->d', 'bb->f', 'bi->d', 'BB->f', 'Bi->d',
     'hh->f', 'HH->f', 'ii->d', 'II->d', 'll->d', 'LL->d', 'qq->d', 'QQ->d',
     'ee->e', 'ff->f', 'dd->d', 'FF->F', 'DD->D'),
    'out0 = (out0_type)in0 / (out0_type)in1',
)


_numpy_scalar_invert = _core.create_ufunc(
    'numpy_scalar_invert',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I',
     'l->l', 'L->L', 'q->q', 'Q->Q'),
    'out0 = ~in0',
)


_numpy_scalar_logical_not = _core.create_ufunc(
    'numpy_scalar_logical_not',
    ('?->?', 'b->?', 'B->?', 'h->?', 'H->?', 'i->?', 'I->?', 'l->?', 'L->?',
     'q->?', 'Q->?', 'e->?', 'f->?', 'd->?',
     ('F->?', 'out0 = !in0.real() && !in0.imag()'),
     ('D->?', 'out0 = !in0.real() && !in0.imag()')),
    'out0 = !in0',
)


_scalar_lt = _core.create_comparison('scalar_less', '<')
_scalar_lte = _core.create_comparison('scalar_less', '<=')
_scalar_gt = _core.create_comparison('scalar_less', '>')
_scalar_gte = _core.create_comparison('scalar_less', '>=')


_py_ops = {
    ast.And: lambda x, y: x and y,
    ast.Or: lambda x, y: x or y,
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Pow: operator.pow,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
    ast.BitXor: operator.xor,
    ast.Invert: operator.invert,
    ast.Not: operator.not_,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.USub: operator.neg,
}


_numpy_ops = {
    ast.And: ops.logical_and,
    ast.Or: ops.logical_or,
    ast.Add: arithmetic.add,
    ast.Sub: arithmetic.subtract,
    ast.Mult: arithmetic.multiply,
    ast.Pow: arithmetic.power,
    ast.Div: _numpy_scalar_true_divide,
    ast.FloorDiv: arithmetic.floor_divide,
    ast.Mod: arithmetic.remainder,
    ast.LShift: elementwise.left_shift,
    ast.RShift: elementwise.right_shift,
    ast.BitOr: elementwise.bitwise_or,
    ast.BitAnd: elementwise.bitwise_and,
    ast.BitXor: elementwise.bitwise_xor,
    ast.Invert: _numpy_scalar_invert,
    ast.Not: _numpy_scalar_logical_not,
    ast.Eq: comparison.equal,
    ast.NotEq: comparison.not_equal,
    ast.Lt: _scalar_lt,
    ast.LtE: _scalar_lte,
    ast.Gt: _scalar_gt,
    ast.GtE: _scalar_gte,
    ast.USub: arithmetic.negative,
}


def get_pyfunc(op_type):
    return _py_ops[op_type]


def get_ufunc(mode, op_type):
    if mode == 'numpy':
        return _numpy_ops[op_type]
    if mode == 'cuda':
        return _numpy_ops[op_type]
    assert False


def get_ctype_from_scalar(mode, x):
    if isinstance(x, numpy.generic):
        return _cuda_types.Scalar(x.dtype)

    if mode == 'numpy':
        if isinstance(x, bool):
            return _cuda_types.Scalar(numpy.bool_)
        if isinstance(x, int):
            # use plain int here for cross-platform portability
            return _cuda_types.Scalar(int)
        if isinstance(x, float):
            return _cuda_types.Scalar(numpy.float64)
        if isinstance(x, complex):
            return _cuda_types.Scalar(numpy.complex128)

    if mode == 'cuda':
        if isinstance(x, bool):
            return _cuda_types.Scalar(numpy.bool_)
        if isinstance(x, int):
            if -(1 << 31) <= x < (1 << 31):
                return _cuda_types.Scalar(numpy.int32)
            return _cuda_types.Scalar(numpy.int64)
        if isinstance(x, float):
            return _cuda_types.Scalar(numpy.float32)
        if isinstance(x, complex):
            return _cuda_types.Scalar(numpy.complex64)

    raise NotImplementedError(f'{x} is not scalar object.')


_typechars = '?bBhHiIlLqQefdFD'


def _cuda_can_cast(from_dtype, to_dtype):
    from_dtype = numpy.dtype(from_dtype)
    to_dtype = numpy.dtype(to_dtype)
    return _typechars.find(from_dtype.char) <= _typechars.find(to_dtype.char)


def guess_routine(ufunc, in_types, dtype, mode):
    if dtype is not None:
        return ufunc._ops._guess_routine_from_dtype(dtype)
    can_cast = numpy.can_cast if mode == 'numpy' else _cuda_can_cast
    return ufunc._ops._guess_routine_from_in_types(tuple(in_types), can_cast)
