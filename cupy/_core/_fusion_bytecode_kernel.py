"""Scheduled single-kernel CUDA lowering for iterative scalar-output functions.

Unlike the tracing frontends, this emits device control flow. Boolean indexing
is a virtual filtered view: reductions visit selected elements without
compacting them. Collectives and range-loop conditions must be block-uniform.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import builtins
import inspect
import sys
import types

import numpy as np
import cupy as cp

from cupy._core._fusion_bytecode import (
    ArraySpec, Plan, Unsupported, _BY_ID, _BINARY, _COMPARE)


_TYPES = {'b': 'signed char', 'B': 'unsigned char', 'h': 'short',
          'H': 'unsigned short', 'i': 'int', 'I': 'unsigned int',
          'l': 'long long', 'L': 'unsigned long long',
          'q': 'long long', 'Q': 'unsigned long long',
          '?': 'bool', 'f': 'float', 'd': 'double'}
_BINARY_UFUNCS = {name: symbol for symbol, name in
                  {**_BINARY, **_COMPARE}.items()}
_UNARY_UFUNCS = {'negative': '-', 'positive': '+', 'invert': '~',
                 'logical_not': '!'}
_MATH_UFUNCS = {
    name: (getattr(np, name), cuda_name)
    for name, cuda_name in {
        **{name: name for name in (
            'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'hypot',
            'exp', 'exp2', 'expm1', 'log', 'log2', 'log10', 'log1p',
            'sqrt', 'cbrt', 'ceil', 'floor', 'trunc', 'rint',
            'isnan', 'isinf', 'isfinite', 'signbit')},
        'arcsin': 'asin', 'arccos': 'acos', 'arctan': 'atan',
        'arctan2': 'atan2', 'arcsinh': 'asinh', 'arccosh': 'acosh',
        'arctanh': 'atanh',
    }.items()}
_PREAMBLE = r'''
#ifdef COOPERATIVE
#include <cooperative_groups.h>
#endif
#define NAN nan("")
template <typename T> __device__ T integer_mod(T a, T b) {
    if (b == 0 || (T(-1) < T(0) && b == T(-1))) return T(0);
    T r = a % b;
    return r != 0 && ((r < 0) != (b < 0)) ? r + b : r;
}
template <typename T> __device__ T floating_mod(T a, T b) {
    T r = fmod(a, b);
    return r != 0 && ((r < 0) != (b < 0)) ? r + b : r;
}
template <typename T> __device__ T block_sum(T value) {
    __shared__ T values[256];
    values[threadIdx.x] = value;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            values[threadIdx.x] += values[threadIdx.x + stride];
        __syncthreads();
    }
    T result = values[0];
    __syncthreads();
    return result;
}
__device__ void phase_barrier() {
#ifdef COOPERATIVE
    cooperative_groups::this_grid().sync();
#else
    __syncthreads();
#endif
}
template <typename T> __device__ T total_sum(
        T value, unsigned long long* storage) {
    T result = block_sum(value);
#ifdef COOPERATIVE
    T* partials = reinterpret_cast<T*>(storage);
    if (threadIdx.x == 0) partials[blockIdx.x] = result;
    phase_barrier();
    // Every block gets the same scalar, so subsequent conditions and range
    // loops remain uniform across the grid. Only block-sized data is reread.
    result = 0;
    for (unsigned int i = threadIdx.x; i < gridDim.x; i += blockDim.x)
        result += partials[i];
    result = block_sum(result);
    // No block may reuse the scratch until all blocks have finished reading.
    phase_barrier();
#endif
    return result;
}
'''


@dataclass
class Expr:
    code: str
    dtype: np.dtype
    array: bool = False
    where: str = 'true'
    weak: type | None = None
    carried: bool = False
    accumulator: str | None = None


@dataclass
class Iterator:
    index: str
    stop: str
    array: Expr | None = None


def ctype(dtype):
    try:
        return _TYPES[np.dtype(dtype).char]
    except KeyError:
        raise Unsupported('Single-kernel loops require bool, integer, '
                          'float32 or float64 values') from None


class Compiler:
    def __init__(self, function, args, analyzer):
        self.function = function
        self.entry = analyzer.analyse(function)
        self.args = args
        self.locals = {}
        self.declarations = []
        self.lines = []
        self.parameters = []
        self.buffers = []
        self.stack_targets = {}
        self.target_defined = {}
        self.visited_defined = {}
        self.defined = set()
        self.stack = []
        self.serial = 0
        self.parallel_loop = None
        self.accumulators = set()
        self.zero_locals = set()
        self.used_loop = False
        self.result_dtype = None
        self.array_arg = None
        self.input_array = None
        self.rebound_input = None
        self.filtered_input = False
        self.element_locals = set()
        self.element_loop_seen = False
        self.current = None
        self.operations = []
        self.phases = []

    def fail(self, message):
        raise Unsupported(f'{self.function.__qualname__}: bytecode '
                          f'{self.current}: {message}')

    def emit(self, line):
        self.lines.append(line)

    def temporary(self, dtype, prefix='t'):
        name = f'{prefix}{self.serial}'
        self.serial += 1
        self.declarations.append(f'{ctype(dtype)} {name};')
        return name

    def scalar(self, value):
        if isinstance(value, Expr):
            return value
        if type(value) not in (bool, int, float):
            self.fail('Expected a numerical expression')
        dtype = np.dtype(type(value))
        ctype(dtype)
        if type(value) is float and not np.isfinite(value):
            self.fail('Nonfinite literals are unsupported')
        if type(value) is int and not -(1 << 63) <= value < (1 << 63):
            self.fail('Integer literals must fit int64')
        code = str(int(value)) + 'LL' if type(value) is int else repr(value)
        if type(value) is bool:
            code = 'true' if value else 'false'
        return Expr(code, dtype, weak=type(value))

    def collective(self):
        if self.parallel_loop is not None:
            self.fail('Collectives and array writes inside element iteration '
                      'are unsupported')

    def binary(self, symbol, left, right):
        if symbol == '**' and type(right) is int and right == 2:
            value = self.scalar(left)
            if value.dtype.kind == 'b':
                value = replace(value, dtype=np.dtype('int64'), weak=None,
                                code=f'((long long)({value.code}))')
            return self.binary('*', value, value)
        left, right = self.scalar(left), self.scalar(right)
        if left.carried or right.carried:
            self.fail('Element-loop accumulators may only be read by +=')
        if left.where != 'true' or right.where != 'true':
            self.fail('Arithmetic on compacted views is unsupported')
        comparison = symbol in ('==', '!=', '<', '<=', '>', '>=')
        ltype = left.weak() if left.weak else left.dtype
        rtype = right.weak() if right.weak else right.dtype
        dtype = np.result_type(ltype, rtype)
        if symbol in ('&', '|', '^') and dtype.kind not in 'biu':
            self.fail('Bitwise operands must be boolean or integer')
        if comparison:
            dtype = np.dtype(bool)
        if symbol == '/' and dtype.kind in 'biu':
            dtype = np.dtype('float64')
        if (dtype.kind == 'b' and not comparison
                and symbol not in ('&', '|', '^')):
            self.fail('Boolean arithmetic requires an explicit integer cast')
        if symbol == '%':
            # Python's sign convention differs from C/C++ for negative inputs.
            typ = ctype(dtype)
            helper = 'integer_mod' if dtype.kind in 'iu' else 'floating_mod'
            code = (f'{helper}(({typ})({left.code}), '
                    f'({typ})({right.code}))')
        elif symbol in ('+', '-', '*', '/', '&', '|', '^',
                        '==', '!=', '<', '<=', '>', '>='):
            # Cast operands too: integer division and mixed signed comparisons
            # must not accidentally use C++'s implicit promotion rules.
            operand_dtype = (np.result_type(ltype, rtype)
                             if comparison else dtype)
            typ = ctype(operand_dtype)
            code = f'(({typ})({left.code}) {symbol} ({typ})({right.code}))'
        else:
            self.fail(f'Unsupported device operator {symbol}')
        weak = (type(np.empty((), dtype=dtype).item())
                if left.weak and right.weak else None)
        return Expr(f'(({ctype(dtype)})({code}))', dtype,
                    left.array or right.array, weak=weak)

    def unary(self, symbol, value):
        value = self.scalar(value)
        if value.carried:
            self.fail('Element-loop accumulators may only be read by +=')
        if value.where != 'true':
            self.fail('Unary operations on compacted views are unsupported')
        if symbol == '~' and value.dtype.kind == 'b':
            symbol = '!'
        if symbol == '~' and value.dtype.kind not in 'iu':
            self.fail('Bitwise inversion requires boolean or integer input')
        if symbol == '!':
            return replace(value, code=f'(!({value.code}))',
                           dtype=np.dtype(bool),
                           weak=bool if value.weak else None)
        return replace(value, code=f'({symbol}({value.code}))')

    def buffer(self, dtype):
        self.collective()
        name = f'w{len(self.buffers)}'
        self.buffers.append(np.dtype(dtype))
        self.parameters.append(f'{ctype(dtype)}* {name}')
        return name

    def store(self, name, value):
        value = self.scalar(value)
        old = self.locals.get(name)
        if (self.parallel_loop is None and old is not None
                and old.code in self.accumulators):
            self.fail('Element accumulators cannot be changed after iteration')
        if self.parallel_loop is not None:
            if (old is not None and old.code in self.loop_carried
                    and value.accumulator != old.code):
                self.fail('Element-loop carried locals require additive '
                          'accumulator updates')
            self.element_locals.add(name)
        self.defined.add(name)
        if value.array:
            self.collective()
            # All dense passes use the same grid-stride ownership. A later
            # pass reads the element written by that very thread; no block or
            # grid barrier is needed until values are combined by a reduction.
            self.phases.append({'kind': 'array_pass', 'local': name,
                                'synchronization': 'thread'})
            rebind = old is self.input_array
            if rebind:
                if (self.filtered_input or value.where != 'true'
                        or value.dtype != old.dtype):
                    self.fail('Input rebinding requires an unfiltered array '
                              'with unchanged dtype')
                if self.rebound_input is None:
                    self.rebound_input = self.buffer(value.dtype)
            if value.where != 'true':
                # Snapshot selection so later mask updates cannot change this
                # view. Input data is immutable and need not be copied.
                dest = self.buffer(bool) if old is None else old.where[:-4]
                if old is not None and old.where == 'true':
                    self.fail('A local cannot change between dense '
                              'and filtered')
                self.emit('for (long long _i = _rank; _i < n; '
                          '_i += _width)')
                self.emit(f'    {dest}[_i] = {value.where};')
                self.locals[name] = replace(value, where=f'{dest}[_i]')
                return
            if old is not None and (not old.array or old.dtype != value.dtype
                                    or old.where != 'true'):
                self.fail('Array locals must retain their dtype and layout')
            dest = (self.rebound_input if rebind else
                    self.buffer(value.dtype) if old is None else old.code[:-4])
            self.emit('for (long long _i = _rank; _i < n; '
                      '_i += _width)')
            self.emit(f'    {dest}[_i] = {value.code};')
            if rebind:
                self.emit(f'input_data = {dest}; input_stride = 1;')
                self.locals[name] = self.input_array
            else:
                self.locals[name] = replace(value, code=f'{dest}[_i]')
        else:
            if old is not None and (old.array or old.dtype != value.dtype
                                    or old.weak != value.weak):
                self.fail('Scalar loop locals must retain their dtype')
            dest = (self.temporary(value.dtype, 'local') if old is None
                    else old.code)
            self.emit(f'{dest} = {value.code};')
            self.locals[name] = replace(value, code=dest, carried=False,
                                        accumulator=None)
            if value.code == '0LL':
                self.zero_locals.add(dest)

    def reduce(self, name, value):
        self.collective()
        value = self.scalar(value)
        if not value.array:
            self.fail('Reductions require an array')
        self.operations.append(name)
        self.phases.append({'kind': 'reduction', 'operation': name,
                            'synchronization': 'collective'})
        dtype = (np.dtype('int64') if name in ('len', 'all') else
                 value.dtype if value.dtype.kind == 'f'
                 else np.dtype('float64'))
        typ = ctype(dtype)
        result = self.temporary(dtype)
        count = self.temporary(np.dtype('int64'))
        self.emit(f'{result} = 0; {count} = 0;')
        self.emit('for (long long _i = _rank; _i < n; '
                  '_i += _width) {')
        expression = ('1' if name == 'len' else
                      f'!({value.code})' if name == 'all' else value.code)
        self.emit(f'    if ({value.where}) {{ '
                  f'{result} += ({typ})({expression}); ++{count}; }}')
        self.emit('}')
        self.emit(f'{result} = total_sum({result}, partials);')
        if name == 'all':
            self.emit(f'{result} = ({result} == 0);')
            return Expr(result, np.dtype(bool))
        if name == 'len':
            return Expr(result, dtype)
        self.emit(f'{count} = total_sum({count}, partials);')
        self.emit(f'{result} = {count} ? {result} / ({typ}){count} '
                  f': ({typ})NAN;')
        if name == 'std':
            variance = self.temporary(dtype)
            self.emit(f'{variance} = 0;')
            self.emit('for (long long _i = _rank; _i < n; '
                      '_i += _width) {')
            self.emit(f'    if ({value.where}) {{ '
                      f'{typ} delta = ({typ})({value.code}) - {result}; '
                      f'{variance} += delta * delta; }}')
            self.emit('}')
            self.emit(f'{variance} = total_sum({variance}, partials);')
            self.emit(f'{result} = {count} ? sqrt({variance} / '
                      f'({typ}){count}) : ({typ})NAN;')
        return Expr(result, dtype)

    def _call_zeros(self, args, kwargs):
        if (len(args) != 1 or args[0] != ('shape',)
                or kwargs != {'dtype': bool}):
            self.fail('Only zeros(input.shape, dtype=bool) is supported')
        return Expr('false', np.dtype(bool), array=True)

    def _call_range(self, args, kwargs):
        if kwargs:
            self.fail('Device calls do not support these keywords')
        if len(args) != 1:
            self.fail('Only range(stop) is supported')
        stop = self.scalar(args[0])
        if stop.array or stop.dtype.kind not in 'iu' or stop.carried:
            self.fail('range requires a uniform integer bound')
        index = self.temporary(np.dtype('int64'), 'index')
        end = self.temporary(np.dtype('int64'), 'end')
        self.emit(f'{index} = -1; {end} = {stop.code};')
        return Iterator(index, end)

    def _call_argument(self, args, kwargs):
        if kwargs:
            self.fail('Device calls do not support these keywords')
        if len(args) != 1:
            self.fail('Device call requires one argument')
        return args[0]

    def _call_len(self, args, kwargs):
        value = self.scalar(self._call_argument(args, kwargs))
        if not value.array:
            self.fail('len requires an array')
        return (Expr('n', np.dtype('int64')) if value.where == 'true'
                else self.reduce('len', value))

    def _call_reduction(self, args, kwargs, *, name):
        return self.reduce(name, self._call_argument(args, kwargs))

    def _call_abs(self, args, kwargs):
        value = self.scalar(self._call_argument(args, kwargs))
        if value.carried or value.where != 'true':
            self.fail('abs requires a dense, non-accumulator expression')
        code = (f'fabs({value.code})' if value.dtype.kind == 'f' else
                f'(({value.code}) < 0 ? -({value.code}) : ({value.code}))')
        return replace(value, code=code)

    def _call_binary_ufunc(self, args, name):
        return self.binary(_BINARY_UFUNCS[name], *args)

    def _call_square(self, args, name):
        return self.binary('*', args[0], args[0])

    def _call_unary_ufunc(self, args, name):
        return self.unary(_UNARY_UFUNCS[name], args[0])

    def _call_math_ufunc(self, args, name):
        values = [self.scalar(arg) for arg in args]
        if any(v.carried or v.where != 'true' for v in values):
            self.fail('Device math requires a dense expression')
        ufunc, cuda_name = _MATH_UFUNCS[name]
        dtypes = ufunc.resolve_dtypes(
            tuple(value.weak if value.weak in (int, float) else value.dtype
                  for value in values) + (None,))
        arguments = ', '.join(
            f'({ctype(dtype)})({value.code})'
            for value, dtype in zip(values, dtypes))
        code = f'{cuda_name}({arguments})'
        return Expr(code, dtypes[-1], array=any(v.array for v in values))

    # Explicit identity dispatch avoids invoking __hash__/__eq__ on an
    # unsupported callable. Handlers own validation and lowering; the tables
    # can later hold operation objects without changing bytecode analysis.
    _CALL_HANDLERS = {
        id(cp.zeros): _call_zeros,
        id(range): _call_range,
        id(len): _call_len,
        id(cp.mean): partial(_call_reduction, name='mean'),
        id(cp.std): partial(_call_reduction, name='std'),
        id(cp.all): partial(_call_reduction, name='all'),
        id(cp.abs): _call_abs,
        id(abs): _call_abs,
    }
    _UFUNC_HANDLERS = {
        **dict.fromkeys(_BINARY_UFUNCS, _call_binary_ufunc),
        **dict.fromkeys(_UNARY_UFUNCS, _call_unary_ufunc),
        **dict.fromkeys(_MATH_UFUNCS, _call_math_ufunc),
        'square': _call_square,
    }

    def call(self, target, args, kwargs):
        handler = self._CALL_HANDLERS.get(id(target))
        if handler is not None:
            return handler(self, args, kwargs)
        if kwargs:
            self.fail('Device calls do not support these keywords')
        operation = _BY_ID.get(id(target))
        if operation is not None and operation.kind == 'map':
            if len(args) != operation.implementation.nin:
                self.fail('Device ufunc arguments do not match its arity')
            handler = self._UFUNC_HANDLERS.get(operation.name)
            if handler is not None:
                return handler(self, args, operation.name)
        self.fail('Call has no registered single-kernel implementation')

    def target(self, offset, stack):
        previous = self.stack_targets.get(offset)
        if previous is not None and previous != stack:
            self.fail('Stack expressions at control-flow joins must match')
        self.stack_targets[offset] = list(stack)
        if offset in self.visited_defined:
            if not self.visited_defined[offset] <= self.defined:
                self.fail('Loop backedge loses initialized locals')
        elif offset in self.target_defined:
            self.target_defined[offset] &= self.defined
        else:
            self.target_defined[offset] = set(self.defined)

    def build(self):
        if sys.version_info[:2] != (3, 14):
            self.fail('Single-kernel iteration currently requires '
                      'CPython 3.14')
        if self.function.__code__.co_flags & (inspect.CO_GENERATOR
                                              | inspect.CO_COROUTINE):
            self.fail('Generators and coroutines are unsupported')
        for i, (name, arg) in enumerate(zip(
                self.entry.signature.parameters, self.args)):
            if type(arg) in (cp.ndarray, ArraySpec):
                if self.array_arg is not None or arg.ndim != 1:
                    self.fail('Single-kernel loops require one 1D input array')
                self.array_arg = i
                typ = ctype(arg.dtype)
                self.parameters.extend([f'const {typ}* a{i}', 'long long n',
                                        'long long stride'])
                self.declarations.extend((f'const {typ}* input_data = a{i};',
                                          'long long input_stride = stride;'))
                self.locals[name] = Expr('input_data[_i * input_stride]',
                                         arg.dtype,
                                         array=True)
                self.input_array = self.locals[name]
            elif type(arg) in (int, float, bool):
                dtype = np.dtype(type(arg))
                self.parameters.append(f'{ctype(dtype)} a{i}')
                self.locals[name] = Expr(f'a{i}', dtype, weak=type(arg))
            else:
                self.fail('Unsupported single-kernel argument type')
        if self.array_arg is None:
            self.fail('Single-kernel loops require a 1D input array')
        self.defined.update(self.locals)
        # Parameters must remain immutable even when assigned by the function.
        for name, value in list(self.locals.items()):
            if not value.array:
                del self.locals[name]
                self.store(name, value)
        stack = self.stack
        reachable = True
        keys = ()
        closure = dict(zip(self.function.__code__.co_freevars,
                           (c.cell_contents for c in
                            self.function.__closure__ or ())))
        for ins in self.entry.instructions:
            self.current = ins.offset
            if ins.offset in self.stack_targets:
                expected = self.stack_targets[ins.offset]
                if reachable and stack != expected:
                    self.fail('Stack expressions at joins must match')
                stack[:] = expected
                if ins.offset in self.target_defined:
                    incoming = self.target_defined[ins.offset]
                    self.defined = (self.defined & incoming if reachable
                                    else set(incoming))
                reachable = True
            if not reachable:
                continue
            self.visited_defined[ins.offset] = set(self.defined)
            self.emit(f'L{ins.offset}:;')
            op = ins.opname
            if op in ('RESUME', 'NOP', 'NOT_TAKEN', 'CACHE', 'PUSH_NULL',
                      'COPY_FREE_VARS', 'EXTENDED_ARG', 'PRECALL', 'TO_BOOL'):
                continue
            if op.startswith('LOAD_FAST'):
                names = ins.argval if isinstance(ins.argval, tuple) else (
                    ins.argval,)
                for name in names:
                    if name not in self.defined:
                        self.fail(f'Uninitialized local {name}')
                    value = self.locals[name]
                    if (self.parallel_loop is None
                            and name in self.element_locals
                            and value.code not in self.accumulators):
                        self.fail('Element-local values cannot escape '
                                  'the loop')
                    if (self.parallel_loop is not None and not value.array
                            and value.code in self.loop_carried):
                        value = replace(value, carried=True)
                    stack.append(value)
            elif op in ('LOAD_CONST', 'LOAD_SMALL_INT',
                        'LOAD_COMMON_CONSTANT'):
                stack.append(ins.argval)
            elif op == 'LOAD_GLOBAL':
                stack.append(self.function.__globals__.get(
                    ins.argval, vars(builtins).get(ins.argval)))
            elif op == 'LOAD_DEREF':
                stack.append(closure[ins.argval])
            elif op in ('LOAD_ATTR', 'LOAD_METHOD'):
                obj = stack.pop()
                if type(obj) is types.ModuleType and obj is cp:
                    stack.append(vars(cp).get(ins.argval))
                elif (isinstance(obj, Expr) and obj.array
                      and ins.argval == 'shape' and obj.where == 'true'):
                    stack.append(('shape',))
                else:
                    self.fail('Unsupported device attribute')
            elif op == 'KW_NAMES':
                keys = self.function.__code__.co_consts[ins.arg]
            elif op in ('CALL', 'CALL_KW'):
                keywords = stack.pop() if op == 'CALL_KW' else keys
                keys = ()
                values = stack[-ins.arg:] if ins.arg else []
                if ins.arg:
                    del stack[-ins.arg:]
                target = stack.pop()
                npos = len(values) - len(keywords)
                stack.append(self.call(target, values[:npos], dict(zip(
                    keywords, values[npos:]))))
            elif op == 'STORE_FAST':
                self.store(ins.argval, stack.pop())
            elif op == 'STORE_FAST_STORE_FAST':
                for name in ins.argval:
                    self.store(name, stack.pop())
            elif op == 'STORE_FAST_LOAD_FAST':
                store, load = ins.argval
                self.store(store, stack.pop())
                stack.append(self.locals[load])
            elif op == 'BINARY_SUBSCR' or (
                    op == 'BINARY_OP' and ins.argrepr == '[]'):
                index, value = stack.pop(), stack.pop()
                if (not isinstance(value, Expr) or not value.array
                        or value is not self.input_array
                        or value.where != 'true' or not isinstance(index, Expr)
                        or not index.array or index.dtype.kind != 'b'
                        or index.where != 'true'):
                    self.fail('Only dense-array boolean indexing is supported')
                self.filtered_input = True
                stack.append(replace(value, where=index.code))
            elif op in ('BINARY_OP', 'COMPARE_OP'):
                right, left = stack.pop(), stack.pop()
                symbol = ins.argrepr.replace('bool(', '').rstrip(')')
                accumulator = None
                if symbol == '+=' and self.parallel_loop is not None:
                    left = self.scalar(left)
                    if (not left.carried or left.code not in self.zero_locals
                            or self.scalar(right).array):
                        self.fail('Element iteration requires '
                                  'zero-initialized additive scalar '
                                  'accumulators')
                    self.accumulators.add(left.code)
                    accumulator = left.code
                    left = replace(left, carried=False)
                if symbol.endswith('=') and symbol not in (
                        '==', '!=', '<=', '>='):
                    symbol = symbol[:-1]
                stack.append(replace(self.binary(symbol, left, right),
                                     accumulator=accumulator))
            elif op in ('UNARY_NEGATIVE', 'UNARY_POSITIVE', 'UNARY_INVERT',
                        'UNARY_NOT'):
                stack.append(self.unary({'UNARY_NEGATIVE': '-',
                                         'UNARY_POSITIVE': '+',
                                         'UNARY_INVERT': '~',
                                         'UNARY_NOT': '!'}[op], stack.pop()))
            elif op == 'GET_ITER':
                value = stack.pop()
                if isinstance(value, Iterator):
                    self.collective()
                    if self.element_loop_seen:
                        self.fail('Element iteration must be the only loop')
                    stack.append(value)
                elif isinstance(value, Expr) and value.array:
                    self.collective()
                    if value.where != 'true' or self.used_loop:
                        self.fail('Element iteration must be the only loop')
                    index = self.temporary(np.dtype('int64'), 'index')
                    self.emit(f'{index} = _rank - _width;')
                    self.phases.append({'kind': 'independent_element_loop',
                                        'synchronization': 'thread'})
                    stack.append(Iterator(index, 'n', value))
                    self.parallel_loop = index
                    self.element_loop_seen = True
                    self.loop_carried = {v.code for v in self.locals.values()
                                         if not v.array}
                else:
                    self.fail('Unsupported iterator')
                self.used_loop = True
            elif op == 'FOR_ITER':
                iterator = stack[-1]
                if not isinstance(iterator, Iterator):
                    self.fail('Unsupported loop stack')
                self.target(ins.argval, stack)
                step = '_width' if iterator.array is not None else '1'
                self.emit(f'{iterator.index} += {step};')
                self.emit(f'if ({iterator.index} >= {iterator.stop}) '
                          f'goto L{ins.argval};')
                stack.append(Expr(iterator.index, np.dtype('int64'))
                             if iterator.array is None else replace(
                                 iterator.array, array=False,
                                 code=iterator.array.code.replace(
                                     '_i', iterator.index)))
            elif op.startswith('POP_JUMP'):
                value = self.scalar(stack.pop())
                if value.array or value.carried:
                    self.fail('Condition must be scalar and independent of '
                              'element accumulators')
                self.target(ins.argval, stack)
                cond = value.code if 'IF_TRUE' in op else f'!({value.code})'
                self.emit(f'if ({cond}) goto L{ins.argval};')
            elif op in ('JUMP_FORWARD', 'JUMP_BACKWARD',
                        'JUMP_BACKWARD_NO_INTERRUPT', 'JUMP_ABSOLUTE'):
                self.target(ins.argval, stack)
                self.emit(f'goto L{ins.argval};')
                reachable = False
            elif op == 'END_FOR':
                pass
            elif op in ('POP_ITER', 'POP_TOP'):
                value = stack.pop()
                if (isinstance(value, Iterator)
                        and value.array is not None):
                    if op == 'POP_TOP':
                        self.fail('break in element iteration is unsupported')
                    self.parallel_loop = None
            elif op in ('RETURN_VALUE', 'RETURN_CONST'):
                value = self.scalar(stack.pop() if op == 'RETURN_VALUE'
                                    else ins.argval)
                if value.array or self.parallel_loop is not None:
                    self.fail('Only uniform scalar outputs are supported')
                if self.accumulators:
                    if value.code not in self.accumulators:
                        self.fail('Element loops must return '
                                  'their accumulator')
                    self.emit(f'{value.code} = '
                              f'total_sum({value.code}, partials);')
                    self.phases.append({'kind': 'reduction',
                                        'operation': 'accumulator_sum',
                                        'synchronization': 'collective'})
                if (self.result_dtype is not None
                        and self.result_dtype != value.dtype):
                    self.fail('All returns must have the same dtype')
                self.result_dtype = value.dtype
                self.emit(f'if (_rank == 0) *output = {value.code};')
                self.emit('return;')
                reachable = False
            else:
                self.fail(f'Unsupported single-kernel instruction {op}')
        if not self.used_loop or self.result_dtype is None:
            self.fail('Single-kernel lowering requires iteration and a return')
        if self.rebound_input is not None and self.filtered_input:
            self.fail('Device input rebinding cannot be combined with '
                      'filtered views')
        self.parameters.append(f'{ctype(self.result_dtype)}* output')
        self.parameters.append('unsigned long long* partials')
        source = (_PREAMBLE + '\nextern "C" __global__ void iterative('
                  + ', '.join(self.parameters) + ') {\n'
                  + 'const long long _rank = (long long)blockIdx.x * '
                  'blockDim.x + threadIdx.x;\n'
                  + 'const long long _width = '
                  '(long long)blockDim.x * gridDim.x;\n'
                  + '\n'.join(self.declarations + self.lines) + '\n}\n')
        execute = Executable(source, self.array_arg, self.buffers,
                             self.result_dtype)
        report = {'status': 'analysed',
                  'callees': [self.function.__qualname__],
                  'guards': [], 'allocations': [], 'regions': [],
                  'backend': 'scheduled_cuda', 'source': source,
                  'operations': self.operations,
                  'phases': self.phases, 'schedule': execute.schedule,
                  'workspace_arrays': len(self.buffers),
                  'reduction_workspace': '8 bytes per cooperative block',
                  'kernel_launches': 1, 'threads': 256,
                  'limitations': 'One 1D input, scalar output; '
                  'zero-seeded additive element loops or uniform range loops'}
        return Plan(execute, lambda *args: True, [], source, report)


class Executable:
    # Initial heuristic; benchmark whole calls before lowering the crossover.
    single_block_limit = 16384
    elements_per_block = 4096

    def __init__(self, source, array_arg, buffers, dtype, *, block_limit=None):
        self.source, self.array_arg = source, array_arg
        self.buffers, self.dtype = buffers, dtype
        if block_limit is not None and block_limit < 1:
            raise ValueError('block_limit must be positive')
        self.block_limit = block_limit
        self.kernels = {}
        self.devices = {}
        self.schedule = {'kind': 'not_selected', 'blocks': None}

    def _launch(self, size):
        device = cp.cuda.runtime.getDevice()
        cooperative = size > self.single_block_limit and self.block_limit != 1
        if cooperative:
            capacity = self.devices.get(device)
            if capacity is None:
                attributes = cp.cuda.Device(device).attributes
                if attributes.get('CooperativeLaunch', 0):
                    from cupy.cuda import driver
                    kernel = self._kernel(device, True)
                    occupancy = (
                        driver.occupancyMaxActiveBlocksPerMultiprocessor)
                    resident = occupancy(kernel.kernel.ptr, 256, 0)
                    # One block per SM already spreads memory work across the
                    # device; the occupancy bound is a correctness constraint.
                    capacity = (attributes['MultiProcessorCount']
                                if resident > 0 else 1)
                else:
                    capacity = 1
                self.devices[device] = capacity
            blocks = min(capacity, (size + self.elements_per_block - 1)
                         // self.elements_per_block)
            if self.block_limit is not None:
                blocks = min(blocks, self.block_limit)
            cooperative = blocks > 1
        else:
            blocks = 1
        kind = 'cooperative_grid' if cooperative else 'single_block'
        if (self.schedule.get('blocks') != blocks
                or self.schedule.get('device') != device):
            self.schedule.update(kind=kind, blocks=blocks,
                                 device=device, cooperative=cooperative,
                                 reduction_workspace_bytes=(8 * blocks
                                                            if blocks > 1
                                                            else 0))
        return self._kernel(device, cooperative), blocks

    def _kernel(self, device, cooperative):
        key = device, cooperative
        kernel = self.kernels.get(key)
        if kernel is None:
            options = ('--std=c++17',)
            if cooperative:
                options += ('-DCOOPERATIVE',)
            kernel = cp.RawKernel(self.source, 'iterative', options=options,
                                  enable_cooperative_groups=cooperative)
            self.kernels[key] = kernel
        return kernel

    def __call__(self, *args):
        array = args[self.array_arg]
        kernel, blocks = self._launch(array.size)
        parameters = []
        for i, arg in enumerate(args):
            if i == self.array_arg:
                parameters.extend((arg, np.int64(arg.size),
                                   np.int64(arg.strides[0] // arg.itemsize)))
            else:
                parameters.append(np.dtype(type(arg)).type(arg))
        workspace = [cp.empty(array.size, dtype=dtype)
                     for dtype in self.buffers]
        output = cp.empty((), dtype=self.dtype)
        # No initialization launch: every block writes its own reduction slot
        # before the first grid barrier. A one-block kernel needs no scratch.
        partials = cp.empty(blocks, dtype=cp.uint64) if blocks > 1 else output
        kernel((blocks,), (256,), (*parameters, *workspace, output, partials))
        return output


def compile_kernel(function, args, analyzer):
    return Compiler(function, args, analyzer).build()
