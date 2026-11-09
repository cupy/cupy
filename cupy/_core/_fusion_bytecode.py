"""Experimental bytecode frontend for the existing CuPy fusion engines.

This module is private. It never executes the decorated function. Instead it
interprets a deliberately small bytecode subset and emits a callable containing
only registered CuPy operations. See examples/fusion/bytecode_prototype.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import builtins
import dis
import inspect
import operator
import sys
import time
import types

import numpy as np

import cupy as cp
from cupy._core import fusion, new_fusion, _fusion_thread_local


class Unsupported(TypeError):
    """An operation cannot be compiled without executing user code."""


_PYTHON_DTYPES = {bool: np.uint8, int: np.int64,
                  float: np.float64, complex: np.complex128}
_SCALAR_TYPES = frozenset(
    np.dtype(code).type for code in '?bBhHiIlLqQefdFD')


def _normalize(value):
    """Validate the kernel representation and retain weak-scalar origin."""
    typ = type(value)
    if typ in _PYTHON_DTYPES:
        converted = _PYTHON_DTYPES[typ](value)
        # Existing fusers distinguish weak Python scalars from NumPy scalars.
        return converted if typ is bool else typ(converted)
    if typ in _SCALAR_TYPES or typ is cp.ndarray:
        return value
    raise Unsupported(f'Unsupported numerical input: {typ.__name__}')


@dataclass(eq=False)
class Value:
    name: str
    kind: str
    sample: object = None
    source: str | None = None


@dataclass
class Metadata:
    """Host metadata expression, evaluated without touching device data."""
    source: str
    sample: object


@dataclass
class Method:
    receiver: Value
    name: str


@dataclass
class Operation:
    """An operation's recognition and implementation definition."""
    name: str
    implementation: object
    kind: str = 'map'
    keywords: frozenset = frozenset()

    def matches(self, function):
        return function is self.implementation

    def emit(self, arguments, keywords, render, bind):
        args = [render(arg) for arg in arguments]
        args.extend(f'{key}={render(value)}'
                    for key, value in keywords.items())
        return f'{bind(self.implementation)}({", ".join(args)})'


class Elementwise(Operation):
    pass


class Reduction(Operation):
    def __init__(self, name, implementation):
        super().__init__(name, implementation, 'reduction',
                         frozenset({'axis', 'dtype', 'keepdims'}))


class Allocation(Operation):
    def __init__(self, name, implementation):
        super().__init__(name, implementation, 'allocation',
                         frozenset({'shape', 'dtype', 'order', 'fill_value'}))


_OPERATIONS = [Elementwise(name, getattr(cp, name)) for name in (
    'add', 'subtract', 'multiply', 'true_divide', 'floor_divide', 'remainder',
    'power', 'negative', 'positive', 'absolute', 'sin', 'cos', 'tan', 'exp',
    'log', 'sqrt', 'square', 'maximum', 'minimum', 'equal', 'not_equal',
    'less', 'less_equal', 'greater', 'greater_equal', 'logical_not',
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'invert', 'left_shift',
    'right_shift', 'tanh')]
_OPERATIONS += [Reduction(name, getattr(cp, name)) for name in (
    'sum', 'prod', 'min', 'max', 'all', 'any')]
_OPERATIONS += [Allocation(name, getattr(cp, name)) for name in (
    'empty', 'zeros', 'ones', 'full', 'empty_like', 'zeros_like', 'ones_like',
    'full_like')]
_BY_NAME = {op.name: op for op in _OPERATIONS}
_BY_ID = {id(op.implementation): op for op in _OPERATIONS}
_BINARY = dict(zip(
    ('+', '-', '*', '/', '//', '%', '**', '&', '|', '^', '<<', '>>'),
    ('add', 'subtract', 'multiply', 'true_divide', 'floor_divide',
     'remainder', 'power', 'bitwise_and', 'bitwise_or', 'bitwise_xor',
     'left_shift', 'right_shift')))
_COMPARE = {'<': 'less', '<=': 'less_equal', '==': 'equal',
            '!=': 'not_equal', '>': 'greater', '>=': 'greater_equal'}
_HOST_COMPARE = {'<': operator.lt, '<=': operator.le, '==': operator.eq,
                 '!=': operator.ne, '>': operator.gt, '>=': operator.ge}
_UNARY = {'UNARY_NEGATIVE': 'negative', 'UNARY_POSITIVE': 'positive',
          'UNARY_INVERT': 'invert', 'UNARY_NOT': 'logical_not'}


@dataclass
class Node:
    operation: Operation
    arguments: tuple
    keywords: dict
    output: Value
    location: dict


@dataclass
class Analysis:
    function: types.FunctionType  # Strong reference protects against id reuse.
    instructions: tuple
    signature: inspect.Signature
    offsets: dict


class Analyzer:
    def __init__(self):
        self.functions = {}
        self.hits = 0

    def analyse(self, function):
        if not (3, 11) <= sys.version_info[:2] <= (3, 14):
            raise Unsupported('Prototype supports CPython 3.11 through 3.14')
        entry = self.functions.get(id(function))
        if entry is not None:
            self.hits += 1
            return entry
        if type(function) is not types.FunctionType:
            raise Unsupported('Callable has no supported Python definition')
        if function.__code__.co_exceptiontable:
            raise Unsupported('Exception handlers are not supported yet')
        instructions = tuple(dis.get_instructions(function))
        entry = Analysis(function, instructions, inspect.signature(function),
                         {i.offset: n for n, i in enumerate(instructions)})
        self.functions[id(function)] = entry
        return entry


def _compile_source(source, bindings, name):
    # CuPy/NumPy C extensions may import helpers using the caller's builtins.
    # The source is compiler-owned, not a sandbox for arbitrary Python text.
    namespace = {'__builtins__': vars(builtins), **bindings}
    exec(compile(source, f'<cupy-bytecode:{name}>', 'exec'), namespace)
    return namespace[name]


def _walk_values(value):
    if isinstance(value, Value):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_values(item)


class Builder:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.nodes = []
        self.guards = []
        self.callees = []
        self.active = set()
        self.location = {}

    def fail(self, reason):
        loc = self.location
        raise Unsupported(
            f'{loc.get("function", "function")}:'
            f'{loc.get("line", "?")} (bytecode '
            f'{loc.get("offset", "?")}): {reason}')

    def input(self, source, sample, *, receiver=False):
        if type(sample) is cp.ndarray:
            kind = 'array'
        elif type(sample) in _PYTHON_DTYPES or type(sample) in _SCALAR_TYPES:
            kind = 'bool' if type(sample) in (bool, np.bool_) else 'scalar'
        elif receiver:
            kind = 'object'
        else:
            self.fail(f'Unsupported input type: {type(sample).__name__}')
        return Value(source, kind, sample, source)

    def operation(self, op, args, kwargs):
        if op.kind == 'map' and len(args) != op.implementation.nin:
            self.fail(f'{op.name} requires {op.implementation.nin} inputs; '
                      'positional output arguments are unsupported')
        if op.kind == 'reduction' and len(args) != 1:
            self.fail('Reduction options must be keywords; '
                      'out= is unsupported')
        if op.kind == 'reduction' and any(
                isinstance(value, (Value, Metadata))
                for value in kwargs.values()):
            self.fail('Reduction options must be static in this prototype')
        if set(kwargs) - op.keywords:
            self.fail(f'Unsupported keyword for {op.name}: '
                      f'{sorted(set(kwargs) - op.keywords)}')
        if any(isinstance(v, Value) and v.kind == 'object'
               for v in _walk_values(args)):
            self.fail('Objects cannot be numerical operands')
        if op.kind != 'allocation' and any(
                isinstance(v, Metadata) for v in (*args, *kwargs.values())):
            self.fail('Runtime metadata operands require '
                      'an allocation boundary')
        kind = ('array' if op.kind in ('allocation', 'reduction') or any(
            v.kind == 'array' for v in _walk_values(args)) else 'scalar')
        value = Value(f'v{len(self.nodes)}', kind)
        self.nodes.append(Node(op, tuple(args), dict(kwargs), value,
                               self.location.copy()))
        return value

    def attribute(self, obj, name):
        if isinstance(obj, Value):
            if obj.kind == 'object':
                sample = obj.sample
                typ = type(sample)
                if (inspect.getattr_static(typ, '__getattribute__')
                        is not object.__getattribute__
                        or inspect.getattr_static(typ, '__getattr__', None)
                        is not None):
                    self.fail('Custom attribute lookup is unsupported')
                definition = inspect.getattr_static(typ, name, None)
                if type(definition) is types.FunctionType:
                    return (definition, obj)
                if (definition is not None and inspect.getattr_static(
                        type(definition), '__get__', None) is not None):
                    self.fail(f'Descriptor {name!r} is unsupported')
                missing = object()
                value = inspect.getattr_static(sample, name, missing)
                if value is missing:
                    self.fail(f'No stored instance attribute {name!r}')
                # Source identifiers come from validated Python attributes.
                source = f'{obj.source}.{name}'
                return self.input(source, value)
            if name in ('shape', 'ndim', 'size', 'dtype'):
                if obj.kind != 'array' or obj.source is None:
                    self.fail('Metadata currently requires an input array')
                return Metadata(f'{obj.source}.{name}',
                                getattr(obj.sample, name))
            if name in ('sum', 'prod', 'min', 'max', 'all', 'any'):
                return Method(obj, name)
            self.fail(f'Unsupported numerical attribute {name!r}')
        if type(obj) is types.ModuleType:
            if name not in vars(obj):
                self.fail(f'Dynamic module attribute {name!r}')
            return vars(obj)[name]
        self.fail(f'Unsupported attribute lookup {name!r}')

    def call(self, function, args, kwargs):
        if isinstance(function, Method):
            return self.operation(_BY_NAME[function.name],
                                  (function.receiver, *args), kwargs)
        if isinstance(function, tuple) and len(function) == 2:
            function, receiver = function
            return self.run(function, (receiver, *args), kwargs)
        if type(function) is types.MethodType:
            # Bound methods captured in globals are not runtime arguments.
            self.fail('Pass the receiver as an argument instead of capturing '
                      'a bound method')
        op = _BY_ID.get(id(function))
        if op is not None and op.matches(function):
            return self.operation(op, args, kwargs)
        if function is abs:
            return self.operation(_BY_NAME['absolute'], args, kwargs)
        if type(function) is FusedFunction:
            function = function.func
        if type(function) is types.FunctionType:
            return self.run(function, args, kwargs)
        self.fail('Call has neither compilable bytecode nor a registered '
                  'CUDA implementation')

    def host(self, value):
        if isinstance(value, Metadata):
            return value.source, value.sample
        if isinstance(value, Value):
            if value.kind == 'bool' and value.source is not None:
                return value.source, bool(value.sample)
            self.fail('Device-data predicates and non-boolean scalar '
                      'predicates are not supported yet')
        if type(value) in (bool, int, str, type(None)):
            return repr(value), value
        if (isinstance(value, np.dtype) or
                type(value) is type and value in _SCALAR_TYPES):
            dtype = np.dtype(value)
            return f'_dtype({dtype.str!r})', dtype
        self.fail('Unsupported host predicate')

    def run(self, function, arguments, keywords):
        if id(function) in self.active:
            self.fail('Recursive call cycles are not supported yet')
        entry = self.analyzer.analyse(function)
        if function.__code__.co_flags & (inspect.CO_GENERATOR
                                         | inspect.CO_COROUTINE):
            self.fail('Generators and coroutines are unsupported')
        self.callees.append(function.__qualname__)
        bound = entry.signature.bind(*arguments, **keywords)
        bound.apply_defaults()
        local = dict(bound.arguments)
        closure = dict(zip(function.__code__.co_freevars,
                           (cell.cell_contents
                            for cell in function.__closure__ or ())))
        stack = []
        kw_names = ()
        index = 0
        self.active.add(id(function))
        previous = self.location
        try:
            while index < len(entry.instructions):
                ins = entry.instructions[index]
                self.location = {'function': function.__qualname__,
                                 'line': ins.positions.lineno,
                                 'offset': ins.offset}
                index += 1
                name = ins.opname
                if name in ('RESUME', 'NOP', 'NOT_TAKEN', 'CACHE', 'PUSH_NULL',
                            'COPY_FREE_VARS', 'EXTENDED_ARG'):
                    continue
                if name.startswith('LOAD_FAST') and name != 'LOAD_FAST_CHECK':
                    names = (ins.argval if isinstance(ins.argval, tuple)
                             else (ins.argval,))
                    stack.extend(local[key] for key in names)
                elif name == 'LOAD_FAST_CHECK':
                    stack.append(local[ins.argval])
                elif name == 'STORE_FAST':
                    local[ins.argval] = stack.pop()
                elif name == 'STORE_FAST_STORE_FAST':
                    for key in ins.argval:
                        local[key] = stack.pop()
                elif name == 'STORE_FAST_LOAD_FAST':
                    store, load = ins.argval
                    local[store] = stack.pop()
                    stack.append(local[load])
                elif name in ('LOAD_CONST', 'LOAD_SMALL_INT',
                              'LOAD_COMMON_CONSTANT'):
                    stack.append(ins.argval)
                elif name == 'LOAD_GLOBAL':
                    key = ins.argval
                    builtins = function.__builtins__
                    if key in function.__globals__:
                        stack.append(function.__globals__[key])
                    elif key in builtins:
                        stack.append(builtins[key])
                    else:
                        self.fail(f'Unbound global {key!r}')
                elif name == 'LOAD_DEREF':
                    stack.append(closure[ins.argval])
                elif name in ('LOAD_ATTR', 'LOAD_METHOD'):
                    stack.append(self.attribute(stack.pop(), ins.argval))
                elif name == 'KW_NAMES':
                    kw_names = function.__code__.co_consts[ins.arg]
                elif name == 'PRECALL':
                    pass
                elif name in ('CALL', 'CALL_KW', 'CALL_FUNCTION',
                              'CALL_METHOD'):
                    keys = stack.pop() if name == 'CALL_KW' else kw_names
                    kw_names = ()
                    count = ins.arg
                    values = stack[-count:] if count else []
                    if count:
                        del stack[-count:]
                    target = stack.pop()
                    positional = values[:count - len(keys)]
                    kwargs = dict(zip(keys, values[count - len(keys):]))
                    stack.append(self.call(target, positional, kwargs))
                elif name == 'BINARY_OP':
                    right, left = stack.pop(), stack.pop()
                    symbol = ins.argrepr
                    if symbol == '[]':
                        if (not isinstance(left, Metadata)
                                or type(right) is not int):
                            self.fail('Only metadata indexing is supported')
                        stack.append(Metadata(f'{left.source}[{right}]',
                                              left.sample[right]))
                    elif symbol in _BINARY:
                        stack.append(self.operation(_BY_NAME[_BINARY[symbol]],
                                                    (left, right), {}))
                    else:
                        self.fail(f'Unsupported operator {symbol!r}')
                elif name == 'BINARY_SUBSCR':
                    right, left = stack.pop(), stack.pop()
                    if (not isinstance(left, Metadata)
                            or type(right) is not int):
                        self.fail('Only metadata indexing is supported')
                    stack.append(Metadata(f'{left.source}[{right}]',
                                          left.sample[right]))
                elif name == 'COMPARE_OP':
                    right, left = stack.pop(), stack.pop()
                    symbol = ins.argrepr.replace('bool(', '').rstrip(')')
                    if (isinstance(left, Metadata)
                            or isinstance(right, Metadata)):
                        lsrc, lval = self.host(left)
                        rsrc, rval = self.host(right)
                        sample = _HOST_COMPARE[symbol](lval, rval)
                        stack.append(Metadata(f'({lsrc} {symbol} {rsrc})',
                                              sample))
                    elif symbol in _COMPARE:
                        stack.append(self.operation(_BY_NAME[_COMPARE[symbol]],
                                                    (left, right), {}))
                    else:
                        self.fail(f'Unsupported comparison {symbol!r}')
                elif name in _UNARY:
                    value = stack.pop()
                    if name == 'UNARY_NOT' and (
                            isinstance(value, Metadata) or
                            isinstance(value, Value) and value.kind == 'bool'
                            or type(value) is bool):
                        source, sample = self.host(value)
                        stack.append(Metadata(f'not ({source})', not sample))
                    else:
                        if name == 'UNARY_NOT':
                            self.fail('Python not requires a host predicate; '
                                      'array truth is not elementwise')
                        stack.append(self.operation(_BY_NAME[_UNARY[name]],
                                                    (value,), {}))
                elif name == 'TO_BOOL':
                    # Branch handling validates provenance without evaluating
                    # array truth or a user-defined __bool__.
                    pass
                elif name.startswith('POP_JUMP') and (
                        'IF_FALSE' in name or 'IF_TRUE' in name):
                    source, sample = self.host(stack.pop())
                    outcome = bool(sample)
                    self.guards.append((source, outcome))
                    take = outcome if 'IF_TRUE' in name else not outcome
                    if take:
                        if ins.argval <= ins.offset:
                            self.fail('Loops are not supported yet')
                        index = entry.offsets[ins.argval]
                elif name in ('JUMP_FORWARD', 'JUMP_ABSOLUTE'):
                    if ins.argval <= ins.offset:
                        self.fail('Loops are not supported yet')
                    index = entry.offsets[ins.argval]
                elif name == 'BUILD_TUPLE':
                    count = ins.arg
                    value = tuple(stack[-count:]) if count else ()
                    if count:
                        del stack[-count:]
                    stack.append(value)
                elif name == 'UNPACK_SEQUENCE':
                    value = stack.pop()
                    if type(value) is not tuple or len(value) != ins.arg:
                        self.fail('Only fixed tuples can be unpacked')
                    stack.extend(reversed(value))
                elif name == 'COPY':
                    stack.append(stack[-ins.arg])
                elif name == 'SWAP':
                    stack[-1], stack[-ins.arg] = stack[-ins.arg], stack[-1]
                elif name == 'POP_TOP':
                    stack.pop()
                elif name == 'RETURN_VALUE':
                    return stack.pop()
                elif name == 'RETURN_CONST':
                    return ins.argval
                else:
                    self.fail(f'Unsupported instruction {name}')
            self.fail('Function has no return')
        finally:
            self.active.remove(id(function))
            self.location = previous


class RegionKernel:
    """Select a compiler before launch, without the old decorator's replay."""

    def __init__(self, function, name):
        self.function, self.name = function, name
        self.cache = {}
        self.backend = 'not_compiled'
        self.legacy_rejection = None

    def __call__(self, *args):
        key = tuple((arg.dtype, arg.ndim) if type(arg) is cp.ndarray
                    else (type(arg), getattr(arg, 'dtype', None))
                    for arg in args)
        cached = self.cache.get(key)
        if cached is None:
            if not any(type(arg) is cp.ndarray for arg in args):
                cached = (self.function, {}, 'registered_cuda_operations')
            else:
                state = _fusion_thread_local.thread_local
                previous = (getattr(state, 'is_old_fusing', False),
                            getattr(state, 'history', None))
                try:
                    history = fusion._FusionHistory()
                    state.history, state.is_old_fusing = history, True
                    kernel, kwargs = history.get_fusion(
                        self.function, args, self.name)
                    cached = (kernel, kwargs, 'legacy_fusion')
                except Exception as exc:
                    if not (isinstance(exc, (NotImplementedError, TypeError,
                                             AttributeError)) or
                            (type(exc) is Exception and
                             str(exc) == 'Shape mismatch')):
                        raise
                    # Only verified pure expressions enter the tracer. No
                    # allocation or user callback can have executed here.
                    self.legacy_rejection = f'{type(exc).__name__}: {exc}'
                    cached = (new_fusion.Fusion(self.function, self.name), {},
                              'new_fusion')
                finally:
                    state.is_old_fusing, state.history = previous
            self.cache[key] = cached
        kernel, kwargs, self.backend = cached
        return kernel(*args, **kwargs)


@dataclass
class Region:
    """Only compiler-owned, verified code is passed to a tracing backend."""
    function: object
    source: str
    fusion: object
    operations: tuple

    @property
    def backend(self):
        return self.fusion.backend


@dataclass
class Plan:
    execute: object
    accepts: object
    regions: list
    source: str
    report: dict


class Emitter:
    def __init__(self, builder, nargs, result):
        self.builder, self.nargs, self.result = builder, nargs, result
        self.bindings = {'_normalize': _normalize, '_bool': bool,
                         '_dtype': np.dtype}
        self.regions = []

    def bind(self, value):
        name = f'_b{len(self.bindings)}'
        self.bindings[name] = value
        return name

    def render(self, value, names=None):
        if isinstance(value, Value):
            if names is not None:
                return names[value]
            if value.source is not None:
                return (value.source if value.kind in ('array', 'object')
                        else f'_normalize({value.source})')
            return value.name
        if isinstance(value, Metadata):
            return value.source
        if type(value) is tuple:
            if not value:
                return '()'
            return '(' + ', '.join(self.render(v, names) for v in value) + ',)'
        if type(value) in (type(None), bool, int, float, complex, str):
            # Keep weak literal origin; scalar arithmetic uses registered
            # ufuncs, never NumPy host arithmetic during analysis or tracing.
            if type(value) is int:
                # Input conversion is checked, but integer constants inside
                # the function are represented directly in fixed-width form.
                value = ((value + (1 << 63)) % (1 << 64)) - (1 << 63)
            elif type(value) in _PYTHON_DTYPES:
                value = _normalize(value)
            return self.bind(value)
        if type(value) in _SCALAR_TYPES or any(value is t for t in (
                np.float16, np.float32, np.float64, np.int32, np.int64,
                np.uint8, np.complex64, np.complex128)):
            return self.bind(value)
        raise Unsupported(f'Unsupported constant: {type(value).__name__}')

    def build(self):
        args = ', '.join(f'a{i}' for i in range(self.nargs))
        lines = [f'def execute({args}):']
        nodes = self.builder.nodes
        # A multi-reduction grid-synchronized trace is often much slower than
        # ordinary ReductionKernel launches. Start with explicit single-
        # reduction stages, retaining the legacy fast path for each one.
        staged_reductions = sum(n.operation.kind == 'reduction'
                                for n in nodes) > 1
        start = 0
        while start < len(nodes):
            node = nodes[start]
            if node.operation.kind == 'allocation':
                expr = node.operation.emit(node.arguments, node.keywords,
                                           self.render, self.bind)
                lines.append(f'    {node.output.name} = {expr}')
                start += 1
                continue
            end = start
            while (end < len(nodes)
                   and nodes[end].operation.kind != 'allocation'):
                end += 1
                if (staged_reductions and
                        nodes[end - 1].operation.kind == 'reduction'):
                    break
            segment = nodes[start:end]
            produced = {n.output for n in segment}
            external = []
            for n in segment:
                for value in _walk_values((n.arguments,
                                           tuple(n.keywords.values()))):
                    if value not in produced and value not in external:
                        external.append(value)
            needed = set(_walk_values(self.result))
            for n in nodes[end:]:
                needed.update(_walk_values((n.arguments,
                                            tuple(n.keywords.values()))))
            outputs = [n.output for n in segment if n.output in needed]
            if not outputs:
                raise Unsupported('Discarded arithmetic regions '
                                  'are unsupported')
            names = {v: f'p{i}' for i, v in enumerate(external)}
            body = [f'def region({", ".join(names.values())}):']
            for n in segment:
                expr = n.operation.emit(n.arguments, n.keywords,
                                        lambda v: self.render(v, names),
                                        self.bind)
                names[n.output] = n.output.name
                body.append(f'    {n.output.name} = {expr}')
            body.append('    return ' + self.render(
                outputs[0] if len(outputs) == 1 else tuple(outputs), names))
            source = '\n'.join(body) + '\n'
            function = _compile_source(source, self.bindings, 'region')
            fused = RegionKernel(function,
                                 f'bytecode_region_{len(self.regions)}')
            operations = tuple(n.operation.name for n in segment)
            self.regions.append(Region(function, source, fused, operations))
            target = ', '.join(v.name for v in outputs)
            params = ', '.join(self.render(v) for v in external)
            lines.append(f'    {target} = {self.bind(fused)}({params})')
            start = end
        lines.append(f'    return {self.render(self.result)}')
        source = '\n'.join(lines) + '\n'
        execute = _compile_source(source, self.bindings, 'execute')
        tests = [f'(_bool({src}) == {outcome!r})'
                 for src, outcome in self.builder.guards]
        guard = f'def accepts({args}):\n    return ' + (
            ' and '.join(tests) if tests else 'True') + '\n'
        accepts = _compile_source(guard, self.bindings, 'accepts')
        report = {
            'status': 'analysed', 'callees': self.builder.callees,
            'guards': [{'expression': s, 'outcome': v}
                       for s, v in self.builder.guards],
            'allocations': [n.location | {'operation': n.operation.name}
                            for n in nodes
                            if n.operation.kind == 'allocation'],
            'regions': [{'operations': list(r.operations), 'source': r.source,
                         'backend': 'legacy_fusion_or_new_fusion'}
                        for r in self.regions],
            'kernel_launches': None,
            'kernel_launch_note': 'Not instrumented; regions are not launches',
            'scalar_contract': {'int': 'int64', 'float': 'float64',
                                'complex': 'complex128', 'bool': 'uint8',
                                'numpy_warnings': False},
        }
        return Plan(execute, accepts, self.regions, source, report)


class FusedFunction:
    """Private prototype; call clear_cache after changing code or bindings."""

    def __init__(self, function):
        if type(function) is not types.FunctionType:
            raise Unsupported('Decorate a Python function definition')
        self.func = function
        self.__name__ = function.__name__
        self.__doc__ = function.__doc__
        self._signature = inspect.signature(function)
        self._positional = tuple(self._signature.parameters)
        self._fast_bind = all(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in self._signature.parameters.values())
        self._analyzer = Analyzer()
        self._plans = {}
        self.diagnostics = None

    def __get__(self, instance, owner=None):
        return self if instance is None else types.MethodType(self, instance)

    def clear_cache(self):
        """Clear this wrapper's plans and transitive callee analyses."""
        self._analyzer = Analyzer()
        self._plans.clear()
        self._signature = inspect.signature(self.func)
        self._positional = tuple(self._signature.parameters)
        self._fast_bind = all(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in self._signature.parameters.values())
        self.diagnostics = None

    def _arguments(self, args, kwargs):
        if (self._fast_bind and not kwargs
                and len(args) == len(self._positional)):
            return args
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
               for p in self._signature.parameters.values()):
            raise Unsupported('Variadic root signatures are unsupported')
        return tuple(bound.arguments[key] for key in self._positional)

    def _key(self, args):
        key = []
        for index, arg in enumerate(args):
            typ = type(arg)
            if typ is cp.ndarray:
                key.append((arg.dtype, arg.ndim))
            elif typ in _PYTHON_DTYPES:
                # Validate before selecting a cached plan, including bools.
                _normalize(arg)
                key.append((typ, bool(arg) if typ is bool else None))
            elif typ in _SCALAR_TYPES:
                key.append((arg.dtype, bool(arg) if typ is np.bool_ else None))
            elif index == 0 and self._positional[0] in ('self', 'cls'):
                key.append(('receiver', typ))
            else:
                raise Unsupported(f'Unsupported input type: {typ.__name__}')
        return tuple(key)

    def _select(self, args):
        try:
            key = self._key(args)
        except Exception as exc:
            self.diagnostics = {'status': 'rejected',
                                'error': f'{type(exc).__name__}: {exc}'}
            raise
        candidates = self._plans.get(key, ())
        for plan in candidates:
            if plan.accepts(*args):
                return plan, True
        before = time.perf_counter()
        builder = Builder(self._analyzer)
        inputs = tuple(builder.input(
            f'a{i}', arg, receiver=(i == 0 and self._positional[0]
                                    in ('self', 'cls')))
            for i, arg in enumerate(args))
        try:
            positional, keywords = [], {}
            for parameter, value in zip(self._signature.parameters.values(),
                                        inputs):
                if parameter.kind == parameter.KEYWORD_ONLY:
                    keywords[parameter.name] = value
                else:
                    positional.append(value)
            try:
                result = builder.run(self.func, positional, keywords)
                plan = Emitter(builder, len(args), result).build()
            except Unsupported as original:
                # Iterative functions need device control flow, which cannot
                # be represented by a straight-line tracing backend.
                from cupy._core._fusion_bytecode_kernel import compile_kernel
                try:
                    plan = compile_kernel(self.func, args, self._analyzer)
                except Unsupported as kernel_error:
                    raise Unsupported(f'{original}\nSingle-kernel lowering: '
                                      f'{kernel_error}') from None
        except Exception as exc:
            self.diagnostics = {
                'status': 'rejected',
                'error': f'{type(exc).__name__}: {exc}',
                'analysis_seconds': time.perf_counter() - before}
            raise
        plan.report['analysis_seconds'] = time.perf_counter() - before
        plan.report['analysed_functions'] = len(self._analyzer.functions)
        plan.report['analysis_cache_hits'] = self._analyzer.hits
        self._plans.setdefault(key, []).append(plan)
        return plan, False

    def explain(self, *args, **kwargs):
        """Analyse without user execution, allocation or CUDA compilation."""
        args = self._arguments(args, kwargs)
        plan, hit = self._select(args)
        self.diagnostics = dict(plan.report, cache_hit=hit)
        return self.diagnostics

    def __call__(self, *args, **kwargs):
        args = self._arguments(args, kwargs)
        plan, hit = self._select(args)
        # Execution errors propagate. Never retry the original function.
        self.diagnostics = dict(plan.report, cache_hit=hit, status='executing')
        try:
            result = plan.execute(*args)
        except Exception as exc:
            self.diagnostics.update(status='execution_failed',
                                    error=f'{type(exc).__name__}: {exc}')
            raise
        self.diagnostics['status'] = 'executed'
        self.diagnostics['regions'] = [
            dict(info, backend=region.backend,
                 legacy_rejection=region.fusion.legacy_rejection)
            for info, region in zip(plan.report['regions'], plan.regions)]
        return result


def fuse(function):
    """Build a private bytecode-fusion prototype without changing cupy.fuse."""
    return FusedFunction(function)
