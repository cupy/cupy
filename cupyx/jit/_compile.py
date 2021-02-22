import ast
import builtins
import inspect
import numbers
import re
import warnings

import numpy

from cupyx.jit._codeblock import CodeBlock
from cupy.core import _kernel
from cupyx.jit import _types
from cupyx.jit import _typerules


_typeclasses = (bool, numpy.bool_, numbers.Number)


def transpile(func, attributes, mode, in_types, ret_type):
    """Transpile the target function
    Args:
        func (function): Target function.
        attributes (list of str): Attributes of the generated CUDA function.
        mode ('numpy' or 'cuda'): The rule for typecast.
        in_types (list of _types.TypeBase): Types of the arguments.
        ret_type (_types.TypeBase or None): Type of the return value.
    """

    if not callable(func):
        raise ValueError('`func` must be a callable object.')

    if func.__name__ == '<lambda>':
        raise NotImplementedError('Lambda function is not supported.')

    attributes = ' '.join(attributes)
    source = inspect.getsource(func)
    lines = source.split('\n')
    num_indent = len(lines[0]) - len(lines[0].lstrip())
    source = '\n'.join([
        line.replace(' ' * num_indent, '', 1) for line in lines])

    global_mems = dict(inspect.getclosurevars(func).globals)
    nonlocals = dict(inspect.getclosurevars(func).nonlocals)
    consts = dict(**global_mems, **nonlocals, **builtins.__dict__)
    tree = ast.parse(source)
    assert isinstance(tree, ast.Module)
    assert len(tree.body) == 1
    cuda_code, env = _transpile_function(
        tree.body[0], attributes, mode, consts, in_types, ret_type)
    cuda_code = ''.join([code + '\n' for code in env.preambles]) + cuda_code
    return cuda_code, env.ret_type


def _indent(lines, spaces='  '):
    return [spaces + line for line in lines]


class CudaObject:
    def __init__(self, code, ctype):
        self.code = code
        self.ctype = ctype

    @property
    def obj(self):
        raise ValueError(f'Constant value is requried: {self.code}')

    def __repr__(self):
        return f'<CudaObject code = "{self.code}", type = {self.ctype}>'


class Constant:
    def __init__(self, obj):
        self._obj = obj

    @property
    def obj(self):
        return self._obj

    def __repr__(self):
        return f'<Constant obj = "{self.obj}">'


class Range:

    def __init__(self, start, stop, step, step_is_positive):
        self.start = start
        self.stop = stop
        self.step = step
        self.ctype = stop.ctype
        self.step_is_positive = step_is_positive  # True, False or None

        if self.ctype.dtype.kind not in 'iu':
            raise TypeError('range supports only for integer type.')
        if self.ctype.dtype != start.ctype.dtype:
            raise TypeError(f'dtype mismatch: {self.ctype} != {start.ctype}')
        if self.ctype.dtype != step.ctype.dtype:
            raise TypeError(f'dtype mismatch: {self.ctype} != {step.ctype}')


def is_constants(values):
    return all(isinstance(x, Constant) for x in values)


class Environment:
    """Environment of the scope

    Attributes:
        mode ('numpy' or 'cuda'): The rule for typecast.
        consts (dict): The dictionary with keys as the variable names and
            the values as the data that is determined at compile-time.
        params (dict): The dictionary of function arguments with keys as
            the variable names and the values as the CudaObject.
        locals (dict): The dictionary with keys as the variable names and the
            values as the CudaObject stored at the local scope of the function.
        ret_type (_types.TypeBase): The type of return value of the function.
            If it is initialized to be ``None``, the return type must be
            inferred until the end of transpilation of the function.
    """

    def __init__(self, mode, consts, params, ret_type):
        self.mode = mode
        self.consts = consts
        self.params = params
        self.locals = {}
        self.ret_type = ret_type
        self.preambles = set()

    def __getitem__(self, key):
        if key in self.locals:
            return self.locals[key]
        if key in self.params:
            return self.params[key]
        if key in self.consts:
            return self.consts[key]
        return None

    def __setitem__(self, key, value):
        self.locals[key] = value


def _transpile_function(
        func, attributes, mode, consts, in_types, ret_type):
    """Transpile the function
    Args:
        func (ast.FunctionDef): Target function.
        attributes (str): The attributes of target function.
        mode ('numpy' or 'cuda'): The rule for typecast.
        consts (dict): The dictionary with keys as variable names and
            values as concrete data object.
        in_types (list of _types.TypeBase): The types of arguments.
        ret_type (_types.TypeBase): The type of return value.

    Returns:
        code (str): The generated CUDA code.
        env (Environment): More details of analysis result of the function,
            which includes preambles, estimated return type and more.
    """
    if not isinstance(func, ast.FunctionDef):
        # TODO(asi1024): Support for `ast.ClassDef`.
        raise NotImplementedError('Not supported: {}'.format(type(func)))
    if len(func.decorator_list) > 0:
        raise NotImplementedError('Decorator is not supported')
    arguments = func.args
    if arguments.vararg is not None:
        raise NotImplementedError('`*args` is not supported currently.')
    if len(arguments.kwonlyargs) > 0:  # same length with `kw_defaults`.
        raise NotImplementedError(
            'keyword only arguments are not supported currently .')
    if arguments.kwarg is not None:
        raise NotImplementedError('`**kwargs` is not supported currently.')
    if len(arguments.defaults) > 0:
        raise NotImplementedError(
            'Default values are not supported currently.')

    args = [arg.arg for arg in arguments.args]
    if len(args) != len(in_types):
        raise TypeError(
            f'{func.name}() takes {len(args)} positional arguments '
            'but {len(in_types)} were given.')
    env = Environment(
        mode,
        dict([(k, Constant(v)) for k, v, in consts.items()]),
        dict([(x, CudaObject(x, t)) for x, t in zip(args, in_types)]),
        ret_type)
    body = _transpile_stmts(func.body, True, env)
    params = ', '.join([f'{env[a].ctype} {a}' for a in args])
    local_vars = [f'{v.ctype} {n};' for n, v in env.locals.items()]
    head = f'{attributes} {env.ret_type} {func.name}({params})'
    code = CodeBlock(head, local_vars + body)
    return str(code), env


def _eval_operand(op, args, env):
    if is_constants(args):
        pyfunc = _typerules.get_pyfunc(type(op))
        return Constant(pyfunc(*[x.obj for x in args]))

    ufunc = _typerules.get_ufunc(env.mode, type(op))
    return _call_ufunc(ufunc, args, None, env)


def _call_ufunc(ufunc, args, dtype, env):
    if len(args) != ufunc.nin:
        raise ValueError('invalid number of arguments')

    in_types = []
    for x in args:
        if is_constants([x]):
            t = _typerules.get_ctype_from_scalar(env.mode, x.obj).dtype
        else:
            t = x.ctype.dtype
        in_types.append(t)

    if dtype is None:
        op = ufunc._ops._guess_routine_from_in_types(tuple(in_types))
    else:
        op = ufunc._ops._guess_routine_from_dtype(dtype)

    if op is None:
        raise TypeError(
            f'"{ufunc.name}" does not support for the input types: {in_types}')

    if op.error_func is not None:
        op.error_func()

    if ufunc.nout == 1 and op.routine.startswith('out0 = '):
        out_type = _types.Scalar(op.out_types[0])
        expr = op.routine.replace('out0 = ', '')

        in_params = []
        for x, t in zip(args, op.in_types):
            x = _astype_scalar(x, _types.Scalar(t), 'same_kind', env)
            x = _to_cuda_object(x, env)
            in_params.append(x)

        can_use_inline_expansion = True
        for i in range(ufunc.nin):
            if len(list(re.finditer(r'in{}'.format(i), op.routine))) > 1:
                can_use_inline_expansion = False

        if can_use_inline_expansion:
            # Code pass for readable generated code
            for i, x in enumerate(in_params):
                expr = expr.replace(f'in{i}', x.code)
            expr = '(' + expr.replace('out0_type', str(out_type)) + ')'
            env.preambles.add(ufunc._preamble)
        else:
            template_typenames = ', '.join([
                f'typename T{i}' for i in range(ufunc.nin)])
            ufunc_name = f'{ufunc.name}_{str(numpy.dtype(op.out_types[0]))}'
            params = ', '.join([f'T{i} in{i}' for i in range(ufunc.nin)])
            ufunc_code = f"""template <{template_typenames}>
__device__ {out_type} {ufunc_name}({params}) {{
    return {expr};
}}
"""
            env.preambles.add(ufunc_code)
            in_params = ', '.join([a.code for a in in_params])
            expr = f'{ufunc_name}({in_params})'
        return CudaObject(expr, out_type)

    raise NotImplementedError(f'ufunc `{ufunc.name}` is not supported.')


def _transpile_stmts(stmts, is_toplevel, env):
    codeblocks = []
    for stmt in stmts:
        codeblocks.extend(_transpile_stmt(stmt, is_toplevel, env))
    return codeblocks


def _transpile_stmt(stmt, is_toplevel, env):
    """Transpile the statement.

    Returns (list of [CodeBlock or str]): The generated CUDA code.
    """

    if isinstance(stmt, ast.ClassDef):
        raise NotImplementedError('class is not supported currently.')
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise NotImplementedError(
            'Nested functions are not supported currently.')
    if isinstance(stmt, ast.Return):
        value = _transpile_expr(stmt.value, env)
        value = _to_cuda_object(value, env)
        t = value.ctype
        if env.ret_type is None:
            env.ret_type = t
        elif env.ret_type != t:
            raise ValueError(
                f'Failed to infer the return type: {env.ret_type} or {t}')
        return [f'return {value.code};']
    if isinstance(stmt, ast.Delete):
        raise NotImplementedError('`del` is not supported currently.')

    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1:
            raise NotImplementedError('Not implemented.')
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            raise NotImplementedError('Tuple is not supported.')
        name = target.id
        value = _transpile_expr(stmt.value, env)

        if is_constants([value]):
            if not isinstance(value.obj, _typeclasses):
                if is_toplevel:
                    if env[name] is not None and not is_constants([env[name]]):
                        raise TypeError(f'Type mismatch of variable: `{name}`')
                    env.consts[name] = value
                    return []
                else:
                    raise TypeError(
                        'Cannot assign constant value not at top-level.')
            value = _to_cuda_object(value, env)

        if env[name] is None:
            env[name] = CudaObject(target.id, value.ctype)
        elif is_constants([env[name]]):
            raise TypeError('Type mismatch of variable: `{name}`')
        elif env[name].ctype.dtype != value.ctype.dtype:
            raise TypeError(
                f'Data type mismatch of variable: `{name}`: '
                f'{env[name].ctype.dtype} != {value.ctype.dtype}')
        return [f'{target.id} = {value.code};']

    if isinstance(stmt, ast.AugAssign):
        value = _transpile_expr(stmt.value, env)
        target = _transpile_expr(stmt.target, env)
        assert isinstance(target, CudaObject)
        value = _to_cuda_object(value, env)
        result = _eval_operand(stmt.op, (target, value), env)
        if not numpy.can_cast(
                result.ctype.dtype, target.ctype.dtype, 'same_kind'):
            raise TypeError('dtype mismatch')
        return [f'{target.code} = {result.code};']

    if isinstance(stmt, ast.For):
        if len(stmt.orelse) > 0:
            raise NotImplementedError('while-else is not supported.')
        name = stmt.target.id
        iters = _transpile_expr(stmt.iter, env)

        if env[name] is None:
            env[name] = CudaObject(stmt.target.id, iters.ctype)
        elif env[name].ctype.dtype != iters.ctype.dtype:
            raise TypeError(
                f'Data type mismatch of variable: `{name}`: '
                f'{env[name].ctype.dtype} != {iters.ctype.dtype}')

        body = _transpile_stmts(stmt.body, False, env)

        if not isinstance(iters, Range):
            raise NotImplementedError(
                'for-loop is supported only for range iterator.')

        init_code = (f'{iters.ctype} '
                     f'__it = {iters.start.code}, '
                     f'__stop = {iters.stop.code}, '
                     f'__step = {iters.step.code}')
        cond = f'__step >= 0 ? __it < __stop : __it > __stop'
        if iters.step_is_positive is True:
            cond = f'__it < __stop'
        elif iters.step_is_positive is False:
            cond = f'__it > __stop'

        head = f'for ({init_code}; {cond}; __it += __step)'
        return [CodeBlock(head, [f'{name} = __it;'] + body)]

    if isinstance(stmt, ast.AsyncFor):
        raise ValueError('`async for` is not allowed.')
    if isinstance(stmt, ast.While):
        if len(stmt.orelse) > 0:
            raise NotImplementedError('while-else is not supported.')
        condition = _transpile_expr(stmt.test, env)
        condition = _astype_scalar(condition, _types.bool_, 'unsafe', env)
        condition = _to_cuda_object(condition, env)
        body = _transpile_stmts(stmt.body, False, env)
        head = f'while ({condition.code})'
        return [CodeBlock(head, body)]
    if isinstance(stmt, ast.If):
        condition = _transpile_expr(stmt.test, env)
        if is_constants([condition]):
            stmts = stmt.body if condition.obj else stmt.orelse
            return _transpile_stmts(stmts, is_toplevel, env)
        head = f'if ({condition.code})'
        then_body = _transpile_stmts(stmt.body, False, env)
        else_body = _transpile_stmts(stmt.orelse, False, env)
        return [CodeBlock(head, then_body), CodeBlock('else', else_body)]
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        raise ValueError('Switching contexts are not allowed.')
    if isinstance(stmt, (ast.Raise, ast.Try)):
        raise ValueError('throw/catch are not allowed.')
    if isinstance(stmt, ast.Assert):
        value = _transpile_expr(stmt.test, env)
        if is_constants([value]):
            assert value.obj
            return [';']
        else:
            return ['assert(' + value + ');']
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        raise ValueError('Cannot import modules from the target functions.')
    if isinstance(stmt, (ast.Global, ast.Nonlocal)):
        raise ValueError('Cannot use global/nonlocal in the target functions.')
    if isinstance(stmt, ast.Expr):
        value = _transpile_expr(stmt.value, env)
        return [';'] if is_constants([value]) else [value + ';']
    if isinstance(stmt, ast.Pass):
        return [';']
    if isinstance(stmt, ast.Break):
        raise NotImplementedError('Not implemented.')
    if isinstance(stmt, ast.Continue):
        raise NotImplementedError('Not implemented.')
    assert False


def _transpile_expr(expr, env):
    """Transpile the statement.

    Returns (CudaObject): The CUDA code and its type of the expression.
    """

    if isinstance(expr, ast.BoolOp):
        values = [_transpile_expr(e, env) for e in expr.values]
        value = values[0]
        for rhs in values[1:]:
            value = _eval_operand(expr.op, (value, rhs), env)
        return value
    if isinstance(expr, ast.BinOp):
        left = _transpile_expr(expr.left, env)
        right = _transpile_expr(expr.right, env)
        return _eval_operand(expr.op, (left, right), env)
    if isinstance(expr, ast.UnaryOp):
        value = _transpile_expr(expr.operand, env)
        return _eval_operand(expr.op, (value,), env)
    if isinstance(expr, ast.Lambda):
        raise NotImplementedError('Not implemented.')
    if isinstance(expr, ast.Compare):
        values = [expr.left] + expr.comparators
        if len(values) != 2:
            raise NotImplementedError(
                'Comparison of 3 or more values is not implemented.')
        values = [_transpile_expr(e, env) for e in values]
        return _eval_operand(expr.ops[0], values, env)
    if isinstance(expr, ast.IfExp):
        cond = _transpile_expr(expr.test, env)
        x = _transpile_expr(expr.body, env)
        y = _transpile_expr(expr.orelse, env)

        if isinstance(expr, Constant):
            return x if expr.obj else y
        if cond.ctype.dtype.kind == 'c':
            raise NotImplementedError('')
        x = _to_cuda_object(x, env)
        y = _to_cuda_object(y, env)
        if x.ctype.dtype != y.ctype.dtype:
            raise TypeError(
                'Type mismatch in conditional expression.: '
                f'{x.ctype.dtype} != {y.ctype.dtype}')
        cond = _astype_scalar(cond, _types.Scalar(numpy.bool_), 'unsafe', env)
        return CudaObject(f'({cond.code} ? {x.code} : {y.code})', x.ctype)

    if isinstance(expr, ast.Call):
        func = _transpile_expr(expr.func, env).obj
        args = [_transpile_expr(x, env) for x in expr.args]
        kwargs = dict([(kw.arg, _transpile_expr(kw.value, env))
                       for kw in expr.keywords])

        if func is range:
            if len(args) == 0:
                raise TypeError('range expected at least 1 argument, got 0')
            elif len(args) == 1:
                start, stop, step = Constant(0), args[0], Constant(1)
            elif len(args) == 2:
                start, stop, step = args[0], args[1], Constant(1)
            elif len(args) == 3:
                start, stop, step = args
            else:
                raise TypeError(
                    f'range expected at most 3 argument, got {len(args)}')
            step_is_positive = step.obj >= 0 if is_constants([step]) else None
            start = _to_cuda_object(start, env)
            stop = _to_cuda_object(stop, env)
            step = _to_cuda_object(step, env)
            return Range(start, stop, step, step_is_positive)

        if is_constants(args) and is_constants(kwargs.values()):
            # compile-time function call
            args = [x.obj for x in args]
            kwargs = dict([(k, v.obj) for k, v in kwargs.items()])
            return Constant(func(*args, **kwargs))

        if isinstance(func, _kernel.ufunc):
            # ufunc call
            dtype = kwargs.pop('dtype', Constant(None)).obj
            if len(kwargs) > 0:
                name = next(iter(kwargs))
                raise TypeError(
                    f"'{name}' is an invalid keyword to ufunc {func.name}")
            return _call_ufunc(func, args, dtype, env)

        if inspect.isclass(func) and issubclass(func, _typeclasses):
            # explicit typecast
            if len(args) != 1:
                raise TypeError(
                    f'function takes {func} invalid number of argument')
            return _astype_scalar(args[0], _types.Scalar(func), 'unsafe', env)

        raise NotImplementedError(
            f'function call of `{func.__name__}` is not implemented')

    if isinstance(expr, ast.Constant):
        return Constant(expr.value)
    if isinstance(expr, ast.Num):
        # Deprecated since py3.8
        return Constant(expr.n)
    if isinstance(expr, ast.Str):
        # Deprecated since py3.8
        return Constant(expr.s)

    if isinstance(expr, ast.Subscript):
        # # TODO(asi1024): Fix.
        # value = _transpile_expr(expr.value, env)
        # if isinstance(expr.slice, ast.Index):
        #     index = _transpile_expr(expr.slice.value, env)
        #     return value + '[' + index + ']'
        raise NotImplementedError('Not implemented.')
    if isinstance(expr, ast.Name):
        value = env[expr.id]
        if value is None:
            raise NameError(
                f'Unbound name: {expr.id} in line {expr.lineno}')
        return env[expr.id]
    if isinstance(expr, ast.Attribute):
        value = _transpile_expr(expr.value, env)
        if is_constants([value]):
            return Constant(getattr(value.obj, expr.attr))
        raise NotImplementedError('Not implemented: __getattr__')
    raise ValueError('Not supported: type {}'.format(type(expr)))


def _astype_scalar(x, ctype, casting, env):
    if is_constants([x]):
        return Constant(ctype.dtype.type(x.obj))

    from_t = x.ctype.dtype
    to_t = ctype.dtype
    if from_t == to_t:
        return x
    # Uses casting rules for scalar values.
    if not numpy.can_cast(from_t.type(0), to_t.type(0), casting):
        raise TypeError(
            f"Cannot cast from '{from_t}' to {to_t} "
            f"with casting rule {casting}.")
    if from_t.kind == 'c' and to_t.kind != 'c':
        if to_t.kind != 'b':
            warnings.warn(
                'Casting complex values to real discards the imaginary part',
                numpy.ComplexWarning)
        return CudaObject(f'({ctype})({x.code}.real())', ctype)
    return CudaObject(f'({ctype})({x.code})', ctype)


def _to_cuda_object(x, env):
    if isinstance(x, CudaObject):
        return x
    if isinstance(x, Constant):
        ctype = _typerules.get_ctype_from_scalar(env.mode, x.obj)
        code = _typerules.get_cuda_code_from_constant(x.obj, ctype)
        return CudaObject(code, ctype)
    if isinstance(x, Range):
        raise TypeError('range object cannot be interpreted as a cuda object.')
    assert False
