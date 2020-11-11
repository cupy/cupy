import ast
import inspect

import numpy

from cupyx.jit import _types
from cupyx.jit import _typerules


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
    consts = dict(**global_mems, **nonlocals)
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

    def __repr__(self):
        return f'<CudaObject code = "{self.code}", type = {self.ctype}>'


class Constant:
    def __init__(self, obj):
        self.obj = obj


def is_constants(values):
    return all(isinstance(x, Constant) for x in values)


class Environment:
    """Environment of the scope

    Attributes:
        mode ('numpy' or 'cuda'): The rule for typecast.
        consts (dict): The dictionary with keys as the variable names and
            the values as the data stored at the global scopes.
        params (dict): The dictionary of function arguments with keys as
            the variable names and the values as the CudaObject.
        values (dict): The dictionary with keys as the variable names and the
            values as the CudaObject stored at the local scope of the function.
        ret_type (_types.TypeBase): The type of return value of the function.
            If it is initialized to be ``None``, the return type must be
            inferred until the end of transpilation of the function.
    """

    def __init__(self, mode, consts, params, ret_type):
        self.mode = mode
        self.consts = consts
        self.params = params
        self.values = {}
        self.ret_type = ret_type
        self.preambles = set()

    def __getitem__(self, key):
        if key in self.values:
            return self.values[key]
        if key in self.params:
            return self.params[key]
        if key in self.consts:
            return self.consts[key]
        return None

    def __setitem__(self, key, value):
        self.values[key] = value


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

    Returns (str):
        The generated CUDA code.
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
    params = ', '.join([f'{env[a].ctype} {a}' for a in args])
    body = _transpile_stmts(func.body, env)
    function_decl = f'{attributes} {env.ret_type} {func.name}({params})'
    local_vars = _indent([f'{v.ctype} {n};' for n, v in env.values.items()])
    return '\n'.join([function_decl + ' {'] + local_vars + body + ['}']), env


def _eval_operand(op, args, env, dtype=None):
    if is_constants(args):
        pyfunc = _typerules.get_pyfunc(type(op))
        return Constant(pyfunc(*[x.obj for x in args]))

    args = [_to_cuda_object(x, env) for x in args]
    ufunc = _typerules.get_ufunc(env.mode, type(op))
    assert ufunc.nin == len(args)
    assert ufunc.nout == 1
    for x in args:
        if not isinstance(x.ctype, _types.Scalar):
            raise NotImplementedError

    in_types = tuple([x.ctype.dtype for x in args])
    if dtype is None:
        op = ufunc._ops._guess_routine_from_in_types(in_types)
    else:
        op = ufunc._ops._guess_routine_from_dtype(dtype)

    if op is None:
        raise TypeError(
            f'"{ufunc.name}" does not support for the input types: {in_types}')
    if op.routine is None:
        op.error_func()

    assert op.routine.startswith('out0 = ')
    out_type = _types.Scalar(op.out_types[0])
    expr = op.routine[7:]
    for i, x in enumerate(args):
        x = astype_scalar(x, _types.Scalar(op.in_types[i]))
        expr = expr.replace(f'in{i}', x.code)
    expr = expr.replace('out0_type', str(out_type))
    env.preambles.add(ufunc._preamble)

    return CudaObject('(' + expr + ')', out_type)


def _transpile_stmts(stmts, env):
    return _indent([_transpile_stmt(stmt, env) for stmt in stmts])


def _transpile_stmt(stmt, env):
    """Transpile the statement.

    Returns (str): The generated CUDA code.
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
        return f'return {value.code};'
    if isinstance(stmt, ast.Delete):
        raise NotImplementedError('`del` is not supported currently.')
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1:
            raise NotImplementedError('Not implemented.')
        target = stmt.targets[0]
        name = target.id
        value = _transpile_expr(stmt.value, env)
        value = _to_cuda_object(value, env)
        if isinstance(target, ast.Name):
            if env[name] is None:
                env[name] = CudaObject(target.id, value.ctype)
            elif env[name].ctype.dtype != value.ctype.dtype:
                raise TypeError('dtype mismatch.')
            return f'{target.id} = {value.code};'
        raise NotImplementedError('Not implemented')
    if isinstance(stmt, ast.AugAssign):
        value = _transpile_expr(stmt.value, env)
        target = _transpile_expr(stmt.target, env)
        assert isinstance(target, CudaObject)
        value = _to_cuda_object(value, env)
        result = _eval_operand(stmt.op, (target, value), env)
        if not numpy.can_cast(
                result.ctype.dtype, target.ctype.dtype, 'same_kind'):
            raise TypeError('dtype mismatch')
        return f'{target.code} = {result.code};'
    if isinstance(stmt, ast.For):
        raise NotImplementedError('Not implemented.')
    if isinstance(stmt, ast.AsyncFor):
        raise ValueError('`async for` is not allowed.')
    if isinstance(stmt, ast.While):
        raise NotImplementedError('Not implemented.')
    if isinstance(stmt, ast.If):
        raise NotImplementedError('Not implemented.')
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        raise ValueError('Switching contexts are not allowed.')
    if isinstance(stmt, (ast.Raise, ast.Try)):
        raise ValueError('throw/catch are not allowed.')
    if isinstance(stmt, ast.Assert):
        value = _transpile_expr(stmt.test, env)
        if is_constants([value]):
            assert value.obj
            return ';'
        else:
            return 'assert(' + value + ');'
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        raise ValueError('Cannot import modules from the target functions.')
    if isinstance(stmt, (ast.Global, ast.Nonlocal)):
        raise ValueError('Cannot use global/nonlocal in the target functions.')
    if isinstance(stmt, ast.Expr):
        value = _transpile_expr(stmt.value, env)
        return ';' if is_constants([value]) else value
    if isinstance(stmt, ast.Pass):
        return ';'
    if isinstance(stmt, ast.Break):
        return NotImplementedError('Not implemented.')
    if isinstance(stmt, ast.Continue):
        return NotImplementedError('Not implemented.')
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
                f'Type mismatch in conditional expression.: '
                '{x.ctype.dtype} != {y.ctype.dtype}')
        cond = astype_scalar(cond, _types.Scalar(numpy.bool_))
        return CudaObject(f'({cond.code} ? {x.code} : {y.code})', x.ctype)
    if isinstance(expr, ast.Call):
        raise NotImplementedError('Not implemented.')
    if isinstance(expr, ast.Constant):
        return Constant(expr.value)
    if isinstance(expr, ast.Num):
        # Deprecated since py3.8
        return Constant(expr.n)
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
                'Unbound name: {} in L{}'.format(expr.id, expr.lineno))
        return env[expr.id]
    if isinstance(expr, ast.Attribute):
        value = _transpile_expr(expr.value, env)
        if is_constants([value]):
            return Constant(getattr(value.obj, expr.attr))
        raise NotImplementedError('Not implemented: __getattr__')
    raise ValueError('Not supported: type {}'.format(type(expr)))


def astype_scalar(x, ctype):
    if x.ctype.dtype == ctype.dtype:
        return x
    return CudaObject(f'({ctype})({x.code})', ctype)


def _to_cuda_object(x, env):
    if isinstance(x, CudaObject):
        return x
    if isinstance(x, Constant):
        ctype = _typerules.get_ctype_from_scalar(env.mode, x.obj)
        return CudaObject(str(x.obj).lower(), ctype)
    assert False
