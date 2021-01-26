import ast
import inspect
import re

import numpy

from cupy.core import _kernel
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


def is_constants(values):
    return all(isinstance(x, Constant) for x in values)


class Environment:
    """Environment of the scope

    Attributes:
        mode ('numpy' or 'cuda'): The rule for typecast.
        globals (dict): The dictionary with keys as the variable names and
            the values as the data stored at the global scopes.
        params (dict): The dictionary of function arguments with keys as
            the variable names and the values as the CudaObject.
        locals (dict): The dictionary with keys as the variable names and the
            values as the CudaObject stored at the local scope of the function.
        ret_type (_types.TypeBase): The type of return value of the function.
            If it is initialized to be ``None``, the return type must be
            inferred until the end of transpilation of the function.
    """

    def __init__(self, mode, globals, params, ret_type):
        self.mode = mode
        self.globals = globals
        self.params = params
        self.locals = {}
        self.ret_type = ret_type
        self.preambles = set()

    def __getitem__(self, key):
        if key in self.locals:
            return self.locals[key]
        if key in self.params:
            return self.params[key]
        if key in self.globals:
            return self.globals[key]
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
    local_vars = _indent([f'{v.ctype} {n};' for n, v in env.locals.items()])
    return '\n'.join([function_decl + ' {'] + local_vars + body + ['}']), env


def _eval_operand(op, args, env):
    if is_constants(args):
        pyfunc = _typerules.get_pyfunc(type(op))
        return Constant(pyfunc(*[x.obj for x in args]))

    ufunc = _typerules.get_ufunc(env.mode, type(op))
    return _call_ufunc(ufunc, args, None, env)


def _call_ufunc(ufunc, args, dtype, env):
    if len(args) != ufunc.nin:
        raise ValueError('invalid number of arguments')

    args = [_to_cuda_object(x, env) for x in args]

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

    if op.error_func is not None:
        op.error_func()

    if ufunc.nout == 1 and op.routine.startswith('out0 = '):
        out_type = _types.Scalar(op.out_types[0])
        expr = op.routine.replace('out0 = ', '')
        args = [_astype_scalar(x, _types.Scalar(t), 'same_kind')
                for x, t in zip(args, op.in_types)]

        can_use_inline_expansion = True
        for i in range(ufunc.nin):
            if len(list(re.finditer(r'in{}'.format(i), op.routine))) > 1:
                can_use_inline_expansion = False

        if can_use_inline_expansion:
            # Code pass for readable generated code
            for i, x in enumerate(args):
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
            in_params = ', '.join([a.code for a in args])
            expr = f'{ufunc_name}({in_params})'
        return CudaObject(expr, out_type)

    raise NotImplementedError(f'ufunc `{ufunc.name}` is not supported.')


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
        return ';' if is_constants([value]) else value + ';'
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
                'Type mismatch in conditional expression.: '
                f'{x.ctype.dtype} != {y.ctype.dtype}')
        cond = _astype_scalar(cond, _types.Scalar(numpy.bool_), 'unsafe')
        return CudaObject(f'({cond.code} ? {x.code} : {y.code})', x.ctype)
    if isinstance(expr, ast.Call):
        func = _transpile_expr(expr.func, env).obj
        args = [_transpile_expr(x, env) for x in expr.args]
        kwargs = dict([(kw.arg, _transpile_expr(kw.value, env))
                       for kw in expr.keywords])
        if isinstance(func, _kernel.ufunc):
            dtype = kwargs.pop('dtype', Constant(None)).obj
            if len(kwargs) > 0:
                name = next(iter(kwargs))
                raise TypeError(
                    f"'{name}' is an invalid keyword to ufunc {func.name}")
            return _call_ufunc(func, args, dtype, env)
        raise NotImplementedError('non-ufunc function call is not implemented')

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


def _astype_scalar(x, ctype, casting):
    from_t = x.ctype.dtype
    to_t = ctype.dtype
    if from_t == to_t:
        return x
    # Uses casting rules for scalar values.
    if not numpy.can_cast(from_t.type(0), to_t.type(0), casting):
        raise TypeError(
            f"Cannot cast from '{from_t}' to {to_t} "
            f"with casting rule {casting}.")
    return CudaObject(f'({ctype})({x.code})', ctype)


def _to_cuda_object(x, env):
    if isinstance(x, CudaObject):
        return x
    if isinstance(x, Constant):
        ctype = _typerules.get_ctype_from_scalar(env.mode, x.obj)
        return CudaObject(str(x.obj).lower(), ctype)
    assert False
