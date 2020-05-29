import cupy.core
from cupyx.jit import _syntax

import ast
import inspect


def transpile(*args, **kwargs):
    codes = []
    for func in args:
        if not isinstance(func, _syntax.CudaFunction):
            raise ValueError(
                'The target function should be decorated with '
                '`cupyx.jit.cuda_function`')

        attributes = ' '.join(func.attributes)
        source = inspect.getsource(func.func)

        # Fix indentation
        lines = source.split('\n')
        num_indent = len(lines[0]) - len(lines[0].lstrip())
        source = '\n'.join([
            line.replace(' ' * num_indent, '', 1) for line in lines])

        global_mems = inspect.getclosurevars(func.func).globals
        global_mems = dict([
            (k, v) for k, v in global_mems.items() if inspect.ismodule(v)])
        tree = ast.parse(source)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        cuda_code = _transpile_function(tree.body[0], attributes, global_mems)
        codes.append(cuda_code)

    code = '\n'.join(codes)
    # print(code)
    return cupy.core.RawModule(code=code, **kwargs)


def _indent(lines, spaces='  '):
    return [spaces + line for line in lines]


class Environment:
    def __init__(self, global_mems, types):
        self.global_mems = global_mems
        self.types = types


def _transpile_function(func, attributes, global_mems):
    if not isinstance(func, ast.FunctionDef):
        # TODO(asi1024): Support for `ast.ClassDef`.
        raise NotImplementedError('Not supported: {}'.format(type(func)))
    if len(func.decorator_list) > 1:
        # TODO(asi1024): Checks if the decorator is `cuda_function`.
        raise NotImplementedError('Decorator is not supported')
    arguments = func.args
    if arguments.vararg is not None:
        raise ValueError('Cannot transpile `*args`.')
    if len(arguments.kwonlyargs) > 0:  # same length with `kw_defaults`.
        raise ValueError('Cannot transpile keyword only arguments.')
    if arguments.kwarg is not None:
        raise ValueError('Cannot transpile `**kwargs`.')
    if len(arguments.defaults) > 0:
        raise NotImplementedError(
            'Default values are not supported currently.')

    args = arguments.args
    arg_names = [arg.arg for arg in args]
    types = dict([(a, '_Type_{}'.format(a)) for a in arg_names])
    type_params = 'template<{}>'.format(
        ', '.join(['typename {}'.format(types[a]) for a in arg_names]))
    params = ', '.join(['{} {}'.format(types[a], a) for a in arg_names])
    function_decl = '{} void {}({})'.format(attributes, func.name, params)
    env = Environment(global_mems, types)
    body = _transpile_stmts(func.body, env)
    return '\n'.join([type_params, function_decl + ' {'] + body + ['}'])


_ops = {
    ast.And: '&&',
    ast.Or: '||',
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.Mod: '%',
    ast.LShift: '<<',
    ast.RShift: '>>',
    ast.BitOr: '|',
    ast.BitAnd: '&',
    ast.BitXor: '^',
    ast.Invert: '~',
    ast.Not: '!',
    ast.UAdd: '+',
    ast.USub: '-',
    ast.Eq: '==',
    ast.NotEq: '!=',
    ast.Lt: '<',
    ast.LtE: '<=',
    ast.Gt: '>',
    ast.GtE: '>=',
}


def _transpile_stmts(stmts, env):
    return _indent([_transpile_stmt(stmt, env) for stmt in stmts])


def _transpile_stmt(stmt, env):
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        raise ValueError('Cannot transpile nested functions.')
    if isinstance(stmt, ast.Return):
        value = _transpile_expr(stmt.value, env)
        return 'return ' + value
    if isinstance(stmt, ast.Delete):
        raise NotImplementedError('Not implemented.')
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1:
            raise NotImplementedError('Not implemented.')
        target = stmt.targets[0]
        value = _transpile_expr(stmt.value, env)
        if isinstance(target, ast.Name):
            env.types[target.id] = None
            return 'auto ' + target.id + ' = ' + value + ';'
        target = _transpile_expr(target, env)
        return target + ' = ' + value + ';'
    if isinstance(stmt, ast.AugAssign):
        value = _transpile_expr(stmt.value, env)
        target = _transpile_expr(stmt.target, env)
        op = ' ' + _ops[type(stmt.op)] + '= '
        return target + op + value + ';'
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
        return 'assert(' + _transpile_expr(stmt.test, env) + ')'
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        raise ValueError('Cannot import modules from the target functions.')
    if isinstance(stmt, (ast.Global, ast.Nonlocal)):
        raise ValueError('Cannot use global/nonlocal in the target functions.')
    if isinstance(stmt, ast.Expr):
        return _transpile_expr(stmt.value, env)
    if isinstance(stmt, ast.Pass):
        return ';'
    if isinstance(stmt, ast.Break):
        return 'break'
    if isinstance(stmt, ast.Continue):
        return 'continue'
    assert False


def _transpile_expr(expr, env):
    if isinstance(expr, ast.BoolOp):
        values = [_transpile_expr(e, env) for e in expr.values]
        op = ' ' + _ops[type(expr.op)] + ' '
        return '(' + op.join(values) + ')'
    if isinstance(expr, ast.BinOp):
        left = _transpile_expr(expr.left, env)
        right = _transpile_expr(expr.right, env)
        op = ' ' + _ops[type(expr.op)] + ' '
        return '(' + left + op + right + ')'
    if isinstance(expr, ast.UnaryOp):
        op = _ops[type(expr.op)]
        operand = _transpile_expr(expr.operand, env)
        return '(' + op + operand + ')'
    if isinstance(expr, ast.Lambda):
        raise NotImplementedError('Not implemented.')
    if isinstance(expr, ast.IfExp):
        raise NotImplementedError('Not implemented.')
    if isinstance(expr, ast.Compare):
        values = expr.left + expr.comparators
        values = [_transpile_expr(e, env) for e in values]
        ops = [_ops[type(op)] for op in expr.ops]
        conditions = [
            l + ' ' + op + ' ' + r
            for op, l, r in zip(ops, values[:-1], values[1:])]
        return '(' + ' && '.join(conditions) + ')'
    if isinstance(expr, ast.Call):
        if len(expr.keywords) > 0:
            raise ValueError('Cannot transpile keyword only arguments.')
        funcname = _transpile_expr(expr.func, env)
        args = [_transpile_expr(e, env) for e in expr.args]
        return funcname + '(' + ', '.join(args) + ')'
    if isinstance(expr, ast.Constant):
        return str(expr.value)
    if isinstance(expr, ast.Num):  # Deprecated since py3.8
        return str(expr.n)
    if isinstance(expr, ast.Subscript):
        value = _transpile_expr(expr.value, env)
        if isinstance(expr.slice, ast.Index):
            index = _transpile_expr(expr.slice.value, env)
            return value + '[' + index + ']'
        raise NotImplementedError('Not implemented.')
    if isinstance(expr, (ast.Name, ast.Attribute)):
        return _transpile_name(expr, env)
    raise ValueError('Not supported: type {}'.format(type(expr)))


def _transpile_name(expr, env):
    def _parse_attribute(expr):
        if isinstance(expr, ast.Name):
            if expr.id in env.types:
                return expr.id
            if expr.id in env.global_mems:
                e = env.global_mems[expr.id]
                return e.s if isinstance(e, _syntax.CudaObject) else e
            raise NameError(
                'Unbound name: {} in L{}'.format(expr.id, expr.lineno))
        if isinstance(expr, ast.Attribute):
            value = _parse_attribute(expr.value)
            if isinstance(value, str):
                return value + '.' + expr.attr
            e = getattr(value, expr.attr)
            return e.s if isinstance(e, _syntax.CudaObject) else e
        return _transpile_expr(expr, env)
    e = _parse_attribute(expr)
    if isinstance(e, str):
        return e
    raise ValueError('Transpile error: {}'.format(e))
