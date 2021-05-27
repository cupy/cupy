import ast
import collections
import inspect
import numbers
import re
import sys
import warnings

import numpy

from cupy._core._codeblock import CodeBlock
from cupy._core import _kernel
from cupyx import jit
from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit import _internal_types
from cupyx.jit._internal_types import Data
from cupyx.jit._internal_types import Constant
from cupyx.jit import _builtin_funcs


_is_debug_mode = False

_typeclasses = (bool, numpy.bool_, numbers.Number)

Result = collections.namedtuple('Result', ['func_name', 'code', 'return_type'])


class _JitCompileError(Exception):

    def __init__(self, e, node):
        self.error_type = type(e)
        self.mes = str(e)
        self.node = node

    def reraise(self, pycode):
        start = self.node.lineno
        end = getattr(self.node, 'end_lineno', start)
        pycode = '\n'.join([
            (f'> {line}' if start <= i + 1 <= end else f'  {line}').rstrip()
            for i, line in enumerate(pycode.split('\n'))])
        raise self.error_type(self.mes + '\n\n' + pycode)


def transpile_function_wrapper(func):
    def new_func(node, *args, **kwargs):
        try:
            return func(node, *args, **kwargs)
        except _JitCompileError:
            raise
        except Exception as e:
            raise _JitCompileError(e, node)

    return new_func


def transpile(func, attributes, mode, in_types, ret_type):
    """Transpile the target function
    Args:
        func (function): Target function.
        attributes (list of str): Attributes of the generated CUDA function.
        mode ('numpy' or 'cuda'): The rule for typecast.
        in_types (list of _cuda_types.TypeBase): Types of the arguments.
        ret_type (_cuda_types.TypeBase or None): Type of the return value.
    """

    if not callable(func):
        raise ValueError('`func` must be a callable object.')

    if func.__name__ == '<lambda>':
        raise NotImplementedError('Lambda function is not supported.')

    attributes = ' '.join(attributes)
    source = jit._getsource_func(func)
    lines = source.split('\n')
    num_indent = len(lines[0]) - len(lines[0].lstrip())
    source = '\n'.join([
        line.replace(' ' * num_indent, '', 1) for line in lines])

    cvars = inspect.getclosurevars(func)
    consts = dict(**cvars.globals, **cvars.nonlocals, **cvars.builtins)
    tree = ast.parse(source)
    assert isinstance(tree, ast.Module)
    assert len(tree.body) == 1
    cuda_code, env = _transpile_function(
        tree.body[0], attributes, mode, consts, in_types, ret_type,
        source=source
    )
    cuda_code = ''.join([code + '\n' for code in env.preambles]) + cuda_code
    return Result(
        func_name=func.__name__,
        code=cuda_code,
        return_type=env.ret_type,
    )


def _indent(lines, spaces='  '):
    return [spaces + line for line in lines]


def is_constants(*values):
    assert all(isinstance(x, _internal_types.Expr) for x in values)
    return all(isinstance(x, Constant) for x in values)


class Environment:
    """Environment of the scope

    Attributes:
        mode ('numpy' or 'cuda'): The rule for typecast.
        consts (dict): The dictionary with keys as the variable names and
            the values as the data that is determined at compile-time.
        params (dict): The dictionary of function arguments with keys as
            the variable names and the values as the Data.
        locals (dict): The dictionary with keys as the variable names and the
            values as the Data stored at the local scope of the function.
        ret_type (_cuda_types.TypeBase):
            The type of return value of the function.
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
        self.count = 0

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

    def get_fresh_variable_name(self, prefix='', suffix=''):
        self.count += 1
        return f'{prefix}{self.count}{suffix}'


def _transpile_function(
        func, attributes, mode, consts, in_types, ret_type, *, source):
    """Transpile the function
    Args:
        func (ast.FunctionDef): Target function.
        attributes (str): The attributes of target function.
        mode ('numpy' or 'cuda'): The rule for typecast.
        consts (dict): The dictionary with keys as variable names and
            values as concrete data object.
        in_types (list of _cuda_types.TypeBase): The types of arguments.
        ret_type (_cuda_types.TypeBase): The type of return value.

    Returns:
        code (str): The generated CUDA code.
        env (Environment): More details of analysis result of the function,
            which includes preambles, estimated return type and more.
    """
    try:
        return _transpile_function_internal(
            func, attributes, mode, consts, in_types, ret_type)
    except _JitCompileError as e:
        exc = e
        if _is_debug_mode:
            exc.reraise(source)

    # Raises the error out of `except` block to clean stack trace.
    exc.reraise(source)
    assert False


def _transpile_function_internal(
        func, attributes, mode, consts, in_types, ret_type):
    consts = dict([(k, Constant(v)) for k, v, in consts.items()])

    if not isinstance(func, ast.FunctionDef):
        # TODO(asi1024): Support for `ast.ClassDef`.
        raise NotImplementedError('Not supported: {}'.format(type(func)))
    if len(func.decorator_list) > 0:
        if sys.version_info >= (3, 9):
            # Code path for Python versions that support `ast.unparse`.
            for deco in func.decorator_list:
                deco_code = ast.unparse(deco)
                if not any(word in deco_code
                           for word in ['rawkernel', 'vectorize']):
                    warnings.warn(
                        f'Decorator {deco_code} may not supported in JIT.',
                        RuntimeWarning)
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
            f'but {len(in_types)} were given.')
    params = dict([(x, Data(x, t)) for x, t in zip(args, in_types)])
    env = Environment(mode, consts, params, ret_type)
    body = _transpile_stmts(func.body, True, env)
    params = ', '.join([env[a].ctype.declvar(a) for a in args])
    local_vars = [v.ctype.declvar(n) + ';' for n, v in env.locals.items()]

    if env.ret_type is None:
        env.ret_type = _cuda_types.void

    head = f'{attributes} {env.ret_type} {func.name}({params})'
    code = CodeBlock(head, local_vars + body)
    return str(code), env


def _eval_operand(op, args, env):
    if is_constants(*args):
        pyfunc = _cuda_typerules.get_pyfunc(type(op))
        return Constant(pyfunc(*[x.obj for x in args]))

    ufunc = _cuda_typerules.get_ufunc(env.mode, type(op))
    return _call_ufunc(ufunc, args, None, env)


def _call_ufunc(ufunc, args, dtype, env):
    if len(args) != ufunc.nin:
        raise ValueError('invalid number of arguments')

    in_types = []
    for x in args:
        if is_constants(x):
            t = _cuda_typerules.get_ctype_from_scalar(env.mode, x.obj).dtype
        else:
            t = x.ctype.dtype
        in_types.append(t)

    op = _cuda_typerules.guess_routine(ufunc, in_types, dtype, env.mode)

    if op is None:
        raise TypeError(
            f'"{ufunc.name}" does not support for the input types: {in_types}')

    if op.error_func is not None:
        op.error_func()

    if ufunc.nout == 1 and op.routine.startswith('out0 = '):
        out_type = _cuda_types.Scalar(op.out_types[0])
        expr = op.routine.replace('out0 = ', '')

        in_params = []
        for x, t in zip(args, op.in_types):
            x = _astype_scalar(x, _cuda_types.Scalar(t), 'same_kind', env)
            x = Data.init(x, env)
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
        return Data(expr, out_type)

    raise NotImplementedError(f'ufunc `{ufunc.name}` is not supported.')


def _transpile_stmts(stmts, is_toplevel, env):
    codeblocks = []
    for stmt in stmts:
        codeblocks.extend(_transpile_stmt(stmt, is_toplevel, env))
    return codeblocks


@transpile_function_wrapper
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
        value = Data.init(value, env)
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

        value = _transpile_expr(stmt.value, env)
        target = stmt.targets[0]

        if is_constants(value) and isinstance(target, ast.Name):
            name = target.id
            if not isinstance(value.obj, _typeclasses):
                if is_toplevel:
                    if env[name] is not None and not is_constants(env[name]):
                        raise TypeError(f'Type mismatch of variable: `{name}`')
                    env.consts[name] = value
                    return []
                else:
                    raise TypeError(
                        'Cannot assign constant value not at top-level.')
            value = Data.init(value, env)

        target = _transpile_lvalue(target, env, value.ctype)
        return [f'{target.code} = {value.code};']

    if isinstance(stmt, ast.AugAssign):
        value = _transpile_expr(stmt.value, env)
        target = _transpile_expr(stmt.target, env)
        assert isinstance(target, Data)
        value = Data.init(value, env)
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
            env[name] = Data(stmt.target.id, iters.ctype)
        elif env[name].ctype.dtype != iters.ctype.dtype:
            raise TypeError(
                f'Data type mismatch of variable: `{name}`: '
                f'{env[name].ctype.dtype} != {iters.ctype.dtype}')

        body = _transpile_stmts(stmt.body, False, env)

        if not isinstance(iters, _internal_types.Range):
            raise NotImplementedError(
                'for-loop is supported only for range iterator.')

        init_code = (f'{iters.ctype} '
                     f'__it = {iters.start.code}, '
                     f'__stop = {iters.stop.code}, '
                     f'__step = {iters.step.code}')
        cond = '__step >= 0 ? __it < __stop : __it > __stop'
        if iters.step_is_positive is True:
            cond = '__it < __stop'
        elif iters.step_is_positive is False:
            cond = '__it > __stop'

        head = f'for ({init_code}; {cond}; __it += __step)'
        return [CodeBlock(head, [f'{name} = __it;'] + body)]

    if isinstance(stmt, ast.AsyncFor):
        raise ValueError('`async for` is not allowed.')
    if isinstance(stmt, ast.While):
        if len(stmt.orelse) > 0:
            raise NotImplementedError('while-else is not supported.')
        condition = _transpile_expr(stmt.test, env)
        condition = _astype_scalar(condition, _cuda_types.bool_, 'unsafe', env)
        condition = Data.init(condition, env)
        body = _transpile_stmts(stmt.body, False, env)
        head = f'while ({condition.code})'
        return [CodeBlock(head, body)]
    if isinstance(stmt, ast.If):
        condition = _transpile_expr(stmt.test, env)
        if is_constants(condition):
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
        if is_constants(value):
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
        return [';'] if is_constants(value) else [value.code + ';']
    if isinstance(stmt, ast.Pass):
        return [';']
    if isinstance(stmt, ast.Break):
        raise NotImplementedError('Not implemented.')
    if isinstance(stmt, ast.Continue):
        raise NotImplementedError('Not implemented.')
    assert False


@transpile_function_wrapper
def _transpile_expr(expr, env):
    """Transpile the statement.

    Returns (Data): The CUDA code and its type of the expression.
    """
    res = _transpile_expr_internal(expr, env)

    if isinstance(res, Constant) and isinstance(res.obj, _internal_types.Expr):
        return res.obj
    else:
        return res


def _transpile_expr_internal(expr, env):
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
        x = Data.init(x, env)
        y = Data.init(y, env)
        if x.ctype.dtype != y.ctype.dtype:
            raise TypeError(
                'Type mismatch in conditional expression.: '
                f'{x.ctype.dtype} != {y.ctype.dtype}')
        cond = _astype_scalar(cond, _cuda_types.bool_, 'unsafe', env)
        return Data(f'({cond.code} ? {x.code} : {y.code})', x.ctype)

    if isinstance(expr, ast.Call):
        func = _transpile_expr(expr.func, env)
        args = [_transpile_expr(x, env) for x in expr.args]
        kwargs = dict([(kw.arg, _transpile_expr(kw.value, env))
                       for kw in expr.keywords])

        builtin_funcs = _builtin_funcs.builtin_functions_dict
        if is_constants(func) and (func.obj in builtin_funcs):
            func = builtin_funcs[func.obj]

        if isinstance(func, _internal_types.BuiltinFunc):
            return func.call(env, *args, **kwargs)

        if not is_constants(func):
            raise NotImplementedError(
                'device function call is not implemented.')

        func = func.obj

        if is_constants(*args, *kwargs.values()):
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
            ctype = _cuda_types.Scalar(func)
            return _astype_scalar(args[0], ctype, 'unsafe', env)

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
    if isinstance(expr, ast.NameConstant):
        # Deprecated since py3.8
        return Constant(expr.value)
    if isinstance(expr, ast.Subscript):
        array = _transpile_expr(expr.value, env)
        index = _transpile_expr(expr.slice, env)
        return _indexing(array, index, env)
    if isinstance(expr, ast.Name):
        value = env[expr.id]
        if value is None:
            raise NameError(f'Unbound name: {expr.id}')
        return env[expr.id]
    if isinstance(expr, ast.Attribute):
        value = _transpile_expr(expr.value, env)
        if is_constants(value):
            return Constant(getattr(value.obj, expr.attr))
        raise NotImplementedError('Not implemented: __getattr__')

    if isinstance(expr, ast.Tuple):
        elts = [_transpile_expr(x, env) for x in expr.elts]
        # TODO: Support compile time constants.
        elts = [Data.init(x, env) for x in elts]
        elts_code = ', '.join([x.code for x in elts])
        ctype = _cuda_types.Tuple([x.ctype for x in elts])
        return Data(f'thrust::make_tuple({elts_code})', ctype)

    if isinstance(expr, ast.Index):
        return _transpile_expr(expr.value, env)

    raise ValueError('Not supported: type {}'.format(type(expr)))


def _transpile_lvalue(target, env, ctype):
    if isinstance(target, ast.Name):
        name = target.id
        if env[name] is None:
            env[name] = Data(name, ctype)
        elif is_constants(env[name]):
            raise TypeError('Type mismatch of variable: `{name}`')
        elif env[name].ctype != ctype:
            raise TypeError(
                f'Data type mismatch of variable: `{name}`: '
                f'{env[name].ctype.dtype} != {ctype.dtype}')
        return env[name]

    if isinstance(target, ast.Subscript):
        return _transpile_expr(target, env)

    if isinstance(target, ast.Tuple):
        if not isinstance(ctype, _cuda_types.Tuple):
            raise ValueError(f'{ctype} cannot be unpack')
        size = len(target.elts)
        if len(ctype.types) > size:
            raise ValueError(f'too many values to unpack (expected {size})')
        if len(ctype.types) < size:
            raise ValueError(f'not enough values to unpack (expected {size})')
        elts = [_transpile_lvalue(x, env, t)
                for x, t in zip(target.elts, ctype.types)]
        # TODO: Support compile time constants.
        elts_code = ', '.join([x.code for x in elts])
        return Data(f'thrust::tie({elts_code})', ctype)


def _indexing(array, index, env):
    if is_constants(array):
        if is_constants(index):
            return Constant(array.obj[index.obj])
        raise TypeError(
            f'{type(array.obj)} is not subscriptable with non-constants.')

    array = Data.init(array, env)

    if isinstance(array.ctype, _cuda_types.Tuple):
        raise NotImplementedError

    if isinstance(array.ctype, _cuda_types.ArrayBase):
        index = Data.init(index, env)
        ndim = array.ctype.ndim
        if isinstance(index.ctype, _cuda_types.Scalar):
            index_dtype = index.ctype.dtype
            if ndim != 1:
                raise TypeError(
                    'Scalar indexing is supported only for 1-dim array.')
            if index_dtype.kind not in 'ui':
                raise TypeError('Array indices must be integers.')
            return Data(
                f'{array.code}[{index.code}]', array.ctype.child_type)
        if isinstance(index.ctype, _cuda_types.Tuple):
            if ndim != len(index.ctype.types):
                raise IndexError(f'The size of index must be {ndim}')
            for t in index.ctype.types:
                if not isinstance(t, _cuda_types.Scalar):
                    raise TypeError('Array indices must be scalar.')
                if t.dtype.kind not in 'iu':
                    raise TypeError('Array indices must be integer.')
            if ndim == 0:
                return Data(
                    f'{array.code}[0]', array.ctype.child_type)
            if ndim == 1:
                return Data(
                    f'{array.code}[thrust::get<0>({index.code})]',
                    array.ctype.child_type)
            return Data(
                f'{array.code}._indexing({index.code})',
                array.ctype.child_type)
        if isinstance(index.ctype, _cuda_types.Array):
            raise TypeError('Advanced indexing is not supported.')
        assert False  # Never reach.

    raise TypeError(f'{array.code} is not subscriptable.')


def _astype_scalar(x, ctype, casting, env):
    if is_constants(x):
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
        return Data(f'({ctype})({x.code}.real())', ctype)
    return Data(f'({ctype})({x.code})', ctype)
