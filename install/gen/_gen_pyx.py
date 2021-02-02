import argparse
import sys

import pycparser.c_ast as c_ast

from install.gen import _gen


# FFI

def transpile_ffi_one(env, node, removed):
    def argaux(env, node):
        name = node.name
        type = _gen.transpile_type_name(env, node.type)
        return '{} {}'.format(type, name)

    assert isinstance(node.type, c_ast.FuncDecl)

    code = []

    if removed:
        code.append('# REMOVED')

    ret_type = _gen.transpile_type_name(env, node.type.type)
    name = node.name
    args = [argaux(env, p) for p in node.type.args.params]
    code.append('{} {}({})'.format(ret_type, name, ', '.join(args)))

    return _gen.join_or_none('\n', code)


def transpile_ffi(env, directives):
    code = []
    for d in directives:
        if _gen.is_comment_directive(d):
            comment = _gen.directive_comment(d)
            code.append('')
            code.append('# ' + comment)
        elif _gen.is_function_directive(d):
            head = _gen.directive_head(d)
            decls, removed = _gen.query_func_decls(head, env)
            for decl in decls:
                code.append(transpile_ffi_one(env, decl, removed))
    return _gen.join_or_none('\n', code)


# Helper class

def is_helper_class_directive(directive):
    return (
        _gen.is_function_directive(directive)
        and _gen.is_directive_multi_out(directive))


def transpile_helper_class(env, directive, node, removed):
    def argaux(env, node):
        # dereference node
        name = _gen.deref_var_name(node.name)
        if isinstance(node.type.type, c_ast.TypeDecl):
            type = _gen.transpile_type_name(env, node.type.type)
        elif isinstance(node.type.type, c_ast.PtrDecl):
            type = _gen.erased_type_name(env, node.type.type)
            assert type is not None
        else:
            assert False
        return '{} {}'.format(type, name)
    
    assert isinstance(node.type, c_ast.FuncDecl)

    out_type, out_args = _gen.directive_multi_out(directive)

    code = []

    if removed:
        code.append('# REMOVED')
    
    code.append('cdef class {}:'.format(out_type))
    code.append('')

    params = [p for p in node.type.args.params if p.name in out_args]
    args = [argaux(env, p) for p in params]
    code.append('    def __init__(self, {}):'.format(', '.join(args)))

    for p in params:
        attr = _gen.deref_var_name(p.name)
        code.append('        self.{attr} = {attr}'.format(attr=attr))

    return _gen.join_or_none('\n', code)


# Assuming multiple functions do not use the same auxiliary structure.
def transpile_helper_classes(env, directives):
    code = []
    for d in directives:
        assert is_helper_class_directive(d)
        if code == []:
            code.append('')
            code.append('')
            code.append('#' * 40)
            code.append('# Helper classes')
        head = _gen.directive_head(d)
        decls, removed = _gen.query_func_decls(head, env)
        assert len(decls) == 1  # assuming not type generic
        code.append('')
        code.append(transpile_helper_class(env, d, decls[0], removed))
    return _gen.join_or_none('\n', code)


# Wrappers

def transpile_wrapper_def(env, directive, node, pass_stream):
    def is_stream_param(node):
        return (isinstance(node.type, c_ast.TypeDecl)
                and node.type.type.names[0] == 'cudaStream_t')

    def argaux(env, node):
        name = node.name
        type = _gen.erased_type_name(env, node.type)
        if type is None:
            type = _gen.transpile_type_name(env, node.type)
        return '{} {}'.format(type, name)

    assert isinstance(node.type, c_ast.FuncDecl)

    if pass_stream:
        params = [p for p in node.type.args.params if not is_stream_param(p)]
        assert len(params) == len(node.type.args.params) - 1
    else:
        params = node.type.args.params

    if _gen.is_directive_none_out(directive):
        assert _gen.directive_except(directive) is None
        name = _gen.transpile_func_name(env, directive, node)
        args = [argaux(env, p) for p in params]
        return '{}({})'.format(name, ', '.join(args))
    elif _gen.is_directive_returned_out(directive):
        assert _gen.directive_except(directive) is None
        ret_type = _gen.erased_type_name(env, node.type.type)
        if ret_type is None:
            ret_type = _gen.transpile_type_name(env, node.type.type)
        name = _gen.transpile_func_name(env, directive, node)
        args = [argaux(env, p) for p in params]
        return '{} {}({})'.format(ret_type, name, ', '.join(args))
    elif _gen.is_directive_single_out(directive):
        out_name = _gen.directive_single_out(directive)
        out, params1 = _gen.partition(
            lambda p: p.name == out_name, params)
        assert len(out) == 1, \
            '`{}` not found in API arguments'.format(out_name)
        # dereference out[0]
        ret_type = _gen.erased_type_name(env, out[0].type.type)
        if ret_type is None:
            ret_type = _gen.transpile_type_name(env, out[0].type.type)
        name = _gen.transpile_func_name(env, directive, node)
        args = [argaux(env, p) for p in params1]
        excpt = _gen.directive_except(directive)
        return '{} {}({}) {}'.format(ret_type, name, ', '.join(args), excpt)
    elif _gen.is_directive_multi_out(directive):
        assert _gen.directive_except(directive) is None
        out_type, out_args = _gen.directive_multi_out(directive)
        outs, params1 = _gen.partition(
            lambda p: p.name in out_args, params)
        assert len(outs) > 1
        name = _gen.transpile_func_name(env, directive, node)
        args = [argaux(env, p) for p in params1]
        return '{} {}({})'.format(out_type, name, ', '.join(args))
    else:
        assert False


def transpile_wrapper_call(env, directive, node):
    def argaux(env, directive, node):
        name = node.name
        if _gen.is_directive_single_out(directive):
            if name == _gen.directive_single_out(directive):
                name1 = _gen.deref_var_name(name)
                return '&{}'.format(name1)
        if _gen.is_directive_multi_out(directive):
            _, out_args = _gen.directive_multi_out(directive)
            if name in out_args:
                name1 = _gen.deref_var_name(name)
                return '&{}'.format(name1)
        erased_type = _gen.erased_type_name(env, node.type)
        if erased_type is not None:
            return _gen.transpile_type_conversion(env, node.type, name)
        return name
    assert isinstance(node.type, c_ast.FuncDecl)
    name = node.name
    args = [argaux(env, directive, p) for p in node.type.args.params]
    return '{}({})'.format(name, ', '.join(args))


def handler_name(node):
    assert isinstance(node, c_ast.Decl)
    # Assuming the handler's name is always 'handle'.
    for param in node.type.args.params:
        if param.name == 'handle':
            return 'handle'
    assert False


def stream_name(node):
    assert isinstance(node, c_ast.Decl)
    # Assuming the stream pointer's name is always 'stream'.
    for param in node.type.args.params:
        if param.name == 'stream':
            return 'stream'
    assert False


def transpile_wrapper(env, directive, node, removed):
    assert isinstance(node.type, c_ast.FuncDecl)

    # Get stream configuration for following steps
    use_stream, fashion, func_name = _gen.directive_use_stream(directive)

    code = []

    # Comment if removed
    if removed:
        code.append('# REMOVED')

    # Function definition
    def_ = transpile_wrapper_def(
        env, directive, node, use_stream and fashion == 'pass')
    code.append('cpdef {}:'.format(def_))

    # Allocate space for the value to return
    if _gen.is_directive_none_out(directive):
        pass
    elif _gen.is_directive_returned_out(directive):
        pass
    elif _gen.is_directive_single_out(directive):
        out_name = _gen.directive_single_out(directive)
        out, params = _gen.partition(
            lambda p: p.name == out_name, node.type.args.params)
        assert len(out) == 1
        # dereference out[0]
        out_type = _gen.transpile_type_name(env, out[0].type.type)
        out_name1 = _gen.deref_var_name(out_name)
        code.append('    cdef {} {}'.format(out_type, out_name1))
    elif _gen.is_directive_multi_out(directive):
        _, out_args = _gen.directive_multi_out(directive)
        outs, params = _gen.partition(
            lambda p: p.name in out_args, node.type.args.params)
        assert len(outs) > 1
        for out, out_arg in zip(outs, out_args):
            # dereference out
            out_arg_type = _gen.transpile_type_name(env, out.type.type)
            out_arg1 = _gen.deref_var_name(out_arg)
            code.append('    cdef {} {}'.format(out_arg_type, out_arg1))
    else:
        assert False

    if use_stream:
        # Set stream if necessary
        if fashion == 'set':
            handle = handler_name(node)
            code.append('    if stream_module.enable_current_stream:')
            code.append(
                '        {}({}, stream_module.get_current_stream_ptr())'
                ''.format(func_name, handle))
        # Assign the current stream pointer to a variable
        elif fashion == 'pass':
            stream = stream_name(node)
            code.append(
                '    cdef intptr_t {} = stream_module.get_current_stream_ptr()'
                ''.format(stream))
        else:
            assert False

    # Call cuSPARSE API and check its returned status if necessary
    if _gen.is_directive_returned_out(directive):
        call = transpile_wrapper_call(env, directive, node)
        code.append('    return {}'.format(call))
    else:
        status_var = 'status'  # assuming cusparse API does not use the name
        call = transpile_wrapper_call(env, directive, node)
        code.append('    {} = {}'.format(status_var, call))
        code.append('    check_status({})'.format(status_var))

    # Return value if necessary
    if _gen.is_directive_none_out(directive):
        pass
    elif _gen.is_directive_returned_out(directive):
        pass
    elif _gen.is_directive_single_out(directive):
        out_name = _gen.directive_single_out(directive)
        # dereference out[0]
        ret_type = _gen.erased_type_name(env, out[0].type.type)
        out_name1 = _gen.deref_var_name(out_name)
        if ret_type is not None:
            code.append('    return <{}>{}'.format(ret_type, out_name1))
        else:
            code.append('    return {}'.format(out_name1))
    elif _gen.is_directive_multi_out(directive):
        out_type, out_args = _gen.directive_multi_out(directive)
        outs, params = _gen.partition(
            lambda p: p.name in out_args, node.type.args.params)
        assert len(outs) > 1
        out_args1 = []
        for out_arg, out in zip(out_args, outs):
            # dereference out
            out_arg_name = _gen.deref_var_name(out_arg)
            if isinstance(out.type.type, c_ast.TypeDecl):
                out_args1.append(out_arg_name)
            elif isinstance(out.type.type, c_ast.PtrDecl):
                out_arg_type = _gen.erased_type_name(env, out.type.type)
                out_args1.append('<{}>{}'.format(out_arg_type, out_arg_name))
            else:
                assert False
        code.append('    return {}({})'.format(out_type, ', '.join(out_args1)))
    else:
        assert False

    return _gen.join_or_none('\n', code)


def transpile_wrappers(env, directives):
    code = []
    for d in directives:
        if _gen.is_comment_directive(d):
            comment = _gen.directive_comment(d)
            code.append('')
            code.append('')
            code.append('#' * max(40, len(comment) + 2))
            code.append('# ' + comment)
        elif _gen.is_function_directive(d):
            head = _gen.directive_head(d)
            decls, removed = _gen.query_func_decls(head, env)
            for decl in decls:
                code.append('')
                code.append(transpile_wrapper(env, d, decl, removed))
    return _gen.join_or_none('\n', code)


# Main

def gen_pyx(directive, template):
    directives = _gen.read_directives(directive)
    env = _gen.make_environment(directives)
    template = _gen.read_template(template)

    # FFI
    ffi_code = _gen.indent(transpile_ffi(env, directives) or '')

    # Helper classes
    directives1 = [d for d in directives if is_helper_class_directive(d)]
    helper_class_code = transpile_helper_classes(env, directives1) or ''

    # Wrappers
    wrapper_code = transpile_wrappers(env, directives) or ''

    code = template.format(
        ffi=ffi_code, helper_class=helper_class_code, wrapper=wrapper_code)

    return code
