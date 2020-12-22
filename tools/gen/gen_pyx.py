import argparse
import sys

import pycparser.c_ast as c_ast

import gen


def transpile_ffi_decl(env, node, removed):
    def argaux(env, node):
        name = node.name
        type = gen.transpile_type_name(env, node.type)
        return '{} {}'.format(type, name)
    assert isinstance(node.type, c_ast.FuncDecl)

    code = []
    if removed:
        code.append('# REMOVED')

    ret_type = gen.transpile_type_name(env, node.type.type)
    name = node.name
    args = [argaux(env, p) for p in node.type.args.params]
    code.append('{} {}({})'.format(ret_type, name, ', '.join(args)))

    return '\n'.join(code)


def transpile_ffi(env, directive):
    if gen.is_cuda_versions_directive(directive):
        return None
    elif gen.is_headers_directive(directive):
        return None
    elif gen.is_regexes_directive(directive):
        return None
    elif gen.is_special_types_directive(directive):
        return None
    elif gen.is_comment_directive(directive):
        comment = gen.directive_comment(directive)
        return '\n# ' + comment
    elif gen.is_raw_directive(directive):
        return None
    elif gen.is_function_directive(directive):
        head = gen.directive_head(directive)
        decls, removed = gen.query_func_decls(head, env)
        return '\n'.join(
            transpile_ffi_decl(env, decl, removed) for decl in decls)
    else:
        assert False


def transpile_aux_struct_decl(env, directive, node, removed):
    def argaux(env, node):
        # dereference node
        name = gen.deref_var_name(node.name)
        if isinstance(node.type.type, c_ast.TypeDecl):
            type = gen.transpile_type_name(env, node.type.type)
        elif isinstance(node.type.type, c_ast.PtrDecl):
            type = gen.erased_type_name(env, node.type.type)
            assert type is not None
        else:
            assert False
        return '{} {}'.format(type, name)

    assert isinstance(node.type, c_ast.FuncDecl)

    out_type, out_args = gen.directive_multi_out(directive)
    code = []

    if removed:
        code.append('# REMOVED')

    code.append('cdef class {}:'.format(out_type))
    code.append('')

    params = [p for p in node.type.args.params if p.name in out_args]
    args = [argaux(env, p) for p in params]
    code.append('    def __init__(self, {}):'.format(', '.join(args)))

    for p in params:
        attr = gen.deref_var_name(p.name)
        code.append('        self.{attr} = {attr}'.format(attr=attr))

    return '\n'.join(code)


# Assuming multiple functions do not use the same auxiliary structure.
def transpile_aux_struct(env, directive):
    if gen.is_cuda_versions_directive(directive):
        return None
    elif gen.is_headers_directive(directive):
        return None
    elif gen.is_regexes_directive(directive):
        return None
    elif gen.is_special_types_directive(directive):
        return None
    elif gen.is_comment_directive(directive):
        return None
    elif gen.is_raw_directive(directive):
        return None
    elif gen.is_function_directive(directive):
        if gen.is_directive_multi_out(directive):
            head = gen.directive_head(directive)
            decls, removed = gen.query_func_decls(head, env)
            assert len(decls) == 1  # assuming not type generic
            return transpile_aux_struct_decl(env, directive, decls[0], removed)
        else:
            return None
    else:
        assert False


def transpile_wrapper_def(env, directive, node, pass_stream):
    def is_stream_param(node):
        return (isinstance(node.type, c_ast.TypeDecl)
                and node.type.type.names[0] == 'cudaStream_t')

    def argaux(env, node):
        name = node.name
        type = gen.erased_type_name(env, node.type)
        if type is None:
            type = gen.transpile_type_name(env, node.type)
        return '{} {}'.format(type, name)

    assert isinstance(node.type, c_ast.FuncDecl)

    if pass_stream:
        params = [p for p in node.type.args.params if not is_stream_param(p)]
        assert len(params) == len(node.type.args.params) - 1
    else:
        params = node.type.args.params

    if gen.is_directive_none_out(directive):
        assert gen.directive_except(directive) is None
        name = gen.transpile_func_name(env, directive, node)
        args = [argaux(env, p) for p in params]
        return '{}({})'.format(name, ', '.join(args))
    elif gen.is_directive_returned_out(directive):
        assert gen.directive_except(directive) is None
        ret_type = gen.erased_type_name(env, node.type.type)
        if ret_type is None:
            ret_type = gen.transpile_type_name(env, node.type.type)
        name = gen.transpile_func_name(env, directive, node)
        args = [argaux(env, p) for p in params]
        return '{} {}({})'.format(ret_type, name, ', '.join(args))
    elif gen.is_directive_single_out(directive):
        out_name = gen.directive_single_out(directive)
        out, params1 = gen.partition(
            lambda p: p.name == out_name, params)
        assert len(out) == 1, \
            '`{}` not found in API arguments'.format(out_name)
        # dereference out[0]
        ret_type = gen.erased_type_name(env, out[0].type.type)
        if ret_type is None:
            ret_type = gen.transpile_type_name(env, out[0].type.type)
        name = gen.transpile_func_name(env, directive, node)
        args = [argaux(env, p) for p in params1]
        excpt = gen.directive_except(directive)
        return '{} {}({}) {}'.format(ret_type, name, ', '.join(args), excpt)
    elif gen.is_directive_multi_out(directive):
        assert gen.directive_except(directive) is None
        out_type, out_args = gen.directive_multi_out(directive)
        outs, params1 = gen.partition(
            lambda p: p.name in out_args, params)
        assert len(outs) > 1
        name = gen.transpile_func_name(env, directive, node)
        args = [argaux(env, p) for p in params1]
        return '{} {}({})'.format(out_type, name, ', '.join(args))
    else:
        assert False


def transpile_wrapper_call(env, directive, node):
    def argaux(env, directive, node):
        name = node.name
        if gen.is_directive_single_out(directive):
            if name == gen.directive_single_out(directive):
                name1 = gen.deref_var_name(name)
                return '&{}'.format(name1)
        if gen.is_directive_multi_out(directive):
            _, out_args = gen.directive_multi_out(directive)
            if name in out_args:
                name1 = gen.deref_var_name(name)
                return '&{}'.format(name1)
        erased_type = gen.erased_type_name(env, node.type)
        if erased_type is not None:
            return gen.transpile_type_conversion(env, node.type, name)
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


def transpile_wrapper_decl(env, directive, node, removed):
    assert isinstance(node.type, c_ast.FuncDecl)

    # Get stream configuration for following steps
    use_stream, fashion, func_name = gen.directive_use_stream(directive)

    code = []

    # Comment if removed
    if removed:
        code.append('# REMOVED')

    # Function definition
    def_ = transpile_wrapper_def(
        env, directive, node, use_stream and fashion == 'pass')
    code.append('cpdef {}:'.format(def_))

    # Allocate space for the value to return
    if gen.is_directive_none_out(directive):
        pass
    elif gen.is_directive_returned_out(directive):
        pass
    elif gen.is_directive_single_out(directive):
        out_name = gen.directive_single_out(directive)
        out, params = gen.partition(
            lambda p: p.name == out_name, node.type.args.params)
        assert len(out) == 1
        # dereference out[0]
        out_type = gen.transpile_type_name(env, out[0].type.type)
        out_name1 = gen.deref_var_name(out_name)
        code.append('    cdef {} {}'.format(out_type, out_name1))
    elif gen.is_directive_multi_out(directive):
        _, out_args = gen.directive_multi_out(directive)
        outs, params = gen.partition(
            lambda p: p.name in out_args, node.type.args.params)
        assert len(outs) > 1
        for out, out_arg in zip(outs, out_args):
            # dereference out
            out_arg_type = gen.transpile_type_name(env, out.type.type)
            out_arg1 = gen.deref_var_name(out_arg)
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
    if gen.is_directive_returned_out(directive):
        call = transpile_wrapper_call(env, directive, node)
        code.append('    return {}'.format(call))
    else:
        status_var = 'status'  # assuming cusparse API does not use the name
        call = transpile_wrapper_call(env, directive, node)
        code.append('    {} = {}'.format(status_var, call))
        code.append('    check_status({})'.format(status_var))

    # Return value if necessary
    if gen.is_directive_none_out(directive):
        pass
    elif gen.is_directive_returned_out(directive):
        pass
    elif gen.is_directive_single_out(directive):
        out_name = gen.directive_single_out(directive)
        # dereference out[0]
        ret_type = gen.erased_type_name(env, out[0].type.type)
        out_name1 = gen.deref_var_name(out_name)
        if ret_type is not None:
            code.append('    return <{}>{}'.format(ret_type, out_name1))
        else:
            code.append('    return {}'.format(out_name1))
    elif gen.is_directive_multi_out(directive):
        out_type, out_args = gen.directive_multi_out(directive)
        outs, params = gen.partition(
            lambda p: p.name in out_args, node.type.args.params)
        assert len(outs) > 1
        out_args1 = []
        for out_arg, out in zip(out_args, outs):
            # dereference out
            out_arg_name = gen.deref_var_name(out_arg)
            if isinstance(out.type.type, c_ast.TypeDecl):
                out_args1.append(out_arg_name)
            elif isinstance(out.type.type, c_ast.PtrDecl):
                out_arg_type = gen.erased_type_name(env, out.type.type)
                out_args1.append('<{}>{}'.format(out_arg_type, out_arg_name))
            else:
                assert False
        code.append('    return {}({})'.format(out_type, ', '.join(out_args1)))
    else:
        assert False

    return '\n'.join(code)


def transpile_wrapper(env, directive):
    if gen.is_cuda_versions_directive(directive):
        return None
    elif gen.is_headers_directive(directive):
        return None
    elif gen.is_regexes_directive(directive):
        return None
    elif gen.is_special_types_directive(directive):
        return None
    elif gen.is_comment_directive(directive):
        comment = gen.directive_comment(directive)
        code = []
        code.append('')
        code.append('#' * max(40, len(comment) + 2))
        code.append('# ' + comment)
        return '\n'.join(code)
    elif gen.is_raw_directive(directive):
        return gen.directive_raw(directive)
    elif gen.is_function_directive(directive):
        head = gen.directive_head(directive)
        decls, removed = gen.query_func_decls(head, env)
        return '\n\n'.join(
            transpile_wrapper_decl(
                env, directive, decl, removed) for decl in decls)
    else:
        assert False


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--directive', required=True, type=str,
        help='Path to directive file')
    parser.add_argument(
        '-t', '--template', required=True, type=str,
        help='Path to template file')
    args = parser.parse_args(args)

    directives = gen.read_directives(args.directive)

    env = gen.make_environment(directives)

    template = gen.read_template(args.template)

    maybe_indent = gen.maybe(gen.indent)
    ffi = '\n'.join(
        gen.compact(maybe_indent(transpile_ffi(env, d)) for d in directives))
    aux_struct = '\n\n'.join(
        gen.compact(transpile_aux_struct(env, d) for d in directives))
    wrapper = '\n\n'.join(
        gen.compact(transpile_wrapper(env, d) for d in directives))
    print(template.format(ffi=ffi, aux_struct=aux_struct, wrapper=wrapper))


if __name__ == '__main__':
    main(sys.argv[1:])
