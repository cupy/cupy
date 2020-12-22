import argparse
import sys

import pycparser.c_ast as c_ast

import gen
import gen_pyx


def transpile_opaque(env, node):
    assert isinstance(node, c_ast.Typedef)
    type_name = node.name
    type_name1 = gen.transpile_type_name(env, node)
    # Use `void*` for opaque data structures.
    return "ctypedef void* {} '{}'".format(type_name1, type_name)


def transpile_enum_type(env, node):
    assert isinstance(node, c_ast.Typedef)
    assert isinstance(node.type, c_ast.TypeDecl)
    assert isinstance(node.type.type, c_ast.Enum)
    type_name = node.name
    type_name1 = gen.transpile_type_name(env, node)
    # Use `int` for enums.
    return "ctypedef int {} '{}'".format(type_name1, type_name)


def transpile_enum_value(node):
    def aux(enumerator):
        name = enumerator.name
        if enumerator.value is not None:
            value = enumerator.value.value
            assert enumerator.value.type == 'int'
            return '{} = {}'.format(name, value)
        else:
            return name
    assert isinstance(node, c_ast.Typedef)
    assert isinstance(node.type, c_ast.TypeDecl)
    assert isinstance(node.type.type, c_ast.Enum)

    code = []
    code.append('cpdef enum:')

    for e in node.type.type.values.enumerators:
        # TODO: Recursively resolve expressions
        name = e.name
        if e.value is None:
            code.append('    ' + name)
        elif isinstance(e.value, c_ast.Constant):
            value = e.value.value
            assert e.value.type in ['int', 'unsigned int']
            code.append('    {} = {}'.format(name, value))
        elif isinstance(e.value, c_ast.UnaryOp):
            op = e.value.op
            expr = e.value.expr
            assert expr.type in ['int', 'unsigned int']
            code.append('    {} = {}{}'.format(name, op, expr.value))
        elif isinstance(e.value, c_ast.BinaryOp):
            op = e.value.op
            left = e.value.left
            assert left.type in ['int', 'unsigned int']
            right = e.value.right
            assert right.type in ['int', 'unsigned int']
            code.append('    {} = ({} {} {})'
                        ''.format(name, left.value, op, right.value))
        else:
            assert False
    return '\n'.join(code)


def transpile_aux_struct_decl(env, directive, node, removed):
    assert isinstance(node.type, c_ast.FuncDecl)

    out_type, out_args = gen.directive_multi_out(directive)
    code = []

    if removed:
        code.append('# REMOVED')

    code.append('cdef class {}:'.format(out_type))
    code.append('    cdef:')

    params = [p for p in node.type.args.params if p.name in out_args]
    for p in params:
        if isinstance(p.type.type, c_ast.TypeDecl):
            type_name = gen.transpile_type_name(env, p.type.type)
        elif isinstance(p.type.type, c_ast.PtrDecl):
            type_name = gen.erased_type_name(env, p.type.type)
            assert type_name is not None
        else:
            assert False
        attr = gen.deref_var_name(p.name)
        code.append('        public {} {}'.format(type_name, attr))

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


def transpile_wrapper_decl(env, directive, node, removed):
    assert isinstance(node.type, c_ast.FuncDecl)

    # Get stream configuration for following steps
    use_stream, fashion, _ = gen.directive_use_stream(directive)

    code = []

    # Comment if removed
    if removed:
        code.append('# REMOVED')

    # Function declaration
    decl = gen_pyx.transpile_wrapper_def(
        env, directive, node, use_stream and fashion == 'pass')
    code.append('cpdef {}'.format(decl))

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
        return None  # planned to be deprecated
    elif gen.is_function_directive(directive):
        head = gen.directive_head(directive)
        decls, removed = gen.query_func_decls(head, env)
        return '\n'.join(
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

    env  = gen.make_environment(directives)

    template = gen.read_template(args.template)

    opaques = gen.environment_opaques(env)
    opaque = '\n'.join(
        gen.indent(transpile_opaque(env, o)) for o in opaques)

    enums = gen.environment_enums(env)
    enum_type = '\n'.join(
        gen.indent(transpile_enum_type(env, e)) for e in enums)
    enum_value = '\n\n\n'.join(transpile_enum_value(e) for e in enums)

    aux_struct = '\n\n\n'.join(
        gen.compact(transpile_aux_struct(env, d) for d in directives))

    wrapper = '\n\n'.join(
        gen.compact(transpile_wrapper(env, d) for d in directives))

    print(template.format(
        opaque=opaque, enum_type=enum_type, enum_value=enum_value,
        aux_struct=aux_struct, wrapper=wrapper))


if __name__ == '__main__':
    main(sys.argv[1:])
