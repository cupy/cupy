import argparse
import sys

import pycparser.c_ast as c_ast

import gen
import gen_pyx


# Opaque pointers

def transpile_opaques(env, opaques):
    code = []
    code.append('')
    code.append('cdef extern from *:')
    for o in opaques:
        type_name = o.name
        type_name1 = gen.transpile_type_name(env, o)
        # Use `void*` for opaque data structures.
        code.append("    ctypedef void* {} '{}'".format(type_name1, type_name))
    return gen.join_or_none('\n', code)


# Enumerators

def transpile_enums(env, enums):
    code = []
    code.append('')
    code.append('')
    code.append('#' * 40)
    code.append('# Enumerators')
    code.append('')
    code.append('cdef extern from *:')
    for e in enums:
        type_name = e.name
        type_name1 = gen.transpile_type_name(env, e)
        # Use `int` for enums.
        code.append("    ctypedef int {} '{}'".format(type_name1, type_name))
    for e in enums:
        code.append('')
        code.append('cpdef enum:')        
        for v in e.type.type.values.enumerators:
            name = v.name
            if v.value is None:
                code.append('    ' + name)
            else:
                value = gen.transpile_expression(v.value)
                code.append('    {} = {}'.format(name, value))
    return gen.join_or_none('\n', code)


# Helper classes

def transpile_helper_class_node(env, directive, node, removed):
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

    return gen.join_or_none('\n', code)


def transpile_helper_class(env, directive):
    head = gen.directive_head(directive)
    decls, removed = gen.query_func_decls(head, env)
    assert len(decls) == 1  # assuming not type generic
    return transpile_helper_class_node(env, directive, decls[0], removed)


def transpile_helper_classes(env, directives):
    # Assuming multiple functions do not use the same helper class.
    code = []
    for d in directives:
        if gen.is_function_directive(d) and gen.is_directive_multi_out(d):
            if code == []:
                code.append('')
                code.append('')
                code.append('#' * 40)
                code.append('# Helper classes')
            code.append('')
            code.append(transpile_helper_class(env, d))
    return gen.join_or_none('\n', code)


# Wrappers

def transpile_wrapper_node(env, directive, node, removed):
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

    return gen.join_or_none('\n', code)


def transpile_wrappers(env, directives):
    code = []
    for d in directives:
        if gen.is_comment_directive(d):
            comment = gen.directive_comment(d)
            code.append('')
            code.append('')
            code.append('#' * max(40, len(comment) + 2))
            code.append('# ' + comment)
        elif gen.is_function_directive(d):
            code.append('')
            head = gen.directive_head(d)
            decls, removed = gen.query_func_decls(head, env)
            for decl in decls:
                code.append(transpile_wrapper_node(env, d, decl, removed))
    return gen.join_or_none('\n', code)


# Main

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'directive', type=str,
        help='Path to directive file for library to generate')
    parser.add_argument(
        'template', type=str,
        help='Path to template file for library to generate')
    args = parser.parse_args(args)

    directives = gen.read_directives(args.directive)

    env = gen.make_environment(directives)

    template = gen.read_template(args.template)

    # Opaque pointers
    opaques = gen.environment_opaques(env)
    opaque_code = transpile_opaques(env, opaques) or ''

    # Enumerators
    enums = gen.environment_enums(env)
    enum_code = transpile_enums(env, enums) or ''

    # Helper classes
    helper_class_code = transpile_helper_classes(env, directives) or ''

    # Wrapper functions
    wrapper_code = transpile_wrappers(env, directives) or ''

    code = template.format(
        opaque=opaque_code, enum=enum_code, helper_class=helper_class_code,
        wrapper=wrapper_code)
    print(code, end='')


if __name__ == '__main__':
    main(sys.argv[1:])
