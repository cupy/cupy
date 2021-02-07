import argparse
import sys

from install.gen.pycparser import c_ast

from install.gen import _gen
from install.gen import _gen_pyx


# Opaque pointers

def transpile_opaques(env, opaques):
    code = []
    code.append('')
    code.append('cdef extern from *:')
    for o in opaques:
        type_name = o.name
        type_name1 = _gen.transpile_type_name(env, o)
        # Use `void*` for opaque data structures.
        code.append("    ctypedef void* {} '{}'".format(type_name1, type_name))
    return _gen.join_or_none('\n', code)


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
        type_name1 = _gen.transpile_type_name(env, e)
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
                value = _gen.transpile_expression(v.value)
                code.append('    {} = {}'.format(name, value))
    return _gen.join_or_none('\n', code)


# Helper classes

def transpile_helper_class_node(env, directive, node):
    out_type, out_args = _gen.directive_multi_out(directive)

    code = []

    code.append('cdef class {}:'.format(out_type))
    code.append('    cdef:')

    params = [p for p in node.type.args.params if p.name in out_args]
    for p in params:
        if isinstance(p.type.type, c_ast.TypeDecl):
            type_name = _gen.transpile_type_name(env, p.type.type)
        elif isinstance(p.type.type, c_ast.PtrDecl):
            type_name = _gen.erased_type_name(env, p.type.type)
            assert type_name is not None
        else:
            assert False
        attr = _gen.deref_var_name(p.name)
        code.append('        public {} {}'.format(type_name, attr))

    return _gen.join_or_none('\n', code)


def transpile_helper_class(env, directive):
    head = _gen.directive_head(directive)
    decls = _gen.query_func_decls(head, env)
    if decls is None:
        return None
    assert len(decls) == 1  # assuming not type generic
    return transpile_helper_class_node(env, directive, decls[0])


def transpile_helper_classes(env, directives):
    # Assuming multiple functions do not use the same helper class.
    code = []
    for d in directives:
        if _gen.is_function_directive(d) and _gen.is_directive_multi_out(d):
            if code == []:
                code.append('')
                code.append('')
                code.append('#' * 40)
                code.append('# Helper classes')
            code.append('')
            code.append(transpile_helper_class(env, d))
    return _gen.join_or_none('\n', code)


# Wrappers

def transpile_wrapper_node(env, directive, node):
    # Get stream configuration for following steps
    use_stream, fashion, _ = _gen.directive_use_stream(directive)

    # Function declaration
    decl = _gen_pyx.transpile_wrapper_def(
        env, directive, node, use_stream and fashion == 'pass')
    return 'cpdef {}'.format(decl)


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
            code.append('')
            head = _gen.directive_head(d)
            decls = _gen.query_func_decls(head, env)
            if decls is not None:
                assert decls != []
                for decl in decls:
                    code.append(transpile_wrapper_node(env, d, decl))
    return _gen.join_or_none('\n', code)


# Main

def gen_pxd(cuda_path, directive, template):
    directives = _gen.read_directives(directive)
    env = _gen.make_environment(cuda_path, directives)
    template = _gen.read_template(template)

    # Opaque pointers
    opaques = _gen.environment_opaques(env)
    opaque_code = transpile_opaques(env, opaques) or ''

    # Enumerators
    enums = _gen.environment_enums(env)
    enum_code = transpile_enums(env, enums) or ''

    # Helper classes
    helper_class_code = transpile_helper_classes(env, directives) or ''

    # Wrapper functions
    wrapper_code = transpile_wrappers(env, directives) or ''

    code = template.format(
        opaque=opaque_code, enum=enum_code, helper_class=helper_class_code,
        wrapper=wrapper_code)

    return code
