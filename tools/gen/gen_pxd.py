import os.path

import pycparser.c_ast as c_ast

import gen


def transpile_opaque(decl, typedef):
    assert isinstance(decl, c_ast.Decl)
    assert isinstance(typedef, c_ast.Typedef)
    type_name = typedef.name
    type_name1 = gen.transpile_type_name(env, typedef)
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
        name = e.name
        if e.value is not None:
            value = e.value.value
            assert e.value.type == 'int'
            code.append('    {} = {}'.format(name, value))
        else:
            code.append('    ' + name)

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
    if gen.is_comment_directive(directive):
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


if __name__ == '__main__':
    directives = gen.read_directives('directives/cusparse.py')

    env = gen.make_environment()

    template = gen.read_template('templates/cusparse.pxd.template')

    opaques = gen.environment_opaques(env)
    opaque = '\n'.join(
        gen.indent(transpile_opaque(decl, tydef)) for decl, tydef in opaques)

    enums = gen.environment_enums(env)
    enum_type = '\n'.join(
        gen.indent(transpile_enum_type(env, e)) for e in enums)
    enum_value = '\n\n\n'.join(transpile_enum_value(e) for e in enums)

    aux_struct = '\n\n\n'.join(
        gen.compact(transpile_aux_struct(env, d) for d in directives))

    print(template.format(
        opaque=opaque, enum_type=enum_type, enum_value=enum_value,
        aux_struct=aux_struct))
