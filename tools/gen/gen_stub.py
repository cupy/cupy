import argparse
import sys

import gen


# Opaque pointers

def transpile_opaques(opaques):
    code = []
    for o in opaques:
        code.append('typedef void* {};'.format(o.name))
    return gen.join_or_none('\n', code)


# Enumerators

def is_status_enum(env, enum):
    status_type = gen.environment_status_type(env)
    return enum.name == status_type


def transpile_status_enum(env, enum):
    success_name, success_expr = gen.environment_status_success(env)
    code = []
    code.append('typedef enum {')
    code.append('  {} = {}'.format(success_name, success_expr))
    code.append('}} {};'.format(enum.name))
    return gen.join_or_none('\n', code)


def transpile_enums(env, enums):
    code = []
    for e in enums:
        if is_status_enum(env, e):
            code.append(transpile_status_enum(env, e))
        else:
            code.append('typedef enum {{}} {};'.format(e.name))
    return gen.join_or_none('\n', code)


# Functions

def transpile_functions(env, directives):
    status_type = gen.environment_status_type(env)
    status_success, _ = gen.environment_status_success(env)
    code = []
    for d in directives:
        if gen.is_function_directive(d):
            head = gen.directive_head(d)
            decls, removed = gen.query_func_decls(head, env)
            if removed:
                continue
            for decl in decls:
                code.append('')
                code.append('{} {}(...) {{'.format(status_type, decl.name))
                code.append('  return {};'.format(status_success))
                code.append('}')
    return gen.join_or_none('\n', code)


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

    cuda_versions = gen.environment_cuda_versions(env)
    latest = cuda_versions[0]

    # Opaque pointers
    opaques = gen.environment_opaques(env, latest)
    opaque_code = transpile_opaques(opaques) or ''

    # Enumerators
    enums = gen.environment_enums(env, latest)
    enum_code = transpile_enums(env, enums) or ''

    # Functions
    function_code = transpile_functions(env, directives) or ''

    # Include guard name
    lib_name = gen.environment_lib_name(env)
    include_guard_name = 'INCLUDE_GUARD_STUB_CUPY_{}_H'.format(
        lib_name.upper())

    code = template.format(
        include_guard_name=include_guard_name, opaque=opaque_code,
        enum=enum_code, function=function_code)
    print(code, end='')


if __name__ == '__main__':
    main(sys.argv[1:])
