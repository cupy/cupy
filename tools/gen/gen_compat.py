import argparse
import math
import sys

import gen


def version_info(env, version):
    lib_name = gen.environment_lib_name(env)
    version = float(version)
    if lib_name == 'cusparse':
        return 'CUSPARSE_VERSION', str(int(version * 1000))
    else:
        major = math.floor(version)
        minor = version - major
        return 'CUDA_VERSION', str(int(major * 1000 + minor * 100))


def compute_diff_opaques(new_ver, old_ver, env):
    new_opaques = gen.environment_opaques(env, new_ver)
    old_opaques = gen.environment_opaques(env, old_ver)

    new_set = set(x.name for x in new_opaques)
    old_set = set(x.name for x in old_opaques)
    added_set = new_set - old_set
    removed_set = old_set - new_set

    # Use lists just for keeping the order of appearance in header files.
    added_list = []
    for o in new_opaques:
        if o.name in added_set:
            added_list.append(o.name)
    removed_list = []
    for o in old_opaques:
        if o.name in removed_set:
            removed_list.append(o.name)

    diff = {}
    if added_list != []:
        diff['added'] = added_list
    if removed_list != []:
        diff['removed'] = removed_list
    return diff


def compute_diff_enums(new_ver, old_ver, env):
    new_enums = gen.environment_enums(env, new_ver)
    old_enums = gen.environment_enums(env, old_ver)

    new_set = set(x.name for x in new_enums)
    old_set = set(x.name for x in old_enums)
    added_set = new_set - old_set
    removed_set = old_set - new_set

    # Use lists just for keeping the order of appearance in header files.
    added_list = []
    for e in new_enums:
        if e.name in added_set:
            added_list.append(e.name)
    removed_list = []
    for e in old_enums:
        if e.name in removed_set:
            removed_list.append(e.name)

    diff = {}
    if added_list != []:
        diff['added'] = added_list
    if removed_list != []:
        diff['removed'] = removed_list
    return diff
    

def compute_diff_functions(new_ver, old_ver, env, directives):
    new_funcs = gen.environment_functions(env, new_ver)
    old_funcs = gen.environment_functions(env, old_ver)

    new_set = set(x.name for x in new_funcs)
    old_set = set(x.name for x in old_funcs)
    added_set = new_set - old_set
    removed_set = old_set - new_set

    # Keep the order of appearance in directive files.
    added_list = []
    removed_list = []
    for d in directives:
        if gen.is_function_directive(d):
            head = gen.directive_head(d)
            decls, _ = gen.query_func_decls(head, env)
            for decl in decls:
                name = decl.name
                if name in added_set:
                    added_list.append(name)
                if name in removed_set:
                    removed_list.append(name)

    diff = {}
    if added_list != []:
        diff['added'] = added_list
    if removed_list != []:
        diff['removed'] = removed_list
    return diff


def compute_diff_version(new_ver, old_ver, env, directives):
    diff = {}

    diff_opaques = compute_diff_opaques(new_ver, old_ver, env)
    if diff_opaques != {}:
        diff['opaques'] = diff_opaques

    diff_enums = compute_diff_enums(new_ver, old_ver, env)
    if diff_enums != {}:
        diff['enums'] = diff_enums

    diff_funcs = compute_diff_functions(new_ver, old_ver, env, directives)
    if diff_funcs != {}:
        diff['functions'] = diff_funcs

    return diff


def compute_diff(env, directives):
    cuda_versions = list(reversed(gen.environment_cuda_versions(env)))
    diff = {}
    for new_ver, old_ver in zip(cuda_versions[1:], cuda_versions[:-1]):
        diff_ver = compute_diff_version(new_ver, old_ver, env, directives)
        if diff_ver != {}:
            diff[new_ver] = diff_ver
    return diff


def transpile_diff_added(env, diff, ver):
    diff_ver = diff.get(ver)
    if diff_ver is None:
        return None

    try:
        opaques = diff_ver['opaques']['added']
    except KeyError:
        opaques = None

    try:
        enums = diff_ver['enums']['added']
    except KeyError:
        enums = None

    try:
        funcs = diff_ver['functions']['added']
    except KeyError:
        funcs = None

    if opaques is None and enums is None and funcs is None:
        return None

    version_macro, version = version_info(env, ver)

    code = []
    code.append('')
    code.append('#if {} < {}'.format(version_macro, version))
    code.append('// Added in {}'.format(ver))

    if opaques is not None:
        code.append('')
        for name in opaques:
            code.append('typedef void* {};'.format(name))

    if enums is not None:
        code.append('')
        for name in enums:
            code.append('typedef enum {{}} {};'.format(name))

    status_type = gen.environment_status_type(env)
    status_success, _ = gen.environment_status_success(env)
    if funcs is not None:
        for name in funcs:
            code.append('')
            code.append('{} {}(...) {{'.format(status_type, name))
            code.append('  return {};'.format(status_success))
            code.append('}')

    code.append('')
    code.append('#endif  // #if {} < {}'.format(version_macro, version))

    return gen.join_or_none('\n', code)


def transpile_diff_removed(env, diff, ver):
    diff_ver = diff.get(ver)
    if diff_ver is None:
        return None

    try:
        opaques = diff_ver['opaques']['removed']
    except KeyError:
        opaques = None

    try:
        enums = diff_ver['enums']['removed']
    except KeyError:
        enums = None

    try:
        funcs = diff_ver['functions']['removed']
    except KeyError:
        funcs = None

    if opaques is None and enums is None and funcs is None:
        return None

    version_macro, version = version_info(env, ver)

    code = []
    code.append('')
    code.append('#if {} >= {}'.format(version_macro, version))
    code.append('// Removed in {}'.format(ver))

    if opaques is not None:
        code.append('')
        for name in opaques:
            code.append('typedef void* {};'.format(name))

    if enums is not None:
        code.append('')
        for name in enums:
            code.append('typedef enum {{}} {};'.format(name))

    status_type = gen.environment_status_type(env)
    status_success, _ = gen.environment_status_success(env)
    if funcs is not None:
        for name in funcs:
            code.append('')
            code.append('{} {}(...) {{'.format(status_type, name))
            code.append('  return {};'.format(status_success))
            code.append('}')

    code.append('')
    code.append('#endif  // #if {} >= {}'.format(version_macro, version))

    return gen.join_or_none('\n', code)


def transpile_diff(env, diff):
    code = []
    cuda_versions = list(reversed(gen.environment_cuda_versions(env)))
    for ver in cuda_versions[1:]:
        code.append(transpile_diff_added(env, diff, ver))
    for ver in cuda_versions[1:]:
        code.append(transpile_diff_removed(env, diff, ver))
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

    diff = compute_diff(env, directives)
    diff_code = transpile_diff(env, diff) or ''
    code = template.format(code=diff_code)
    print(code, end='')


if __name__ == '__main__':
    main(sys.argv[1:])
