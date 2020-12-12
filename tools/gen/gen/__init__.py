import os.path
import re

import pycparser
import pycparser.c_ast as c_ast


# Utilities

def partition(pred, seq):
    a, b = [], []
    for item in seq:
        (a if pred(item) else b).append(item)
    return a, b


def compact(iterable):
    return (x for x in iterable if x is not None)


def indent(code):
    return '\n'.join('    ' + l if l != '' else '' for l in code.split('\n'))


def maybe(fn):
    return lambda x: fn(x) if x is not None else None


# Directive

def read_directives(path):
    with open(path) as f:
        directives = eval(f.read())
    headers, regexes, *rest = directives
    assert is_headers_directive(headers)
    assert is_regexes_directive(regexes)
    assert not any(is_headers_directive(d) for d in rest)
    assert not any(is_regexes_directive(d) for d in rest)
    assert not any(is_special_types_directive(d) for d in rest[1:])
    return directives


def is_headers_directive(directive):
    return directive[0] == 'Headers'


def is_regexes_directive(directive):
    return directive[0] == 'Regexes'


def is_special_types_directive(directive):
    return directive[0] == 'SpecialTypes'


def is_comment_directive(directive):
    return directive[0] == 'Comment'


def is_raw_directive(directive):
    return directive[0] == 'Raw'


def is_function_directive(directive):
    return (
        isinstance(directive[0], str)
        and not is_headers_directive(directive)
        and not is_regexes_directive(directive)
        and not is_special_types_directive(directive)
        and not is_comment_directive(directive)
        and not is_raw_directive(directive)
    )


def directive_head(directive):
    return directive[0]


def directive_headers(directive):
    assert is_headers_directive(directive)
    return directive[1]


def directive_regexes(directive):
    assert is_regexes_directive(directive)
    return directive[1]


def directive_special_types(directive):
    assert is_special_types_directive(directive)
    return directive[1]


def directive_comment(directive):
    assert is_comment_directive(directive)
    return directive[1]


def directive_raw(directive):
    assert is_raw_directive(directive)
    return directive[1]


def directive_transpiled_name(directive):
    assert is_function_directive(directive)
    return directive[1].get('transpiled', None)


def is_directive_none_out(directive):
    assert is_function_directive(directive)
    return directive[1]['out'] is None


def is_directive_returned_out(directive):
    assert is_function_directive(directive)
    out_spec = directive[1]['out']
    return isinstance(out_spec, str) and out_spec == 'Returned'


def is_directive_single_out(directive):
    assert is_function_directive(directive)
    out_spec = directive[1]['out']
    return isinstance(out_spec, str) and out_spec != 'Returned'


def is_directive_multi_out(directive):
    assert is_function_directive(directive)
    out_spec = directive[1]['out']
    if not isinstance(out_spec, tuple):
        return False
    assert len(out_spec) == 2
    assert isinstance(out_spec[0], str)
    assert isinstance(out_spec[1], tuple)
    return True


def directive_single_out(directive):
    assert is_directive_single_out(directive)
    return directive[1]['out']


def directive_multi_out(directive):
    assert is_directive_multi_out(directive)
    return directive[1]['out']


def directive_use_stream(directive):
    assert is_function_directive(directive)
    use_stream = directive[1].get('use_stream', False)
    if isinstance(use_stream, bool):
        if use_stream:
            return True, 'setStream'
        else:
            return False, None
    elif isinstance(use_stream, str):
        return True, use_stream
    else:
        assert False


def directive_except(directive):
    config = directive[1]
    if is_directive_none_out(directive):
        assert 'except' not in config and 'except?' not in config
        return None
    elif is_directive_returned_out(directive):
        assert 'except' not in config and 'except?' not in config
        return None
    elif is_directive_single_out(directive):
        excpt = config.get('except?')
        if excpt is not None:
            assert 'except' not in config  # either 'except?' or 'except'
            return 'except? {}'.format(excpt)
        excpt = config.get('except')
        if excpt is not None:
            return 'except {}'.format(excpt)
        assert False, "Either 'except?' or 'except' must be given"
    elif is_directive_multi_out(directive):
        assert 'except' not in config and 'except?' not in config
        return None
    else:
        assert False


# Transpilation

def transpile_func_name(env, directive, node):
    assert isinstance(node.type, c_ast.FuncDecl)
    name = directive_transpiled_name(directive)
    if name is not None:
        return name
    regex = regex_func_name(env)
    name = regex.fullmatch(node.name)[1]
    return name[0:2].lower() + name[2:]


def transpile_type_name(env, node):
    def transpile(env, names, quals):
        name = names[0]
        if len(names) > 1:
            return ' '.join(quals + names)
        elif is_special_type(name, env):
            name1 = special_type_transpiled(name, env)
            return ' '.join(quals + [name1])
        elif is_opaque_data_structure(name, env):
            regex = regex_type_name(env)
            m = regex.fullmatch(name)
            if m is not None:
                return ' '.join(quals + [m[1]])
            else:
                # Opaque data structures that do not follow the pattern above
                return ' '.join(quals + [name])
        elif is_enum(name, env):
            regex = regex_type_name(env)
            name1 = re.fullmatch(regex, name)[1]
            return ' '.join(quals + [name1])
        else:
            return ' '.join(quals + names)
    if isinstance(node, c_ast.TypeDecl):
        names = node.type.names
        quals = node.quals
        return transpile(env, names, quals)
    elif isinstance(node, c_ast.PtrDecl):
        return transpile_type_name(env, node.type) + '*'
    elif isinstance(node, c_ast.ArrayDecl):
        return transpile_type_name(env, node.type) + '*'
    elif isinstance(node, c_ast.Typedef):
        return transpile(env, [node.name], [])
    else:
        assert False


def erased_type_name(env, node):
    if isinstance(node, c_ast.TypeDecl):
        names = node.type.names
        name = names[0]
        if len(names) > 1:
            return None
        elif re.fullmatch(r'cu.+Handle_t', name) is not None:
            # We can remove this adhoc branch if we also use `intptr_t` for
            # opaque data structures. See #2716 and #3081.
            return 'intptr_t'
        elif is_special_type(name, env):
            return special_type_erased(name, env)
        elif is_opaque_data_structure(name, env):
            return 'size_t'  # TODO: should be 'intptr_t'?
        elif is_enum(name, env):
            return 'int'
        else:
            return None
    elif isinstance(node, c_ast.PtrDecl):
        return 'intptr_t'
    elif isinstance(node, c_ast.ArrayDecl):
        return 'intptr_t'
    else:
        assert False


def deref_var_name(name):
    m = re.fullmatch(r'p([A-Z].*)', name)
    if m is not None:
        name1 = m[1]
        return name1[0].lower() + name1[1:]
    else:
        return name


def transpile_type_conversion(env, node, var_name):
    if isinstance(node, c_ast.TypeDecl):
        type_names = node.type.names
        type_name = type_names[0]
        if len(type_names) == 1 and is_special_type(type_name, env):
            conversion = special_type_conversion(type_name, env)
            return conversion(var_name)
        else:
            cast_type = transpile_type_name(env, node)
            return '<{}>{}'.format(cast_type, var_name)
    elif isinstance(node, c_ast.PtrDecl):
        cast_type = transpile_type_name(env, node)
        return '<{}>{}'.format(cast_type, var_name)
    elif isinstance(node, c_ast.ArrayDecl):
        cast_type = transpile_type_name(env, node)
        return '<{}>{}'.format(cast_type, var_name)
    else:
        assert False


# Environment

def _collect_opaque_decls(nodes):
    opaques = []
    unique = set()
    for n in nodes:
        if (
            isinstance(n, c_ast.Typedef)
            and isinstance(n.type, c_ast.PtrDecl)
            and isinstance(n.type.type, c_ast.TypeDecl)
            and isinstance(n.type.type.type, c_ast.Struct)
            and n.name not in unique
        ):
            opaques.append(n)
            unique.add(n.name)
    return opaques


def _collect_enum_decls(nodes):
    enums = []
    unique = set()
    for n in nodes:
        if (
            isinstance(n, c_ast.Typedef)
            and isinstance(n.type, c_ast.TypeDecl)
            and isinstance(n.type.type, c_ast.Enum)
            and n.name not in unique
        ):
            enums.append(n)
            unique.add(n.name)
    return enums


def _collect_func_decls(nodes):
    def pred(node):
        return (isinstance(node, c_ast.Decl) and
                isinstance(node.type, c_ast.FuncDecl))
    return [n for n in nodes if pred(n)]


def _parse_headers(headers, version):
    assert version in ['11.0', '10.2']
    include_path = '/usr/local/cuda-{}/include/'.format(version)
    nodes = []
    for h in headers:
        path = os.path.join(include_path, h)        
        ast = pycparser.parse_file(path, use_cpp=True, cpp_args=[
            r'-I{}'.format(include_path),
            r'-I/home/ext-mtakagi/pycparser/utils/fake_libc_include',
            r'-D __attribute__(n)=',
            r'-D __inline__='])
        nodes.extend(ast.ext)
    return nodes


SPECIAL_TYPES = {
    'cudaDataType': {
        'transpiled': 'DataType',
        'erased': 'size_t',
        'conversion': '<DataType>{}',
    },
    'libraryPropertyType': {
        'transpiled': 'LibraryPropertyType',
        'erased': 'int',
        'conversion': '<LibraryPropertyType>{}',
    },
    'cudaStream_t': {
        'transpiled': 'driver.Stream',
        'erased': 'size_t',
        'conversion': '<driver.Stream>{}',
    },
}


def make_environment(directives):
    headers = directive_headers(directives[0])
    nodes_110 = _parse_headers(headers, '11.0')
    nodes_102 = _parse_headers(headers, '10.2')

    # Assuming the letters before the first appearance of `_` or `.` make the
    # library name.
    lib_name = re.match(r'^([a-z]+)(:?_|.)', headers[0])[1]

    patterns = directive_regexes(directives[1])
    regexes = {
        'func': re.compile(patterns['func']),
        'type': re.compile(patterns['type']),
    }

    special_types = {}
    special_types.update(SPECIAL_TYPES)
    if is_special_types_directive(directives[2]):
        special_types1 = directive_special_types(directives[2])
        special_types.update(special_types1)

    opaques = _collect_opaque_decls(nodes_110)  # assuming no opaques removed
    enums = _collect_enum_decls(nodes_110)  # assuming no enums removed
    funcs_110 = _collect_func_decls(nodes_110)
    funcs_102 = _collect_func_decls(nodes_102)
    return ('environment', special_types, opaques, enums,
            (funcs_110, funcs_102), regexes, lib_name)


def _environment_specials(env):
    assert env[0] == 'environment'
    return env[1]


def environment_opaques(env):
    assert env[0] == 'environment'
    lib_name = env[6]
    return [n for n in env[2] if lib_name in str(n.coord)]


def environment_enums(env):
    assert env[0] == 'environment'
    lib_name = env[6]
    return [n for n in env[3] if lib_name in str(n.coord)]


def _environment_funcs(cuda_ver, env):
    assert env[0] == 'environment'
    if cuda_ver == 'cuda-11.0':
        return env[4][0]
    elif cuda_ver == 'cuda-10.2':
        return env[4][1]
    else:
        assert False


def _environment_regexes(env):
    assert env[0] == 'environment'
    return env[5]


def is_special_type(name, env):
    return name in _environment_specials(env)


def is_opaque_data_structure(name, env):
    assert env[0] == 'environment'
    return name in (n.name for n in env[2])


def is_enum(name, env):
    assert env[0] == 'environment'
    return name in (n.name for n in env[3])


def query_func_decls(name, env):
    def compile_pattern(string):
        p = re.sub(r'<t\d?>', r'[ZCKEYDSHBX]', string)
        p = p.replace('{', '(|').replace(',', '|').replace('}', ')')
        return p, p != string
    def query(nodes):
        pat, generic = compile_pattern(name)
        if generic:
            pat = re.compile(pat, flags=re.I)
            return [n for n in nodes if pat.fullmatch(n.name) is not None]
        else:
            try:
                return [next(n for n in nodes if n.name == name)]
            except StopIteration:
                return []
    nodes_110 = _environment_funcs('cuda-11.0', env)
    decls = query(nodes_110)
    if decls != []:
        return decls, False

    nodes_102 = _environment_funcs('cuda-10.2', env)
    decls = query(nodes_102)
    if decls != []:
        return decls, True

    assert False, '`{}` not found'.format(name)


def special_type_conversion(name, env):
    return _environment_specials(env)[name]['conversion'].format


def special_type_transpiled(name, env):
    return _environment_specials(env)[name]['transpiled']


def special_type_erased(name, env):
    return _environment_specials(env)[name]['erased']


def regex_func_name(env):
    return _environment_regexes(env)['func']


def regex_type_name(env):
    return _environment_regexes(env)['type']


# Template

def read_template(path):
    with open(path) as f:
        return f.read()
