import os
import os.path
import re
import tempfile

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
    first, second, third, *rest = directives
    assert is_cuda_versions_directive(first)
    assert is_headers_directive(second)
    assert is_patterns_directive(third)
    assert not any(is_cuda_versions_directive(d) for d in rest)
    assert not any(is_headers_directive(d) for d in rest)
    assert not any(is_patterns_directive(d) for d in rest)
    assert not any(is_special_types_directive(d) for d in rest[1:])
    return directives


def is_cuda_versions_directive(directive):
    return directive[0] == 'CudaVersions'


def is_headers_directive(directive):
    return directive[0] == 'Headers'


def is_patterns_directive(directive):
    return directive[0] == 'Patterns'


def is_special_types_directive(directive):
    return directive[0] == 'SpecialTypes'


def is_comment_directive(directive):
    return directive[0] == 'Comment'


def is_function_directive(directive):
    return (
        isinstance(directive[0], str)
        and not is_cuda_versions_directive(directive)
        and not is_headers_directive(directive)
        and not is_patterns_directive(directive)
        and not is_special_types_directive(directive)
        and not is_comment_directive(directive)
    )


def directive_head(directive):
    return directive[0]


def directive_cuda_versions(directive):
    assert is_cuda_versions_directive(directive)
    versions = directive[1]
    assert len(versions) in [1, 2]
    assert all(isinstance(v, str) for v in versions)
    if len(versions) == 2:
        assert versions[0] > versions[1]
    return versions


def directive_headers(directive):
    assert is_headers_directive(directive)
    return directive[1]


def directive_patterns(directive):
    assert is_patterns_directive(directive)
    return directive[1]


def directive_special_types(directive):
    assert is_special_types_directive(directive)
    return directive[1]


def directive_comment(directive):
    assert is_comment_directive(directive)
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
            assert False
        else:
            return False, None, None
    elif isinstance(use_stream, str):
        if use_stream == 'set':
            return True, 'set', 'setStream'
        elif use_stream == 'pass':
            return True, 'pass', None
        else:
            assert False
    elif isinstance(use_stream, tuple):
        head, func_name = use_stream
        assert head == 'set'
        assert isinstance(func_name, str)
        return True, 'set', func_name
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
    pattern = pattern_func_name(env)
    name = pattern.fullmatch(node.name)[1]
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
            pattern = pattern_type_name(env)
            m = pattern.fullmatch(name)
            if m is not None:
                return ' '.join(quals + [m[1]])
            else:
                # Opaque data structures that do not follow the pattern above
                return ' '.join(quals + [name])
        elif is_enum(name, env):
            pattern = pattern_type_name(env)
            name1 = pattern.fullmatch(name)[1]
            return ' '.join(quals + [name1])
        else:
            return ' '.join(quals + names)
    if isinstance(node, c_ast.TypeDecl):
        names = node.type.names
        quals = node.quals
        return transpile(env, names, quals)
    elif isinstance(node, c_ast.PtrDecl):
        quals = node.quals
        assert(len(quals) in [0, 1])  # assuming no qualifier or only 'const'
        # In case a special type as a pointer
        if isinstance(node.type, c_ast.TypeDecl):
            # Currently non-recursive pointer types only
            type_names = node.type.type.names
            type_name = type_names[0] + '*'
            if len(type_names) == 1 and is_special_type(type_name, env):
                # Currently support one-token types only
                type_name1 = special_type_transpiled(type_name, env)
                quals = node.type.quals
                return ' '.join(quals + [type_name1]) + (' ' + quals[0] if quals != [] else '')
        return transpile_type_name(env, node.type) + '*' + (' ' + quals[0] if quals != [] else '')
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
        # In case a special type as a pointer
        if isinstance(node.type, c_ast.TypeDecl):
            # Currently non-recursive pointer types only
            assert node.quals == []  # assuming PtrDecl has no qualifiers
            type_names = node.type.type.names
            type_name = type_names[0] + '*'
            if len(type_names) == 1 and is_special_type(type_name, env):
                # Currently support one-token types only
                return special_type_erased(type_name, env)
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
            # Currently for one-token types only
            conversion = special_type_conversion(type_name, env)
            quals = ''.join([q + ' ' for q in node.quals])
            return conversion(var=var_name, quals=quals)
        cast_type = transpile_type_name(env, node)
        return '<{}>{}'.format(cast_type, var_name)
    elif isinstance(node, c_ast.PtrDecl):
        # In case a special type as a pointer
        if isinstance(node.type, c_ast.TypeDecl):
            # Currently non-recursive pointer types only
            assert node.quals == []  # assuming PtrDecl has no qualifiers
            type_names = node.type.type.names
            type_name = type_names[0] + '*'
            if len(type_names) == 1 and is_special_type(type_name, env):
                # Currently support one-token types only
                conversion = special_type_conversion(type_name, env)
                quals = ''.join([q + ' ' for q in node.type.quals])
                return conversion(var=var_name, quals=quals)
        cast_type = transpile_type_name(env, node)
        return '<{}>{}'.format(cast_type, var_name)
    elif isinstance(node, c_ast.ArrayDecl):
        cast_type = transpile_type_name(env, node)
        return '<{}>{}'.format(cast_type, var_name)
    else:
        assert False


def transpile_expression(node):
    if isinstance(node, c_ast.Constant):
        value = node.value
        assert node.type in ['int', 'unsigned int']
        return value
    elif isinstance(node, c_ast.UnaryOp):
        op = node.op
        expr = transpile_expression(node.expr)
        return '{}{}'.format(op, expr)
    elif isinstance(node, c_ast.BinaryOp):
        op = node.op
        left = transpile_expression(node.left)
        right = transpile_expression(node.right)
        return '({} {} {})'.format(left, op, right)
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
    cuda_path = '/usr/local/cuda-{}/'.format(version)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_c_path = os.path.join(temp_dir, 'temp.c')
        with open(temp_c_path, 'w') as f:
            for h in headers:
                f.write('#include "{}"\n'.format(h))
        ast = pycparser.parse_file(temp_c_path, use_cpp=True, cpp_args=[
            os.path.expandvars('$CFLAGS'),  # use CFLAGS as CuPy does
            '-I{}include/'.format(cuda_path),
            '-I/home/ext-mtakagi/pycparser/utils/fake_libc_include',
            '-D __attribute__(n)=',
            '-D __inline__='])
    return ast.ext


SPECIAL_TYPES = {
    'cudaDataType': {
        'transpiled': 'DataType',
        'erased': 'size_t',
        'conversion': '<{quals}DataType>{var}',
    },
    'cudaDataType_t': {
        'transpiled': 'DataType',
        'erased': 'size_t',
        'conversion': '<{quals}DataType>{var}',
    },
    'libraryPropertyType': {
        'transpiled': 'LibraryPropertyType',
        'erased': 'int',
        'conversion': '<{quals}LibraryPropertyType>{var}',
    },
    'cudaStream_t': {
        'transpiled': 'driver.Stream',
        'erased': 'size_t',
        'conversion': '<{quals}driver.Stream>{var}',
    },
}


def make_environment(directives):
    cuda_versions = directive_cuda_versions(directives[0])
    headers = directive_headers(directives[1])

    patterns = directive_patterns(directives[2])
    compiled_patterns = {
        'func': re.compile(patterns['func']),
        'type': re.compile(patterns['type']),
    }

    special_types = {}
    special_types.update(SPECIAL_TYPES)
    if is_special_types_directive(directives[3]):
        special_types1 = directive_special_types(directives[3])
        special_types.update(special_types1)

    nodes_per_version = [_parse_headers(headers, ver) for ver in cuda_versions]
    # assuming no opaque pointers removed in a newer CUDA version
    opaques = _collect_opaque_decls(nodes_per_version[0])
    # assuming no enumerators removed in a newer CUDA version
    enums = _collect_enum_decls(nodes_per_version[0])
    funcs_per_version = [
        _collect_func_decls(nodes) for nodes in nodes_per_version]
    # FIXME
    if len(funcs_per_version) == 1:
        funcs_per_version.append(None)

    # Assuming the letters before the first appearance of `_` or `.` make the
    # library name.
    lib_name = re.match(r'^([a-z]+)(:?_|.)', headers[0])[1]

    return ('environment', special_types, opaques, enums, funcs_per_version,
            compiled_patterns, lib_name)


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


def _environment_funcs(env):
    assert env[0] == 'environment'
    return env[4]


def _environment_patterns(env):
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

    nodes0, nodes1 = _environment_funcs(env)

    decls = query(nodes0)
    if decls != []:
        return decls, False

    if nodes1 is not None:
        decls = query(nodes1)
        if decls != []:
            return decls, True

    assert False, '`{}` not found'.format(name)


def special_type_conversion(name, env):
    return _environment_specials(env)[name]['conversion'].format


def special_type_transpiled(name, env):
    return _environment_specials(env)[name]['transpiled']


def special_type_erased(name, env):
    return _environment_specials(env)[name]['erased']


def pattern_func_name(env):
    return _environment_patterns(env)['func']


def pattern_type_name(env):
    return _environment_patterns(env)['type']


# Template

def read_template(path):
    with open(path) as f:
        return f.read()
