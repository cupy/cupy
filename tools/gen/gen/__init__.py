import os.path
import re

import pycparser
import pycparser.c_ast as c_ast


FILENAME_110 = '/usr/local/cuda-11.0/include/cusparse.h'
FILENAME_102 = '/usr/local/cuda-10.2/include/cusparse.h'

SPECIAL_TYPES = {
    'cudaDataType': {
        'transpiled': 'DataType',
        'erased': 'size_t',
        'conversion': '<DataType>{}'.format,
    },
    'cudaStream_t': {
        'transpiled': 'driver.Stream',
        'erased': 'size_t',
        'conversion': '<driver.Stream>{}'.format,
    },
    'cuComplex': {
        'transpiled': 'cuComplex',
        'erased': 'complex',
        'conversion': 'complex_to_cuda({})'.format,
    },
    'cuDoubleComplex': {
        'transpiled': 'cuDoubleComplex',
        'erased': 'double complex',
        'conversion': 'double_complex_to_cuda({})'.format,
    },
}


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


# Directive

def read_directives(path):
    # TODO: properly resolve path
    path1 = os.path.join(os.path.dirname(__file__), '../', path)
    with open(path1) as f:
        directives = eval(f.read())
    assert isinstance(directives, list)
    return directives


def is_comment_directive(directive):
    return directive[0] == 'Comment'


def is_function_directive(directive):
    head = directive[0]
    return isinstance(directive[0], str) and head != 'Comment'


def directive_head(directive):
    return directive[0]


def directive_comment(directive):
    assert is_comment_directive(directive)
    return directive[1]


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
    return directive[1]['use_stream']


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

def transpile_func_name(node):
    assert isinstance(node.type, c_ast.FuncDecl)
    name = re.match(r'cusparse([A-Z].*)', node.name)[1]
    return name[0].lower() + name[1:]


def transpile_type_name(env, node):
    def transpile(env, name, quals):
        if is_special_type(name, env):
            name1 = special_type_transpiled(name, env)
            return ' '.join(quals + [name1])
        elif is_opaque_data_structure(name, env):
            m = re.match(r'cusparse([A-Z].*)_t', name)
            if m is not None:
                return ' '.join(quals + [m[1]])
            else:
                # Opaque data structures that do not follow the pattern above
                return ' '.join(quals + [name])
        elif is_enum(name, env):
            name1 = re.match(r'cusparse([A-Z].*)_t', name)[1]
            return ' '.join(quals + [name1])
        else:
            return ' '.join(quals + [name])
    if isinstance(node, c_ast.TypeDecl):
        assert len(node.type.names) == 1
        name = node.type.names[0]
        quals = node.quals
        return transpile(env, name, quals)
    elif isinstance(node, c_ast.PtrDecl):
        return transpile_type_name(env, node.type) + '*'
    elif isinstance(node, c_ast.Typedef):
        return transpile(env, node.name, [])
    else:
        assert False


def erased_type_name(env, node):
    if isinstance(node, c_ast.TypeDecl):
        assert len(node.type.names) == 1
        name = node.type.names[0]
        if name == 'cusparseHandle_t':
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
    else:
        assert False


def deref_var_name(name):
    m = re.match(r'p([A-Z].*)', name)
    if m is not None:
        name1 = m[1]
        return name1[0].lower() + name1[1:]
    else:
        return name


def transpile_type_conversion(env, node, var_name):
    if isinstance(node, c_ast.TypeDecl):
        assert len(node.type.names) == 1
        type_name = node.type.names[0]
        if is_special_type(type_name, env):
            conversion = special_type_conversion(type_name, env)
            return conversion(var_name)
        else:
            cast_type = transpile_type_name(env, node)
            return '<{}>{}'.format(cast_type, var_name)
    elif isinstance(node, c_ast.PtrDecl):
        cast_type = transpile_type_name(env, node)
        return '<{}>{}'.format(cast_type, var_name)
    else:
        assert False


# Environment

def _collect_cusparse_decls(nodes):
    return [n for n in nodes if 'cusparse.h' in str(n.coord)]


def _collect_opaque_decls(nodes):
    tmp_decls = {}
    for n in nodes:
        if (isinstance(n, c_ast.Decl) and isinstance(n.type, c_ast.Struct)):
            name = n.type.name
            tmp_decls[name] = n
    opaques = []
    for n in nodes:
        if (isinstance(n, c_ast.Typedef)
                and isinstance(n.type, c_ast.PtrDecl)
                and isinstance(n.type.type, c_ast.TypeDecl)
                and isinstance(n.type.type.type, c_ast.Struct)):
            decl = tmp_decls.pop(n.type.type.type.name)
            opaques.append((decl, n))
    assert tmp_decls == {}
    return opaques


def _collect_enum_decls(nodes):
    def is_enum(node):
        return (isinstance(node, c_ast.Typedef) and
                isinstance(node.type, c_ast.TypeDecl) and
                isinstance(node.type.type, c_ast.Enum))
    return [n for n in nodes if is_enum(n)]


def _collect_func_decls(nodes):
    def pred(node):
        return (isinstance(node, c_ast.Decl) and
                isinstance(node.type, c_ast.FuncDecl))
    return [n for n in nodes if pred(n)]


def _parse_file(filename):
    ast = pycparser.parse_file(filename, use_cpp=True, cpp_args=[
        r'-I/usr/local/cuda/include',
        r'-I/home/ext-mtakagi/pycparser/utils/fake_libc_include',
        r'-D __attribute__(n)=',
        r'-D __inline__='])
    nodes = ast.ext
    nodes = _collect_cusparse_decls(nodes)
    return nodes


def make_environment():
    nodes_110 = _parse_file(FILENAME_110)
    nodes_102 = _parse_file(FILENAME_102)

    specials = SPECIAL_TYPES
    opaques = _collect_opaque_decls(nodes_110)  # assuming no opaques removed
    enums = _collect_enum_decls(nodes_110)  # assuming no enums removed
    funcs_110 = _collect_func_decls(nodes_110)
    funcs_102 = _collect_func_decls(nodes_102)
    return ('environment', specials, opaques, enums, (funcs_110, funcs_102))


def environment_specials(env):
    assert env[0] == 'environment'
    return env[1]

def environment_opaques(env):
    assert env[0] == 'environment'
    return env[2]


def environment_enums(env):
    assert env[0] == 'environment'
    return env[3]


def environment_funcs(cuda_ver, env):
    assert env[0] == 'environment'
    if cuda_ver == 'cuda-11.0':
        return env[4][0]
    elif cuda_ver == 'cuda-10.2':
        return env[4][1]
    else:
        assert False


def is_special_type(name, env):
    return name in environment_specials(env)


def is_opaque_data_structure(name, env):
    return name in (n.name for _, n in environment_opaques(env))


def is_enum(name, env):
    return name in (n.name for n in environment_enums(env))


def query_special_type(name, env):
    return environment_specials(env)[name]


def special_type_conversion(name, env):
    return query_special_type(name, env)['conversion']


def special_type_transpiled(name, env):
    return query_special_type(name, env)['transpiled']


def special_type_erased(name, env):
    return query_special_type(name, env)['erased']


def query_func_decls(name, env):
    def aux(node, t):
        return node.name == name.replace('<t>', t)
    def query(nodes):
        if '<t>' in name:
            return [n for n in nodes for t in 'SDCZ' if aux(n, t)]
        else:
            try:
                return [next(n for n in nodes if n.name == name)]
            except StopIteration:
                return []
    nodes_110 = environment_funcs('cuda-11.0', env)
    decls = query(nodes_110)
    if decls != []:
        return decls, False

    nodes_102 = environment_funcs('cuda-10.2', env)
    decls = query(nodes_102)
    if decls != []:
        return decls, True

    assert False, '`{}` not found'.format(name)


# Template

def read_template(path):
    # TODO: properly resolve path
    path1 = os.path.join(os.path.dirname(__file__), '../', path)
    with open(path1) as f:
        return f.read()
