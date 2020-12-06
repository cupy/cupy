import os.path
import re

import pycparser
from pycparser import c_ast


FILENAME = '/usr/local/cuda/include/cusparse.h'


# functions, orders, configurations
# The reference and the header have different orders, even different sections.
# https://docs.nvidia.com/cuda/cusparse/index.html
config = [  # TODO: The name `config` is fine?
    # cuSPARSE Management Function
    ('Comment', 'cuSPARSE Management Function'),
    ('cusparseCreate', {
        'out': 'handle',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroy', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseGetVersion', {
        'out': 'version',
        'except': -1,
        'use_stream': False,
    }),
    ('cusparseGetPointerMode', {
        'out': 'mode',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseSetPointerMode', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseGetStream', {
        'out': 'streamId',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseSetStream', {
        'out': None,
        'use_stream': False,
    }),

    # cuSPARSE Helper Function
    ('Comment', 'cuSPARSE Helper Function'),
    ('cusparseCreateMatDescr', {
        'out': 'descrA',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroyMatDescr', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseGetMatDiagType', {
        'out': 'Returned',
        'use_stream': False,
    }),
    ('cusparseGetMatFillMode', {
        'out': 'Returned',
        'use_stream': False,
    }),
    ('cusparseGetMatIndexBase', {
        'out': 'Returned',
        'use_stream': False,
    }),
    ('cusparseGetMatType', {
        'out': 'Returned',
        'use_stream': False,
    }),
    ('cusparseSetMatDiagType', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseSetMatFillMode', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseSetMatIndexBase', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseSetMatType', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseCreateCsrgemm2Info', {
        'out': 'info',
        'except?': 0,
        'use_stream': False,
    }),

    # cuSPARSE Level 1 Function
    ('Comment', 'cuSPARSE Level 1 Function'),
    ('cusparse<t>axpyi', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>gthr', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>gthrz', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>roti', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>sctr', {
        'out': None,
        'use_stream': True,
    }),

    # cuSPARSE Level 2 Function
    ('Comment', 'cuSPARSE Level 2 Function'),
    ('cusparse<t>bsrmv', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>bsrxmv', {
        'out': None,
        'use_stream': True,
    }),
    # ...
    ('cusparse<t>gemvi', {
        'out': None,
        'use_stream': True,
    }),

    # cuSPARSE Level 3 Function
    ('Comment', 'cuSPARSE Level 3 Function'),
    ('cusparse<t>bsrmm', {
        'out': None,
        'use_stream': True,
    }),
    # ...

    # cuSPARSE Extra Function
    ('Comment', 'cuSPARSE Extra Function'),
    ('cusparse<t>csrgeam2_bufferSizeExt', {
        'out': 'pBufferSizeInBytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparse<t>csrgeam2', {
        'out': None,
        'use_stream': True,
    }),
    # ...

    # cuSPARSE Preconditioners - Incomplete Cholesky Factorization: level 0
    ('Comment', ('cuSPARSE Preconditioners - '
                 'Incomplete Cholesky Factorization: level 0')),
    ('cusparse<t>csric02_bufferSize', {
        'out': 'pBufferSizeInBytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparse<t>csric02_analysis', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparse<t>csric02', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseXcsric02_zeroPivot', {
        'out': 'position',
        'except?': 0,
        'use_stream': False,
    }),

    # cuSPARSE Preconditioners - Incomplete LU Factorization: level 0
    ('Comment', ('cuSPARSE Preconditioners - '
                 'Incomplete LU Factorization: level 0')),
    ('cusparse<t>csrilu02_numericBoost', {
        'out': None,
        'use_stream': True,
    }),
    # ...

    # cuSPARSE Reordering
    ('Comment', 'cuSPARSE Reorderings'),

    # cuSPARSE Format Conversion
    ('Comment', 'cuSPARSE Format Conversion'),
    ('cusparseXcoo2csr', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>dense2csr', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>nnz', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseXcoosortByRow', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>nnz_compress', {
        'out': None,
        'use_stream': False,
    }),
    # ...

    # cuSPARSE Generic API - Sparse Vector APIs
    ('Comment', 'cuSPARSE Generic API - Sparse Vector APIs'),
    ('cusparseCreateSpVec', {
        'out': 'spVecDescr',
        'except?': 0,
        'use_stream': False,
    }),

    # cuSPARSE Generic API - Dense Vector APIs
    ('Comment', 'cuSPARSE Generic API - Dense Vector APIs'),
    ('cusparseCreateDnVec', {
        'out': 'dnVecDescr',
        'except?': 0,
        'use_stream': False,
    }),
    # ...
]


def partition(pred, seq):
    a, b = [], []
    for item in seq:
        (a if pred(item) else b).append(item)
    return a, b
        

# not used
def get_config(config, name):
    def possible_generic_func_name(name):
        # Can not determine the data type specifiers ('SDCZ') or the first
        # letter of a word ('C'reate)
        return re.sub(r'(cusparse)[SDCZ](.*)', r'\1<t>\2', name)
    config1 = config.get(name)
    if config1 is not None:
        return config1, None
    name1 = possible_generic_func_name(name)
    config1 = config.get(name1)
    if config1 is not None:
        type_spec = name[8]  # one of 'SDCZ'
        return config.get(name1), type_spec
    return None


def collect_cusparse_decls(nodes):
    return [n for n in nodes if 'cusparse.h' in str(n.coord)]


def collect_opaque_decls(nodes):
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


def collect_enum_decls(nodes):
    def is_enum(node):
        return (isinstance(node, c_ast.Typedef) and
                isinstance(node.type, c_ast.TypeDecl) and
                isinstance(node.type.type, c_ast.Enum))
    return [n for n in nodes if is_enum(n)]


def collect_func_decls(nodes):
    def pred(node):
        return (isinstance(node, c_ast.Decl) and
                isinstance(node.type, c_ast.FuncDecl))
    return [n for n in nodes if pred(n)]


def transpile_func_name(node):
    assert isinstance(node.type, c_ast.FuncDecl)
    name = re.match(r'cusparse([A-Z].*)', node.name)[1]
    return name[0].lower() + name[1:]


def transpile_type_name(env, node):
    def qualified(name):
        return ' '.join(node.quals + [name])
    if isinstance(node, c_ast.TypeDecl):
        assert len(node.type.names) == 1
        name = node.type.names[0]
        if is_special_type(name, env):
            name1 = special_type_transpiled(name, env)
            return qualified(name1)
        elif is_opaque_data_structure(name, env):
            m = re.match(r'cusparse([A-Z].*)_t', name)
            if m is not None:
                return  qualified(m[1])
            else:
                # Opaque data structures that do not follow the pattern above
                return qualified(name)
        elif is_enum(name, env):
            name1 = re.match(r'cusparse([A-Z].*)_t', name)[1]
            return qualified(name1)
        else:
            return qualified(name)
    elif isinstance(node, c_ast.PtrDecl):
        return transpile_type_name(env, node.type) + ' *'
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
            return 'size_t'
        elif is_enum(name, env):
            return 'size_t'
        else:
            return None
    elif isinstance(node, c_ast.PtrDecl):
        assert len(node.type.type.names) == 1
        name = node.type.type.names[0]
        if name == 'cusparseHandle_t':  # for cusparseHandle_t *
            return 'intptr_t'
        else:
            return 'size_t'  # TODO: should use 'intptr_t'?
    else:
        assert False


def transpile_ffi_decl(env, node):
    def argaux(env, node):
        name = node.name
        type = transpile_type_name(env, node.type)
        if type[-1] == '*':
            return '{}{}'.format(type, name)
        else:
            return '{} {}'.format(type, name)

    assert isinstance(node.type, c_ast.FuncDecl)
    ret_type = transpile_type_name(env, node.type.type)
    name = node.name
    args = [argaux(env, p) for p in node.type.args.params]
    return '{} {}({})'.format(ret_type, name, ', '.join(args))


def transpile_ffi(env, item):
    head = item[0]
    if head == 'Comment':
        comment = item[1]
        return '\n# ' + comment
    else:
        decls = query_func_decls(head, env)
        return '\n'.join(transpile_ffi_decl(env, decl) for decl in decls)


def transpile_wrapper_def(env, config, node):
    def argaux(env, node):
        name = node.name
        type = erased_type_name(env, node.type)
        if type is None:
            type = transpile_type_name(env, node.type)
        return '{} {}'.format(type, name)
    def config_except(config):
        excpt_ret = config.get('except?')
        if excpt_ret is not None:
            return 'except? {}'.format(excpt_ret)
        excpt_ret = config.get('except')
        if excpt_ret is not None:
            return 'except {}'.format(excpt_ret)
        assert False
    assert isinstance(node.type, c_ast.FuncDecl)
    out_name = config.get('out')
    if out_name is None:
        name = transpile_func_name(node)
        args = [argaux(env, p) for p in node.type.args.params]
        return '{}({}) except *'.format(name, ', '.join(args))
    elif out_name == 'Returned':
        ret_type = erased_type_name(env, node.type.type)
        if ret_type is None:
            ret_type = transpile_type_name(env, node.type.type)
        name = transpile_func_name(node)
        args = [argaux(env, p) for p in node.type.args.params]
        return '{} {}({})'.format(ret_type, name, ', '.join(args))
    else:
        out, params = partition(
            lambda p: p.name == out_name, node.type.args.params)
        assert len(out) == 1
        # dereference out[0]
        ret_type = erased_type_name(env, out[0].type.type)
        if ret_type is None:
            ret_type = transpile_type_name(env, out[0].type.type)
        name = transpile_func_name(node)
        args = [argaux(env, p) for p in params]
        excpt = config_except(config)
        return '{} {}({}) {}'.format(ret_type, name, ', '.join(args), excpt)


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


def transpile_wrapper_call(env, config, node):
    def argaux(env, config, node):
        name = node.name
        out_name = config.get('out')
        if out_name is not None and name == out_name:
            name1 = deref_var_name(name)
            return '&{}'.format(name1)
        else:
            erased_type = erased_type_name(env, node.type)
            if erased_type is not None:
                return transpile_type_conversion(env, node.type, name)
            else:
                return name
    assert isinstance(node.type, c_ast.FuncDecl)
    name = node.name
    args = [argaux(env, config, p) for p in node.type.args.params]
    return '{}({})'.format(name, ', '.join(args))


def handle_arg_name(node):
    assert isinstance(node, c_ast.Decl)
    for param in node.type.args.params:
        assert len(param.type.type.names) == 1
        type_name = param.type.type.names[0]
        if type_name == 'cusparseHandle_t':
            return param.name
    assert False


def transpile_wrapper_decl(env, config, node):
    assert isinstance(node.type, c_ast.FuncDecl)

    code = []

    # Function definition
    def_ = transpile_wrapper_def(env, config, node)
    code.append('cpdef {}:'.format(def_))

    # Allocate space for the value to return
    out_name = config.get('out')
    if out_name is not None and out_name != 'Returned':
        out, params = partition(
            lambda p: p.name == out_name, node.type.args.params)
        assert len(out) == 1
        # dereference out[0]
        out_type = transpile_type_name(env, out[0].type.type)
        out_name1 = deref_var_name(out_name)
        code.append('    cdef {} {}'.format(out_type, out_name1))

    # Set stream if necessary
    if config.get('use_stream', False):
        handle_name = handle_arg_name(node)
        code.append('    if stream_module.enable_current_stream:')
        code.append(
            '        setStream({}, stream_module.get_current_stream_ptr())'
            ''.format(handle_name))

    if out_name is None or out_name != 'Returned':
        # Call cuSPARSE API and check its status returned
        status_var = 'status'  # assuming cusparse API does not use the name
        call = transpile_wrapper_call(env, config, node)
        code.append('    {} = {}'.format(status_var, call))
        code.append('    check_status({})'.format(status_var))
    elif out_name == 'Returned':
        call = transpile_wrapper_call(env, config, node)
        code.append('    return {}'.format(call))
    else:
        assert False

    # Return value if necessary
    if out_name is not None and out_name != 'Returned':
        # dereference out[0]
        ret_type = erased_type_name(env, out[0].type.type)
        # `out_name1` should have been assigned on `cdef` above
        if ret_type is not None:
            code.append('    return <{}>{}'.format(ret_type, out_name1))
        else:
            code.append('    return {}'.format(out_name1))

    return '\n'.join(code)


def transpile_wrapper(env, item):
    head = item[0]
    if head == 'Comment':
        comment = item[1]
        code = []
        code.append('')
        code.append('# ' + '-' * len(comment))
        code.append('# ' + comment)
        code.append('# ' + '-' * len(comment))
        return '\n'.join(code)
    else:
        decls = query_func_decls(head, env)
        config = item[1]
        return '\n\n'.join(
            transpile_wrapper_decl(env, config, decl) for decl in decls)


def validate_config(config):
    # too much, too less
    pass


special_types = {
    'cuComplex': {
        'conversion': 'complex_to_cuda({})'.format,
        'transpiled': 'cuComplex',
        'erased': 'complex',
    },
    'cuDoubleComplex': {
        'conversion': 'double_complex_to_cuda({})'.format,
        'transpiled': 'cuDoubleComplex',
        'erased': 'double complex',
    },
}

def make_environment(nodes):
    specials = special_types
    opaques = collect_opaque_decls(nodes)
    enums = collect_enum_decls(nodes)
    funcs = collect_func_decls(nodes)
    return ('environment', specials, opaques, enums, funcs)


def environment_specials(env):
    assert env[0] == 'environment'
    return env[1]

def environment_opaques(env):
    assert env[0] == 'environment'
    return env[2]


def environment_enums(env):
    assert env[0] == 'environment'
    return env[3]


def environment_funcs(env):
    assert env[0] == 'environment'
    return env[4]


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
    nodes = environment_funcs(env)
    if '<t>' in name:
        decls = [n for n in nodes for t in 'SDCZ' if aux(n, t)]
        assert decls != []
        return decls
    else:
        return [next(n for n in nodes if n.name == name)]


def indent(code):
   return '\n'.join('    ' + l if l != '' else '' for l in code.split('\n'))


if __name__ == '__main__':
    validate_config(config)

    ast = pycparser.parse_file(FILENAME, use_cpp=True, cpp_args=[
        r'-I/usr/local/cuda/include',
        r'-I/home/ext-mtakagi/pycparser/utils/fake_libc_include',
        r'-D __attribute__(n)=',
        r'-D __inline__='])
    nodes = ast.ext
    nodes = collect_cusparse_decls(nodes)
    env = make_environment(nodes)

    path = os.path.join(
        os.path.dirname(__file__), 'templates/cusparse.pyx.template')
    with open(path) as f:
        template = f.read()

    ffi = '\n'.join(indent(transpile_ffi(env, item)) for item in config)
    wrapper = '\n\n'.join(transpile_wrapper(env, item) for item in config)
    print(template.format(ffi=ffi, wrapper=wrapper))
