import os.path
import re

import pycparser
from pycparser import c_ast


FILENAME_110 = '/usr/local/cuda-11.0/include/cusparse.h'
FILENAME_102 = '/usr/local/cuda-10.2/include/cusparse.h'


# functions, orders, configurations
# The reference and the header have different orders, even different sections.
# https://docs.nvidia.com/cuda/cusparse/index.html
DIRECTIVES = [
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
    ('cusparseCreateCsrsv2Info', {
        'out': 'info',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroyCsrsv2Info', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseCreateCsrsm2Info', {
        'out': 'info',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroyCsrsm2Info', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseCreateCsric02Info', {
        'out': 'info',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroyCsric02Info', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseCreateCsrilu02Info', {
        'out': 'info',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroyCsrilu02Info', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseCreateBsric02Info', {
        'out': 'info',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroyBsric02Info', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseCreateBsrilu02Info', {
        'out': 'info',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroyBsrilu02Info', {
        'out': None,
        'use_stream': False,
    }),
    # cuSPARSE Level 1 Function
    ('Comment', 'cuSPARSE Level 1 Function'),
    ('cusparse<t>gthr', {
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
    ('cusparseSpVecGet', {
        'out': ('SpVecAttributes',
                ('size', 'nnz', 'indices', 'values', 'idxType', 'idxBase',
                 'valueType')),
        'use_stream': False,
    }),
    # ...

    # cuSPARSE Generic API - Dense Vector APIs
    ('Comment', 'cuSPARSE Generic API - Dense Vector APIs'),
    ('cusparseCreateDnVec', {
        'out': 'dnVecDescr',
        'except?': 0,
        'use_stream': False,
    }),
    # ...
]


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
        assert False
    elif is_directive_multi_out(directive):
        assert 'except' not in config and 'except?' not in config
        return None
    else:
        assert False


def partition(pred, seq):
    a, b = [], []
    for item in seq:
        (a if pred(item) else b).append(item)
    return a, b
        

def compact(iterable):
    return (x for x in iterable if x is not None)


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
        return transpile_type_name(env, node.type) + '*'
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
        if isinstance(node.type, c_ast.TypeDecl):
            assert len(node.type.type.names) == 1
            name = node.type.type.names[0]
            if name == 'cusparseHandle_t':  # for cusparseHandle_t *
                return 'intptr_t'
            else:
                return 'size_t'  # TODO: should use 'intptr_t'?
        elif isinstance(node.type, c_ast.PtrDecl):
            return 'size_t'  # TODO: should use 'intptr_t'?
        else:
            assert False
    else:
        assert False


def transpile_ffi_decl(env, node, removed):
    def argaux(env, node):
        name = node.name
        type = transpile_type_name(env, node.type)
        return '{} {}'.format(type, name)
    assert isinstance(node.type, c_ast.FuncDecl)

    code = []
    if removed:
        code.append('# REMOVED')

    ret_type = transpile_type_name(env, node.type.type)
    name = node.name
    args = [argaux(env, p) for p in node.type.args.params]
    code.append('{} {}({})'.format(ret_type, name, ', '.join(args)))

    return '\n'.join(code)


def transpile_ffi(env, directive):
    if is_comment_directive(directive):
        comment = directive_comment(directive)
        return '\n# ' + comment
    elif is_function_directive(directive):
        head = directive_head(directive)
        decls, removed = query_func_decls(head, env)
        return '\n'.join(
            transpile_ffi_decl(env, decl, removed) for decl in decls)
    else:
        assert False


def transpile_aux_struct_decl(env, directive, node, removed):
    def argaux(env, node):
        name = deref_var_name(node.name)
        type = erased_type_name(env, node.type.type)
        if type is None:
            type = transpile_type_name(env, node.type.type)
        return '{} {}'.format(type, name)

    assert isinstance(node.type, c_ast.FuncDecl)

    out_type, out_args = directive_multi_out(directive)
    code = []

    if removed:
        code.append('# REMOVED')

    code.append('cdef class {}'.format(out_type))
    code.append('')

    params = [p for p in node.type.args.params if p.name in out_args]
    args = [argaux(env, p) for p in params]
    code.append('    def __init__(self, {})'.format(', '.join(args)))

    for p in params:
        attr = deref_var_name(p.name)
        code.append('        self.{attr} = {attr}'.format(attr=attr))

    return '\n'.join(code)


# Assuming multiple functions do not use the same auxiliary structure.
def transpile_aux_struct(env, directive):
    if is_comment_directive(directive):
        return None
    elif is_function_directive(directive):
        if is_directive_multi_out(directive):
            head = directive_head(directive)
            decls, removed = query_func_decls(head, env)
            assert len(decls) == 1  # assuming not type generic
            return transpile_aux_struct_decl(env, directive, decls[0], removed)
        else:
            return None
    else:
        assert False


def transpile_wrapper_def(env, directive, node):
    def argaux(env, node):
        name = node.name
        type = erased_type_name(env, node.type)
        if type is None:
            type = transpile_type_name(env, node.type)
        return '{} {}'.format(type, name)
    assert isinstance(node.type, c_ast.FuncDecl)
    if is_directive_none_out(directive):
        assert directive_except(directive) is None
        name = transpile_func_name(node)
        args = [argaux(env, p) for p in node.type.args.params]
        return '{}({})'.format(name, ', '.join(args))
    elif is_directive_returned_out(directive):
        assert directive_except(directive) is None
        ret_type = erased_type_name(env, node.type.type)
        if ret_type is None:
            ret_type = transpile_type_name(env, node.type.type)
        name = transpile_func_name(node)
        args = [argaux(env, p) for p in node.type.args.params]
        return '{} {}({})'.format(ret_type, name, ', '.join(args))
    elif is_directive_single_out(directive):
        out_name = directive_single_out(directive)
        out, params = partition(
            lambda p: p.name == out_name, node.type.args.params)
        assert len(out) == 1
        # dereference out[0]
        ret_type = erased_type_name(env, out[0].type.type)
        if ret_type is None:
            ret_type = transpile_type_name(env, out[0].type.type)
        name = transpile_func_name(node)
        args = [argaux(env, p) for p in params]
        excpt = directive_except(directive)
        return '{} {}({}) {}'.format(ret_type, name, ', '.join(args), excpt)
    elif is_directive_multi_out(directive):
        assert directive_except(directive) is None
        out_type, out_args = directive_multi_out(directive)
        outs, params = partition(
            lambda p: p.name in out_args, node.type.args.params)
        assert len(outs) > 1
        name = transpile_func_name(node)
        args = [argaux(env, p) for p in params]
        return '{} {}({})'.format(out_type, name, ', '.join(args))
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


def transpile_wrapper_call(env, directive, node):
    def argaux(env, directive, node):
        name = node.name
        if is_directive_single_out(directive):
            if name == directive_single_out(directive):
                name1 = deref_var_name(name)
                return '&{}'.format(name1)
        if is_directive_multi_out(directive):
            _, out_args = directive_multi_out(directive)
            if name in out_args:
                name1 = deref_var_name(name)
                return '&{}'.format(name1)
        erased_type = erased_type_name(env, node.type)
        if erased_type is not None:
            return transpile_type_conversion(env, node.type, name)
        return name
    assert isinstance(node.type, c_ast.FuncDecl)
    name = node.name
    args = [argaux(env, directive, p) for p in node.type.args.params]
    return '{}({})'.format(name, ', '.join(args))


def handler_name(node):
    assert isinstance(node, c_ast.Decl)
    for param in node.type.args.params:
        assert len(param.type.type.names) == 1
        type_name = param.type.type.names[0]
        if type_name == 'cusparseHandle_t':
            return param.name
    assert False


def transpile_wrapper_decl(env, directive, node, removed):
    assert isinstance(node.type, c_ast.FuncDecl)

    code = []

    # Comment if removed
    if removed:
        code.append('# REMOVED')

    # Function definition
    def_ = transpile_wrapper_def(env, directive, node)
    code.append('cpdef {}:'.format(def_))

    # Allocate space for the value to return
    if is_directive_none_out(directive):
        pass
    elif is_directive_returned_out(directive):
        pass
    elif is_directive_single_out(directive):
        out_name = directive_single_out(directive)
        out, params = partition(
            lambda p: p.name == out_name, node.type.args.params)
        assert len(out) == 1
        # dereference out[0]
        out_type = transpile_type_name(env, out[0].type.type)
        out_name1 = deref_var_name(out_name)
        code.append('    cdef {} {}'.format(out_type, out_name1))
    elif is_directive_multi_out(directive):
        _, out_args = directive_multi_out(directive)
        outs, params = partition(
            lambda p: p.name in out_args, node.type.args.params)
        assert len(outs) > 1
        for out, out_arg in zip(outs, out_args):
            # dereference out
            out_arg_type = transpile_type_name(env, out.type.type)
            out_arg1 = deref_var_name(out_arg)
            code.append('    cdef {} {}'.format(out_arg_type, out_arg1))
    else:
        assert False

    # Set stream if necessary
    if directive_use_stream(directive):
        handle = handler_name(node)
        code.append('    if stream_module.enable_current_stream:')
        code.append(
            '        setStream({}, stream_module.get_current_stream_ptr())'
            ''.format(handle))

    # Call cuSPARSE API and check its returned status if necessary
    if is_directive_returned_out(directive):
        call = transpile_wrapper_call(env, directive, node)
        code.append('    return {}'.format(call))
    else:
        status_var = 'status'  # assuming cusparse API does not use the name
        call = transpile_wrapper_call(env, directive, node)
        code.append('    {} = {}'.format(status_var, call))
        code.append('    check_status({})'.format(status_var))

    # Return value if necessary
    if is_directive_none_out(directive):
        pass
    elif is_directive_returned_out(directive):
        pass
    elif is_directive_single_out(directive):
        out_name = directive_single_out(directive)
        # dereference out[0]
        ret_type = erased_type_name(env, out[0].type.type)
        out_name1 = deref_var_name(out_name)
        if ret_type is not None:
            code.append('    return <{}>{}'.format(ret_type, out_name1))
        else:
            code.append('    return {}'.format(out_name1))
    elif is_directive_multi_out(directive):
        out_type, out_args = directive_multi_out(directive)
        outs, params = partition(
            lambda p: p.name in out_args, node.type.args.params)
        assert len(outs) > 1
        out_args1 = []
        for out_arg, out in zip(out_args, outs):
            # dereference out
            out_arg_type = erased_type_name(env, out.type.type)
            out_arg_name = deref_var_name(out_arg)
            if out_arg_type is not None:
                out_args1.append('<{}>{}'.format(out_arg_type, out_arg_name))
            else:
                out_args1.append(out_arg_name)
        code.append('    return {}({})'.format(out_type, ', '.join(out_args1)))
    else:
        assert False

    return '\n'.join(code)


def transpile_wrapper(env, directive):
    if is_comment_directive(directive):
        comment = directive_comment(directive)
        code = []
        code.append('')
        code.append('#' * max(40, len(comment) + 2))
        code.append('# ' + comment)
        return '\n'.join(code)
    elif is_function_directive(directive):
        head = directive_head(directive)
        decls, removed = query_func_decls(head, env)
        return '\n\n'.join(
            transpile_wrapper_decl(
                env, directive, decl, removed) for decl in decls)
    else:
        assert False


def validate_directives(directives):
    # too much, too less
    pass


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

def make_environment(nodes_110, nodes_102):
    specials = SPECIAL_TYPES
    opaques = collect_opaque_decls(nodes_110)  # assuming no opaques removed
    enums = collect_enum_decls(nodes_110)  # assuming no enums removed
    funcs_110 = collect_func_decls(nodes_110)
    funcs_102 = collect_func_decls(nodes_102)
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


def parse_file(filename):
    ast = pycparser.parse_file(filename, use_cpp=True, cpp_args=[
        r'-I/usr/local/cuda/include',
        r'-I/home/ext-mtakagi/pycparser/utils/fake_libc_include',
        r'-D __attribute__(n)=',
        r'-D __inline__='])
    nodes = ast.ext
    nodes = collect_cusparse_decls(nodes)
    return nodes


def indent(code):
   return '\n'.join('    ' + l if l != '' else '' for l in code.split('\n'))


if __name__ == '__main__':
    directives = DIRECTIVES
    validate_directives(directives)

    nodes_110 = parse_file(FILENAME_110)
    nodes_102 = parse_file(FILENAME_102)
    env = make_environment(nodes_110, nodes_102)

    path = os.path.join(
        os.path.dirname(__file__), 'templates/cusparse.pyx.template')
    with open(path) as f:
        template = f.read()

    ffi = '\n'.join(indent(transpile_ffi(env, d)) for d in directives)
    aux_struct = '\n'.join(
        compact(transpile_aux_struct(env, d) for d in directives))
    wrapper = '\n\n'.join(transpile_wrapper(env, d) for d in directives)
    print(template.format(ffi=ffi, aux_struct=aux_struct, wrapper=wrapper))
