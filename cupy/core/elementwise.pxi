import numpy
import six

from cupy.cuda import compiler
from cupy import util

from cupy.cuda cimport device
from cupy.cuda cimport function
from cupy.cuda import stream as stream_module


cdef dict _typenames_base = {
    numpy.dtype('float64'): 'double',
    numpy.dtype('float32'): 'float',
    numpy.dtype('float16'): 'float16',
    numpy.dtype('complex128'): 'complex<double>',
    numpy.dtype('complex64'): 'complex<float>',
    numpy.dtype('int64'): 'long long',
    numpy.dtype('int32'): 'int',
    numpy.dtype('int16'): 'short',
    numpy.dtype('int8'): 'signed char',
    numpy.dtype('uint64'): 'unsigned long long',
    numpy.dtype('uint32'): 'unsigned int',
    numpy.dtype('uint16'): 'unsigned short',
    numpy.dtype('uint8'): 'unsigned char',
    numpy.dtype('bool'): 'bool',
}

cdef str _all_type_chars = 'dfDFeqlihbQLIHB?'
# for c in 'dDfFeqlihbQLIHB?':
#    print('#', c, '...', np.dtype(c).name)
# d ... float64
# D ... complex128
# f ... float32
# F ... complex64
# e ... float16
# q ... int64
# l ... int64
# i ... int32
# h ... int16
# b ... int8
# Q ... uint64
# L ... uint64
# I ... uint32
# H ... uint16
# B ... uint8
# ? ... bool

cdef dict _typenames = {
    numpy.dtype(i).type: _typenames_base[numpy.dtype(i)]
    for i in _all_type_chars}

cdef tuple _python_scalar_type = six.integer_types + (float, bool, complex)
cdef tuple _numpy_scalar_type = tuple([numpy.dtype(i).type
                                       for i in _all_type_chars])

cdef set _python_scalar_type_set = set(_python_scalar_type)
cdef set _numpy_scalar_type_set = set(_numpy_scalar_type)

cdef dict _kind_score = {
    'b': 0,
    'u': 1,
    'i': 1,
    'f': 2,
    'c': 3,
}


cdef dict _python_type_to_numpy_type = {
    float: numpy.dtype(float).type,
    complex: numpy.dtype(complex).type,
    bool: numpy.dtype(bool).type}
for i in six.integer_types:
    _python_type_to_numpy_type[i] = numpy.int64


cdef str _get_ctype_name(dtype):
    if dtype is None:
        raise ValueError('dtype is None')
    name = _typenames.get(dtype)
    if name is None:
        name = _typenames[numpy.dtype(dtype).type]
    return name


cdef void _preprocess_args(list args, list arg_infos):
    """Preprocesses arguments for kernel invocation

    This function modifies args and arg_infos in-place.
    - Checks device compatibility for ndarrays
    - Converts Python scalars into NumPy scalars
    """
    cdef int dev_id = device.get_device_id()
    cdef ArgInfo ai
    cdef Py_ssize_t i

    for i, ai in enumerate(arg_infos):
        if ai.is_ndarray:
            arr_dev = (<ndarray>args[i]).data.device
            if arr_dev is not None and arr_dev.id != dev_id:
                raise ValueError(
                    'Array device must be same as the current '
                    'device: array device = %d while current = %d'
                    % (arr_dev.id, dev_id))
        elif ai.typ in _python_scalar_type_set:
            args[i] = _python_type_to_numpy_type[ai.typ](args[i])
            arg_infos[i] = ArgInfo_from_arg(args[i], True)
        elif ai.typ in _numpy_scalar_type_set:
            pass
        else:
            raise TypeError('Unsupported type %s' % ai.typ)


cpdef tuple _reduce_dims(list args, list arg_infos, tuple params, tuple shape):
    """Reduces the dimensions of arrays into the minimum without copy.

    This function modifies args and arg_infos in-place.
    """
    cdef Py_ssize_t i, j, n, ndim, cnt, axis, s
    cdef vector.vector[Py_ssize_t] vecshape, newshape, newstrides
    cdef vector.vector[bint] is_array_flags
    cdef vector.vector[vector.vector[Py_ssize_t]] args_strides
    cdef ParameterInfo p
    cdef ndarray arr, view
    cdef ArgInfo ai
    cdef bint is_array

    ndim = len(shape)
    if ndim <= 1:
        return shape

    n = len(args)
    for i, (p, ai) in enumerate(zip(params, arg_infos)):
        is_array = not p.raw and ai.is_ndarray
        is_array_flags.push_back(is_array)
        if is_array:
            arr = args[i]
            args_strides.push_back(arr._strides)

    vecshape = shape
    axis = -1
    cnt = 0
    for i in range(1, ndim):
        if vecshape[i - 1] == 1:
            continue
        for j in range(<Py_ssize_t>args_strides.size()):
            if args_strides[j][i] * vecshape[i] != args_strides[j][i - 1]:
                cnt += 1
                axis = i - 1
                break
        else:
            vecshape[i] *= vecshape[i - 1]
            vecshape[i - 1] = 1
    if vecshape[ndim - 1] != 1:
        cnt += 1
        axis = ndim - 1

    if cnt == ndim:
        return shape

    if cnt == 1:
        newshape.assign(<Py_ssize_t>1, <Py_ssize_t>vecshape[axis])
        new_args = []
        new_arg_infos = []
        for i in range(len(arg_infos)):
            if is_array_flags[i]:
                arr = args[i]
                arr = arr.view()
                newstrides.assign(
                    <Py_ssize_t>1, <Py_ssize_t>arr._strides[axis])
                arr._set_shape_and_strides(newshape, newstrides, False)
                args[i] = arr
                arg_infos[i] = ArgInfo_from_arg(arr)
        return tuple(newshape)

    for i in range(ndim):
        if vecshape[i] != 1:
            newshape.push_back(vecshape[i])

    new_args = []
    new_arg_infos = []
    for i in range(len(args)):
        if is_array_flags[i]:
            arr = args[i]
            arr = arr.view()
            newstrides.clear()
            for j in range(ndim):
                if vecshape[j] != 1:
                    newstrides.push_back(arr._strides[j])
            arr._set_shape_and_strides(newshape, newstrides, False)
            args[i] = arr
            arg_infos[i] = ArgInfo_from_arg(arr)
    return tuple(newshape)


cdef class ParameterInfo:
    cdef:
        readonly str name
        readonly object dtype
        readonly str ctype
        readonly bint raw
        readonly bint is_const

    def __init__(
            self, str name, object dtype, str ctype, bint raw, bint is_const):

        self.name = name
        self.dtype = dtype
        self.ctype = ctype
        self.raw = raw
        self.is_const = is_const

    @staticmethod
    def indexer(str name, bint raw):
        return ParameterInfo(name, None, None, raw, False)

    @staticmethod
    def parse(str param, bint is_const):
        name = None
        dtype = None
        ctype = None
        raw = False

        s = [i for i in param.split() if len(i) != 0]
        if len(s) < 2:
            raise Exception('Syntax error: %s' % param)

        for i in s[:-2]:
            if i == 'raw':
                raw = True
            else:
                raise Exception('Unknown keyword "%s"' % i)

        t, name = s[-2:]
        if t == 'CIndexer':
            return ParameterInfo.indexer(name, raw)
        else:
            if len(t) == 1:
                ctype = t
            else:
                dtype_ = numpy.dtype(t)
                dtype = dtype_.type
                if dtype_.name != t:
                    raise ValueError('Wrong type %s' % t)
                ctype = _get_ctype_name(dtype)

            return ParameterInfo(name, dtype, ctype, raw, is_const)

    def __repr__(self):
        return ('<ParameterInfo name={!r} dtype={} ctype={} raw={} '
                'is_const={}>').format(
                    self.name, self.dtype, self.ctype, self.raw,
                    self.is_const)


cdef class ArgInfo:
    cdef:
        readonly object arg
        readonly object typ
        readonly object dtype
        readonly tuple shape
        readonly int ndim
        readonly bint is_ndarray
        readonly tuple strides

    def __init__(
            self, object arg, type typ, dtype, tuple shape, int ndim,
            tuple strides):

        self.arg = arg
        self.typ = typ
        self.dtype = dtype
        self.shape = shape
        self.ndim = ndim
        self.strides = strides
        self.is_ndarray = typ is ndarray

    def __repr__(self):
        return '<ArgInfo typ={} shape={} dtype={} ndim={}>'.format(
            self.typ.__name__,
            self.shape,
            'None' if self.dtype is None else self.dtype.name,
            self.ndim)

    def __hash__(self):
        return (hash(self.typ) ^ hash(self.dtype) ^ hash(self.shape) ^
                hash(self.arg) ^ hash(self.strides))

    def __richcmp__(ArgInfo x, ArgInfo y, int op):
        if op == 2:
            if x is y:
                return True
            return (
                # arg is either None or a scalar
                x.arg == y.arg and
                x.typ is y.typ and
                x.dtype == y.dtype and
                x.shape == y.shape and
                x.strides == y.strides)
        raise NotImplementedError()

    cpdef str get_base_type_expr(self):
        if self.typ is Indexer:
            t = 'CIndexer<%d>' % self.ndim
        else:
            dt = self.get_ctype_name()
            if self.typ is ndarray:
                t = 'CArray<%s, %d>' % (dt, self.ndim)
            else:
                t = dt
        return t

    cdef str get_ctype_name(self):
        return _get_ctype_name(self.dtype)


cpdef ArgInfo ArgInfo_from_arg(arg, bint hold_strides=False):
    typ = type(arg)
    strides = None
    arg_ = None

    if arg is None:
        dtype = None
        shape = None
        ndim = -1
    elif typ is ndarray:
        dtype = (<ndarray>arg).dtype
        # Note: (<ndarray>arg).shape incurs a symbolic lookup and thus slower.
        shape = tuple((<ndarray>arg)._shape)
        ndim = len(shape)
        if hold_strides:
            strides = tuple((<ndarray>arg)._strides)
    elif typ is Indexer:
        dtype = None
        shape = (<Indexer>arg).shape
        ndim = len(shape)
    elif typ is slice:
        dtype = None
        shape = None
        ndim = -1
    elif typ in _python_scalar_type_set:
        arg_ = arg
        dtype = None
        shape = ()
        ndim = 0
    elif typ in _numpy_scalar_type_set:
        arg_ = arg
        dtype = arg.dtype
        shape = arg.shape
        ndim = len(shape)
    else:
        dtype = arg.dtype
        shape = arg.shape
        ndim = len(shape)
        if hold_strides:
            strides = arg.strides

    return ArgInfo(arg_, typ, dtype, shape, ndim, strides)


cpdef list ArgInfo_from_args(args, bint hold_strides=False):
    return [ArgInfo_from_arg(arg, hold_strides) for arg in args]


cdef class ParameterList:
    cdef:
        readonly tuple params  # tuple of ParameterInfo
        readonly tuple arg_infos  # tuple of ArgInfo
        readonly list _var_names
        readonly list _base_types
        readonly int nparams
        readonly int nin
        readonly int nout

    def __init__(self, tuple params, tuple arg_infos, int nin, int nout):
        assert len(params) == len(arg_infos), (len(params), len(arg_infos))
        assert nin >= 0, nin
        assert nout >= 0, nout
        assert nin + nout <= len(params), (nin, nout, len(params))
        self.params = params
        self.arg_infos = arg_infos
        self.nparams = len(params)
        self.nin = nin
        self.nout = nout

        self._var_names = None
        self._base_types = None

    def __hash__(self):
        return hash(self.params) ^ hash(self.arg_infos)

    def __richcmp__(ParameterList x, ParameterList y, int op):
        if op == 2:
            return (x.params == y.params and
                    x.arg_infos == y.arg_infos)
        raise NotImplementedError()

    cpdef list var_names(self):
        self._ensure_var_names()
        return self._var_names

    cdef _ensure_var_names(self):
        cdef ParameterInfo p
        cdef ArgInfo a
        if self._var_names is not None:
            return
        ret = []
        for p, a in zip(self.params, self.arg_infos):
            if not p.raw and a.is_ndarray:
                var_name = '_raw_' + p.name
            else:
                var_name = p.name
            ret.append(var_name)
        self._var_names = ret

    cpdef list base_types(self):
        self._ensure_base_types()
        return self._base_types

    cdef _ensure_base_types(self):
        if self._base_types is not None:
            return
        ret = []
        for i in range(self.nparams):
            arg_info = <ArgInfo>self.arg_infos[i]
            ret.append(arg_info.get_base_type_expr())
        self._base_types = ret

    cpdef str get_kernel_params_decl(self):
        self._ensure_var_names()
        self._ensure_base_types()
        return ', '.join([
            '%s %s' % (self._base_types[i], self._var_names[i])
            for i in range(self.nparams)])

    cpdef str get_entry_function_params_decl(self):
        self._ensure_var_names()
        self._ensure_base_types()
        return ', '.join([
            '%s %s' % (self._base_types[i], self._var_names[i])
            for i in range(self.nparams)])

    cpdef str get_entry_function_param_list(self):
        self._ensure_var_names()
        return ', '.join(self._var_names)

    cpdef list generate_ref_variable_decl_init_stmts(self):
        cdef ParameterInfo p
        cdef ArgInfo a
        stmts = []
        for p, a in zip(self.params, self.arg_infos):
            if not p.raw and a.typ is ndarray:
                stmts.append(
                    '{t} &{n} = _raw_{n}[_ind.get()];'.format(
                        t=p.ctype, n=p.name))
        return stmts


@util.memoize()
def _parse_param_infos(str s, bint is_const):
    """Returns a tuple of ParameterInfo's specified by a string."""

    if len(s) == 0:
        return ()
    return tuple([
        ParameterInfo.parse(_, is_const) for _ in s.strip().split(',')])


@util.memoize()
def _decide_param_types(
        tuple in_params, tuple out_params,
        tuple in_arg_dtypes, tuple out_arg_dtypes):
    """Determines the dtypes of input/output arguments in the generated kernel.

    Args:
        in_params: ParameterInfo's of input arguments.
        out_params: ParameterInfo's of output arguments..
        in_arg_dtypes: Dtypes of input arguments.
        out_arg_dtypes: Dtypes of output arguments.

    Returns:
        A 3-element tuple where the first 2 elements are the tuples of
        input/output dtypes in the generated kernel corresponding to each
        input/output argument, and the last element is a tuple containing the
        unique pairs of (dtype, ctype) in undefined order.
    """
    type_dict = {}
    if out_arg_dtypes:
        assert len(out_params) == len(out_arg_dtypes)
        for p, a in zip(out_params, out_arg_dtypes):
            if a is None:
                raise TypeError('Output arguments must be cupy.ndarray')
            if p.dtype is not None:
                if numpy.dtype(a) != numpy.dtype(p.dtype):
                    raise TypeError(
                        'Type is mismatched. %s %s %s' % (p.name, a, p.dtype))
            elif p.ctype in type_dict:
                t = type_dict[p.ctype]
                if numpy.dtype(t) != numpy.dtype(a):
                    raise TypeError(
                        'Type is mismatched. %s %s %s %s' % (
                            p.name, a, t, p.ctype))
            else:
                type_dict[p.ctype] = a

    assert len(in_params) == len(in_arg_dtypes)
    unknown_ctype = []
    for p, a in zip(in_params, in_arg_dtypes):
        if a is None:
            if p.dtype is None:
                unknown_ctype.append(p.ctype)
        else:
            if p.dtype is not None:
                if numpy.dtype(a) != numpy.dtype(p.dtype):
                    raise TypeError(
                        'Type is mismatched. %s %s %s' % (p.name, a, p.dtype))
            elif p.ctype in type_dict:
                t = type_dict[p.ctype]
                if numpy.dtype(t) != numpy.dtype(a):
                    raise TypeError(
                        'Type is mismatched. %s %s %s %s' % (
                            p.name, a, t, p.ctype))
            else:
                type_dict[p.ctype] = a

    in_types = tuple([type_dict[p.ctype] if p.dtype is None else p.dtype
                      for p in in_params])
    out_types = tuple([type_dict[p.ctype] if p.dtype is None else p.dtype
                       for p in out_params])
    return in_types, out_types, tuple(type_dict.items())


cdef _broadcast_impl _broadcast(list arg_infos, tuple params, int size):
    cdef ParameterInfo p
    cdef bint all_none = True
    cdef bint use_size = size >= 0
    cdef ArgInfo a
    if params is not None:
        assert len(arg_infos) <= len(params)
        arg_infos_ = []
        for i in range(len(arg_infos)):
            a = arg_infos[i]
            p = params[i]
            if not p.raw and a.is_ndarray:
                all_none = False
                arg_infos_.append(a)
            else:
                arg_infos_.append(None)
        arg_infos = arg_infos_
        if use_size:
            if not all_none:
                raise ValueError("Specified 'size' can be used only "
                                 "if all of the ndarray are 'raw'.")
        else:
            if all_none:
                raise ValueError('Loop size is Undecided')

    brod_impl = _broadcast_impl(arg_infos)
    return brod_impl


cdef list _allocate_out_args(
        tuple out_types, tuple out_shape, tuple out_params, bint use_size):
    """Allocates output arguments as needed."""

    # Check: if there is a raw parameter, size must be specified.
    if out_params is not None and not use_size:
        if any(p.raw for p in out_params):
            raise ValueError('Output array size is Undecided')
    return [ndarray(out_shape, t) for t in out_types]


cdef int _check_out_args(
        list out_args, tuple out_types, tuple out_shape, tuple out_params,
        str casting) except -1:
    cdef bint raw

    for i, (a, t) in enumerate(zip(out_args, out_types)):
        if not isinstance(a, ndarray):
            raise TypeError(
                'Output arguments type must be cupy.ndarray')
        if out_params is None:
            raw = False
        else:
            raw = (<ParameterInfo>out_params[i]).raw
        if not raw and a.shape != out_shape:
            raise ValueError('Out shape is mismatched')

        if casting and not numpy.can_cast(t, a.dtype, casting=casting):
            msg = 'output (typecode \'{}\') could not be coerced to ' \
                  'provided output parameter (typecode \'{}\') according to ' \
                  'the casting rule "{}"'.format(
                      numpy.dtype(t).char,
                      a.dtype.char,
                      casting)
            raise TypeError(msg)


cdef class _BaseKernelCallContext(object):
    cdef:
        readonly _BaseKernel kernel

    def __init__(self, _BaseKernel kernel):
        self.kernel = kernel

    cdef decide_param_types(self, list in_arg_infos, list out_arg_infos):
        raise NotImplementedError()

    cpdef broadcast_and_cast(self, arg_infos, int size, bint skip_raw):
        kernel = self.kernel
        nin = kernel.nin

        key = ('broadcast', kernel, size, skip_raw, tuple(arg_infos))
        tup = kernel.kernel_cache.get(key)
        if tup is None:
            if skip_raw:
                params = kernel.inout_params
            else:
                params = None

            # Broadcast
            brod_impl = _broadcast(arg_infos, params, size)
            shape = (size,) if size >= 0 else brod_impl.shape()

            arg_infos_ = brod_impl.apply_infos(arg_infos)

            tup = (brod_impl, shape, arg_infos_)
            kernel.kernel_cache[key] = tup

        return tup


cdef class _BaseElementwiseKernelCallContext(_BaseKernelCallContext):
    cdef:
        readonly arg_infos
        readonly int size
        readonly str casting
        readonly int nin
        readonly int nout
        readonly object stream

        str _preamble
        str _operation

    def __init__(self, _BaseElementwiseKernel kernel, arg_infos,
                 int size, str casting, stream):

        assert stream is None or isinstance(stream, stream_module.Stream), \
            type(stream)

        super(_BaseElementwiseKernelCallContext, self).__init__(kernel)

        self.arg_infos = arg_infos
        self.size = size
        self.casting = casting
        self.nin = kernel.nin
        self.nout = kernel.nout
        self.stream = stream

    cpdef call(self, args):
        cdef _broadcast_impl brod_impl
        cdef ArgInfo ai
        cdef Py_ssize_t i

        size = self.size
        casting = self.casting
        stream = self.stream
        kernel = <_BaseElementwiseKernel>self.kernel
        nin = kernel.nin
        nout = kernel.nout
        reduce_dims = kernel.reduce_dims
        out_params = kernel.out_params
        inout_params = kernel.inout_params
        params = kernel.params

        # Preprocess
        args = list(args)
        arg_infos = ArgInfo_from_args(args, True)
        _preprocess_args(args, arg_infos)

        # Decide parameter dtypes.
        in_types, out_types = self.decide_param_types(
            arg_infos[:nin],
            arg_infos[nin:])

        # Broadcast
        brod_impl, shape, arg_infos = self.broadcast_and_cast(
            arg_infos, size, kernel.broadcast_skip_raw)
        args = brod_impl.apply(args)

        #
        in_args = args[:nin]
        out_args = args[nin:]
        in_arg_infos = arg_infos[:nin]
        out_arg_infos = arg_infos[nin:]

        # Cast scalar args to ndarrays
        for i, ai in enumerate(in_arg_infos):
            if not ai.is_ndarray:
                in_args[i] = in_types[i](in_args[i])
                in_arg_infos[i] = ArgInfo_from_arg(in_args[i])
        for i, ai in enumerate(out_arg_infos):
            if not ai.is_ndarray:
                out_args[i] = out_types[i](out_args[i])
                out_arg_infos[i] = ArgInfo_from_arg(out_args[i])

        # Allocate output args as needed.
        if len(out_args) == 0:
            out_args = _allocate_out_args(
                out_types, shape, out_params, size >= 0)
            out_arg_infos = ArgInfo_from_args(out_args)
        else:
            _check_out_args(out_args, out_types, shape, out_params, casting)

        if nout == 1:
            ret = out_args[0]
        else:
            ret = tuple(out_args)

        # If the shape is 0-sized, return immediately without any computation
        if 0 in shape:
            return ret

        inout_args = in_args + out_args
        inout_arg_infos = in_arg_infos + out_arg_infos

        # Reduce array dimensions
        if reduce_dims:
            shape = _reduce_dims(
                inout_args, inout_arg_infos, inout_params, shape)

        # Append indexer
        indexer = Indexer(shape)
        inout_args.append(indexer)
        inout_arg_infos.append(ArgInfo_from_arg(indexer))

        inout_arg_infos = tuple(inout_arg_infos)

        key = ('kernel', device.get_device_id(), inout_arg_infos) + \
            self.get_code_args()
        kern = kernel.kernel_cache.get(key)
        if kern is None:
            param_list = ParameterList(params, inout_arg_infos, nin, nout)
            params_decl = param_list.get_entry_function_params_decl()
            name = kernel.name
            op_class_name = name + '__'
            loop_class_name = name + '__loop_'
            emitter = emit.KernelCodeEmitter(name)

            # Retrieve source
            self._operation, self._preamble = self.get_code(
                param_list, self.get_code_args())

            # Emit elementwise op class
            self.emit_op_class(emitter, op_class_name, param_list)

            # Emit elementwise loop computation class
            self.emit_loop_class(
                emitter, loop_class_name, op_class_name, param_list)

            # Emit kernel entry function
            self.emit_kernel_entry_function(
                emitter, loop_class_name, param_list)

            kern = emitter.get_function(kernel.options)
            kernel.kernel_cache[key] = kern

        kern.linear_launch(indexer.size, inout_args, stream=stream)
        return ret

    cpdef emit_call_stmt(self, emitter, param_list):
        pass

    cdef tuple get_code_args(self):
        return ()

    cdef get_code(self, ParameterList param_list, tuple args):
        raise NotImplementedError()

    cpdef get_op_param_list(self, ParameterList param_list):
        lst = ['i'] + param_list.var_names()
        return ', '.join(lst)

    cpdef get_op_params_decl(self, ParameterList param_list):
        lst = ['const ptrdiff_t& i'] + \
              ['{}& {}'.format(base_type, var_name)
               for base_type, var_name
               in zip(param_list.base_types(), param_list.var_names())]
        return ', '.join(lst)

    cpdef emit_op_class(
            self, emitter, class_name, ParameterList param_list):

        op_params_decl = self.get_op_params_decl(param_list)
        temp = emit.Templater(
            class_name=class_name,
            preamble=emitter.indent_lines(self._preamble, 2),
            op_params_decl=op_params_decl,
            operation=emitter.indent_lines(self._operation, 4))

        emitter.emit_lines(temp('''
class ${class_name} {
private:
${preamble}
public:
  __device__ void op(${op_params_decl}) {
${operation}
    ;
  }
};
'''))

    cdef emit_kernel_entry_function(
            self, emitter, str loop_class_name, ParameterList param_list):

        # Emit kernel entry function
        with emitter.construct_kernel_entry_function(param_list):
            temp = emit.Templater(
                loop_class_name=loop_class_name,
                param_list_expr=param_list.get_entry_function_param_list())
            emitter.emit_line(temp(
                '''${loop_class_name}().compute(${param_list_expr});'''))

    cpdef emit_op_caller(
            self, emitter, str class_name, ParameterList param_list):

        op_param_list = self.get_op_param_list(param_list)
        emitter.emit_line(
            '{}().op({});'.format(class_name, op_param_list))

    cpdef emit_loop_class(
            self, emitter, str class_name, str op_class_name,
            ParameterList param_list):

        kernel = <_ElementwiseKernel>self.kernel
        loop_preamble = self._preamble
        loop_prep = kernel.loop_prep
        after_loop = kernel.after_loop

        temp = emit.Templater(
            params_decl=param_list.get_kernel_params_decl(),
            class_name=class_name,
            loop_preamble=emitter.indent_lines(loop_preamble, 2),
            loop_prep=emitter.indent_lines(loop_prep, 4),
            after_loop=emitter.indent_lines(after_loop, 4))

        with emitter.indented_construct(6) as c:

            c.emit_construct(temp('''
class ${class_name} {
private:
${loop_preamble}
public:
  __device__ void compute(${params_decl}) {
${loop_prep};
    CUPY_FOR(i, _ind.size()) {
      _ind.set(i);
'''))

            self.emit_op_caller(emitter, op_class_name, param_list)

            c.emit_construct(temp('''
    }
${after_loop};
  }
};
'''))


cdef class _ElementwiseKernelCallContext(_BaseElementwiseKernelCallContext):

    cdef:
        tuple _types

    def __init__(
            self, _BaseElementwiseKernel elementwise, arg_infos,
            int size, stream):

        super(_ElementwiseKernelCallContext, self).__init__(
            elementwise, arg_infos, size, None, stream)

    cdef decide_param_types(self, list in_arg_infos, list out_arg_infos):
        cdef ArgInfo a
        kernel = <_ElementwiseKernel>self.kernel

        key = (
            'param_types',
            self.kernel, tuple(in_arg_infos), tuple(out_arg_infos))
        tup = kernel.kernel_cache.get(key)
        if tup is None:
            in_ndarray_types = tuple(
                [a.dtype.type if a.is_ndarray else None for a in in_arg_infos])
            out_ndarray_types = tuple([a.dtype.type for a in out_arg_infos])

            tup = _decide_param_types(
                kernel.in_params, kernel.out_params,
                in_ndarray_types, out_ndarray_types)

            kernel.kernel_cache[key] = tup

        in_types, out_types, types = tup

        # Store types for succeeding get_code()
        self._types = types

        return in_types, out_types

    cdef tuple get_code_args(self):
        return (self._types,)

    cdef get_code(self, ParameterList param_list, tuple code_args):
        cdef tuple types
        types, = code_args
        operation_ = (<_ElementwiseKernel>self.kernel).operation
        preamble_ = (<_ElementwiseKernel>self.kernel).preamble

        types_preamble = '\n'.join([
            'typedef %s %s;' % (_get_ctype_name(v), k) for k, v in types])
        preamble = types_preamble + '\n' + preamble_

        op = []
        for stmt in param_list.generate_ref_variable_decl_init_stmts():
            op.append(stmt)
        op.append(operation_)
        operation = '\n'.join(op)
        return operation, preamble


cdef class _BaseKernel(object):

    cdef:
        readonly str name
        readonly int nin
        readonly int nout
        readonly int nargs
        readonly tuple in_params
        readonly tuple out_params
        readonly tuple inout_params
        readonly dict kernel_cache
        readonly tuple options

    def __init__(self, in_params, out_params, name, options):
        if not compiler.is_valid_kernel_name(name):
            raise ValueError(
                'Invalid kernel name: "%s"' % name)

        self.in_params = in_params
        self.out_params = out_params
        self.nin = len(in_params)
        self.nout = len(out_params)
        self.nargs = self.nin + self.nout
        self.inout_params = in_params + out_params
        self.name = name
        self.options = options
        self.kernel_cache = {}


cdef class _BaseElementwiseKernel(_BaseKernel):
    cdef:
        readonly tuple params
        readonly bint reduce_dims
        readonly bint broadcast_skip_raw
        readonly str loop_prep
        readonly str after_loop

    def __init__(
            self, in_params, out_params, name, bint reduce_dims,
            bint broadcast_skip_raw,
            loop_prep, after_loop, options):
        cdef ParameterInfo p

        if any([p.name == 'i' for p in in_params]):
            raise ValueError("Can not use 'i' as a parameter name")
        if any([p.name == 'i' for p in out_params]):
            raise ValueError("Can not use 'i' as a parameter name")

        super(_BaseElementwiseKernel, self).__init__(
            in_params, out_params, name, options)

        self.inout_params = in_params + out_params
        self.params = self.inout_params + (
            ParameterInfo.parse('CIndexer _ind', False),)
        self.reduce_dims = reduce_dims
        self.broadcast_skip_raw = broadcast_skip_raw
        self.loop_prep = loop_prep
        self.after_loop = after_loop

    cpdef _BaseElementwiseKernelCallContext create_call_context(
            self, arg_infos, kwargs):

        raise NotImplementedError()

cdef class _ElementwiseKernel(_BaseElementwiseKernel):

    cdef:
        readonly str operation
        readonly str preamble

    def __init__(
            self, str in_params, str out_params, str operation,
            str name, bint reduce_dims, str preamble,
            str loop_prep, str after_loop, tuple options):

        in_params_ = _parse_param_infos(in_params, True)
        out_params_ = _parse_param_infos(out_params, False)

        super(_ElementwiseKernel, self).__init__(
            in_params_, out_params_, name, reduce_dims, True,
            loop_prep, after_loop, options)

        self.operation = operation
        self.preamble = preamble

    def __call__(self, *args, **kwargs):
        """Compiles and invokes the elementwise kernel.

        The compilation runs only if the kernel is not cached. Note that the
        kernels with different argument dtypes or dimensions are not
        compatible. It means that single ElementwiseKernel object may be
        compiled into multiple kernel binaries.

        Args:
            args: Arguments of the kernel.
            size (int): The size of index range. If specified, ``_ind.size()``
                in the kernel code will evaluate to this value. Otherwise,
                it's determined by the result of broadcast.

        Returns:
            Arrays are returned according to the ``out_params`` argument of the
            ``__init__`` method.

        """

        arg_infos = ArgInfo_from_args(args)
        call_ctx = self.create_call_context(arg_infos, kwargs)
        return call_ctx.call(args)

    cpdef _BaseElementwiseKernelCallContext create_call_context(
            self, arg_infos, kwargs):

        size = kwargs.pop('size', None)
        stream = kwargs.pop('stream', None)

        nargs = len(arg_infos)
        if nargs != self.nin and nargs != self.nargs:
            raise TypeError('Wrong number of arguments for %s' % self.name)

        if size is None:
            size = -1

        return _ElementwiseKernelCallContext(
            self, arg_infos, size, stream)


cdef class ElementwiseKernel(object):

    """User-defined elementwise kernel.

    This class can be used to define an elementwise kernel with or without
    broadcasting.

    The kernel is compiled at an invocation of the
    :meth:`~ElementwiseKernel.__call__` method,
    which is cached for each device.
    The compiled binary is also cached into a file under the
    ``$HOME/.cupy/kernel_cache/`` directory with a hashed file name. The cached
    binary is reused by other processes.

    Args:
        in_params (str): Input argument list.
        out_params (str): Output argument list.
        operation (str): The body in the loop written in CUDA-C/C++.
        name (str): Name of the kernel function. It should be set for
            readability of the performance profiling.
        reduce_dims (bool): If ``False``, the shapes of array arguments are
            kept within the kernel invocation. The shapes are reduced
            (i.e., the arrays are reshaped without copy to the minimum
            dimension) by default. It may make the kernel fast by reducing the
            index calculations.
        options (tuple, list or str): Options passed to the ``nvcc`` command.
        preamble (str): Fragment of the CUDA-C/C++ code that is inserted at the
            top of the cu file.
        loop_prep (str): Fragment of the CUDA-C/C++ code that is inserted at
            the top of the kernel function definition and above the ``for``
            loop.
        after_loop (str): Fragment of the CUDA-C/C++ code that is inserted at
            the bottom of the kernel function definition.

    """

    _cache = {}

    cdef:
        readonly tuple in_params
        readonly tuple out_params
        readonly int nin
        readonly int nout
        readonly int nargs
        readonly tuple params
        readonly str operation
        readonly str name
        readonly bint reduce_dims
        readonly str preamble
        readonly object kwargs

        readonly _ElementwiseKernel _kernel

    def __init__(
            self, str in_params, str out_params, str operation,
            str name='kernel', bint reduce_dims=True, str preamble='',
            **kwargs):
        cdef _ElementwiseKernel kernel

        loop_prep = kwargs.pop('loop_prep', '')
        after_loop = kwargs.pop('after_loop', '')
        options = kwargs.pop('options', ())
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)

        if isinstance(options, str):
            options = (options,)
        elif isinstance(options, list):
            options = tuple(options)

        key = (
            in_params, out_params, operation, name, reduce_dims, preamble,
            loop_prep, after_loop, options)
        kernel = self._cache.get(key)
        if kernel is None:
            kernel = _ElementwiseKernel(
                in_params, out_params, operation, name, reduce_dims, preamble,
                loop_prep, after_loop, options)
            self._cache[key] = kernel

        self.in_params = kernel.in_params
        self.out_params = kernel.out_params
        self.nin = kernel.nin
        self.nout = kernel.nout
        self.nargs = kernel.nargs
        self.params = kernel.params
        self.operation = kernel.operation
        self.name = kernel.name
        self.reduce_dims = kernel.reduce_dims
        self.preamble = kernel.preamble
        self._kernel = kernel

    def __call__(self, *args, **kwargs):
        """Compiles and invokes the elementwise kernel.

        The compilation runs only if the kernel is not cached. Note that the
        kernels with different argument dtypes or dimensions are not
        compatible. It means that single ElementwiseKernel object may be
        compiled into multiple kernel binaries.

        Args:
            args: Arguments of the kernel.
            size (int): The size of index range. If specified, ``_ind.size()``
                in the kernel code will evaluate to this value. Otherwise,
                it's determined by the result of broadcast.

        Returns:
            Arrays are returned according to the ``out_params`` argument of the
            ``__init__`` method.

        """

        return self._kernel(*args, **kwargs)


cdef class _UfuncKernelCallContext(_BaseElementwiseKernelCallContext):

    cdef:
        readonly object dtype

        str _routine
        tuple _in_types
        tuple _out_types

    def __init__(
            self, _BaseElementwiseKernel elementwise, arg_infos,
            dtype, str casting):

        super(_UfuncKernelCallContext, self).__init__(
            elementwise, arg_infos, -1, casting, None)

        self.dtype = dtype
        self._routine = None

    cdef decide_param_types(self, list in_arg_infos, list out_arg_infos):
        kernel = <_UfuncKernel>self.kernel

        key = (
            'param_types',
            tuple(in_arg_infos), tuple(out_arg_infos), self.dtype)
        tup = kernel.kernel_cache.get(key)
        if tup is None:
            tup = _guess_routine(
                kernel.name,
                kernel._routine_cache,
                kernel._ops,
                in_arg_infos, self.dtype)
            kernel.kernel_cache[key] = tup

        in_types, out_types, routine = tup

        # Store variables for succeeding get_code()
        self._routine = routine
        self._in_types = in_types
        self._out_types = out_types

        return in_types, out_types

    cdef tuple get_code_args(self):
        return (self._routine, self._in_types, self._out_types)

    cdef get_code(self, ParameterList param_list, tuple code_args):
        cdef ArgInfo a
        cdef int i, nin
        preamble_ = (<_UfuncKernel>self.kernel)._preamble
        routine_, in_types, out_types = code_args

        nin = len(in_types)
        types = []
        op = []
        for i, x in enumerate(in_types):
            types.append('typedef %s in%d_type;' % (_get_ctype_name(x), i))
            a = param_list.arg_infos[i]
            if a.is_ndarray:
                op.append(
                    'const in{0}_type in{0}(_raw_in{0}[_ind.get()]);'.format(
                        i))

        for i, x in enumerate(out_types):
            a = param_list.arg_infos[i + nin]
            types.append('typedef %s out%d_type;' % (
                _get_ctype_name(a.dtype), i))
            op.append('out{0}_type &out{0} = _raw_out{0}[_ind.get()];'.format(
                i, a.get_ctype_name()))

        op.append(routine_)
        operation = '\n'.join(op)

        types.append(preamble_)
        preamble = '\n'.join(types)

        return operation, preamble


cdef class _UfuncKernel(_BaseElementwiseKernel):

    cdef:
        tuple _ops
        str _preamble
        dict _routine_cache

    def __init__(self, nin, nout, name, ops, preamble):
        self.nin = nin
        self.nout = nout
        self.nargs = nin + nout
        self._ops = ops
        self._preamble = preamble
        in_params = tuple(
            ParameterInfo.parse('T in%d' % i, True)
            for i in range(nin))
        out_params = tuple(
            ParameterInfo.parse('T out%d' % i, False)
            for i in range(nout))
        self.params = in_params + out_params + \
            (ParameterInfo.parse('CIndexer _ind', False),)
        self._routine_cache = {}

        super(_UfuncKernel, self).__init__(
            in_params, out_params, name, True, False, '', '', ())

    def __call__(self, *args, **kwargs):
        out = kwargs.pop('out', None)
        if out is not None:
            if self.nout != 1:
                raise ValueError("Cannot use 'out' in %s" % self.name)
            if len(args) != self.nin:
                raise ValueError("Cannot specify 'out' as both "
                                 "a positional and keyword argument")
            args += (out,)

        arg_infos = ArgInfo_from_args(args)
        call_ctx = self.create_call_context(arg_infos, kwargs)
        return call_ctx.call(args)

    cpdef _BaseElementwiseKernelCallContext create_call_context(
            self, arg_infos, kwargs):

        dtype = kwargs.pop('dtype', None)
        # Note default behavior of casting is 'same_kind' on numpy>=1.10
        casting = kwargs.pop('casting', 'same_kind')
        if dtype is not None and not isinstance(dtype, numpy.dtype):
            dtype = numpy.dtype(dtype)
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)

        return _UfuncKernelCallContext(
            self, arg_infos, dtype, casting)


cdef tuple _guess_routine_from_in_types(tuple ops, tuple in_types):
    cdef Py_ssize_t i, n
    cdef tuple op, op_types
    n = len(in_types)
    can_cast = numpy.can_cast
    for op in ops:
        op_types = op[0]
        for i in range(n):
            if not can_cast(in_types[i], op_types[i]):
                break
        else:
            return op
    return None


cdef tuple _guess_routine_from_dtype(tuple ops, object dtype):
    cdef tuple op, op_types
    for op in ops:
        op_types = op[1]
        for t in op_types:
            if t != dtype.type:
                break
        else:
            return op
    return None


cdef bint _check_should_use_min_scalar(list in_arg_infos) except *:
    cdef int kind, max_array_kind, max_scalar_kind
    cdef bint all_scalars
    cdef ArgInfo a
    all_scalars = True
    max_array_kind = -1
    max_scalar_kind = -1
    for a in in_arg_infos:
        kind = _kind_score[a.dtype.kind]
        if a.is_ndarray:
            all_scalars = False
            max_array_kind = max(max_array_kind, kind)
        else:
            max_scalar_kind = max(max_scalar_kind, kind)
    return (max_scalar_kind != -1 and
            not all_scalars and
            max_array_kind >= max_scalar_kind)


cdef tuple _guess_routine(
        str name, dict cache, tuple ops, list in_arg_infos, dtype):
    """Find the best-matching operation from given dtype or input arguments.

    Args:
        name: ufunc name. Just used in error message.
        cache: Cache
        ops: List of candidate oprations.
        in_args: Input arguments.
        dtype: dtype

    Returns:
        One of the elements in op argument; a 3-element tuple where the first 2
        elements are input/output dtypes and the last element is the operation
        code.
    """
    cdef ArgInfo a

    if dtype is None:
        # dtype is not given. Guess operation from input arguments.
        use_raw_value = _check_should_use_min_scalar(in_arg_infos)
        if use_raw_value:
            in_types = tuple([
                a.dtype.type if a.is_ndarray else a.arg for a in in_arg_infos])
            op = None
        else:
            in_types = tuple([a.dtype.type for a in in_arg_infos])
            op = cache.get(in_types)

        if op is None:
            # Not found in cache
            op = _guess_routine_from_in_types(ops, in_types)
            if not use_raw_value:
                cache[in_types] = op
    else:
        # dtype is given. Guess operation from dtype.
        op = cache.get(dtype)
        if op is None:
            # Not found in cache
            op = _guess_routine_from_dtype(ops, dtype)
            cache[dtype] = op

    if op is not None:
        return op
    if dtype is None:
        dtype = tuple([a.dtype.type for a in in_arg_infos])
    raise TypeError('Wrong type (%s) of arguments for %s' %
                    (dtype, name))


class ufunc(object):

    """Universal function.

    Arguments:
        name: Kernel name
        nin:
        nout:
        ops: A tuple which specifies the possible dtype combinations. Each
             element is a 3-element tuple, where first 2 elements are a tuples
             of input/output dtypes and the last element is the operation code.
        preamble:
        doc:

    Attributes:
        ~ufunc.name (str): The name of the universal function.
        nin (int): Number of input arguments.
        nout (int): Number of output arguments.
        nargs (int): Number of all arguments.

    """
    def __init__(self, name, nin, nout, ops, preamble='', doc=''):
        self.name = name
        self.nin = nin
        self.nout = nout
        self.nargs = nin + nout
        self._ops = ops
        assert len(preamble) == 0, preamble
        self.__doc__ = doc
        self.k = _UfuncKernel(nin, nout, name, tuple(ops), preamble)

    def __repr__(self):
        return "<ufunc '%s'>" % self.name

    @property
    def types(self):
        """A list of type signatures.

        Each type signature is represented by type character codes of inputs
        and outputs separated by '->'.

        """
        types = []
        for in_types, out_types, _ in self._ops:
            in_str = ''.join([<str>numpy.dtype(t).char for t in in_types])
            out_str = ''.join([<str>numpy.dtype(t).char for t in out_types])
            types.append('%s->%s' % (in_str, out_str))
        return types

    def __call__(self, *args, **kwargs):
        """__call__(*args, **kwargs)

        Applies the universal function to arguments elementwise.

        Args:
            args: Input arguments. Each of them can be a :class:`cupy.ndarray`
                object or a scalar. The output arguments can be omitted or be
                specified by the ``out`` argument.
            out (cupy.ndarray): Output array. It outputs to new arrays
                default.
            dtype: Data type specifier.

        Returns:
            Output array or a tuple of output arrays.

        """

        return self.k(*args, **kwargs)


cpdef create_ufunc(name, ops, routine=None, preamble='', doc=''):
    """ Creates ufunc instance.

    Arguments:
        name: Kernel name
        ops: A tuple which specifies the possible dtype combinations. Each
             element can be either a string which represents input-output
             dtype correspondence (e.g. ''bb->bb'), or a 2-element tuple
             in which the first element is a string described above, and the
             second element is the operation code for that dtype combination,
             which overrides `routine` argument.
        routine: Default operation code.
        preamble:
        doc:
    """
    _ops = []
    for t in ops:
        if isinstance(t, str):
            typ = t
            rt = routine
        elif isinstance(t, tuple):
            typ, rt = t
            assert isinstance(typ, str)
            assert isinstance(rt, str)
        else:
            assert False

        types = typ.split('->')
        if len(types) == 1:
            in_types = out_types = tuple(types)
        else:
            in_types, out_types = map(tuple, types)
        in_types = tuple([numpy.dtype(t).type for t in in_types])
        out_types = tuple([numpy.dtype(t).type for t in out_types])
        _ops.append((in_types, out_types, rt))

    return ufunc(name, len(_ops[0][0]), len(_ops[0][1]), _ops, preamble, doc)
