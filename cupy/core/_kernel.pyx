import string
import threading

import numpy

from cupy.cuda import compiler
from cupy import util

cimport cpython  # NOQA
cimport cython  # NOQA

from libcpp cimport vector

from cupy.cuda cimport device
from cupy.cuda cimport function
from cupy.cuda.function cimport Arg
from cupy.cuda.function cimport NdarrayArg
from cupy.cuda.function cimport ScalarArg
from cupy.core cimport _carray
from cupy.core cimport _scalar
from cupy.core._dtype cimport get_dtype
from cupy.core._scalar import get_typename as _get_typename
from cupy.core.core cimport _convert_object_with_cuda_array_interface
from cupy.core.core cimport compile_with_cache
from cupy.core.core cimport ndarray
from cupy.core cimport internal
import cupy

_thread_local = threading.local()


cpdef inline bint _is_fusing() except? -1:
    try:
        return _thread_local.history is not None
    except AttributeError:
        _thread_local.history = None
    return False


cdef function.Function _create_elementwise_function(
        params, args, operation, name, _TypeMap type_map,
        preamble, loop_prep='', after_loop='', options=()):
    module_code = string.Template('''
    ${typedef_preamble}
    ${preamble}
    extern "C" __global__ void ${name}(${params}) {
      ${loop_prep};
      CUPY_FOR(i, _ind.size()) {
        _ind.set(i);
        ${operation};
      }
      ${after_loop};
    }
    ''').substitute(
        typedef_preamble=type_map.get_typedef_code(),
        params=_get_kernel_params(params, args),
        operation=operation,
        name=name,
        preamble=preamble,
        loop_prep=loop_prep,
        after_loop=after_loop)
    module = compile_with_cache(module_code, options)
    return module.get_function(name)


cdef inline int get_kind_score(int kind):
    if b'b' == kind:
        return 0
    if b'u' == kind or b'i' == kind:
        return 1
    if b'f' == kind or b'c' == kind:
        return 2
    return -1


@cython.profile(False)
cdef inline _check_array_device_id(ndarray arr, int device_id):
    if arr.data.device_id != device_id:
        raise ValueError(
            'Array device must be same as the current '
            'device: array device = %d while current = %d'
            % (arr.data.device_id, device_id))


# TODO(imanishi): Move to `_reduction.pyx`
cdef list _preprocess_args(int dev_id, args):
    """Preprocesses arguments for kernel invocation

    - Checks device compatibility for ndarrays
    - Converts Python/NumPy scalars into CScalars.
    """
    cdef list ret = []

    for arg in args:
        if type(arg) is not ndarray:
            if isinstance(arg, ndarray):
                _check_array_device_id(arg, dev_id)
            elif hasattr(arg, '__cuda_array_interface__'):
                arg = _convert_object_with_cuda_array_interface(arg)
        ret.append(arg)

    return ret


cdef str _get_c_type(Arg arg):
    # Returns the C type representation.
    if arg.is_ndarray():
        return 'CArray<%s, %d, %d>' % (
            _get_typename(arg.dtype), arg.ndim, arg.c_contiguous)
    if arg.is_scalar():
        return _get_typename(arg.dtype)
    else:  # indexer
        return 'CIndexer<%d>' % arg.ndim


cdef str _get_param_c_type(Arg arg, ParameterInfo p):
    # Returns the C type representation in the global function's
    # parameter list.
    cdef str ctyp = _get_c_type(arg)
    if p.is_const:
        return 'const ' + ctyp
    return ctyp


cdef str _get_c_var_name(Arg arg, ParameterInfo p):
    if arg.is_ndarray() and not p.raw:
        return '_raw_' + p.name
    return p.name


cpdef str _get_kernel_params(tuple params, tuple args):
    cdef ParameterInfo p
    cdef Arg arg
    assert len(params) == len(args), (len(params), len(args))
    lst = []
    for i in range(len(params)):
        p = params[i]
        arg = args[i]
        lst.append('{} {}'.format(
            _get_param_c_type(arg, p),
            _get_c_var_name(arg, p)))
    return ', '.join(lst)


cdef class _Args:

    def __init__(
            self, in_params, in_args, out_params, out_args, int device_id):
        cdef int nin = len(in_params)
        cdef int nout = len(out_params)
        assert nin == len(in_args)
        assert out_args is None or nout == len(out_args)
        in_args = _Args._preprocess(in_args, device_id)
        if out_args is not None:
            out_args = _Args._preprocess(out_args, device_id)
            # Avoid memory overlap between input and output arrays.
            _Args._copy_in_args_if_needed(in_args, out_args)

        self.in_params = in_params
        self.out_params = out_params
        self.in_args = in_args
        self.out_args = out_args

    cdef bint is_in_ndarray(self, int index):
        cdef Arg arg = self.in_args[index]
        return arg.is_ndarray()

    cdef list all_args(self):
        if self.out_args is None:
            return self.in_args
        return self.in_args + self.out_args

    cdef tuple all_params(self):
        if self.out_args is None:
            return self.in_params
        return self.in_params + self.out_params

    @staticmethod
    cdef list _preprocess(list objs, int device_id):
        cdef Arg arg
        args = []
        for obj in objs:
            arg = Arg.from_obj(obj)
            # Check device of an ndarray
            if arg.is_ndarray():
                _check_array_device_id(arg.obj, device_id)
            args.append(arg)
        return args

    @staticmethod
    cdef _copy_in_args_if_needed(list in_args, list out_args):
        cdef Arg arg
        cdef NdarrayArg arr_arg
        cdef int i
        for i in range(len(in_args)):
            arg = in_args[i]
            if arg.is_ndarray():
                arr_arg = in_args[i]
                arr_arg.copy_in_arg_if_needed(out_args)

    cdef broadcast(self):
        cdef int nin = len(self.in_params)
        cdef int nout
        cdef int i
        cdef ParameterInfo p
        cdef Arg arg
        cdef bint is_ndarray
        cdef ndarray arr
        cdef ndarray arr_orig
        cdef vector.vector[Py_ssize_t] shape

        if self.out_args is not None:
            nout = len(self.out_params)
            params = self.in_params + self.out_params
            args = self.in_args + self.out_args
        else:
            nout = 0
            params = self.in_params
            args = self.in_args
        assert len(params) == len(args) == nin + nout

        # Collect non-raw arrays (including out args)
        arrays = []
        for i, p in enumerate(params):
            is_ndarray = nin <= i or self.is_in_ndarray(i)
            if is_ndarray and not p.raw:
                arrays.append((<Arg>args[i]).obj)
            else:
                arrays.append(None)

        # Perform broadcast.
        # Note that arrays in `arrays` are replaced with broadcasted ones.
        internal._broadcast_core(arrays, shape)

        # Raise error if any of output arguments would be affected.
        for i in range(nout):
            arr = arrays[nin + i]
            arr_orig = (<Arg>args[nin + i]).obj
            if arr is not None and arr.shape != arr_orig.shape:
                raise ValueError(
                    'Out shape is mismatched: '
                    '{} while {} is expected.'.format(
                        arr.shape, arr_orig.shape))

        # Get broadcasted arrays back.
        for i in range(nin):
            if arrays[i] is not None:
                arg = self.in_args[i]
                arg.obj = arrays[i]

        return tuple(shape)

    cdef set_scalar_dtypes(self, in_types):
        cdef nin = len(self.in_params)
        cdef Arg arg
        cdef int i
        assert len(in_types) == nin
        for i in range(nin):
            arg = self.in_args[i]
            if arg.is_scalar():
                (<ScalarArg>arg).apply_dtype(in_types[i])

    cdef set_out_args(self, out_args):
        self.out_args = out_args

    cdef tuple get_out_arrays(self):
        cdef Arg arg
        return tuple([arg.obj for arg in self.out_args])

    cdef tuple reduce_dims(self, shape):
        assert self.out_args is not None
        cdef Arg arg
        cdef int nin = len(self.in_params)
        cdef int nout = len(self.out_params)
        cdef ndarray arr

        if len(shape) <= 1:
            return shape

        args = self.in_args + self.out_args
        params = self.in_params + self.out_params
        objs = [arg.obj for arg in args]
        new_shape = _reduced_view_core(objs, params, shape)

        # Update args
        for i in range(nin):
            arg = self.in_args[i]
            if arg.is_ndarray():
                self.in_args[i] = Arg.from_obj(objs[i])
        for i in range(nout):
            arg = self.out_args[i]
            if arg.is_ndarray():
                self.out_args[i] = Arg.from_obj(objs[nin + i])
        return new_shape

    cdef tuple get_kernel_args(self, shape):
        assert self.out_args is not None
        args = tuple(self.in_args + self.out_args)
        indexer_args = (Arg.from_indexer(shape),)
        return args, indexer_args


cdef tuple _reduce_dims(list args, tuple params, tuple shape):
    """ Remove contiguous stride to optimize CUDA kernel."""
    cdef ndarray arr

    if len(shape) <= 1 or len(args) == 0:
        return shape

    if len(args) == 1:  # fast path for reduction
        a = args[0]
        if (<ParameterInfo>params[0]).raw or not isinstance(a, ndarray):
            return shape
        arr = a
        arr = arr.reduced_view()
        if arr is a:
            return shape
        else:
            args[0] = arr
            return arr.shape
    return _reduced_view_core(args, params, shape)


cdef tuple _reduced_view_core(list args, tuple params, tuple shape):
    cdef int i, ax, last_ax, ndim
    cdef Py_ssize_t x, total_size
    cdef vector.vector[Py_ssize_t] vecshape, newshape, newstrides
    cdef vector.vector[int] array_indexes, axes
    cdef vector.vector[int] strides_indexes
    cdef ParameterInfo p
    cdef ndarray arr

    ndim = len(shape)
    array_indexes.reserve(len(args))
    strides_indexes.reserve(len(args))
    for i in range(len(args)):
        p = params[i]
        if not p.raw and isinstance(args[i], ndarray):
            array_indexes.push_back(i)
            arr = args[i]
            if not arr._c_contiguous:
                strides_indexes.push_back(i)

    if array_indexes.size() == 0:
        return shape

    if strides_indexes.size() == 0:
        # The input arrays are all c_contiguous
        i = array_indexes[0]
        arr = args[i]
        total_size = arr.size
        newshape.assign(<Py_ssize_t>1, total_size)
        newstrides.resize(1)
        for i in array_indexes:
            arr = args[i]
            newstrides[0] = arr.dtype.itemsize
            # TODO(niboshi): Confirm update_x_contiguity flags
            args[i] = arr._view(newshape, newstrides, False, True)
        return total_size,

    axes.reserve(ndim)
    vecshape.reserve(ndim)
    for x in shape:
        vecshape.push_back(x)
    last_ax = -1
    for ax in range(ndim):
        if vecshape[ax] == 1:
            continue
        if last_ax < 0:
            last_ax = ax
            continue
        for i in strides_indexes:
            arr = args[i]
            if arr._strides[ax] * vecshape[ax] != arr._strides[last_ax]:
                axes.push_back(last_ax)
                break
        else:
            vecshape[ax] *= vecshape[last_ax]
        last_ax = ax
    if last_ax >= 0:
        axes.push_back(last_ax)
    if <int>axes.size() == ndim:
        return shape

    newshape.reserve(axes.size())
    newstrides.reserve(axes.size())
    for ax in axes:
        newshape.push_back(vecshape[ax])
    for i in array_indexes:
        arr = args[i]
        newstrides.clear()
        for ax in axes:
            newstrides.push_back(arr._strides[ax])
        # TODO(niboshi): Confirm update_x_contiguity flags
        args[i] = arr._view(newshape, newstrides, False, True)
    return tuple(newshape)


cdef class ParameterInfo:

    def __init__(self, str param, bint is_const):
        self.name = None
        self.dtype = None
        self.ctype = None
        self.raw = False
        self.is_const = is_const
        s = tuple([i for i in param.split() if len(i) != 0])
        if len(s) < 2:
            raise Exception('Syntax error: %r' % param)

        t, self.name = s[-2:]
        if t == 'CIndexer':
            pass
        elif len(t) == 1:
            self.ctype = t
        else:
            dtype = get_dtype(t)
            self.dtype = dtype.type
            if dtype.name != t:
                raise ValueError('Wrong type %s' % t)
            self.ctype = _get_typename(self.dtype)

        for i in s[:-2]:
            if i == 'raw':
                self.raw = True
            elif i == '_non_const':
                self.is_const = False
            else:
                raise Exception('Unknown keyword "%s"' % i)

    def __hash__(self):
        return hash((
            self.name, self.dtype, self.ctype, self.raw, self.is_const))

    def __eq__(self, other):
        cdef ParameterInfo oth
        if not isinstance(other, ParameterInfo):
            return False
        oth = other
        return (
            self.name == oth.name
            and self.dtype == oth.dtype
            and self.ctype == oth.ctype
            and self.raw == oth.raw
            and self.is_const == oth.is_const)

    def __repr__(self):
        return '<ParameterInfo({})>'.format(
            ' '.join([
                'name={!r}'.format(self.name),
                'dtype={!r}'.format(self.dtype),
                'ctype={!r}'.format(self.ctype),
                'raw={!r}'.format(self.raw),
                'is_const={!r}'.format(self.is_const),
            ]))


@util.memoize()
def _get_param_info(str s, is_const):
    if len(s) == 0:
        return ()
    return tuple([ParameterInfo(i, is_const) for i in s.strip().split(',')])


@util.memoize()
def _decide_params_type(in_params, out_params, in_args_dtype, out_args_dtype):
    return _decide_params_type_core(in_params, out_params, in_args_dtype,
                                    out_args_dtype)


cdef class _TypeMap:

    def __init__(self, pairs):
        self._pairs = pairs

    def __hash__(self):
        return hash(self._pairs)

    def __eq__(self, other):
        if not isinstance(other, _TypeMap):
            return False
        return self._pairs == (<_TypeMap>other)._pairs

    def __repr__(self):
        return '<_TypeMap {}>'.format(
            ' '.join(
                '{}={!r}'.format(ctype1, type2)
                for ctype1, type2 in self._pairs))

    cdef str get_typedef_code(self):
        # Returns a code fragment of typedef statements used as preamble.
        return ''.join([
            'typedef %s %s;\n' % (_get_typename(type2), ctype1)
            for ctype1, type2 in self._pairs])


cdef tuple _decide_params_type_core(
        tuple in_params, tuple out_params, tuple in_args_dtype,
        tuple out_args_dtype):
    type_dict = {}
    if out_args_dtype:
        assert len(out_params) == len(out_args_dtype)
        for p, a in zip(out_params, out_args_dtype):
            if a is None:
                raise TypeError('Output arguments must be cupy.ndarray')
            if p.dtype is not None:
                if get_dtype(a) != get_dtype(p.dtype):
                    raise TypeError(
                        'Type is mismatched. %s %s %s' % (p.name, a, p.dtype))
            elif p.ctype in type_dict:
                t = type_dict[p.ctype]
                if get_dtype(t) != get_dtype(a):
                    raise TypeError(
                        'Type is mismatched. %s %s %s %s' % (
                            p.name, a, t, p.ctype))
            else:
                type_dict[p.ctype] = a

    assert len(in_params) == len(in_args_dtype)
    unknown_ctype = []
    for p, a in zip(in_params, in_args_dtype):
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
    type_map = _TypeMap(tuple(sorted(type_dict.items())))
    return in_types, out_types, type_map


cdef tuple _broadcast(list args, tuple params, bint use_size):
    cdef Py_ssize_t i
    cdef ParameterInfo p
    cdef bint any_nonraw_array = False
    cdef vector.vector[Py_ssize_t] shape

    # Collect non-raw arrays
    value = []
    for i in range(len(args)):
        p = params[i]
        a = args[i]
        if not p.raw and isinstance(a, ndarray):
            # Non-raw array
            any_nonraw_array = True
            value.append(a)
        else:
            value.append(None)

    if use_size:
        if any_nonraw_array:
            raise ValueError('Specified \'size\' can be used only '
                             'if all of the ndarray are \'raw\'.')
    else:
        if not any_nonraw_array:
            raise ValueError('Loop size is undecided.')

    # Perform broadcast.
    # Note that arrays in `value` are replaced with broadcasted ones.
    internal._broadcast_core(value, shape)

    # Restore raw arrays and scalars from the original list.
    for i, a in enumerate(value):
        if a is None:
            value[i] = args[i]

    return value, tuple(shape)


cdef class _AbstractElementwiseKernel:

    cdef:
        readonly str name
        readonly Py_ssize_t nin
        readonly Py_ssize_t nout
        readonly Py_ssize_t nargs
        readonly tuple _in_params
        readonly tuple _out_params
        readonly tuple _params
        readonly object _preamble
        readonly dict _kernel_cache
        readonly bint _reduce_dims

        # Arguments to _create_elementwise_funtion
        readonly str _loop_prep
        readonly str _after_loop
        readonly tuple _options

    def __init__(
            self, str name, tuple in_params, tuple out_params, str preamble,
            str loop_prep, str after_loop, tuple options,
            bint reduce_dims,
    ):
        assert name is not None
        assert all(isinstance(p, str) for p in in_params)
        assert all(isinstance(p, str) for p in out_params)
        assert preamble is not None
        assert loop_prep is not None
        assert after_loop is not None
        assert options is not None
        if not compiler.is_valid_kernel_name(name):
            raise ValueError(
                'Invalid kernel name: "%s"' % name)

        in_params_ = tuple([ParameterInfo(p, True) for p in in_params])
        out_params_ = tuple([ParameterInfo(p, False) for p in out_params])
        params = (
            in_params_
            + out_params_
            + (ParameterInfo('CIndexer _ind', False),))

        self.name = name
        self.nin = len(in_params)
        self.nout = len(out_params)
        self.nargs = self.nin + self.nout
        self._in_params = in_params_
        self._out_params = out_params_
        self._params = params
        self._preamble = preamble
        self._loop_prep = loop_prep
        self._after_loop = after_loop
        self._options = options
        self._reduce_dims = reduce_dims

    cdef _Args make_args(self, tuple args, int device_id):
        cdef int nin = self.nin

        # Check the number of arguments
        cdef int nargs = len(args)
        if nargs != nin and nargs != self.nargs:
            raise TypeError('Wrong number of arguments for %s' % self.name)

        in_args = list(args[:nin])
        if len(args) == nin:
            out_args = None
        else:
            out_args = list(args[nin:])
        return _Args(
            self._in_params, in_args, self._out_params, out_args,
            device_id)

    cdef check_args(self, launch_ctx, _Args args):
        pass

    cdef tuple launch(self, launch_ctx, tuple args, stream):
        cdef tuple in_types, out_types, shape
        cdef _Args args_
        cdef Arg arg

        dev_id = device.get_device_id()

        args_ = self.make_args(args, dev_id)
        self.check_args(launch_ctx, args_)

        # Decide types.
        in_types, out_types = self.decide_types(launch_ctx, args_)

        # Broadcast.
        broadcasted_shape = args_.broadcast()

        # Make output arrays.
        out_args, shape = self.get_out_args(
            launch_ctx, args_.out_args, out_types, broadcasted_shape)
        args_.set_out_args(out_args)
        ret_arrs = tuple([(<Arg>arg).obj for arg in out_args])

        # Return without launching a kernel if no computation is necessary.
        if 0 in shape:
            return ret_arrs

        # Set scalar dtypes.
        args_.set_scalar_dtypes(in_types)

        # Squash dimensions.
        if self._reduce_dims:
            shape = args_.reduce_dims(shape)

        # Launch the kernel.
        args, indexer_args = args_.get_kernel_args(shape)
        type_map = self.get_type_map(launch_ctx, args)
        all_args = args + indexer_args
        func = self.get_function(launch_ctx, all_args, type_map, dev_id)
        func.linear_launch(
            internal.prod_sequence(shape),
            all_args,
            shared_mem=0, block_max_size=128,
            stream=stream)
        return ret_arrs

    cdef function.Function create_function(
            self, launch_ctx, tuple args, _TypeMap type_map):
        return _create_elementwise_function(
            self._params,
            args,
            self.get_operation_code(launch_ctx, args),
            self.get_kernel_name(launch_ctx, args),
            type_map,
            self._preamble,
            self._loop_prep,
            self._after_loop,
            self._options)

    cdef tuple decide_types(self, launch_ctx, _Args args):
        # Decides the types of inputs and outputs given actual arguments.
        # Returns a tuple (in_types, out_types).
        raise NotImplementedError()

    cdef _TypeMap get_type_map(self, launch_ctx, args):
        # Returns the type map.
        raise NotImplementedError()

    cdef get_out_args(
            self, launch_ctx, out_args, out_types, broadcasted_shape):
        # Makes output arrays.
        raise NotImplementedError()

    cdef str get_kernel_name(self, launch_ctx, args):
        # TODO: comment
        raise NotImplementedError()

    cdef str get_operation_code(self, launch_ctx, tuple args):
        # TODO: comment
        raise NotImplementedError()

    cdef function.Function get_function(
            self, launch_ctx, args, _TypeMap type_map, int device_id):
        # Returns a kernel function.
        raise NotImplementedError()


# -----------------------------------------------------------------------------
# ElementwiseKernel
# -----------------------------------------------------------------------------

cdef dict _elementwise_kernel_memo = {}


cdef list _get_out_args_with_params(
        list out_args, tuple out_types, tuple out_shape, tuple out_params,
        bint is_size_specified):
    cdef ParameterInfo p
    if out_args is None or len(out_args) == 0:
        for p in out_params:
            if p.raw and not is_size_specified:
                raise ValueError('Output array size is Undecided')
        return [Arg.from_ndarray(ndarray(out_shape, t)) for t in out_types]

    # Check for errors
    cdef ndarray arr
    cdef vector.vector[Py_ssize_t] shape
    cdef Py_ssize_t x
    cdef Arg arg
    shape.reserve(len(out_shape))
    for x in out_shape:
        shape.push_back(x)
    for i, p in enumerate(out_params):
        arg = out_args[i]
        if not arg.is_ndarray():
            raise TypeError(
                'Output arguments type must be cupy.ndarray, '
                'not {}'.format(type(arg.obj)))
        arr = arg.obj
        if not p.raw and not internal.vector_equal(arr._shape, shape):
            raise ValueError(
                'Out shape is mismatched: '
                '{} while {} is expected.'.format(tuple(arr.shape), out_shape))

    return out_args


cdef class _ElementwiseKernelLaunchContext:

    cdef:
        readonly int _size
        _TypeMap _type_map  # Assigned only after decide_types() is called.

    def __init__(self, int size):
        self._size = size


cdef class ElementwiseKernel(_AbstractElementwiseKernel):

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
        options (tuple): Compile options passed to NVRTC. For details, see
            https://docs.nvidia.com/cuda/nvrtc/index.html#group__options.
        preamble (str): Fragment of the CUDA-C/C++ code that is inserted at the
            top of the cu file.
        no_return (bool): If ``True``, __call__ returns ``None``.
        return_tuple (bool): If ``True``, __call__ always returns tuple of
            array even if single value is returned.
        loop_prep (str): Fragment of the CUDA-C/C++ code that is inserted at
            the top of the kernel function definition and above the ``for``
            loop.
        after_loop (str): Fragment of the CUDA-C/C++ code that is inserted at
            the bottom of the kernel function definition.

    """

    cdef:
        readonly object operation
        readonly bint no_return
        readonly bint return_tuple
        readonly dict kwargs
        readonly dict _params_type_memo
        readonly dict _elementwise_kernel_memo

    def __init__(self, str in_params, str out_params, operation,
                 name='kernel', reduce_dims=True, preamble='',
                 no_return=False, return_tuple=False, *,
                 str loop_prep='', str after_loop='', tuple options=()):
        super().__init__(
            name,
            tuple([p.strip() for p in in_params.split(',') if len(p) > 0]),
            tuple([p.strip() for p in out_params.split(',') if len(p) > 0]),
            preamble,
            loop_prep,
            after_loop,
            options,
            reduce_dims,
        )

        self.operation = operation
        self.no_return = no_return
        self.return_tuple = return_tuple
        self._params_type_memo = {}
        names = [p.name for p in self._in_params + self._out_params]
        if 'i' in names:
            raise ValueError('Can not use \'i\' as a parameter name')
        self._elementwise_kernel_memo = {}

    @property
    def preamble(self):
        return self._preamble

    @property
    def params(self):
        return self._params

    @property
    def reduce_dims(self):
        return self._reduce_dims

    def __call__(self, *args, **kwargs):
        """Compiles and invokes the elementwise kernel.

        The compilation runs only if the kernel is not cached. Note that the
        kernels with different argument dtypes or dimensions are not
        compatible. It means that single ElementwiseKernel object may be
        compiled into multiple kernel binaries.

        Args:
            args: Arguments of the kernel.
            size (int): Range size of the indices.  By default, the range size
                is automatically determined from the result of broadcasting.
                This parameter must be specified if and only if all ndarrays
                are `raw` and the range size cannot be determined
                automatically.
            block_size (int): Number of threads per block. By default, the
                value is set to 128.

        Returns:
            If ``no_return`` has not set, arrays are returned according to the
            ``out_params`` argument of the ``__init__`` method.
            If ``no_return`` has set, ``None`` is returned.

        """
        cdef Py_ssize_t size = kwargs.pop('size', -1)
        stream = kwargs.pop('stream', None)
        block_size = kwargs.pop('block_size', 128)
        if len(kwargs):
            raise TypeError('Wrong arguments %s' % kwargs)
        if block_size <= 0:
            raise ValueError('block_size must be greater than zero')

        out_args = self.launch(
            _ElementwiseKernelLaunchContext(size),
            args,
            stream)
        assert len(out_args) == self.nout
        if self.no_return:
            return None
        if not self.return_tuple and self.nout == 1:
            return out_args[0]
        return out_args

    cpdef tuple _decide_params_type(
            self, tuple in_args_dtype, tuple out_args_dtype):
        key = (in_args_dtype, out_args_dtype)
        ret = self._params_type_memo.get(key, None)
        if ret is not None:
            return ret
        ret = _decide_params_type_core(
            self._in_params, self._out_params, in_args_dtype, out_args_dtype)
        self._params_type_memo[key] = ret
        return ret

    # override
    cdef check_args(self, launch_ctx, _Args args):
        cdef _ElementwiseKernelLaunchContext ctx = launch_ctx
        cdef int nin = len(args.in_params)
        cdef ParameterInfo p
        cdef bint any_nonraw_array = False
        cdef bint is_ndarray

        # Check size arguments
        params = args.all_params()
        for i, p in enumerate(params):
            is_ndarray = nin <= i or args.is_in_ndarray(i)
            if is_ndarray and not p.raw:
                any_nonraw_array = True
                break
        if ctx._size != -1:  # size is given
            if any_nonraw_array:
                raise ValueError('Specified \'size\' can be used only '
                                 'if all of the ndarray are \'raw\'.')
        else:
            if not any_nonraw_array:
                raise ValueError('Loop size is undecided.')

    # override
    cdef tuple decide_types(self, launch_ctx, _Args args):
        cdef _ElementwiseKernelLaunchContext ctx = launch_ctx
        cdef Arg arg
        cdef int i
        in_ndarray_types = []
        out_ndarray_types = []
        for i in range(len(args.in_args)):
            if args.is_in_ndarray(i):
                arg = args.in_args[i]
                in_ndarray_types.append(arg.dtype.type)
            else:
                in_ndarray_types.append(None)
        if args.out_args is not None:
            for i in range(len(args.out_args)):
                arg = args.out_args[i]
                out_ndarray_types.append(arg.dtype.type)

        in_types, out_types, type_map = self._decide_params_type(
            tuple(in_ndarray_types), tuple(out_ndarray_types))
        ctx._type_map = type_map  # Keep for later use
        return in_types, out_types

    # override
    cdef _TypeMap get_type_map(self, launch_ctx, args):
        cdef _ElementwiseKernelLaunchContext ctx = launch_ctx
        assert ctx._type_map is not None
        return ctx._type_map

    # override
    cdef get_out_args(
            self, launch_ctx, out_args, out_types, broadcasted_shape):
        cdef _ElementwiseKernelLaunchContext ctx = launch_ctx
        cdef int size = ctx._size
        cdef Arg arg
        if size != -1:
            shape = (size,)
        else:
            shape = broadcasted_shape
        out_args = _get_out_args_with_params(
            out_args, out_types, shape, self._out_params, size != -1)
        return out_args, shape

    # override
    cdef str get_kernel_name(self, launch_ctx, args):
        return self.name

    # override
    cdef str get_operation_code(self, launch_ctx, tuple args):
        cdef int i
        cdef Arg arg
        cdef tuple params = self._in_params + self._out_params
        op = []
        for i in range(len(params)):
            p = params[i]
            arg = args[i]
            if not p.raw and arg.is_ndarray():
                if p.is_const:
                    fmt = 'const {t} &{n} = _raw_{n}[_ind.get()];'
                else:
                    fmt = '{t} &{n} = _raw_{n}[_ind.get()];'
                op.append(fmt.format(t=p.ctype, n=p.name))
        op.append(self.operation)
        operation = '\n'.join(op)
        return operation

    # override
    cdef function.Function get_function(
            self, launch_ctx, args, _TypeMap type_map, int device_id):
        cdef Arg arg
        memo = _elementwise_kernel_memo
        key = (
            self._params,
            self.operation,
            self.name,
            self.preamble,
            self._loop_prep,
            self._after_loop,
            self._options,
            device_id,
            tuple([arg.get_immutable_key() for arg in args]),
            type_map)
        func = memo.get(key, None)
        if func is not None:
            return func
        func = self.create_function(launch_ctx, args, type_map)

        # Store the compiled kernel in the cache.
        # Potentially overwrite a duplicate cache entry because
        # _get_elementwise_kernel() may include IO wait.
        memo[key] = func
        return func


# -----------------------------------------------------------------------------
# ufunc
# -----------------------------------------------------------------------------


cdef inline bint _check_should_use_min_scalar(list in_args) except? -1:
    cdef int kind, max_array_kind, max_scalar_kind
    cdef bint all_scalars
    cdef Arg arg
    all_scalars = True
    max_array_kind = -1
    max_scalar_kind = -1
    for arg in in_args:
        kind = get_kind_score(ord(arg.dtype.kind))
        if arg.is_ndarray():
            all_scalars = False
            max_array_kind = max(max_array_kind, kind)
        else:
            max_scalar_kind = max(max_scalar_kind, kind)
    return (max_scalar_kind != -1 and
            not all_scalars and
            max_array_kind >= max_scalar_kind)


cdef _numpy_can_cast = numpy.can_cast


cdef bint _can_cast(d1, d2, casting):
    # most ufunc passes `same_kind`
    if casting == 'same_kind' and get_dtype(d1).kind == d2.kind:
        return True
    return _numpy_can_cast(d1, d2, casting=casting)


cdef list _get_out_args(
        list out_args, tuple out_types, tuple out_shape, casting):
    if out_args is None or len(out_args) == 0:
        return [Arg.from_ndarray(ndarray(out_shape, t)) for t in out_types]

    # Check for errors
    cdef Arg arg
    cdef ndarray a
    for i, arg in enumerate(out_args):
        if not arg.is_ndarray():
            raise TypeError(
                'Output arguments type must be cupy.ndarray')
        a = arg.obj
        if a.shape != out_shape:
            raise ValueError(
                'Out shape is mismatched: '
                '{} while {} is expected.'.format(a.shape, out_shape))
        out_type = out_types[i]
        if not _can_cast(out_type, a.dtype, casting):
            msg = (
                'output (typecode \'{}\') could not be coerced to '
                'provided output parameter (typecode \'{}\') according to '
                'the casting rule "{}"'.format(
                    get_dtype(out_type).char,
                    a.dtype.char,
                    casting))
            raise TypeError(msg)
    return out_args


cdef class _UfuncLaunchContext:

    cdef:
        readonly object _dtype
        readonly str _casting
        _Op _op  # Assigned only after decide_types() is called.

    def __init__(self, dtype, casting):
        self._dtype = dtype
        self._casting = casting


cdef class ufunc(_AbstractElementwiseKernel):

    """Universal function.

    Attributes:
        ~ufunc.name (str): The name of the universal function.
        ~ufunc.nin (int): Number of input arguments.
        ~ufunc.nout (int): Number of output arguments.
        ~ufunc.nargs (int): Number of all arguments.

    """

    cdef:
        readonly _Ops _ops  # normal routines
        # routines based on explicitly given output dtype
        readonly _Ops _out_ops
        readonly object _default_casting
        readonly dict _routine_cache
        readonly dict _func_memo
        readonly object __doc__
        readonly object __name__
        readonly object __module__

        # Cached map from (_Op, immutable key of Arg) => _TypeMap
        readonly dict _type_map_cache

    def __init__(
            self, str name, Py_ssize_t nin, Py_ssize_t nout, _Ops ops,
            preamble='', loop_prep='', doc='',
            default_casting=None, *, _Ops out_ops=None):
        super().__init__(
            name,
            tuple(['T in%d' % i for i in range(nin)]),
            tuple(['T out%d' % i for i in range(nout)]),
            preamble,
            loop_prep,
            '',  # after_loop
            (),  # options
            True,  # reduce_dims
        )

        self.__name__ = name
        self._ops = ops
        self._out_ops = out_ops
        self.__doc__ = doc
        if default_casting is None:
            self._default_casting = 'same_kind'
        else:
            self._default_casting = default_casting
        self._routine_cache = {}
        self._func_memo = {}
        self._type_map_cache = {}

    def __repr__(self):
        return '<ufunc \'%s\'>' % self.name

    @property
    def types(self):
        """A list of type signatures.

        Each type signature is represented by type character codes of inputs
        and outputs separated by '->'.

        """
        types = []
        for op in self._ops:
            in_str = ''.join([<str>get_dtype(t).char for t in op.in_types])
            out_str = ''.join([<str>get_dtype(t).char for t in op.out_types])
            types.append('%s->%s' % (in_str, out_str))
        return types

    def __call__(self, *args, **kwargs):
        """Applies the universal function to arguments elementwise.

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
        if _is_fusing():
            return _thread_local.history.call_ufunc(self, args, kwargs)

        cdef function.Function kern
        cdef list broad_values
        cdef vector.vector[Py_ssize_t] vec_shape
        cdef tuple shape
        cdef Py_ssize_t s

        out = kwargs.pop('out', None)
        dtype = kwargs.pop('dtype', None)
        # Note default behavior of casting is 'same_kind' on numpy>=1.10
        casting = kwargs.pop('casting', self._default_casting)
        if dtype is not None:
            dtype = get_dtype(dtype).type
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)

        if out is not None:
            if self.nout != 1:
                raise ValueError('Cannot use \'out\' in %s' % self.name)
            if len(args) != self.nin:
                raise ValueError('Cannot specify \'out\' as both '
                                 'a positional and keyword argument')
            args = args + (out,)

        stream = None
        out_args = self.launch(
            _UfuncLaunchContext(dtype, casting),
            args,
            stream)
        if len(out_args) == 1:
            return out_args[0]
        return out_args

    cdef _Op _get_op(self, in_args, dtype):
        return self._ops.guess_routine(
            self.name, self._routine_cache, in_args, dtype, self._out_ops)

    cdef _TypeMap _get_type_map(self, _Op op, args):
        cdef _TypeMap type_map
        cdef Arg arg
        memo = self._type_map_cache
        key = (op, tuple([arg.get_immutable_key() for arg in args]))
        type_map = memo.get(key)
        if type_map is not None:
            return type_map

        cdef list lst = []
        for i, x in enumerate(op.in_types):
            lst.append(('in%d_type' % i, x))
        for i in range(self.nout):
            arg = args[self.nin + i]
            lst.append(('out%d_type' % i, arg.dtype))
        type_map = _TypeMap(tuple(lst))

        memo[key] = type_map
        return type_map

    # override
    cdef tuple decide_types(self, launch_ctx, _Args args):
        cdef _UfuncLaunchContext ctx = launch_ctx
        cdef Arg arg
        cdef int i
        op = self._get_op(args.in_args, launch_ctx._dtype)
        ctx._op = op  # Keep op for use in function creation.
        return op.in_types, op.out_types

    # override
    cdef _TypeMap get_type_map(self, launch_ctx, args):
        cdef _UfuncLaunchContext ctx = launch_ctx
        assert ctx._op is not None
        return self._get_type_map(ctx._op, args)

    # override
    cdef get_out_args(
            self, launch_ctx, out_args, out_types, broadcasted_shape):
        cdef _UfuncLaunchContext ctx = launch_ctx
        cdef Arg arg
        shape = broadcasted_shape
        out_args = _get_out_args(out_args, out_types, shape, ctx._casting)
        return out_args, shape

    # override
    cdef str get_kernel_name(self, launch_ctx, args):
        cdef Arg arg
        inout_type_words = []
        for arg in args:
            dtype = str(numpy.dtype(arg.dtype))
            if arg.is_scalar():
                inout_type_words.append(dtype.rstrip('0123456789'))
            elif arg.is_ndarray():
                inout_type_words.append(dtype)
            elif arg.is_scalar():
                inout_type_words.append(dtype.rstrip('0123456789'))
        return '{}__{}'.format(self.name, '_'.join(inout_type_words))

    # override
    cdef str get_operation_code(self, launch_ctx, tuple args):
        cdef _UfuncLaunchContext ctx = launch_ctx
        cdef Arg arg
        cdef int i
        assert ctx._op is not None
        op = []
        for i in range(self.nin):
            arg = args[i]
            if arg.is_ndarray():
                op.append(
                    'const in{0}_type in{0}(_raw_in{0}[_ind.get()]);'
                    .format(i))
        for i in range(self.nout):
            op.append(
                'out{0}_type &out{0} = _raw_out{0}[_ind.get()];'.format(i))

        op.append(ctx._op.routine)
        operation_code = '\n'.join(op)
        return operation_code

    # override
    cdef function.Function get_function(
            self, launch_ctx, args, _TypeMap type_map, int device_id):
        cdef _UfuncLaunchContext ctx = launch_ctx
        cdef function.Function func
        cdef Arg arg
        assert ctx._op is not None
        memo = self._func_memo
        key = (
            device_id,
            ctx._op,
            tuple([arg.get_immutable_key() for arg in args]),
            type_map)
        func = memo.get(key, None)
        if func is None:
            func = self.create_function(ctx, args, type_map)
            memo[key] = func
        return func


cdef class _Op:

    def __init__(
            self, tuple in_types, tuple out_types, object routine,
            object error_func):
        if error_func is None:
            assert routine is not None
        else:
            assert callable(error_func)
        self.in_types = in_types
        self.out_types = out_types
        self.nin = len(in_types)
        self.nout = len(out_types)
        self.routine = routine
        self.error_func = error_func

    @staticmethod
    cdef _Op _from_type_and_routine_or_error_func(
            str typ, object routine, object error_func):
        # TODO(niboshi): Write type mapping specification.
        types = typ.split('->')
        if len(types) == 1:
            in_types = out_types = tuple(types)
        else:
            in_types, out_types = map(tuple, types)
        in_types = tuple([get_dtype(t).type for t in in_types])
        out_types = tuple([get_dtype(t).type for t in out_types])
        return _Op(in_types, out_types, routine, error_func)

    @staticmethod
    cdef _Op from_type_and_routine(str typ, routine):
        return _Op._from_type_and_routine_or_error_func(typ, routine, None)

    @staticmethod
    cdef _Op from_type_and_error_func(str typ, error_func):
        return _Op._from_type_and_routine_or_error_func(typ, None, error_func)

    def __repr__(self):
        data = [
            'in_types={!r}'.format(self.in_types),
            'out_types={!r}'.format(self.out_types),
        ]
        if self.routine is not None:
            data.append('routine={!r}'.format(self.routine))
        if self.error_func is not None:
            data.append('error_func={!r}'.format(self.error_func))
        return '<_Op {}>'.format(' '.join(data))

    cdef check_valid(self):
        if self.error_func is not None:
            self.error_func()


cdef class _Ops:

    def __init__(self, tuple ops):
        assert len(ops) > 0
        nin = ops[0].nin
        nout = ops[0].nout
        assert all(op.nin == nin for op in ops)
        assert all(op.nout == nout for op in ops)
        self.ops = ops
        self.nin = nin
        self.nout = nout

    @staticmethod
    cdef _Ops from_tuples(object ops, routine):
        ops_ = []
        for t in ops:
            if isinstance(t, tuple):
                typ, rt = t
                if isinstance(rt, tuple):
                    rt = tuple([r1 or r2 for r1, r2 in zip(rt, routine)])
                elif not isinstance(rt, str):
                    assert callable(rt)
                    ops_.append(_Op.from_type_and_error_func(typ, rt))
                    continue
            else:
                assert isinstance(t, str)
                typ, rt = t, routine
            ops_.append(_Op.from_type_and_routine(typ, rt))
        return _Ops(tuple(ops_))

    def __repr__(self):
        return '<_Ops nin={} nout={} ops={!r}>'.format(
            self.nin, self.nout, self.ops)

    cdef _Op guess_routine(
            self, str name, dict cache, list in_args, dtype, _Ops out_ops):
        cdef _Ops ops_
        cdef Arg a
        if dtype is None:
            use_raw_value = _check_should_use_min_scalar(in_args)
            if use_raw_value:
                in_types = tuple([
                    a.dtype.type if a.is_ndarray()
                    else (<ScalarArg>a).get_min_scalar_type()
                    for a in in_args])
            else:
                in_types = tuple([a.dtype.type for a in in_args])
            op = cache.get(in_types, ())
            if op is ():
                op = self._guess_routine_from_in_types(in_types)
                cache[in_types] = op
        else:
            op = cache.get(dtype, ())
            if op is ():
                ops_ = out_ops or self
                op = ops_._guess_routine_from_dtype(dtype)
                cache[dtype] = op

        if op is not None:
            # raise TypeError if the type combination is disallowed
            (<_Op>op).check_valid()
            return op

        if dtype is None:
            dtype = tuple([a.dtype.type for a in in_args])
        raise TypeError('Wrong type (%s) of arguments for %s' %
                        (dtype, name))

    cdef _Op _guess_routine_from_in_types(self, tuple in_types):
        cdef _Op op
        cdef tuple op_types
        cdef Py_ssize_t n = len(in_types)
        cdef Py_ssize_t i
        can_cast = numpy.can_cast
        for op in self.ops:
            op_types = op.in_types
            for i in range(n):
                it = in_types[i]
                ot = op_types[i]
                if isinstance(it, tuple):
                    if not can_cast(it[0], ot) and not can_cast(it[1], ot):
                        break
                elif not can_cast(it, ot):
                    break
            else:
                return op
        return None

    cdef _Op _guess_routine_from_dtype(self, object dtype):
        cdef _Op op
        cdef tuple op_types
        for op in self.ops:
            op_types = op.out_types
            for t in op_types:
                if t != dtype:
                    break
            else:
                return op
        return None


cpdef create_ufunc(name, ops, routine=None, preamble='', doc='',
                   default_casting=None, loop_prep='', out_ops=None):
    ops_ = _Ops.from_tuples(ops, routine)
    _out_ops = None if out_ops is None else _Ops.from_tuples(out_ops, routine)
    return ufunc(
        name, ops_.nin, ops_.nout, ops_, preamble,
        loop_prep, doc, default_casting=default_casting, out_ops=_out_ops)
