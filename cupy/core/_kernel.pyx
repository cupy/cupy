from __future__ import division
import string
import threading

import numpy
import six

from cupy.cuda import compiler
from cupy import util

cimport cpython  # NOQA
cimport cython  # NOQA

from libcpp cimport vector

from cupy.cuda cimport device
from cupy.cuda cimport function
from cupy.core cimport _scalar
from cupy.core._dtype cimport get_dtype
from cupy.core._routines_manipulation cimport _broadcast_core
from cupy.core._scalar import get_typename as _get_typename
from cupy.core.core cimport _convert_object_with_cuda_array_interface
from cupy.core.core cimport compile_with_cache
from cupy.core.core cimport Indexer
from cupy.core.core cimport ndarray
from cupy.core cimport internal


_thread_local = threading.local()


cpdef inline bint _is_fusing() except? -1:
    try:
        return _thread_local.history is not None
    except AttributeError:
        _thread_local.history = None
    return False


cpdef function.Function _get_simple_elementwise_kernel(
        params, operation, name, preamble,
        loop_prep='', after_loop='', options=()):
    module_code = string.Template('''
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
        params=params,
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


cpdef list _preprocess_args(int dev_id, args, bint use_c_scalar):
    """Preprocesses arguments for kernel invocation

    - Checks device compatibility for ndarrays
    - Converts Python scalars into NumPy scalars
    """
    cdef list ret = []

    for arg in args:
        if type(arg) is not ndarray:
            s = _scalar.convert_scalar(arg, use_c_scalar)
            if s is not None:
                ret.append(s)
                continue
            if not hasattr(arg, '__cuda_array_interface__'):
                raise TypeError('Unsupported type %s' % type(arg))
            arg = _convert_object_with_cuda_array_interface(arg)

        arr_dev_id = (<ndarray>arg).data.device_id
        if arr_dev_id != dev_id:
            raise ValueError(
                'Array device must be same as the current '
                'device: array device = %d while current = %d'
                % (arr_dev_id, dev_id))
        ret.append(arg)

    return ret


cpdef tuple _get_args_info(list args):
    ret = []
    for a in args:
        t = type(a)
        if t is Indexer:
            dtype = None
        elif t is _scalar.CScalar:
            dtype = (<_scalar.CScalar>a).get_numpy_type()
        else:
            dtype = a.dtype.type
        ret.append((t, dtype, a.ndim))
    return tuple(ret)


cpdef str _get_kernel_params(tuple params, tuple args_info):
    cdef ParameterInfo p
    ret = []
    for i in range(len(params)):
        p = params[i]
        type, dtype, ndim = <tuple>(args_info[i])
        is_array = type is ndarray
        if type is Indexer:
            t = 'CIndexer<%d>' % ndim
        else:
            t = _get_typename(dtype)
            if is_array:
                t = 'CArray<%s, %d>' % (t, ndim)
        ret.append('{}{} {}{}'.format(
            'const ' if p.is_const else '',
            t,
            '_raw_' if is_array and not p.raw else '',
            p.name))
    return ', '.join(ret)


cpdef tuple _reduce_dims(list args, tuple params, tuple shape):
    """ Remove contiguous stride to optimize CUDA kernel."""
    cdef int i, ax, last_ax, ndim
    cdef Py_ssize_t total_size
    cdef vector.vector[Py_ssize_t] vecshape, newshape, newstrides
    cdef vector.vector[int] array_indexes, axes
    cdef vector.vector[vector.vector[Py_ssize_t]] args_strides
    cdef ParameterInfo p
    cdef ndarray arr
    cdef bint flag

    ndim = len(shape)
    if ndim <= 1:
        return shape

    for i in range(len(args)):
        p = params[i]
        if not p.raw and isinstance(args[i], ndarray):
            array_indexes.push_back(i)
            arr = args[i]
            if not arr._c_contiguous:
                args_strides.push_back(arr._strides)

    if args_strides.size() == 0:
        # The input arrays are all c_contiguous
        total_size = internal.prod(shape)
        newshape.assign(<Py_ssize_t>1, total_size)
        for i in array_indexes:
            arr = args[i]
            newstrides.assign(<Py_ssize_t>1, arr.dtype.itemsize)
            # TODO(niboshi): Confirm update_x_contiguity flags
            args[i] = arr._view(newshape, newstrides, False, True)
        return total_size,

    vecshape = shape
    last_ax = -1
    for ax in range(ndim):
        if vecshape[ax] == 1:
            continue
        if last_ax < 0:
            last_ax = ax
            continue
        for st in args_strides:
            if st[ax] * vecshape[ax] != st[last_ax]:
                axes.push_back(last_ax)
                break
        else:
            vecshape[ax] *= vecshape[last_ax]
        last_ax = ax
    if last_ax >= 0:
        axes.push_back(last_ax)
    if axes.size() == ndim:
        return shape

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
    cdef:
        readonly str name
        readonly object dtype
        readonly str ctype
        readonly bint raw
        readonly bint is_const

    def __init__(self, str param, bint is_const):
        self.name = None
        self.dtype = None
        self.ctype = None
        self.raw = False
        self.is_const = is_const
        s = tuple([i for i in param.split() if len(i) != 0])
        if len(s) < 2:
            raise Exception('Syntax error: %s' % param)

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


@util.memoize()
def _get_param_info(s, is_const):
    if len(s) == 0:
        return ()
    return tuple([ParameterInfo(i, is_const) for i in s.strip().split(',')])


@util.memoize()
def _decide_params_type(in_params, out_params, in_args_dtype, out_args_dtype):
    return _decide_params_type_core(in_params, out_params, in_args_dtype,
                                    out_args_dtype)


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
    return in_types, out_types, tuple(type_dict.items())


cdef tuple _broadcast(list args, tuple params, bint use_size):
    cpdef Py_ssize_t i
    cpdef ParameterInfo p
    cpdef bint is_none, is_not_none
    value = []
    is_none = False
    is_not_none = False
    for i in range(len(args)):
        p = params[i]
        a = args[i]
        if not p.raw and isinstance(a, ndarray):
            is_not_none = True
            value.append(a)
        else:
            is_none = True
            value.append(None)

    if use_size:
        if not is_none:
            raise ValueError('Specified \'size\' can be used only '
                             'if all of the ndarray are \'raw\'.')
    else:
        if not is_not_none:
            raise ValueError('Loop size is Undecided')
    value, shape = _broadcast_core(value)
    for i, a in enumerate(value):
        if a is None:
            value[i] = args[i]
    return value, shape


cdef list _get_out_args(list out_args, tuple out_types, tuple out_shape,
                        casting):
    if not out_args:
        return [ndarray(out_shape, t) for t in out_types]

    for i, a in enumerate(out_args):
        if not isinstance(a, ndarray):
            raise TypeError(
                'Output arguments type must be cupy.ndarray')
        if a.shape != out_shape:
            raise ValueError('Out shape is mismatched')
        out_type = out_types[i]
        if not numpy.can_cast(out_type, a.dtype, casting=casting):
            msg = 'output (typecode \'{}\') could not be coerced to ' \
                  'provided output parameter (typecode \'{}\') according to ' \
                  'the casting rule "{}"'.format(
                      get_dtype(out_type).char,
                      a.dtype.char,
                      casting)
            raise TypeError(msg)
    return out_args


cdef list _get_out_args_with_params(
        list out_args, tuple out_types, tuple out_shape, tuple out_params,
        bint is_size_specified):
    cdef ParameterInfo p
    cdef ndarray arr
    cdef vector.vector[Py_ssize_t] shape
    if not out_args:
        for p in out_params:
            if p.raw and not is_size_specified:
                raise ValueError('Output array size is Undecided')
        return [ndarray(out_shape, t) for t in out_types]

    shape = out_shape
    for i in range(len(out_params)):
        a = out_args[i]
        p = out_params[i]
        if not isinstance(a, ndarray):
            raise TypeError(
                'Output arguments type must be cupy.ndarray')
        arr = a
        if not p.raw and not internal.vector_equal(arr._shape, shape):
            raise ValueError('Out shape is mismatched')
    return out_args


cdef function.Function _get_elementwise_kernel(
        tuple args_info, tuple types, tuple params, operation, name,
        preamble, dict kwargs):
    kernel_params = _get_kernel_params(params, args_info)
    types_preamble = '\n'.join(
        'typedef %s %s;' % (_get_typename(v), k) for k, v in types)
    preamble = types_preamble + '\n' + preamble

    op = []
    for p, a in zip(params, args_info):
        if not p.raw and a[0] == ndarray:
            if p.is_const:
                fmt = 'const {t} &{n} = _raw_{n}[_ind.get()];'
            else:
                fmt = '{t} &{n} = _raw_{n}[_ind.get()];'
            op.append(fmt.format(t=p.ctype, n=p.name))
    op.append(operation)
    operation = '\n'.join(op)
    return _get_simple_elementwise_kernel(
        kernel_params, operation, name,
        preamble, **kwargs)


cdef class ElementwiseKernel:

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
        readonly tuple in_params
        readonly tuple out_params
        readonly Py_ssize_t nin
        readonly Py_ssize_t nout
        readonly Py_ssize_t nargs
        readonly tuple params
        readonly object operation
        readonly object name
        readonly bint reduce_dims
        readonly object preamble
        readonly bint no_return
        readonly bint return_tuple
        readonly dict kwargs
        readonly dict _kernel_memo
        readonly dict _params_type_memo

    def __init__(self, in_params, out_params, operation,
                 name='kernel', reduce_dims=True, preamble='',
                 no_return=False, return_tuple=False, **kwargs):
        if not compiler.is_valid_kernel_name(name):
            raise ValueError(
                'Invalid kernel name: "%s"' % name)

        self.in_params = _get_param_info(in_params, True)
        self.out_params = _get_param_info(out_params, False)
        self.nin = len(self.in_params)
        self.nout = len(self.out_params)
        self.nargs = self.nin + self.nout
        param_rest = _get_param_info('CIndexer _ind', False)
        self.params = self.in_params + self.out_params + param_rest
        self.operation = operation
        self.name = name
        self.reduce_dims = reduce_dims
        self.preamble = preamble
        self.no_return = no_return
        self.return_tuple = return_tuple
        self.kwargs = kwargs
        self._kernel_memo = {}
        self._params_type_memo = {}
        names = [p.name for p in self.in_params + self.out_params]
        if 'i' in names:
            raise ValueError('Can not use \'i\' as a parameter name')

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

        Returns:
            If ``no_return`` has not set, arrays are returned according to the
            ``out_params`` argument of the ``__init__`` method.
            If ``no_return`` has set, ``None`` is returned.

        """
        cdef function.Function kern
        cdef Py_ssize_t size, i
        cdef list in_args, out_args
        cdef tuple in_types, out_types, types, shape

        size = -1
        size = kwargs.pop('size', -1)
        stream = kwargs.pop('stream', None)
        if len(kwargs):
            raise TypeError('Wrong arguments %s' % kwargs)

        n_args = len(args)
        if n_args != self.nin and n_args != self.nargs:
            raise TypeError('Wrong number of arguments for %s' % self.name)
        dev_id = device.get_device_id()
        args = _preprocess_args(dev_id, args, True)

        values, shape = _broadcast(args, self.params, size != -1)
        in_args = values[:self.nin]
        out_args = args[self.nin:]

        in_ndarray_types = tuple(
            [a.dtype.type if isinstance(a, ndarray) else None
             for a in in_args])
        out_ndarray_types = tuple([a.dtype.type for a in out_args])

        in_types, out_types, types = self._decide_params_type(
            in_ndarray_types, out_ndarray_types)

        is_size_specified = False
        if size != -1:
            shape = size,
            is_size_specified = True

        out_args = _get_out_args_with_params(
            out_args, out_types, shape, self.out_params, is_size_specified)
        if self.no_return:
            ret = None
        elif not self.return_tuple and self.nout == 1:
            ret = out_args[0]
        else:
            ret = tuple(out_args)

        if 0 in shape:
            return ret

        for i, x in enumerate(in_args):
            if type(x) is _scalar.CScalar:
                (<_scalar.CScalar>x).apply_dtype(in_types[i])

        inout_args = in_args + out_args

        if self.reduce_dims:
            shape = _reduce_dims(inout_args, self.params, shape)
        indexer = Indexer(shape)
        inout_args.append(indexer)

        args_info = _get_args_info(inout_args)
        kern = self._get_elementwise_kernel(dev_id, args_info, types)
        kern.linear_launch(indexer.size, inout_args, shared_mem=0,
                           block_max_size=128, stream=stream)
        return ret

    cpdef tuple _decide_params_type(
            self, tuple in_args_dtype, tuple out_args_dtype):
        key = (in_args_dtype, out_args_dtype)
        ret = self._params_type_memo.get(key, None)
        if ret is not None:
            return ret
        ret = _decide_params_type_core(
            self.in_params, self.out_params, in_args_dtype, out_args_dtype)
        self._params_type_memo[key] = ret
        return ret

    cpdef function.Function _get_elementwise_kernel(
            self, int dev_id, tuple args_info, tuple types):
        key = (dev_id, args_info, types)
        kern = self._kernel_memo.get(key, None)
        if kern is not None:
            return kern
        kern = _get_elementwise_kernel(
            args_info, types, self.params, self.operation,
            self.name, self.preamble, self.kwargs)
        self._kernel_memo[key] = kern
        return kern


cdef function.Function _get_ufunc_kernel(
        tuple in_types, tuple out_types, routine, tuple args_info, params,
        name, preamble, loop_prep):
    kernel_params = _get_kernel_params(params, args_info)

    types = []
    op = []
    for i, x in enumerate(in_types):
        types.append('typedef %s in%d_type;' % (_get_typename(x), i))
        if args_info[i][0] is ndarray:
            op.append(
                'const in{0}_type in{0}(_raw_in{0}[_ind.get()]);'
                .format(i))

    for i, x in enumerate(out_types):
        types.append('typedef %s out%d_type;' % (
            _get_typename(args_info[i + len(in_types)][1]), i))
        op.append('out{0}_type &out{0} = _raw_out{0}[_ind.get()];'.format(i))

    op.append(routine)
    operation = '\n'.join(op)

    types.append(preamble)
    preamble = '\n'.join(types)

    return _get_simple_elementwise_kernel(
        kernel_params, operation, name, preamble, loop_prep=loop_prep)


cdef tuple _guess_routine_from_in_types(list ops, tuple in_types):
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


cdef tuple _guess_routine_from_dtype(list ops, object dtype):
    cdef tuple op, op_types
    for op in ops:
        op_types = op[1]
        for t in op_types:
            if t != dtype:
                break
        else:
            return op
    return None


cdef inline bint _check_should_use_min_scalar(list in_args) except? -1:
    cdef int kind, max_array_kind, max_scalar_kind
    cdef bint all_scalars
    all_scalars = True
    max_array_kind = -1
    max_scalar_kind = -1
    for i in in_args:
        kind = get_kind_score(ord(i.dtype.kind))
        if isinstance(i, ndarray):
            all_scalars = False
            max_array_kind = max(max_array_kind, kind)
        else:
            max_scalar_kind = max(max_scalar_kind, kind)
    return (max_scalar_kind != -1 and
            not all_scalars and
            max_array_kind >= max_scalar_kind)


cdef tuple _guess_routine(name, dict cache, list ops, list in_args, dtype):
    if dtype is None:
        use_raw_value = _check_should_use_min_scalar(in_args)
        if use_raw_value:
            in_types = tuple([i.dtype if isinstance(i, ndarray) else i
                              for i in in_args])
            op = ()
        else:
            in_types = tuple([i.dtype.type for i in in_args])
            op = cache.get(in_types, ())

        if op is ():
            op = _guess_routine_from_in_types(ops, in_types)
            if not use_raw_value:
                cache[in_types] = op
    else:
        op = cache.get(dtype, ())
        if op is ():
            op = _guess_routine_from_dtype(ops, dtype)
            cache[dtype] = op

    if op:
        return op
    if dtype is None:
        dtype = tuple([i.dtype.type for i in in_args])
    raise TypeError('Wrong type (%s) of arguments for %s' %
                    (dtype, name))


cdef class ufunc:

    """Universal function.

    Attributes:
        ~ufunc.name (str): The name of the universal function.
        ~ufunc.nin (int): Number of input arguments.
        ~ufunc.nout (int): Number of output arguments.
        ~ufunc.nargs (int): Number of all arguments.

    """

    cdef:
        readonly Py_ssize_t nin
        readonly Py_ssize_t nout
        readonly Py_ssize_t nargs
        readonly object name
        readonly list _ops
        readonly object _preamble
        readonly object _loop_prep
        readonly object _default_casting
        readonly tuple _params
        readonly dict _routine_cache
        readonly dict _kernel_memo
        readonly object __doc__
        readonly object __name__
        readonly object __module__

    def __init__(self, name, nin, nout, ops, preamble='', loop_prep='', doc='',
                 default_casting=None):
        self.name = name
        self.__name__ = name
        self.nin = nin
        self.nout = nout
        self.nargs = nin + nout
        self._ops = ops
        self._preamble = preamble
        self._loop_prep = loop_prep
        self.__doc__ = doc
        if default_casting is None:
            self._default_casting = 'same_kind'
        else:
            self._default_casting = default_casting
        _in_params = tuple(
            ParameterInfo('T in%d' % i, True)
            for i in range(nin))
        _out_params = tuple(
            ParameterInfo('T out%d' % i, False)
            for i in range(nout))
        self._params = _in_params + _out_params + (
            ParameterInfo('CIndexer _ind', False),)
        self._routine_cache = {}
        self._kernel_memo = {}

    def __repr__(self):
        return '<ufunc \'%s\'>' % self.name

    @property
    def types(self):
        """A list of type signatures.

        Each type signature is represented by type character codes of inputs
        and outputs separated by '->'.

        """
        types = []
        for in_types, out_types, _ in self._ops:
            in_str = ''.join([<str>get_dtype(t).char for t in in_types])
            out_str = ''.join([<str>get_dtype(t).char for t in out_types])
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
        cdef tuple shape

        out = kwargs.pop('out', None)
        dtype = kwargs.pop('dtype', None)
        # Note default behavior of casting is 'same_kind' on numpy>=1.10
        casting = kwargs.pop('casting', self._default_casting)
        if dtype is not None:
            dtype = get_dtype(dtype).type
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)

        n_args = len(args)
        if n_args != self.nin and n_args != self.nargs:
            raise TypeError('Wrong number of arguments for %s' % self.name)

        dev_id = device.get_device_id()
        args = _preprocess_args(dev_id, args, False)
        if out is None:
            in_args = args[:self.nin]
            out_args = args[self.nin:]
        else:
            if self.nout != 1:
                raise ValueError('Cannot use \'out\' in %s' % self.name)
            if n_args != self.nin:
                raise ValueError('Cannot specify \'out\' as both '
                                 'a positional and keyword argument')

            in_args = list(args)
            out_args = _preprocess_args(dev_id, (out,), False)
            args += out_args

        broad_values, shape = _broadcast_core(args)

        op = _guess_routine(
            self.name, self._routine_cache, self._ops, in_args, dtype)
        in_types, out_types, routine = op
        out_args = _get_out_args(out_args, out_types, shape, casting)
        if self.nout == 1:
            ret = out_args[0]
        else:
            ret = tuple(out_args)

        if 0 in shape:
            return ret

        inout_args = []
        for i, t in enumerate(in_types):
            x = broad_values[i]
            inout_args.append(x if isinstance(x, ndarray) else
                              _scalar.get_scalar_from_numpy(x, t))
        inout_args.extend(out_args)
        shape = _reduce_dims(inout_args, self._params, shape)
        indexer = Indexer(shape)
        inout_args.append(indexer)
        args_info = _get_args_info(inout_args)

        kern = self._get_ufunc_kernel(dev_id, op, args_info)

        kern.linear_launch(indexer.size, inout_args)
        return ret

    cdef str _get_name_with_type(self, tuple args_info):
        inout_type_words = []
        for t, dtype, ndim in args_info:
            dtype = str(numpy.dtype(dtype))
            if t is _scalar.CScalar:
                inout_type_words.append(dtype.rstrip('0123456789'))
            elif t is not Indexer:
                inout_type_words.append(dtype)
        return '{}__{}'.format(self.name, '_'.join(inout_type_words))

    cdef function.Function _get_ufunc_kernel(
            self, int dev_id, tuple op, tuple args_info):
        cdef function.Function kern
        key = (dev_id, op, args_info)
        kern = self._kernel_memo.get(key, None)
        if kern is None:
            in_types, out_types, routine = op
            name = self._get_name_with_type(args_info)
            kern = _get_ufunc_kernel(
                in_types, out_types, routine, args_info,
                self._params, name, self._preamble, self._loop_prep)
            self._kernel_memo[key] = kern
        return kern


cpdef create_ufunc(name, ops, routine=None, preamble='', doc='',
                   default_casting=None, loop_prep=''):
    _ops = []
    for t in ops:
        if not isinstance(t, tuple):
            typ = t
            rt = routine
        else:
            typ, rt = t

        types = typ.split('->')
        if len(types) == 1:
            in_types = out_types = tuple(types)
        else:
            in_types, out_types = map(tuple, types)
        in_types = tuple([get_dtype(t).type for t in in_types])
        out_types = tuple([get_dtype(t).type for t in out_types])
        _ops.append((in_types, out_types, rt))

    ret = ufunc(name, len(_ops[0][0]), len(_ops[0][1]), _ops, preamble,
                loop_prep, doc, default_casting=default_casting)
    return ret

include 'reduction.pxi'
