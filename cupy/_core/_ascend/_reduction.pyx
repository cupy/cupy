# Ascend 实现的 reduction 派发（模块名 cupy._core._reduction，由
# features/ascend.py 映射到本文件；CUDA 构建使用 cupy/_core/_gpu/_reduction.pyx，
# 两侧独立演进）。声明层 cupy/_core/_reduction.pxd 为双方共享 —— 修改本文件中
# 被 pxd 声明的 cpdef/cdef 签名时必须同步 pxd，否则其它模块的 cimport 会失配。
# CUDA 专用代码（kernel 代码生成、cub、axis permute 等）属于 _gpu 版，不在本文件。
from cpython cimport sequence

from cupy._core cimport _carray
from cupy._core cimport _accelerator
from cupy._core._carray cimport shape_t
#from cupy._core cimport _cub_reduction
from cupy._core._dtype cimport get_dtype
from cupy._core cimport _kernel
from cupy._core._kernel cimport _broadcast
from cupy._core._kernel cimport _check_peer_access
from cupy._core._kernel cimport _get_arginfos
from cupy._core._kernel cimport _get_out_args_from_optionals
from cupy._core._kernel cimport _get_out_args_with_params
from cupy._core._kernel cimport _preprocess_args
from cupy._core._kernel cimport _reduce_dims
from cupy._core._kernel cimport ParameterInfo, _ArgInfo
from cupy._core cimport _optimize_config
from cupy._core cimport _routines_manipulation as _manipulation
from cupy._core cimport _scalar
from cupy._core._scalar import get_typename as _get_typename
#from cupy._core._routines_creation cimport _convert_object_with_cuda_array_interface
from cupy._core._routines_creation cimport _create_ndarray_from_shape_strides
#from cupy._core._compile_with_cache cimport compile_with_cache
from cupy._core.core cimport _ndarray_base
from cupy._core cimport internal
from cupy.xpu cimport device
from cupy.xpu cimport function
from cupy.backends.backend.api cimport runtime

import math
import warnings
import numpy

import cupy
from cupy._core._kernel import _get_param_info
from cupy._core._kernel import _decide_params_type
from cupy._core._ufuncs import elementwise_copy
#from cupy.cuda import compiler
from cupy import _util

from cupy.backends.ascend.api.acl_utils cimport launch_reduction_op
from cupy.backends.ascend.api.acl_utils cimport ascend_float64_promote_enabled

from cupy.xpu cimport stream as stream_module

# uint -> signed promotion for aclnn reductions (docs/ascend/
# AscendSpecialization.md §A.1). Kept in sync with the always-on layer of
# _ASCEND_DTYPE_PROMOTE in cupy/backends/ascend/api/acl_utils.pyx, which
# serves only the elementwise/general launchers (the reduction dispatcher no
# longer promotes). Narrow ints (b/h) are NOT promoted here: aclnnAmax/Amin/
# mean/sum take them natively; all/any handle their dtype needs in C++
# (aclop_Any/aclop_All). The optional float64/complex128 demotion layer DOES
# apply here, live-gated by enable_float64_to_float32 (see _FLOAT64_DEMOTE).
cdef dict _UINT_PROMOTE = {
    'B': 'i',   # uint8  -> int32
    'H': 'i',   # uint16 -> int32
    'I': 'i',   # uint32 -> int32
    'Q': 'q',   # uint64 -> int64 (numpy 1.x char)
    'L': 'q',   # uint64 -> int64 (numpy 2.x char)
}

# 可选层（enable_float64_to_float32 开关，见 AscendSpecialization.md A.1.1）：
# float64 -> float32、complex128 -> complex64。开关打开后并入 `_call` 的有效
# 提升表，结果经 `ret[...] = promoted_out`（elementwise copy，dtype 转换）
# 写回原 float64/complex128 out —— 用户可见 dtype 不变，精度降为单精度。
# 注意 aclnnAmax/Amin/mean/sum 本身原生收 DOUBLE：开关打开后这些归约也要多
# 付两次 cast kernel；开关用于不收 DOUBLE 的归约/组合算子和 910B（无 float64
# 硬件吞吐）。状态经 acl_utils.ascend_float64_promote_enabled() 实时读取，
# py_enable_float64_to_float32 的运行时切换对 reduction 同样生效。
cdef dict _FLOAT64_DEMOTE = {
    'd': 'f',   # float64    -> float32
    'G': 'F',   # complex128 -> complex64
}
cdef inline size_t _get_stream(stream) except *:
    if stream is None:
        return stream_module.get_current_stream_ptr()
    else:
        return stream.ptr

cpdef tuple _get_axis(object axis, Py_ssize_t ndim):
    cdef Py_ssize_t dim
    if axis is None:
        return (tuple(range(ndim)), ())
    elif sequence.PySequence_Check(axis):
        axis = tuple(axis)
    else:
        axis = axis,

    reduce_axis = tuple(sorted(
        [internal._normalize_axis_index(dim, ndim) for dim in axis]))
    out_axis = tuple([dim for dim in range(ndim) if dim not in reduce_axis])
    if len(reduce_axis) + len(out_axis) != ndim:
        raise ValueError("duplicate value in 'axis'")
    return reduce_axis, out_axis


cpdef shape_t _get_out_shape(
        const shape_t& shape, tuple reduce_axis, tuple out_axis,
        bint keepdims):
    cdef shape_t out_shape
    if keepdims:
        out_shape = shape
        for i in reduce_axis:
            out_shape[i] = 1
    else:
        out_shape.reserve(len(out_axis))
        for i in out_axis:
            out_shape.push_back(shape[i])
    return out_shape


cdef shape_t _set_permuted_args(
        list args, tuple axis_permutes, const shape_t& shape, tuple params):
    # This function updates `args`
    cdef ParameterInfo p
    cdef Py_ssize_t i, s
    cdef bint need_permutation = False
    cdef shape_t out_shape
    for i, s in enumerate(axis_permutes):
        if i != s:
            need_permutation = True
            break
    if need_permutation:
        for p in params:
            if p.raw:
                raise NotImplementedError('Illegal conditions')
        for i, a in enumerate(args):
            if isinstance(a, _ndarray_base):
                args[i] = _manipulation._transpose(a, axis_permutes)
        out_shape.reserve(len(axis_permutes))
        for i in axis_permutes:
            out_shape.push_back(shape[i])
        return out_shape
    else:
        return shape



cdef Py_ssize_t _get_contiguous_size(
        list args, tuple params, list out_shape, Py_ssize_t ndim) except -1:
    '''
    get contiguous size in the *output* axis (not *reduce* axis!)
    '''
    cdef int i, j
    cdef ParameterInfo p
    cdef Py_ssize_t contiguous_size, tmp_contiguous_size, itemsize
    out_ndim = len(out_shape)
    contiguous_size = 1
    for i, a in enumerate(args):
        if not isinstance(a, _ndarray_base):
            continue
        p = params[i]
        if p.raw:
            continue
        tmp_contiguous_size = 1
        itemsize = a.dtype.itemsize
        for j in range(out_ndim):
            if a._strides[ndim-j-1] != tmp_contiguous_size * itemsize:
                break
            tmp_contiguous_size *= out_shape[out_ndim-j-1]
        contiguous_size = max(contiguous_size, tmp_contiguous_size)
    return contiguous_size


cdef Py_ssize_t _default_block_size = (
    256 if runtime._is_hip_environment else 512)
cdef Py_ssize_t _min_block_size_log = 5
cdef Py_ssize_t _max_block_size_log = (
    8 if runtime._is_hip_environment else 9)


cpdef (Py_ssize_t, Py_ssize_t, Py_ssize_t) _get_block_specs(  # NOQA
        Py_ssize_t in_size, Py_ssize_t out_size,
        Py_ssize_t contiguous_size,
        Py_ssize_t block_size) except*:
    cdef Py_ssize_t reduce_block_size, block_stride, out_block_num
    if block_size == -1:
        block_size = _default_block_size

    reduce_block_size = max(1, in_size // out_size)
    contiguous_size = min(contiguous_size, 32)
    block_stride = max(contiguous_size, block_size // reduce_block_size)
    block_stride = internal.clp2(block_stride // 2 + 1)  # floor
    out_block_num = (out_size + block_stride - 1) // block_stride

    return block_size, block_stride, out_block_num


cdef tuple _sort_axis(tuple axis, tuple strides):
    # Sorts axis in the decreasing order of absolute values of strides.
    return tuple(sorted(axis, key=lambda i: -abs(strides[i])))


cdef tuple _get_shape_and_strides(list in_args, list out_args):
    cdef list shape_and_strides = []
    for x in in_args + out_args:
        if isinstance(x, _ndarray_base):
            shape_and_strides.append(x.shape)
            shape_and_strides.append(x.strides)
        else:
            shape_and_strides.append(None)
            shape_and_strides.append(None)
    return tuple(shape_and_strides)


cdef _optimizer_copy_arg(a):
    if isinstance(a, _ndarray_base):
        x = _create_ndarray_from_shape_strides(
            cupy.ndarray, a._shape, a._strides, a.dtype, None)
        assert a.data.device_id == x.data.device_id
        elementwise_copy(a, x)
        return x
    return a




cdef class _AbstractReductionKernel:

    def __init__(
            self, str name, str identity, str in_params, str out_params):
        assert name is not None
        assert identity is not None
        assert in_params is not None
        assert out_params is not None

        in_params_ = _get_param_info(in_params, True)
        out_params_ = _get_param_info(out_params, False)
        params = (
            in_params_
            + out_params_
            + _get_param_info('CIndexer _in_ind, CIndexer _out_ind', False)
            + _get_param_info('int32 _block_stride', True))

        self.name = name
        self.identity = identity
        self.in_params = in_params_
        self.out_params = out_params_
        self._params = params
        # This is for profiling mechanisms to auto infer a name
        self.__name__ = name
        self._cached_codes = {}

    cpdef _ndarray_base _call(
            self,
            list in_args, list out_args,
            const shape_t& a_shape, axis, dtype,
            bint keepdims, bint reduce_dims, int device_id,
            stream, bint try_use_cub=False, bint sort_reduce_axis=True):

        cdef tuple reduce_axis, out_axis, axis_permutes
        cdef tuple params, opt_params
        cdef tuple shape_and_strides
        cdef Py_ssize_t contiguous_size = -1
        cdef shape_t in_shape, out_shape
        cdef _ndarray_base ret
        cdef tuple ops

        if dtype is not None:
            dtype = get_dtype(dtype).type
        # not needed for ASCEND?
        (
            map_expr, reduce_expr, post_map_expr,
            in_types, out_types, reduce_type,
            type_map,
        ) = self._get_expressions_and_types(in_args, out_args, dtype)

        reduce_axis, out_axis = _get_axis(axis, a_shape.size())

        # When there is only one input array, sort the axes in such a way that
        # contiguous (C or F) axes can be squashed in _reduce_dims() later.
        # TODO(niboshi): Support (out_axis) > 1
        if (len(in_args) == 1
                and len(out_axis) <= 1
                and not in_args[0]._c_contiguous):
            strides = in_args[0].strides
            if sort_reduce_axis:
                reduce_axis = _sort_axis(reduce_axis, strides)
            out_axis = _sort_axis(out_axis, strides)

        out_shape = _get_out_shape(a_shape, reduce_axis, out_axis, keepdims)
        out_args = self._get_out_args(out_args, out_types, out_shape)
        ret = out_args[0]
        if ret.size == 0:
            return ret

        if self.identity == '' and internal.is_in(a_shape, 0):
            raise ValueError(('zero-size array to reduction operation'
                            ' %s which has no identity') % self.name)

        if internal.prod(a_shape) / internal.prod(out_shape) > 0x7fffffff:
            index_type = ('IndexT', 'int64')
        else:
            index_type = ('IndexT', 'int32')
        type_map = _kernel._TypeMap(type_map._pairs + (index_type,))

        in_args = [x if isinstance(x, _ndarray_base) else
                _scalar.CScalar.from_numpy_scalar_with_dtype(x, t)
                for x, t in zip(in_args, in_types)]

        key = ()

        # ASCEND: Special NOTE: aclnn reduction ops take the real ndim + the `original` axis pos
        # so keep the input un-permuted and pass reduce_axis directly
        # CUDA permutes axes to the front for the kernel contiguity

        # NOTE: ASCEND special: reduce_dims sequeeze C-continguous dims (2, 3) -> (6,)
        # which corrupts the shape semantics of the alcnn reduction ops
        # which got 1D inptut but he original reduce_axis/dim, aclnn relies on real ndim
        # so skip the dim-squashing optimization on ASCEND
        #if reduce_dims:
        #    in_shape = _reduce_dims(in_args, self.in_params, in_shape)
        #    out_shape = _reduce_dims(out_args, self.out_params, out_shape)

        params = self._params
        cdef s = _get_stream(stream)
        # NOTE: launch_reduction_op 的 kwargs 形参类型是 dict，传 None 会
        # 直接 TypeError（reduction 全部不可用）。当前没有需要透传的关键字参数，传空 dict。

        # ASCEND: aclnn reductions take no unsigned integer inputs. Promote
        # uint -> signed per the table above, reduce into a promoted temp
        # out, then cast the result back into `ret` (whose dtype is what the
        # CUDA loop types selected). This replaces the uint-promotion block
        # that used to live in launch_reduction_op_raw.
        # 可选层：enable_float64_to_float32 打开时（env var 或运行时 setter，
        # 实时读取），float64->float32、complex128->complex64 并入有效表；
        # 归约写进单精度临时 out，再经 `ret[...] = promoted_out` cast 回原
        # float64/complex128 ret（用户可见 dtype 不变，精度单精度，
        # AscendSpecialization.md §A.1.1）。
        cdef dict _promote = _UINT_PROMOTE
        cdef list launch_ins = list(in_args)
        cdef list launch_outs = [ret]
        cdef bint promoted = False
        cdef Py_ssize_t _pi
        cdef object _x
        cdef _ndarray_base promoted_out = None
        if ascend_float64_promote_enabled():
            _promote = dict(_UINT_PROMOTE)
            _promote.update(_FLOAT64_DEMOTE)
        for _pi in range(len(launch_ins)):
            _x = launch_ins[_pi]
            if isinstance(_x, _ndarray_base) and _x.dtype.char in _promote:
                launch_ins[_pi] = _x.astype(_promote[_x.dtype.char])
                promoted = True
        if ret.dtype.char in _promote:
            promoted_out = cupy.empty(ret.shape, _promote[ret.dtype.char])
            launch_outs = [promoted_out]
            promoted = True
        if promoted:
            launch_reduction_op(self.name, launch_ins, launch_outs,
                                axis, keepdims, {}, s)
            # 仅当 out 也被提升（promoted_out 非 None）才需要 cast-back：
            # 只有输入被提升而 ret 本身不在提升表时，归约已直接写进 ret。
            if promoted_out is not None:
                ret[...] = promoted_out
            return ret
        launch_reduction_op(self.name, list(in_args), [ret], axis, keepdims, {}, s)
        return ret

    def _get_optimized_params(
            self, optimize_config, in_args, out_args, in_shape, out_shape,
            type_map, map_expr, reduce_expr, post_map_expr, reduce_type,
            stream):
        out_size = internal.prod(out_shape)
        in_args = [_optimizer_copy_arg(a) for a in in_args]
        out_args = [_optimizer_copy_arg(a) for a in out_args]

        contiguous_size = _get_contiguous_size(
            in_args, self.in_params, out_shape, len(in_shape))
        block_size, block_stride, default_out_block_num = _get_block_specs(
            internal.prod(in_shape),
            internal.prod(out_shape),
            contiguous_size, -1)
        default_block_size_log = math.floor(math.log2(block_size))
        default_block_stride_log = math.floor(math.log2(block_stride))

        def target_func(block_size, block_stride, out_block_num):
            self._launch(
                out_block_num, block_size, block_stride, in_args, out_args,
                in_shape, out_shape, type_map, map_expr, reduce_expr,
                post_map_expr, reduce_type, stream, self._params)

        def suggest_func(trial):
            block_size_log = trial.suggest_int(
                'block_size_log', _min_block_size_log, _max_block_size_log)
            block_size = 2 ** block_size_log
            block_stride_log = trial.suggest_int(
                'block_stride_log', 0, block_size_log)
            block_stride = 2 ** block_stride_log
            max_out_block_num = (out_size + block_stride - 1) // block_stride
            out_block_num = trial.suggest_int(
                'out_block_num', 1, max_out_block_num)

            trial.set_user_attr('block_size', block_size)
            trial.set_user_attr('block_stride', block_stride)
            return block_size, block_stride, out_block_num

        optimize_impl = optimize_config.optimize_impl
        best = optimize_impl(
            optimize_config, target_func, suggest_func,
            default_best={
                'block_size_log': default_block_size_log,
                'block_stride_log': default_block_stride_log,
                'out_block_num': default_out_block_num,
            }
        )
        return (
            best.user_attrs['block_size'],
            best.user_attrs['block_stride'],
            best.params['out_block_num'])

    cdef inline void _launch(
            self, out_block_num, block_size, block_stride,
            in_args, out_args, in_shape, out_shape, type_map,
            map_expr, reduce_expr, post_map_expr, reduce_type,
            stream, params):
        cdef function.Function func

        inout_args = (
            in_args
            + out_args
            + [
                _carray._indexer_init(in_shape),
                _carray._indexer_init(out_shape),
                # block_stride is passed as the last argument.
                _scalar.CScalar.from_int32(block_stride),
            ])

        # Retrieve the kernel function
        func = self._get_function(
            params,
            _get_arginfos(inout_args),
            type_map,
            map_expr, reduce_expr, post_map_expr, reduce_type,
            block_size)

        # Launch the kernel
        func.linear_launch(
            out_block_num * block_size, inout_args, 0, block_size, stream)

    cdef tuple _get_expressions_and_types(
            self, list in_args, list out_args, dtype):
        raise NotImplementedError()

    cdef list _get_out_args(
            self, list out_args, tuple out_types, const shape_t& out_shape):
        raise NotImplementedError()

    cdef function.Function _get_function(
            self,
            tuple params, tuple arginfos, _kernel._TypeMap type_map,
            str map_expr, str reduce_expr, str post_map_expr, str reduce_type,
            Py_ssize_t block_size):
        raise NotImplementedError()

    @property
    def cached_codes(self):
        """Returns a dict that has input types as keys and codes values.

        This proprety method is for debugging purpose.
        The return value is not guaranteed to keep backward compatibility.
        """
        if len(self._cached_codes) == 0:
            warnings.warn(
                'No codes are cached because compilation is deferred until '
                'the first function call or CUB is enabled.')
        return dict([(k, v) for k, v in self._cached_codes.items()])

    @property
    def cached_code(self):
        """Returns `next(iter(self.cached_codes.values()))`.

        This proprety method is for debugging purpose.
        The return value is not guaranteed to keep backward compatibility.
        """
        codes = self._cached_codes
        if len(codes) > 1:
            warnings.warn(
                'The input types of the kernel could not be inferred. '
                'Please use `.cached_codes` instead.')
        return next(iter(codes.values()))


# -----------------------------------------------------------------------------
# create_reduction_func
# -----------------------------------------------------------------------------

cpdef _SimpleReductionKernel create_reduction_func(
        name, ops, routine=None, identity=None, preamble='',
        sort_reduce_axis=True):
    ops = _kernel._Ops.from_tuples(ops, routine)
    return _SimpleReductionKernel(
        name, ops, identity, preamble, sort_reduce_axis)


cdef class _SimpleReductionKernel(_AbstractReductionKernel):

    cdef:
        readonly _kernel._Ops _ops
        readonly str preamble
        readonly int nin
        readonly int nout
        readonly str _input_expr
        readonly str _output_expr
        readonly dict _routine_cache
        readonly bint _sort_reduce_axis

    def __init__(
            self, name, _kernel._Ops ops, identity, preamble,
            sort_reduce_axis=True):
        super().__init__(
            name,
            '' if identity is None else str(identity),
            'T in0',
            'T out0',
        )
        self._ops = ops
        self.preamble = preamble
        self.nin = 1
        self.nout = 1
        self._input_expr = 'const type_in0_raw in0 = _raw_in0[_in_ind.get()];'
        self._output_expr = 'type_out0_raw &out0 = _raw_out0[_out_ind.get()];'
        self._routine_cache = {}
        self._sort_reduce_axis = sort_reduce_axis

    def __call__(self, object a, axis=None, dtype=None, _ndarray_base out=None,
                 bint keepdims=False):

        if hasattr(a, '__cupy_override_reduction_kernel__'):
            return a.__cupy_override_reduction_kernel__(
                self, axis, dtype, out, keepdims)

        cdef _ndarray_base arr

        if isinstance(a, _ndarray_base):
            arr = a
        #elif hasattr(a, '__cuda_array_interface__'):
        elif hasattr(a, '__cupy_get_ndarray__'):
            arr = a.__cupy_get_ndarray__()
        else:
            raise TypeError(
                'Argument \'a\' has incorrect type (expected %s, got %s)' %
                (cupy.ndarray, type(a)))
        in_args = [arr]

        dev_id = device.get_device_id()
        _check_peer_access(arr, dev_id)

        if out is None:
            out_args = []
        else:
            _check_peer_access(out, dev_id)
            out_args = [out]

        reduce_dims = True
        return self._call(
            in_args, out_args,
            arr._shape, axis, dtype, keepdims, reduce_dims, dev_id,
            None, True, self._sort_reduce_axis)

    cdef tuple _get_expressions_and_types(
            self, list in_args, list out_args, dtype):
        cdef _kernel._Op op

        # XXX: weaks
        weaks = None
        op = self._ops.guess_routine(
            self.name, self._routine_cache, in_args, weaks, dtype, self._ops)
        map_expr, reduce_expr, post_map_expr, reduce_type = op.routine

        if reduce_type is None:
            reduce_type = _get_typename(op.out_types[0])

        if out_args:
            out_type = out_args[0].dtype.type
        else:
            out_type = op.out_types[0]

        # We guessed a routine that requires a C2R casting for the input
        if (in_args[0].dtype.kind == 'c'
                and numpy.dtype(op.in_types[0]).kind == 'f'):
            warnings.warn(
                'Casting complex values to real discards the imaginary part',
                cupy.exceptions.ComplexWarning)
            in_args[0] = in_args[0].real

        type_map = _kernel._TypeMap((
            ('type_in0_raw', in_args[0].dtype.type),
            ('type_out0_raw', out_type),
        ))

        return (
            map_expr, reduce_expr, post_map_expr,
            op.in_types, op.out_types, reduce_type,
            type_map)

    cdef list _get_out_args(
            self, list out_args, tuple out_types, const shape_t& out_shape):
        return _get_out_args_from_optionals(
            cupy.ndarray, out_args, out_types, out_shape, 'unsafe', None)

    cdef function.Function _get_function(
            self,
            tuple params, tuple arginfos, _kernel._TypeMap type_map,
            str map_expr, str reduce_expr, str post_map_expr, str reduce_type,
            Py_ssize_t block_size):

        in_types = []
        for x in arginfos:
            if x.type is cupy.ndarray:
                in_types.append(cupy.dtype(x.dtype).char)
        in_types = tuple(in_types)
        return None

cdef class ReductionKernel(_AbstractReductionKernel):

    """User-defined reduction kernel.

    This class can be used to define a reduction kernel with or without
    broadcasting.

    The kernel is compiled at an invocation of the
    :meth:`~ReductionKernel.__call__` method, which is cached for each device.
    The compiled binary is also cached into a file under the
    ``$HOME/.cupy/kernel_cache/`` directory with a hashed file name. The cached
    binary is reused by other processes.

    Args:
        in_params (str): Input argument list.
        out_params (str): Output argument list.
        map_expr (str): Mapping expression for input values.
        reduce_expr (str): Reduction expression.
        post_map_expr (str): Mapping expression for reduced values.
        identity (str): Identity value for starting the reduction.
        name (str): Name of the kernel function. It should be set for
            readability of the performance profiling.
        reduce_type (str): Type of values to be used for reduction. This type
            is used to store the special variables ``a``.
        reduce_dims (bool): If ``True``, input arrays are reshaped without copy
            to smaller dimensions for efficiency.
        preamble (str): Fragment of the CUDA-C/C++ code that is inserted at the
            top of the cu file.
        options (tuple of str): Additional compilation options.

    """

    def __init__(self, str in_params, str out_params,
                 map_expr, reduce_expr, post_map_expr,
                 identity, name='reduce_kernel', reduce_type=None,
                 reduce_dims=True, preamble='', options=()):
        #if not compiler.is_valid_kernel_name(name):
        #    raise ValueError(
        #        'Invalid kernel name: "%s"' % name)

        super().__init__(
            name,
            '' if identity is None else str(identity),
            in_params,
            out_params,
        )
        self.nin = len(self.in_params)
        self.nout = len(self.out_params)
        self.nargs = self.nin + self.nout
        self.reduce_expr = reduce_expr
        self.map_expr = map_expr
        self.post_map_expr = post_map_expr
        self.options = options
        self.reduce_dims = reduce_dims
        if reduce_type is None:
            self.reduce_type = self.out_params[0].ctype
        else:
            self.reduce_type = reduce_type
        self.preamble = preamble

    def __call__(self, *args, **kwargs):
        """Compiles and invokes the reduction kernel.

        The compilation runs only if the kernel is not cached. Note that the
        kernels with different argument dtypes, ndims, or axis are not
        compatible. It means that single ReductionKernel object may be compiled
        into multiple kernel binaries.

        Args:
            args: Arguments of the kernel.
            out (cupy.ndarray): The output array. This can only be specified if
                ``args`` does not contain the output array.
            axis (int or tuple of ints): Axis or axes along which the
                reduction is performed.
            keepdims (bool): If ``True``, the specified axes are remained as
                axes of length one.
            stream (cupy.xpu.Stream, optional): The CUDA stream to launch the
                kernel on. If not given, the current stream will be used.

        Returns:
            Arrays are returned according to the ``out_params`` argument of the
            ``__init__`` method.

        """
        cdef shape_t broad_shape

        out = kwargs.pop('out', None)
        axis = kwargs.pop('axis', None)
        keepdims = kwargs.pop('keepdims', False)
        stream = kwargs.pop('stream', None)
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)

        n_args = len(args)
        if n_args != self.nin and n_args != self.nargs:
            raise TypeError('Wrong number of arguments for %s' % self.name)

        out_args = list(args[self.nin:])
        if out is not None:
            if self.nout != 1:
                raise NotImplementedError('')
            if len(out_args) != 0:
                raise ValueError("cannot specify 'out' as both "
                                 "a positional and keyword argument")
            out_args = [out]

        # XXX: needs to handle weak scalars from _preprocess_args?
        dev_id = device.get_device_id()
        in_args, _ = _preprocess_args(dev_id, args[:self.nin], False)
        out_args, _ = _preprocess_args(dev_id, out_args, False)
        in_args = _broadcast(in_args, self.in_params, False, broad_shape)

        return self._call(
            in_args, out_args,
            broad_shape, axis, None,
            keepdims, self.reduce_dims, dev_id, stream, True, True)

    cdef tuple _get_expressions_and_types(
            self, list in_args, list out_args, dtype):

        in_ndarray_types = tuple(
            [a.dtype.type if isinstance(a, _ndarray_base) else None
             for a in in_args])
        out_ndarray_types = tuple(
            [a.dtype.type if isinstance(a, _ndarray_base) else None
             for a in out_args])
        in_types, out_types, type_map = _decide_params_type(
            self.in_params, self.out_params,
            in_ndarray_types, out_ndarray_types)
        return (
            self.map_expr, self.reduce_expr, self.post_map_expr,
            in_types, out_types, self.reduce_type,
            type_map)

    cdef list _get_out_args(
            self, list out_args, tuple out_types, const shape_t& out_shape):
        return _get_out_args_with_params(
            out_args, out_types, out_shape, self.out_params, False)

    cdef function.Function _get_function(
            self,
            tuple params, tuple arginfos, _kernel._TypeMap type_map,
            str map_expr, str reduce_expr, str post_map_expr, str reduce_type,
            Py_ssize_t block_size):

        in_types = []
        for x in arginfos:
            if x.type is cupy.ndarray:
                in_types.append(cupy.dtype(x.dtype).char)
        in_types = tuple(in_types)
        
        return None # TODO

