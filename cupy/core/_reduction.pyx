from cpython cimport sequence

from cupy.core cimport _carray
from cupy.core._dtype cimport get_dtype
from cupy.core cimport _kernel
from cupy.core._kernel cimport _broadcast
from cupy.core._kernel cimport _check_array_device_id
from cupy.core._kernel cimport _get_arginfos
from cupy.core._kernel cimport _get_kernel_params
from cupy.core._kernel cimport _get_out_args
from cupy.core._kernel cimport _get_out_args_with_params
from cupy.core._kernel cimport _preprocess_args
from cupy.core._kernel cimport _reduce_dims
from cupy.core._kernel cimport ParameterInfo, _ArgInfo
from cupy.core cimport _routines_manipulation as _manipulation
from cupy.core cimport _scalar
from cupy.core._scalar import get_typename as _get_typename
from cupy.core.core cimport _convert_object_with_cuda_array_interface
from cupy.core.core cimport _internal_asfortranarray
from cupy.core.core cimport compile_with_cache
from cupy.core.core cimport ndarray
from cupy.core cimport internal
from cupy.cuda cimport device
from cupy.cuda cimport function
from cupy.cuda cimport memory
from cupy.cuda cimport runtime
from cupy.cuda cimport driver

import string

import numpy

from cupy.core._kernel import _get_param_info
from cupy.core._kernel import _decide_params_type
from cupy.cuda import compiler
from cupy import util


cpdef function.Function _create_reduction_function(
        name, block_size, reduce_type, params, arginfos, identity,
        pre_map_expr, reduce_expr, post_map_expr,
        _kernel._TypeMap type_map, input_expr, output_expr, preamble, options):
    module_code = string.Template('''
${type_preamble}
${preamble}
#define REDUCE(a, b) (${reduce_expr})
#define POST_MAP(a) (${post_map_expr})
#define _REDUCE(_offset) if (_tid < _offset) { \
  _type_reduce _a = _sdata[_tid], _b = _sdata[(_tid + _offset)]; \
  _sdata[_tid] = REDUCE(_a, _b); \
}

typedef ${reduce_type} _type_reduce;
extern "C" __global__ void ${name}(${params}) {
  __shared__ char _sdata_raw[${block_size} * sizeof(_type_reduce)];
  _type_reduce *_sdata = reinterpret_cast<_type_reduce*>(_sdata_raw);
  unsigned int _tid = threadIdx.x;

  int _J_offset = _tid >> __popc(_block_stride - 1);  // _tid / _block_stride
  ptrdiff_t _j_offset = (ptrdiff_t)_J_offset * _out_ind.size();
  int _J_stride = ${block_size} >> __popc(_block_stride - 1);
  ptrdiff_t _j_stride = (ptrdiff_t)_J_stride * _out_ind.size();

  for (ptrdiff_t _i_base = (ptrdiff_t)blockIdx.x * _block_stride;
       _i_base < _out_ind.size();
       _i_base += (ptrdiff_t)gridDim.x * _block_stride) {
    _type_reduce _s = _type_reduce(${identity});
    ptrdiff_t _i =
        _i_base + (_tid & (_block_stride - 1));  // _tid % _block_stride
    int _J = _J_offset;
    for (ptrdiff_t _j = _i + _j_offset; _j < _in_ind.size();
         _j += _j_stride, _J += _J_stride) {
      _in_ind.set(_j);
      ${input_expr}
      _type_reduce _a = static_cast<_type_reduce>(${pre_map_expr});
      _s = REDUCE(_s, _a);
    }
    _sdata[_tid] = _s;
    __syncthreads();
    for (unsigned int _block = ${block_size} / 2;
         _block >= _block_stride; _block >>= 1) {
      if (_tid < _block) {
        _REDUCE(_block);
      }
      __syncthreads();
    }
    if (_tid < _block_stride) {
      _s = _sdata[_tid];
    }
    if (_tid < _block_stride && _i < _out_ind.size()) {
      _out_ind.set(static_cast<ptrdiff_t>(_i));
      ${output_expr}
      POST_MAP(_s);
    }
  }
}''').substitute(
        name=name,
        block_size=block_size,
        reduce_type=reduce_type,
        params=_kernel._get_kernel_params(params, arginfos),
        identity=identity,
        reduce_expr=reduce_expr,
        pre_map_expr=pre_map_expr,
        post_map_expr=post_map_expr,
        type_preamble=type_map.get_typedef_code(),
        input_expr=input_expr,
        output_expr=output_expr,
        preamble=preamble)
    module = compile_with_cache(module_code, options)
    return module.get_function(name)


cpdef function.Function _create_cub_reduction_function(
        name, block_size, items_per_thread,
        reduce_type, params, arginfos, identity,
        pre_map_expr, reduce_expr, post_map_expr,
        _kernel._TypeMap type_map, input_expr, output_expr, preamble, options):

    # move this part to a Python file for faster test cycles
    # TODO: move it back to here!
    from ._cub_simple_reduction import _get_cub_reduction_function_code
    module_code = _get_cub_reduction_function_code(
        name, block_size, items_per_thread,
        reduce_type, params, arginfos, identity,
        pre_map_expr, reduce_expr, post_map_expr,
        type_map, input_expr, output_expr, preamble, options)

    # CUB is not natively supported by NVRTC (NVlabs/cub#131), so use nvcc
    # instead. For this, we have to explicitly spell out the default values for
    # arch, cachd, and prepend_cupy_headers to bypass cdef/cpdef limitation...
    # TODO(leofang): investigate Jitify for using NVRTC (also NVlabs/cub#171)
    module = compile_with_cache(module_code, options, arch=None, cachd_dir=None,
        prepend_cupy_headers=True, backend='nvcc')
    return module.get_function(name)


cdef tuple _can_use_cub_block_reduction(
        list in_args, list out_args, tuple reduce_axis, tuple out_axis):
    '''
    If CUB BlockReduce can be used, this function returns the axes,
    otherwise returns None.
    '''
    from cupy.cuda import _environment
    from cupy import core

    cdef tuple axis_permutes_cub
    cdef ndarray in_arr, out_arr

    # first check the flag settable at runtime from outside
    if not core.cub_block_reduction_enabled:
        return None

    # rare event (mainly for conda-forge users): nvcc is not found!
    if _environment.get_nvcc_path() is None:
        return None

    # we currently support only _SimpleReductionKernel
    if len(in_args) != 1 or len(out_args) != 1:
        return None

    in_arr = in_args[0]
    out_arr = out_args[0]

    # To support generic reductions, note that some NumPy casting rules
    # are not applicable in the C++ space (unless we tweak the type
    # definitions). To circumvent this, we fall back to the old kernel.
    # TODO(leofang): can we relax this?
    if in_arr.dtype.kind != out_arr.dtype.kind:
        # cannot cast complex to anything else
        if in_arr.dtype.kind == 'c':
            return None
        # cannot cast float16 to complex
        if in_arr.dtype.char == 'e' and out_arr.dtype.kind == 'c':
            return None

    # check reduction axes, if not contiguous then fall back to old kernel
    reduce_axis = tuple(sorted(reduce_axis))
    out_axis = tuple(sorted(out_axis))
    if in_arr.flags.f_contiguous:
        axis_permutes_cub = reduce_axis + out_axis
    elif in_arr.flags.c_contiguous:
        axis_permutes_cub = out_axis + reduce_axis
    else:
        axis_permutes_cub = None
    if axis_permutes_cub == tuple(range(in_arr.ndim)):
        return axis_permutes_cub
    else:
        return None


# similar to cupy.core._kernel._get_kernel_params()
cpdef str _get_cub_kernel_params(tuple params, tuple arginfos):
    cdef ParameterInfo p
    cdef _ArgInfo arginfo
    cdef lst = []
    cdef str c_type
    cdef int i
    assert len(params) == len(arginfos)

    for i, (p, arginfo) in enumerate(zip(params, arginfos)):
        if i != len(params) - 1:
            c_type = 'const void*' if p.is_const else 'void*'
        else:
            # for segment size
            c_type = arginfo.get_param_c_type(p)
        lst.append('{} {}'.format(
            c_type,
            arginfo.get_c_var_name(p)))
    return ', '.join(lst)


cdef _cub_two_pass_launch(
        str name, Py_ssize_t block_size, Py_ssize_t segment_size,
        Py_ssize_t items_per_thread, str reduce_type, tuple params,
        list in_args, list out_args,
        str identity, str pre_map_expr, str reduce_expr, str post_map_expr,
        _kernel._TypeMap type_map, str input_expr, str output_expr, str preamble, tuple options, stream):
    # TODO: CANNOT use type_map here, because the intermediate type might not be a legit NumPy dtype


    cdef list out_args_2nd_pass = [out_args[0]]
    cdef Py_ssize_t contiguous_size = block_size * items_per_thread
    cdef Py_ssize_t out_block_num = (segment_size + contiguous_size - 1) // contiguous_size # fair share
    cdef function.Function func
    cdef memory.MemoryPointer ptr

    # Because we can't know sizeof(reduce_type) in advance, here we
    # conservatively assume it's 32 bytes and allocate a work area
    #print("\n************************ allocating", out_block_num * 32, " bytes ************************\n")
    #print("in size", in_args[0].size, block_size, segment_size, out_block_num, contiguous_size)
    ptr = memory.alloc(out_block_num * 32)
    out_args[0] = ptr
    #print(in_args[0].data.ptr, ptr.ptr, contiguous_size)

    #print("out_block_num:", out_block_num, "contiguous_size:", contiguous_size)

    # ****** First pass ******
    name += '_pass1'
    inout_args = (in_args + out_args
                  + [_scalar.CScalar.from_int32(contiguous_size)])


    cdef tuple params1 = (params + _get_param_info('int32 _segment_size', True))
    cub_params = (True, items_per_thread )#, False)
    cdef str input_expr1 = input_expr
    cdef str output_expr1 = '_type_reduce* _out0 = static_cast<_type_reduce*>(_raw_out0);'

    # Kernel arguments passed to the __global__ function.
    gridx = <size_t>(out_block_num*block_size)
    blockx = <size_t>block_size

    # Retrieve the kernel function
    func = _SimpleReductionKernel_get_cached_function(
            pre_map_expr, reduce_expr, post_map_expr, reduce_type,
            #pre_map_expr, reduce_expr, 'out0 = _type_reduce(a)', reduce_type,
            params1,
            _get_arginfos(inout_args), 
            type_map,  # TODO: update this
            name, block_size, identity,
            input_expr1, output_expr1, preamble, ('-DFIRST_PASS=1',), cub_params)

    # Launch the kernel
    func.linear_launch(
        gridx, inout_args, 0, blockx, stream)

    # ****** Second pass ******
    name = name[:-1] + '2'
    contiguous_size = out_block_num
    out_block_num = 1 # (out_block_num + block_size - 1) // block_size
    in_args = out_args
    out_args = out_args_2nd_pass
    cdef str input_expr2 = 'const _type_reduce* _in0 = static_cast<const _type_reduce*>(_raw_in0);'
    cdef str output_expr2 = output_expr

    # Kernel arguments passed to the __global__ function.
    gridx = <size_t>(out_block_num*block_size)
    blockx = <size_t>block_size
    inout_args = (in_args + out_args
                  + [_scalar.CScalar.from_int32(contiguous_size)])

    cdef tuple params2 = (params + _get_param_info('int32 _segment_size', True))
    cub_params = (True, items_per_thread )#, True)

    # Retrieve the kernel function
    func = _SimpleReductionKernel_get_cached_function(
            '', reduce_expr, post_map_expr, reduce_type,
            params2,
            _get_arginfos(inout_args), 
            type_map,  # TODO: update this
            name, block_size, identity,
            input_expr2, output_expr2, preamble, ('-DSECOND_PASS=1',), cub_params)

    # Launch the kernel
    func.linear_launch(
        gridx, inout_args, 0, blockx, stream)


cpdef tuple _get_axis(object axis, Py_ssize_t ndim):
    cdef Py_ssize_t dim
    if axis is None:
        return (tuple(range(ndim)), ())
    elif sequence.PySequence_Check(axis):
        axis = tuple(axis)
    else:
        axis = axis,

    for dim in axis:
        if dim < -ndim or dim >= ndim:
            raise numpy.AxisError('Axis overrun')
    reduce_axis = tuple(sorted([dim % ndim for dim in axis]))
    out_axis = tuple([dim for dim in range(ndim) if dim not in reduce_axis])
    return reduce_axis, out_axis


cpdef tuple _get_out_shape(
        tuple shape, tuple reduce_axis, tuple out_axis, bint keepdims):
    if keepdims:
        out_shape = list(shape)
        for i in reduce_axis:
            out_shape[i] = 1
        return tuple(out_shape)
    return tuple([shape[i] for i in out_axis])


cdef tuple _set_permuted_args(
        list args, tuple axis_permutes, tuple shape, tuple params):
    # This function updates `args`
    cdef ParameterInfo p
    cdef Py_ssize_t i, s
    cdef bint need_permutation = False
    for i, s in enumerate(axis_permutes):
        if i != s:
            need_permutation = True
            break
    if need_permutation:
        for p in params:
            if p.raw:
                raise NotImplementedError('Illegal conditions')
        for i, a in enumerate(args):
            if isinstance(a, ndarray):
                args[i] = _manipulation._transpose(a, axis_permutes)
        shape = tuple([shape[i] for i in axis_permutes])
    return shape


cdef Py_ssize_t _get_contiguous_size(
        list args, tuple params, Py_ssize_t ndim,
        Py_ssize_t out_ndim) except -1:
    '''
    get contiguous size in the *output* axis (not *reduce* axis!)
    '''
    cdef int i, j
    cdef ParameterInfo p
    cdef Py_ssize_t contiguous_size, tmp_contiguous_size, itemsize
    contiguous_size = 1
    for i, a in enumerate(args):
        if not isinstance(a, ndarray):
            continue
        p = params[i]
        if p.raw:
            continue
        tmp_contiguous_size = 1
        itemsize = a.dtype.itemsize
        for j in range(out_ndim):
            if a._strides[ndim-j-1] != tmp_contiguous_size * itemsize:
                break
            tmp_contiguous_size *= a._shape[ndim-j-1]
        contiguous_size = max(contiguous_size, tmp_contiguous_size)
    return contiguous_size


cpdef (Py_ssize_t, Py_ssize_t, Py_ssize_t) _get_block_specs(  # NOQA
        Py_ssize_t in_size, Py_ssize_t out_size,
        Py_ssize_t contiguous_size) except*:
    cdef Py_ssize_t reduce_block_size, block_stride, out_block_num

    reduce_block_size = max(1, in_size // out_size)
    contiguous_size = min(contiguous_size, 32)
    block_stride = max(contiguous_size, _block_size // reduce_block_size)
    block_stride = internal.clp2(block_stride // 2 + 1)  # floor
    out_block_num = (out_size + block_stride - 1) // block_stride

    return _block_size, block_stride, out_block_num


cdef Py_ssize_t _block_size = 256 if runtime._is_hip_environment else 512


def _sort_axis(tuple axis, tuple strides):
    # Sorts axis in the decreasing order of absolute values of strides.
    return tuple(sorted(axis, key=lambda i: -abs(strides[i])))


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

    cpdef ndarray _call(
            self,
            list in_args, list out_args,
            tuple a_shape, axis, dtype,
            bint keepdims, bint reduce_dims,
            stream, bint try_use_cub=False):
        cdef tuple reduce_axis, out_axis, axis_permutes
        cdef tuple axis_permutes_cub, cub_params
        cdef tuple params
        cdef Py_ssize_t i
        cdef Py_ssize_t contiguous_size, items_per_thread
        cdef Py_ssize_t block_size, block_stride, out_block_num
        cdef ndarray ret
        cdef function.Function func
        cdef bint use_cub, full_reduction
        cdef size_t gridx, blockx
        cdef list out_args_2nd_pass = []

        if dtype is not None:
            dtype = get_dtype(dtype).type

        (
            map_expr, reduce_expr, post_map_expr,
            in_types, out_types, reduce_type,
            type_map,
        ) = self._get_expressions_and_types(in_args, out_args, dtype)

        reduce_axis, out_axis = _get_axis(axis, len(a_shape))
        #print("reduce_axis", reduce_axis, "out_axis", out_axis)

        # When there is only one input array, sort the axes in such a way that
        # contiguous (C or F) axes can be squashed in _reduce_dims() later.
        # TODO(niboshi): Support (out_axis) > 1
        if (len(in_args) == 1
                and len(out_axis) <= 1
                and not in_args[0]._c_contiguous):
            strides = in_args[0].strides
            reduce_axis = _sort_axis(reduce_axis, strides)
            out_axis = _sort_axis(out_axis, strides)
            #print("reduce_axis", reduce_axis, "out_axis", out_axis)

        out_shape = _get_out_shape(a_shape, reduce_axis, out_axis, keepdims)
        out_args = self._get_out_args(out_args, out_types, out_shape)
        ret = out_args[0]
        if ret.size == 0:
            return ret

        if self.identity == '' and 0 in a_shape:
            raise ValueError(('zero-size array to reduction operation'
                              ' %s which has no identity') % self.name)

        in_args = [x if isinstance(x, ndarray) else
                   _scalar.CScalar.from_numpy_scalar_with_dtype(x, t)
                   for x, t in zip(in_args, in_types)]

        # TODO: need two code paths for use_cub and the old behavior

        # decide to use CUB or not
        use_cub = False
        axis_permutes = reduce_axis + out_axis
        #print("reduce_axis", reduce_axis, "out_axis", out_axis, "axis_permutes", axis_permutes)
        if try_use_cub:
            axis_permutes_cub = _can_use_cub_block_reduction(
                in_args, out_args, reduce_axis, out_axis)
            if axis_permutes_cub is not None:
                use_cub = True
                axis_permutes = axis_permutes_cub

        in_shape = _set_permuted_args(in_args, axis_permutes,
                                      a_shape, self.in_params)

        if not use_cub:
            if reduce_dims:
                in_shape = _reduce_dims(in_args, self.in_params, in_shape)
                out_shape = _reduce_dims(out_args, self.out_params, out_shape)
            #print("in_shape:", in_shape)
            #print("out_shape:", out_shape)

            # Calculate the reduction block dimensions.
            contiguous_size = _get_contiguous_size(
                in_args, self.in_params, len(in_shape), len(out_shape))
            block_size, block_stride, out_block_num = _get_block_specs(
                internal.prod_sequence(in_shape),
                internal.prod_sequence(out_shape),
                contiguous_size)

            # Kernel arguments passed to the __global__ function.
            inout_args = (
                in_args
                + out_args
                + [
                    _carray.Indexer(in_shape),
                    _carray.Indexer(out_shape),
                    # block_stride is passed as the last argument.
                    _scalar.CScalar.from_int32(block_stride),
                ])

            params = self._params
            cub_params = (False, None)
        else:
            # TODO(leofang): fix reduce_dims

            self._input_expr = 'const type_in0_raw* _in0 = static_cast<const type_in0_raw*>(_raw_in0);'
            self._output_expr = 'type_out0_raw* _out0 = static_cast<type_out0_raw*>(_raw_out0);'

            contiguous_size = 1
            for i in reduce_axis:
                contiguous_size *= in_shape[i]

            # This should be an even number
            # TODO(leofang): this is recommended in the CUB internal, but
            # perhaps we could do some auto-tuning to determine this?
            items_per_thread = 4

            # Calculate the reduction block dimensions.
            # Ideally, we want each block to handle one segment, so:
            # - block size < segment size: the block loops over the segment
            # - block size >= segment size: the segment can fit in the entire block
            # TODO(leofang): also auto-tune the block size?
            block_size = (contiguous_size + items_per_thread - 1) // items_per_thread
            block_size = internal.clp2(block_size)
            if block_size < 32:
                block_size = 32  # warp size
            elif block_size > _block_size:
                block_size = _block_size

            ## fine tuning
            #block_size = 1024
            #items_per_thread = 4

            if in_args[0].flags.f_contiguous:
                ret = out_args[0] = _internal_asfortranarray(ret)
                #print(ret.flags)

            if len(reduce_axis) > 1 and len(out_axis) == 0:
                # full-reduction of N-D array: need to invoke the kernel twice,
                # the first pass to do an even share distributed over block_size threads,
                # and the second pass to do reduction over the threads
                full_reduction = True
                out_block_num = (in_args[0].size + block_size - 1) \
                    // block_size
            else:
                # just need one pass
                full_reduction = False
                out_block_num = 1  # = number of segments
                for i in out_axis:
                    out_block_num *= in_shape[i]

            #print("out_block_num:", out_block_num, "contiguous_size:", contiguous_size)

            # Kernel arguments passed to the __global__ function.
            inout_args = (in_args + out_args
                          + [_scalar.CScalar.from_int32(contiguous_size)])

            params = (self._params[0:2] + _get_param_info('int32 _segment_size', True))
            cub_params = (True, items_per_thread)

        #cdef Py_ssize_t _old_block_size = block_size
        #cdef Py_ssize_t _old_items_per_thread = items_per_thread
        #for block_size in (32, 64, 128, 256, 512, 1024):
        #    for items_per_thread in (4, 8, 16, 32):
        #        cub_params = (True, items_per_thread)
        #        func = self._get_function(
        #            params,
        #            _get_arginfos(inout_args),
        #            type_map,
        #            map_expr, reduce_expr, post_map_expr, reduce_type,
        #            block_size, cub_params)
        #        #print("kernel pointer:", func.ptr)
        #        print("max active blocks for block size", block_size, " and items per thread", items_per_thread, ":",
        #               driver.occupancyMaxActiveBlocksPerMultiprocessor(func.ptr, block_size, 0))
        #        print("max potential block size for items per thread", items_per_thread, ":",
        #               driver.occupancyMaxPotentialBlockSize(func.ptr, 0, 0))
        #block_size = _old_block_size
        #items_per_thread =  _old_items_per_thread
        #cub_params = (True, items_per_thread)

        if use_cub and full_reduction:
            _cub_two_pass_launch(
                self.name, block_size, contiguous_size, items_per_thread,
                reduce_type, self._params[0:2], in_args, out_args, self.identity,
                map_expr, reduce_expr, post_map_expr,
                type_map, self._input_expr, self._output_expr, self._preamble,
                (), stream)
        else:
            # Retrieve the kernel function
            func = self._get_function(
                params,
                _get_arginfos(inout_args),
                type_map,
                map_expr, reduce_expr, post_map_expr, reduce_type,
                block_size, cub_params)

            # Launch the kernel
            gridx = <size_t>(out_block_num*block_size)
            blockx = <size_t>block_size
            func.linear_launch(gridx, inout_args, 0, blockx, stream)

        return ret

    cdef tuple _get_expressions_and_types(
            self, list in_args, list out_args, dtype):
        raise NotImplementedError()

    cdef list _get_out_args(
            self, list out_args, tuple out_types, tuple out_shape):
        raise NotImplementedError()

    cdef function.Function _get_function(
            self,
            tuple params, tuple arginfos, _kernel._TypeMap type_map,
            str map_expr, str reduce_expr, str post_map_expr, str reduce_type,
            Py_ssize_t block_size, tuple cub_params=(False, None)):
        '''
        cub_params (tuple): (use_cub, items_per_thread)
        '''
        raise NotImplementedError()


# -----------------------------------------------------------------------------
# create_reduction_func
# -----------------------------------------------------------------------------

cpdef _SimpleReductionKernel create_reduction_func(
        name, ops, routine=None, identity=None, preamble=''):
    ops = _kernel._Ops.from_tuples(ops, routine)
    return _SimpleReductionKernel(name, ops, identity, preamble)


cdef class _SimpleReductionKernel(_AbstractReductionKernel):

    cdef:
        readonly _kernel._Ops _ops
        readonly _preamble
        readonly int nin
        readonly int nout
        public str _input_expr
        public str _output_expr
        readonly dict _routine_cache

    def __init__(self, name, _kernel._Ops ops, identity, preamble):
        super().__init__(
            name,
            '' if identity is None else str(identity),
            'T in0',
            'T out0',
        )
        self._ops = ops
        self._preamble = preamble
        self.nin = 1
        self.nout = 1
        self._input_expr = 'const type_in0_raw in0 = _raw_in0[_in_ind.get()];'
        self._output_expr = 'type_out0_raw &out0 = _raw_out0[_out_ind.get()];'
        self._routine_cache = {}

    def __call__(self, object a, axis=None, dtype=None, ndarray out=None,
                 bint keepdims=False):

        cdef ndarray arr

        if isinstance(a, ndarray):
            arr = a
        elif hasattr(a, '__cuda_array_interface__'):
            arr = _convert_object_with_cuda_array_interface(a)
        else:
            raise TypeError(
                'Argument \'a\' has incorrect type (expected %s, got %s)' %
                (ndarray, type(a)))
        in_args = [arr]

        dev_id = device.get_device_id()
        _check_array_device_id(arr, dev_id)

        if out is None:
            out_args = []
        else:
            _check_array_device_id(out, dev_id)
            out_args = [out]

        reduce_dims = True
        return self._call(
            in_args, out_args,
            arr.shape, axis, dtype, keepdims, reduce_dims, None, True)

    cdef tuple _get_expressions_and_types(
            self, list in_args, list out_args, dtype):
        cdef _kernel._Op op

        op = self._ops.guess_routine(
            self.name, self._routine_cache, in_args, dtype, self._ops)
        map_expr, reduce_expr, post_map_expr, reduce_type = op.routine

        if reduce_type is None:
            reduce_type = _get_typename(op.out_types[0])

        if out_args:
            out_type = out_args[0].dtype.type
        else:
            out_type = op.out_types[0]

        type_map = _kernel._TypeMap((
            ('type_in0_raw', in_args[0].dtype.type),
            ('type_out0_raw', out_type),
        ))

        #print (
        #    map_expr, reduce_expr, post_map_expr,
        #    op.in_types, op.out_types, reduce_type,
        #    type_map)
        return (
            map_expr, reduce_expr, post_map_expr,
            op.in_types, op.out_types, reduce_type,
            type_map)

    cdef list _get_out_args(
            self, list out_args, tuple out_types, tuple out_shape):
        return _get_out_args(
            out_args, out_types, out_shape, 'unsafe')

    cdef function.Function _get_function(
            self,
            tuple params, tuple arginfos, _kernel._TypeMap type_map,
            str map_expr, str reduce_expr, str post_map_expr, str reduce_type,
            Py_ssize_t block_size, tuple cub_params=(False, None)):
        return _SimpleReductionKernel_get_cached_function(
            map_expr, reduce_expr, post_map_expr, reduce_type,
            params, arginfos, type_map,
            self.name, block_size, self.identity,
            self._input_expr, self._output_expr, self._preamble, (), cub_params)


@util.memoize(for_each_device=True)
def _SimpleReductionKernel_get_cached_function(
        map_expr, reduce_expr, post_map_expr, reduce_type,
        params, arginfos, _kernel._TypeMap type_map,
        name, block_size, identity, input_expr, output_expr, _preamble,
        options, cub_params):

    DEBUG = False
    if DEBUG:
        print("name:",          name,          type(name), "\n")           
        print("block_size:",    block_size,    type(block_size), "\n")     
        print("reduce_type:",   reduce_type,   type(reduce_type), "\n")   
        print("params:",        params,        type(params), "\n")        
        print("arginfos:",      arginfos,      type(arginfos), "\n")       
        print("identity:",      identity,      type(identity), "\n")       

        print("map_expr:",      map_expr,      type(map_expr), "\n")          
        print("reduce_expr:",   reduce_expr,   type(reduce_expr), "\n")  
        print("post_map_expr:", post_map_expr, type(post_map_expr), "\n") 

        print("type_map:",      type_map,      type(type_map), "\n")       
        print("input_expr:",    input_expr,    type(input_expr), "\n")     
        print("output_expr:",   output_expr,   type(output_expr), "\n")    
        print("_preamble:",     _preamble,     type(_preamble), "\n")      
        print("options:",       options,       type(options), "\n")        

        #temp =  _kernel._get_cub_kernel_params(params, arginfos)
        temp2 = type_map.get_typedef_code()
        #print("final params:",  temp,          type(temp), "\n")
        print("modified params:", _get_cub_kernel_params(params, arginfos))
        print("type_preamble:", temp2,         type(temp2), "\n")

    use_cub, items_per_thread = cub_params
    #use_cub, items_per_thread, is_1st_pass = cub_params
    if not use_cub:
        return _create_reduction_function(
            name, block_size, reduce_type, params, arginfos, identity,
            map_expr, reduce_expr, post_map_expr,
            type_map, input_expr, output_expr, _preamble, options)
    else:
        name = name.replace('cupy_', 'cupy_cub_')
        name = name.replace('cupyx_', 'cupyx_cub_')
        return _create_cub_reduction_function(
            name, block_size, items_per_thread,
            reduce_type, params, arginfos, identity,
            map_expr, reduce_expr, post_map_expr,
            type_map, input_expr, output_expr, #is_1st_pass,
            _preamble, options)


# -----------------------------------------------------------------------------
# ReductionKernel
# -----------------------------------------------------------------------------


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
        if not compiler.is_valid_kernel_name(name):
            raise ValueError(
                'Invalid kernel name: "%s"' % name)

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
            axis (int or tuple of ints): Axis or axes along which the
                reduction is performed.
            keepdims (bool): If ``True``, the specified axes are remained as
                axes of length one.

        Returns:
            Arrays are returned according to the ``out_params`` argument of the
            ``__init__`` method.

        """

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

        dev_id = device.get_device_id()
        in_args = _preprocess_args(dev_id, args[:self.nin], False)
        out_args = _preprocess_args(dev_id, out_args, False)
        in_args, broad_shape = _broadcast(in_args, self.in_params, False)

        return self._call(
            in_args, out_args,
            broad_shape, axis, None,
            keepdims, self.reduce_dims, stream, False)

    cdef tuple _get_expressions_and_types(
            self, list in_args, list out_args, dtype):

        in_ndarray_types = tuple(
            [a.dtype.type if isinstance(a, ndarray) else None
             for a in in_args])
        out_ndarray_types = tuple(
            [a.dtype.type if isinstance(a, ndarray) else None
             for a in out_args])
        in_types, out_types, type_map = _decide_params_type(
            self.in_params, self.out_params,
            in_ndarray_types, out_ndarray_types)
        return (
            self.map_expr, self.reduce_expr, self.post_map_expr,
            in_types, out_types, self.reduce_type,
            type_map)

    cdef list _get_out_args(
            self, list out_args, tuple out_types, tuple out_shape):
        return _get_out_args_with_params(
            out_args, out_types, out_shape, self.out_params, False)

    cdef function.Function _get_function(
            self,
            tuple params, tuple arginfos, _kernel._TypeMap type_map,
            str map_expr, str reduce_expr, str post_map_expr, str reduce_type,
            Py_ssize_t block_size, tuple cub_params=(False, None)):
        return _ReductionKernel_get_cached_function(
            self.nin, self.nout, params, arginfos, type_map,
            self.name, block_size, reduce_type, self.identity,
            map_expr, reduce_expr, post_map_expr,
            self.preamble, self.options)


@util.memoize(for_each_device=True)
def _ReductionKernel_get_cached_function(
        nin, nout, params, arginfos, _kernel._TypeMap type_map,
        name, block_size, reduce_type, identity, map_expr, reduce_expr,
        post_map_expr, preamble, options):
    cdef ParameterInfo p
    cdef _ArgInfo arginfo
    in_arrays = [
        p for p, arginfo in zip(params[:nin], arginfos[:nin])
        if not p.raw and arginfo.is_ndarray()]
    out_arrays = [
        p for p, arginfo in zip(params[nin:nin+nout], arginfos[nin:nin+nout])
        if not p.raw and arginfo.is_ndarray()]
    input_expr = '\n'.join(
        [(('const {0} {1}' if p.is_const else '{0}& {1}') +
          ' = _raw_{1}[_in_ind.get()];').format(p.ctype, p.name)
         for p in in_arrays])
    output_expr = '\n'.join(
        ['{0} &{1} = _raw_{1}[_out_ind.get()];'.format(p.ctype, p.name)
         for p in out_arrays if not p.is_const])

    return _create_reduction_function(
        name, block_size, reduce_type, params, arginfos, identity,
        map_expr, reduce_expr, post_map_expr,
        type_map, input_expr, output_expr, preamble, options)
