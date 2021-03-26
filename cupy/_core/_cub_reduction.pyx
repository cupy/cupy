from cupy._core._carray cimport shape_t
from cupy._core cimport _kernel
from cupy._core cimport _optimize_config
from cupy._core cimport _reduction
from cupy._core cimport _scalar
from cupy._core.core cimport compile_with_cache
from cupy._core.core cimport ndarray
from cupy._core.core cimport _internal_ascontiguousarray
from cupy._core cimport internal
from cupy.cuda cimport cub
from cupy.cuda cimport function
from cupy.cuda cimport memory
from cupy_backends.cuda.api cimport runtime

import math
import string
import sys
from cupy import _environment
from cupy._core._kernel import _get_param_info
from cupy.cuda import driver
from cupy import _util


cdef function.Function _create_cub_reduction_function(
        name, block_size, items_per_thread,
        reduce_type, params, arginfos, identity,
        pre_map_expr, reduce_expr, post_map_expr,
        _kernel._TypeMap type_map, preamble, options):
    # A (incomplete) list of internal variables:
    # _J            : the index of an element in the array

    # static_assert needs at least C++11 in NVRTC
    options += ('--std=c++11',)

    cdef str backend
    if runtime._is_hip_environment:
        # In ROCm, we need to set the include path. This does not work for
        # hiprtc as of ROCm 3.5.0, so we must use hipcc.
        options += ('-I' + _rocm_path + '/include', '-O2')
        backend = 'nvcc'  # this is confusing...
    elif sys.platform.startswith('win32'):
        # See #4771. NVRTC on Windows seems to have problems in handling empty
        # macros, so any usage like this:
        #     #ifndef CUB_NS_PREFIX
        #     #define CUB_NS_PREFIX
        #     #endif
        # will drive NVRTC nuts (error: this declaration has no storage class
        # or type specifier). However, we cannot find a minimum reproducer to
        # confirm this is the root cause, so we work around by using nvcc.
        backend = 'nvcc'
    else:
        # use jitify + nvrtc
        # TODO(leofang): how about simply specifying jitify=True when calling
        # compile_with_cache()?
        options += ('-DCUPY_USE_JITIFY',)
        backend = 'nvrtc'

    # TODO(leofang): try splitting the for-loop into full tiles and partial
    # tiles to utilize LoadDirectBlockedVectorized? See, for example,
    # https://github.com/NVlabs/cub/blob/c3cceac115c072fb63df1836ff46d8c60d9eb304/cub/agent/agent_reduce.cuh#L311-L346

    cdef str module_code = _get_cub_header_include()
    module_code += '''
${type_preamble}
${preamble}

typedef ${reduce_type} _type_reduce;

static_assert(sizeof(_type_reduce) <= 32,
    "The intermediate reduction type is assumed to be at most 32 bytes.");

// Compile-time constants for CUB template specializations
#define ITEMS_PER_THREAD  ${items_per_thread}
#define BLOCK_SIZE        ${block_size}

// for hipCUB: use the hipcub namespace
#ifdef __HIP_DEVICE_COMPILE__
#define cub hipcub
#endif

#if defined FIRST_PASS
    typedef type_in0_raw  type_mid_in;
    typedef _type_reduce  type_mid_out;
    #define POST_MAP(a)   out0 = a;
#elif defined SECOND_PASS
    typedef _type_reduce  type_mid_in;
    typedef type_out0_raw type_mid_out;
    #define POST_MAP(a)   (${post_map_expr})
#else  // one-pass reduction
    typedef type_in0_raw  type_mid_in;
    typedef type_out0_raw type_mid_out;
    #define POST_MAP(a)   (${post_map_expr})
#endif

struct _reduction_op {
    __device__ __forceinline__ _type_reduce operator()(
        const _type_reduce &a, const _type_reduce &b) const {
        return ${reduce_expr};
    }
};

extern "C"
__global__ void ${name}(${params}) {
  unsigned int _tid = threadIdx.x;
'''

    if pre_map_expr == 'in0':
        module_code += '''
  // Specialize BlockLoad type for faster (?) loading
  typedef cub::BlockLoad<_type_reduce, BLOCK_SIZE,
                         ITEMS_PER_THREAD, cub::BLOCK_LOAD_DIRECT> BlockLoadT;

  // Shared memory for loading
  __shared__ typename BlockLoadT::TempStorage temp_storage_load;
'''

    module_code += '''
  // Specialize BlockReduce type for our thread block
  typedef cub::BlockReduce<_type_reduce, BLOCK_SIZE> BlockReduceT;

  // Shared memory for reduction
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  // Declare reduction operation
  _reduction_op op;

  // input & output raw pointers
  const type_mid_in* _in0 = static_cast<const type_mid_in*>(_raw_in0);
  type_mid_out* _out0 = static_cast<type_mid_out*>(_raw_out0);

  // Per-thread tile data
  _type_reduce _sdata[ITEMS_PER_THREAD];
  #pragma unroll
  for (int j = 0; j < ITEMS_PER_THREAD; j++) {
      _sdata[j] = _type_reduce(${identity});
  }

  // each block handles the reduction of 1 segment
  size_t segment_idx = blockIdx.x * _segment_size;
  const type_mid_in* segment_head = _in0 + segment_idx;
  size_t i = 0;  // tile head within the segment
  int tile_size = (BLOCK_SIZE * ITEMS_PER_THREAD < _segment_size ?
                   BLOCK_SIZE * ITEMS_PER_THREAD :
                   _segment_size);
  sizeT _seg_size = _segment_size;

  #if defined FIRST_PASS
  // for two-pass reduction only: "last segment" is special
  if (_array_size > 0) {
      if (_array_size - segment_idx <= _segment_size) {
          _seg_size = _array_size - segment_idx;
      }
      #ifdef __HIP_DEVICE_COMPILE__
      // We don't understand HIP...
      __syncthreads();  // Propagate the new value back to memory
      #endif
  }
  #endif

  // loop over tiles within 1 segment
  _type_reduce aggregate = _type_reduce(${identity});
  for (i = 0; i < _seg_size; i += BLOCK_SIZE * ITEMS_PER_THREAD) {
      // for the last tile
      if (_seg_size - i <= tile_size) { tile_size = _seg_size - i; }
'''

    if pre_map_expr == 'in0':
        module_code += '''
      // load a tile
      BlockLoadT(temp_storage_load).Load(segment_head + i, _sdata, tile_size,
                                         _type_reduce(${identity}));
'''
    else:  # pre_map_expr could be something like "in0 != type_in0_raw(0)"
        module_code += '''
      // load a tile
      #pragma unroll
      for (int j = 0; j < ITEMS_PER_THREAD; j++) {
          // index of the element in a tile
          int e_idx = _tid * ITEMS_PER_THREAD + j;

          // some pre_map_expr uses _J internally...
          #if defined FIRST_PASS
          int _J = (segment_idx + i + e_idx);
          #else  // only one pass
          int _J = (segment_idx + i + e_idx) % _seg_size;
          #endif

          if (e_idx < tile_size) {
              const type_mid_in in0 = *(segment_head + i + e_idx);
              _sdata[j] = static_cast<_type_reduce>(${pre_map_expr});
          } else {
              _sdata[j] = _type_reduce(${identity});
          }
      }
'''

    module_code += '''
      // Compute block reduction
      // Note that the output is only meaningful for thread 0
      aggregate = op(aggregate, BlockReduceT(temp_storage).Reduce(_sdata, op));

      __syncthreads();  // for reusing temp_storage
  }

  if (_tid == 0) {
      type_mid_out& out0 = *(_out0 + blockIdx.x);
      POST_MAP(aggregate);
  }
}
'''

    module_code = string.Template(module_code).substitute(
        name=name,
        block_size=block_size,
        items_per_thread=items_per_thread,
        reduce_type=reduce_type,
        params=_get_cub_kernel_params(params, arginfos),
        identity=identity,
        reduce_expr=reduce_expr,
        pre_map_expr=pre_map_expr,
        post_map_expr=post_map_expr,
        type_preamble=type_map.get_typedef_code(),
        preamble=preamble)

    # To specify the backend, we have to explicitly spell out the default
    # values for arch, cachd, and prepend_cupy_headers to bypass cdef/cpdef
    # limitation...
    module = compile_with_cache(
        module_code, options, arch=None, cachd_dir=None,
        prepend_cupy_headers=True, backend=backend)
    return module.get_function(name)


@_util.memoize(for_each_device=True)
def _SimpleCubReductionKernel_get_cached_function(
        map_expr, reduce_expr, post_map_expr, reduce_type,
        params, arginfos, _kernel._TypeMap type_map,
        name, block_size, identity, preamble,
        options, cub_params):
    items_per_thread = cub_params[0]
    name = name.replace('cupy_', 'cupy_cub_')
    name = name.replace('cupyx_', 'cupyx_cub_')
    return _create_cub_reduction_function(
        name, block_size, items_per_thread,
        reduce_type, params, arginfos, identity,
        map_expr, reduce_expr, post_map_expr,
        type_map, preamble, options)


cdef str _cub_path = _environment.get_cub_path()
cdef str _nvcc_path = _environment.get_nvcc_path()
cdef str _rocm_path = _environment.get_rocm_path()
cdef str _hipcc_path = _environment.get_hipcc_path()
cdef str _cub_header = None


cdef str _get_cub_header_include():
    global _cub_header
    if _cub_header is not None:
        return _cub_header

    assert _cub_path is not None
    cdef str rocm_path = None
    if _cub_path == '<bundle>':
        _cub_header = '''
#include <cupy/cub/cub/block/block_reduce.cuh>
#include <cupy/cub/cub/block/block_load.cuh>
'''
    elif _cub_path == '<CUDA>':
        _cub_header = '''
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_load.cuh>
'''
    elif _cub_path == '<ROCm>':
        # As of ROCm 3.5.0, the block headers cannot be included by themselves
        # (many macros left undefined), so we must use the master header.
        _cub_header = '''
#include <hipcub/hipcub.hpp>
'''
    return _cub_header


# make it cpdef'd for unit tests
cpdef inline tuple _can_use_cub_block_reduction(
        list in_args, list out_args, tuple reduce_axis, tuple out_axis):
    '''
    If CUB BlockReduce can be used, this function returns a tuple of the needed
    parameters, otherwise returns None.
    '''
    cdef tuple axis_permutes_cub
    cdef ndarray in_arr, out_arr
    cdef Py_ssize_t contiguous_size = 1
    cdef str order

    # detect whether CUB headers exists somewhere:
    if _cub_path is None:
        import warnings
        warnings.warn('CUB headers are not found.', RuntimeWarning)
        return None

    # we currently support reductions with 1 input and 1 output
    if len(in_args) != 1 or len(out_args) != 1:
        return None

    in_arr = in_args[0]
    out_arr = out_args[0]

    # the axes might not be sorted when we arrive here...
    reduce_axis = tuple(sorted(reduce_axis))
    out_axis = tuple(sorted(out_axis))

    # check reduction axes, if not contiguous then fall back to old kernel
    if in_arr._f_contiguous:
        order = 'F'
        if not cub._cub_device_segmented_reduce_axis_compatible(
                reduce_axis, in_arr.ndim, order):
            return None
        axis_permutes_cub = reduce_axis + out_axis
    elif in_arr._c_contiguous:
        order = 'C'
        if not cub._cub_device_segmented_reduce_axis_compatible(
                reduce_axis, in_arr.ndim, order):
            return None
        axis_permutes_cub = out_axis + reduce_axis
    else:
        return None
    if axis_permutes_cub != tuple(range(in_arr.ndim)):
        return None

    # full-reduction of N-D array: need to invoke the kernel twice
    cdef bint full_reduction = True if len(out_axis) == 0 else False

    # check if the number of elements is too large
    # (ref: cupy/cupy#3309 for CUB limit)
    for i in reduce_axis:
        contiguous_size *= in_arr.shape[i]
    if contiguous_size > 0x7fffffffffffffff or contiguous_size == 0:
        return None
    if full_reduction:
        # assume a GPU has at most 64 GB of physical memory
        if contiguous_size > 0x1000000000:
            return None
    else:
        # the number of blocks to be launched exceeds INT_MAX:
        if in_arr.size // contiguous_size > 0x7fffffff:
            return None

    # rare event (mainly for conda-forge users): nvcc is not found!
    if not runtime._is_hip_environment:
        if _nvcc_path is None:
            return None
    else:
        if _hipcc_path is None:
            return None

    return (axis_permutes_cub, contiguous_size, full_reduction)


# similar to cupy._core._kernel._get_kernel_params()
cdef str _get_cub_kernel_params(tuple params, tuple arginfos):
    cdef _kernel.ParameterInfo p
    cdef _kernel._ArgInfo arginfo
    cdef lst = []
    cdef str c_type, c_name
    cdef int i
    assert len(params) == len(arginfos)

    for i, (p, arginfo) in enumerate(zip(params, arginfos)):
        c_name = arginfo.get_c_var_name(p)
        if i < len(params) - 2:
            c_type = 'const void*' if p.is_const else 'void*'
        else:
            # for segment size and array size
            c_type = arginfo.get_param_c_type(p)
        lst.append('{} {}'.format(c_type, c_name))
    return ', '.join(lst)


cdef Py_ssize_t _cub_default_block_size = (
    256 if runtime._is_hip_environment else 512)


cdef (Py_ssize_t, Py_ssize_t) _get_cub_block_specs(  # NOQA
        Py_ssize_t contiguous_size):
    # This is recommended in the CUB internal and should be an
    # even number
    items_per_thread = 4

    # Calculate the reduction block dimensions.
    # Ideally, we want each block to handle one segment, so:
    # 1. block size < segment size: the block loops over the segment
    # 2. block size >= segment size: the segment fits in the block
    block_size = (contiguous_size + items_per_thread - 1) // items_per_thread
    block_size = internal.clp2(block_size)
    warp_size = 32 if not runtime._is_hip_environment else 64
    if block_size < warp_size:
        block_size = warp_size
    elif block_size > _cub_default_block_size:
        block_size = _cub_default_block_size

    return items_per_thread, block_size


cdef _scalar.CScalar _cub_convert_to_c_scalar(
        Py_ssize_t segment_size, Py_ssize_t value):
    if segment_size > 0x7fffffff:
        return _scalar.scalar_to_c_scalar(value)
    else:
        return _scalar.CScalar.from_int32(value)


cdef inline void _cub_two_pass_launch(
        str name, Py_ssize_t block_size, Py_ssize_t segment_size,
        Py_ssize_t items_per_thread, str reduce_type, tuple params,
        list in_args, list out_args,
        str identity, str pre_map_expr, str reduce_expr, str post_map_expr,
        _kernel._TypeMap type_map, str preamble,
        tuple options, stream) except*:
    '''
    Notes:
    1. Two-pass reduction: the first pass distributes an even share over
       a number of blocks (with block_size threads), and the second pass
       does reduction over 1 block of threads
    '''

    cdef list out_args_2nd_pass = [out_args[0]]
    cdef Py_ssize_t contiguous_size, out_block_num
    cdef function.Function func
    cdef memory.MemoryPointer memptr
    cdef str post_map_expr1, post_map_expr2, f
    cdef list inout_args
    cdef tuple cub_params
    cdef size_t gridx, blockx
    cdef ndarray in_arr

    # fair share
    contiguous_size = min(segment_size, block_size * items_per_thread)
    out_block_num = (segment_size + contiguous_size - 1) // contiguous_size
    assert out_block_num <= 0x7fffffff

    # Because we can't know sizeof(reduce_type) in advance, here we
    # conservatively assume it's 32 bytes and allocate a work area
    memptr = memory.alloc(out_block_num * 32)
    out_args[0] = memptr

    # ************************ 1st pass ************************
    name += '_pass1'
    inout_args = [in_args[0], out_args[0],
                  _cub_convert_to_c_scalar(segment_size, contiguous_size),
                  _cub_convert_to_c_scalar(segment_size, segment_size)]
    cub_params = (items_per_thread,)

    if 'mean' in name:
        post_map_expr1 = post_map_expr.replace('_in_ind.size()', '1.0')
        post_map_expr1 = post_map_expr1.replace('_out_ind.size()', '1.0')
    elif any((f in name for f in ('argmax', 'argmin'))):
        # Workaround: in NumPy the indices are always generated based on
        # a C-order array (since PyArray_ContiguousFromAny was called).
        # We have to do a conversion here (?) since we do not retain the
        # info on strides.
        # TODO(leofang): improve this workaround
        in_arr = in_args[0]
        if in_arr.ndim > 1 and in_arr._f_contiguous:
            in_arr = _internal_ascontiguousarray(in_arr)
            inout_args[0] = in_args[0] = in_arr
        post_map_expr1 = post_map_expr
    else:
        post_map_expr1 = post_map_expr

    # Retrieve the kernel function
    func = _SimpleCubReductionKernel_get_cached_function(
        pre_map_expr, reduce_expr, post_map_expr1, reduce_type,
        params,
        _kernel._get_arginfos(inout_args),
        type_map,
        name, block_size, identity, preamble,
        ('-DFIRST_PASS=1',), cub_params)

    # Kernel arguments passed to the __global__ function.
    gridx = <size_t>(out_block_num * block_size)
    blockx = <size_t>block_size

    # Launch the kernel
    func.linear_launch(gridx, inout_args, 0, blockx, stream)

    # ************************ 2nd pass ************************
    name = name[:-1] + '2'
    contiguous_size = out_block_num
    out_block_num = 1
    in_args = out_args
    out_args = out_args_2nd_pass
    inout_args = [in_args[0], out_args[0],
                  _cub_convert_to_c_scalar(segment_size, contiguous_size),
                  _cub_convert_to_c_scalar(segment_size, segment_size)]

    # For mean()
    if 'mean' in name:
        post_map_expr2 = post_map_expr.replace('_in_ind.size()',
                                               '_array_size')
        post_map_expr2 = post_map_expr2.replace('_out_ind.size()', '1.0')
    else:
        post_map_expr2 = post_map_expr

    # Retrieve the kernel function
    func = _SimpleCubReductionKernel_get_cached_function(
        'in0', reduce_expr, post_map_expr2, reduce_type,
        params,
        _kernel._get_arginfos(inout_args),
        type_map,
        name, block_size, identity, preamble,
        ('-DSECOND_PASS=1',), cub_params)

    # Kernel arguments passed to the __global__ function.
    gridx = <size_t>(out_block_num * block_size)
    blockx = <size_t>block_size

    # Launch the kernel
    func.linear_launch(gridx, inout_args, 0, blockx, stream)


cdef inline void _launch_cub(
        self, out_block_num, block_size, block_stride,
        in_args, out_args, in_shape, out_shape, type_map,
        map_expr, reduce_expr, post_map_expr, reduce_type,
        stream, params, cub_params) except *:
    cdef bint full_reduction
    cdef Py_ssize_t contiguous_size, items_per_thread
    cdef function.Function func

    # Kernel arguments passed to the __global__ function.
    items_per_thread = cub_params[0]
    contiguous_size = cub_params[1]
    full_reduction = cub_params[2]

    if full_reduction:
        _cub_two_pass_launch(
            self.name, block_size, contiguous_size, items_per_thread,
            reduce_type, params, in_args, out_args, self.identity,
            map_expr, reduce_expr, post_map_expr,
            type_map, self.preamble, (), stream)
        return
    else:
        inout_args = (
            in_args + out_args +
            [_cub_convert_to_c_scalar(
                contiguous_size, contiguous_size),
             _cub_convert_to_c_scalar(
                 contiguous_size, 0)])
        arginfos = _kernel._get_arginfos(inout_args)
        func = _SimpleCubReductionKernel_get_cached_function(
            map_expr, reduce_expr, post_map_expr, reduce_type,
            params, arginfos, type_map,
            self.name, block_size, self.identity, self.preamble,
            (), cub_params)

        func.linear_launch(
            out_block_num * block_size, inout_args, 0, block_size, stream)


def _get_cub_optimized_params(
        self, optimize_config, in_args, out_args, in_shape, out_shape,
        type_map, map_expr, reduce_expr, post_map_expr, reduce_type,
        stream, full_reduction, out_block_num, contiguous_size, params):
    out_size = internal.prod(out_shape)
    in_args = [_reduction._optimizer_copy_arg(a) for a in in_args]
    out_args = [_reduction._optimizer_copy_arg(a) for a in out_args]

    items_per_thread, block_size = (
        _get_cub_block_specs(contiguous_size))
    default_block_size_log = math.floor(math.log2(block_size))
    default_items_per_thread = items_per_thread

    def target_func(block_size, items_per_thread):
        block_stride = block_size * items_per_thread
        cub_params = (
            items_per_thread, contiguous_size, full_reduction)
        _launch_cub(
            self,
            out_block_num, block_size, block_stride, in_args, out_args,
            in_shape, out_shape, type_map, map_expr, reduce_expr,
            post_map_expr, reduce_type, stream, params, cub_params)

    def suggest_func(trial):
        block_size_log = trial.suggest_int('block_size_log', 5, 10)
        block_size = 2 ** block_size_log
        items_per_thread = trial.suggest_int(
            'items_per_thread', 2, 32, step=2)

        trial.set_user_attr('block_size', block_size)
        return block_size, items_per_thread

    # CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES is a possible error
    optimize_impl = optimize_config.optimize_impl
    best = optimize_impl(
        optimize_config, target_func, suggest_func,
        default_best={
            'block_size_log': default_block_size_log,
            'items_per_thread': default_items_per_thread,
        }, ignore_error=(driver.CUDADriverError,))

    return best.params['items_per_thread'], best.user_attrs['block_size']


cdef bint _try_to_call_cub_reduction(
        self, list in_args, list out_args, const shape_t& a_shape,
        stream, optimize_context, tuple key,
        map_expr, reduce_expr, post_map_expr,
        reduce_type, _kernel._TypeMap type_map,
        tuple reduce_axis, tuple out_axis, const shape_t& out_shape,
        ndarray ret) except *:
    """Try to use cub.

    Updates `ret` and returns a boolean value whether cub is used.

    Note: input_expr and output_expr are not used in CUB kernels.
    """
    cdef tuple axis_permutes
    cdef tuple params, opt_params
    cdef shape_t in_shape
    cdef Py_ssize_t i
    cdef Py_ssize_t contiguous_size = -1
    cdef Py_ssize_t block_size, block_stride, out_block_num = 0

    # decide to use CUB or not
    can_use_cub = _can_use_cub_block_reduction(
        in_args, out_args, reduce_axis, out_axis)

    if can_use_cub is None:
        return False

    axis_permutes, contiguous_size, full_reduction = can_use_cub

    in_shape = _reduction._set_permuted_args(
        in_args, axis_permutes, a_shape, self.in_params)

    if in_args[0]._f_contiguous:
        ret._set_contiguous_strides(ret.dtype.itemsize, False)
        out_args[0] = ret

    if not full_reduction:  # just need one pass
        out_block_num = 1  # = number of segments
        for i in out_axis:
            out_block_num *= in_shape[i]

        if 'mean' in self.name:
            post_map_expr = post_map_expr.replace(
                '_in_ind.size()', '_segment_size')
            post_map_expr = post_map_expr.replace(
                '_out_ind.size()', '1.0')

    if contiguous_size > 0x7fffffff:  # INT_MAX
        size_type = 'uint64'
    else:
        size_type = 'int32'
    type_map = _kernel._TypeMap(type_map._pairs + (('sizeT', size_type),))
    params = (self._params[0:2]
              + _get_param_info(size_type + ' _segment_size', True)
              + _get_param_info(size_type + ' _array_size', True))

    # HACK for ReductionKernel:
    # 1. input/output arguments might not be named as in0/out0
    # 2. pre-/post- maps might not contain in0/out0
    # 3. type_map does not contain the expected names (type_in0_raw and
    #    type_out0_raw)
    cdef str old_in0 = params[0].name, old_out0 = params[1].name
    if old_in0 != 'in0' or old_out0 != 'out0':
        # avoid overwriting self's attributes
        params = (_get_param_info('T in0', True)
                  + _get_param_info('T out0', False)
                  + params[2:])
        map_expr = map_expr.replace(old_in0, 'in0')
        post_map_expr = post_map_expr.replace(old_out0, 'out0')
        type_map = _kernel._TypeMap(type_map._pairs + (
            ('type_in0_raw', in_args[0].dtype.type),
            ('type_out0_raw', out_args[0].dtype.type),
        ))

    # Calculate the reduction block dimensions.
    optimize_context = _optimize_config.get_current_context()
    if optimize_context is None:
        # Calculate manually
        items_per_thread, block_size = _get_cub_block_specs(contiguous_size)
    else:
        # Optimize dynamically
        key = ('cub_reduction',) + key
        opt_params = optimize_context.get_params(key)
        if opt_params is None:
            opt_params = _get_cub_optimized_params(
                self,
                optimize_context.config, in_args, out_args,
                in_shape, out_shape, type_map, map_expr, reduce_expr,
                post_map_expr, reduce_type, stream,
                full_reduction, out_block_num, contiguous_size, params)
            optimize_context.set_params(key, opt_params)
        items_per_thread, block_size = opt_params

    block_stride = block_size * items_per_thread
    cub_params = (items_per_thread, contiguous_size, full_reduction)

    _launch_cub(
        self,
        out_block_num,
        block_size,
        block_stride,
        in_args, out_args,
        in_shape, out_shape,
        type_map,
        map_expr, reduce_expr, post_map_expr, reduce_type,
        stream, params, cub_params)

    return True
