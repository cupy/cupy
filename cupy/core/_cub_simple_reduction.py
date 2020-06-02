import string

from cupy.core import _kernel
from cupy.core import _reduction


def _get_cub_reduction_function_code(
        name, block_size, items_per_thread,
        reduce_type, params, arginfos, identity,
        pre_map_expr, reduce_expr, post_map_expr,
        type_map, preamble, options):
    # TODO(leofang): try splitting the for-loop into full tiles and partial
    # tiles to utilize LoadDirectBlockedVectorized? See, for example,
    # https://github.com/NVlabs/cub/blob/c3cceac115c072fb63df1836ff46d8c60d9eb304/cub/agent/agent_reduce.cuh#L311-L346

    module_code = '''
#include <cupy/cub/cub/block/block_reduce.cuh>
#include <cupy/cub/cub/block/block_load.cuh>

${preamble}

${type_preamble}
typedef ${reduce_type} _type_reduce;

// Compile-time constants for CUB template specializations
#define ITEMS_PER_THREAD ${items_per_thread}
#define BLOCK_SIZE ${block_size}

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
  unsigned int _bid = blockIdx.x * BLOCK_SIZE + _tid;
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
  for (int j = 0; j < ITEMS_PER_THREAD; j++) {
      _sdata[j] = _type_reduce(${identity});
  }

  // each block handles the reduction of 1 segment
  size_t segment_id = blockIdx.x * _segment_size;
  const type_mid_in* segment_head = _in0 + segment_id;
  size_t i = 0;  // tile head within the segment
  int tile_size = (BLOCK_SIZE * ITEMS_PER_THREAD < _segment_size ?
                   BLOCK_SIZE * ITEMS_PER_THREAD :
                   _segment_size);

  #if defined FIRST_PASS
  // for two-pass reduction only: "last segment" is special
  if (_array_size > 0) {
      if (_array_size - segment_id <= _segment_size) {
          _segment_size = _array_size - segment_id;
      }
  }
  #endif

  // loop over tiles within 1 segment
  _type_reduce aggregate = _type_reduce(${identity});
  for (i = 0; i < _segment_size; i += BLOCK_SIZE * ITEMS_PER_THREAD) {
      // for the last tile
      if (_segment_size - i <= tile_size) {
          tile_size = _segment_size - i;
      }
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
          int _J = (segment_id + i + e_idx);
          #else  // only one pass
          int _J = (segment_id + i + e_idx) % _segment_size;
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
        params=_reduction._get_cub_kernel_params(params, arginfos),
        identity=identity,
        reduce_expr=reduce_expr,
        pre_map_expr=pre_map_expr,
        post_map_expr=post_map_expr,
        type_preamble=type_map.get_typedef_code(),
        preamble=preamble)

    return module_code
