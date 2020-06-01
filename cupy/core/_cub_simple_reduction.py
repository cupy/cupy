import string

from cupy.core import _kernel
from cupy.core import _reduction


def _get_cub_reduction_function_code(
        name, block_size, items_per_thread,
        reduce_type, params, arginfos, identity,
        pre_map_expr, reduce_expr, post_map_expr,
        type_map, preamble, options):
    #  TODO: try splitting the for loop into full tiles and partil tiles to utilize
    #  LoadDirectBlockedVectorized? See, for example,
    #  https://github.com/NVlabs/cub/blob/c3cceac115c072fb63df1836ff46d8c60d9eb304/cub/agent/agent_reduce.cuh#L311-L346

    module_code = '''
#include <cupy/cub/cub/block/block_reduce.cuh>
#include <cupy/cub/cub/block/block_load.cuh>

${preamble}

${type_preamble}
typedef ${reduce_type} _type_reduce;

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

// Compile-time constants for CUB template specializations
#define ITEMS_PER_THREAD ${items_per_thread}
#define BLOCK_SIZE ${block_size}

struct _reduction_op
{
    __device__ __forceinline__ _type_reduce operator()(const _type_reduce &a, const _type_reduce &b) const
    {
        return ${reduce_expr};
    }
};

extern "C"
__global__ void ${name}(${params}) {
  unsigned int _tid = threadIdx.x;
  unsigned int _bid = blockIdx.x * BLOCK_SIZE + _tid;

  // Specialize BlockReduce type for our thread block
  typedef cub::BlockReduce<_type_reduce, BLOCK_SIZE> BlockReduceT;

  // Shared memory
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  // Declare reduction operation
  _reduction_op op;

  // input & output raw pointers
  const type_mid_in* _in0 = static_cast<const type_mid_in*>(_raw_in0);
  type_mid_out* _out0 = static_cast<type_mid_out*>(_raw_out0);

  // Per-thread tile data
  //_type_reduce _sdata[ITEMS_PER_THREAD] = {_type_reduce(${identity})};
  _type_reduce _sdata[ITEMS_PER_THREAD];
  for (int j = 0; j < ITEMS_PER_THREAD; j++) { _sdata[j] = _type_reduce(${identity}); }
  //__syncthreads();
  //for (int j = 0; j < ITEMS_PER_THREAD; j++) { printf("%s, before: %i, %lld\\n", __func__, _bid, _sdata[j]); }
  //__syncthreads();

  // each block handles the reduction of 1 segment
  const type_mid_in* segment_head = _in0 + blockIdx.x * _segment_size;
  _type_reduce aggregate = _type_reduce(${identity});
  size_t i = 0;  // tile head within the segment
  int tile_size = (BLOCK_SIZE * ITEMS_PER_THREAD < _segment_size ? BLOCK_SIZE * ITEMS_PER_THREAD : _segment_size);

  #if defined FIRST_PASS
  // for two-pass reduction only: "last segment" is special
  if (_array_size > 0) {
      _segment_size = (_array_size - blockIdx.x * _segment_size <= _segment_size ?
                       _array_size - blockIdx.x * _segment_size :
                       _segment_size);
  }
  #endif

  // loop over tiles within 1 segment
  for (i = 0; i < _segment_size; i += BLOCK_SIZE * ITEMS_PER_THREAD) {
      if (_tid == 0) printf("At tile: %i, i=%llu\\n", blockIdx.x, i);
      if (_segment_size - i <= tile_size) { // for the last tile
          tile_size = _segment_size - i;
          //if (_tid == 0) printf("last tile: i=%llu\\n", i);
      }
'''

#    if pre_map_expr == 'in0':
#        module_code += '''
#      typedef cub::BlockLoad<_type_reduce, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_DIRECT> BlockLoadT;
#
#      __shared__ typename BlockLoadT::TempStorage temp_storage_load;
#
#      // load a tile
#      for (int j = 0; j < ITEMS_PER_THREAD; j++) { printf("%s, before: %i, %i\\n", __func__, _bid, _sdata[j]); }
#      BlockLoadT(temp_storage_load).Load(segment_head + i, _sdata, tile_size, _type_reduce(${identity}));
#      for (int j = 0; j < ITEMS_PER_THREAD; j++) { printf("after: %i, %i\\n", _bid, _sdata[j]); }
#'''
#    else:  # pre_map_expr could be something like "in0 != type_in0_raw(0)"
#        module_code += '''
    module_code += '''
      // load a tile
      #pragma unroll
      for (int j = 0; j < ITEMS_PER_THREAD; j++) {
          // some pre_map_expr uses _J internally...
          #if defined FIRST_PASS
          int _J = (blockIdx.x * _segment_size + i + _tid * ITEMS_PER_THREAD + j);
          #else  // only one pass
          int _J = (blockIdx.x * _segment_size + i + _tid * ITEMS_PER_THREAD + j) % _segment_size;
          #endif

          if ((_tid * ITEMS_PER_THREAD) + j < tile_size) {
              const type_mid_in in0 = *(segment_head + i + _tid * ITEMS_PER_THREAD + j);
              //const type_mid_in in0 = segment_head[i + _tid * ITEMS_PER_THREAD + j];
              _sdata[j] = static_cast<_type_reduce>(${pre_map_expr});
              printf("%s, element %lld loads %lld\\n", __func__, blockIdx.x * _segment_size + i + _tid * ITEMS_PER_THREAD + j, _sdata[j]);
          } else { 
              _sdata[j] = _type_reduce(${identity});
              printf("%s, element %lld resets %lld\\n", __func__, blockIdx.x * _segment_size + i + _tid * ITEMS_PER_THREAD + j, _sdata[j]);
          }
          //printf("%s, after: %i, %lld\\n", __func__, _bid, _sdata[j]);
          //printf("%s, element %lld: %lld\\n", __func__, blockIdx.x * _segment_size + i + _tid * ITEMS_PER_THREAD + j, _sdata[j]);
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
        name=name,  # used
        block_size=block_size,  # used
        items_per_thread=items_per_thread,  # used
        reduce_type=reduce_type,  # used
        params=_reduction._get_cub_kernel_params(params, arginfos),  # used
        identity=identity,  # used
        reduce_expr=reduce_expr,  # used
        pre_map_expr=pre_map_expr,  # used
        post_map_expr=post_map_expr,  # used
        type_preamble=type_map.get_typedef_code(),  # used
        preamble=preamble)  # used

    return module_code
