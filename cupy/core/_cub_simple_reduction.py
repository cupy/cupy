import string

from cupy.core import _kernel
from cupy.core import _reduction


def _get_cub_reduction_function_code(
        name, block_size, items_per_thread,
        reduce_type, params, arginfos, identity,
        pre_map_expr, reduce_expr, post_map_expr,
        type_map, input_expr, output_expr, preamble, options):
    # TODO: clean up
    # TODO: remove input_expr & output_expr

    # For mean()
    post_map_expr = post_map_expr.replace('_in_ind.size()', '_segment_size')
    post_map_expr = post_map_expr.replace('_out_ind.size()', '1.0')

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
    #define POST_MAP(a) (${post_map_expr})
#else  // one-pass reduction
    typedef type_in0_raw  type_mid_in;
    typedef type_out0_raw type_mid_out; 
    #define POST_MAP(a) (${post_map_expr})
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
  //printf("%d\\n", _bid);

  //if (_bid == 0)
  //  printf("%p, %p, %i\\n", _raw_in0, _raw_out0, _segment_size);



  // Specialize BlockReduce type for our thread block
  typedef cub::BlockReduce<_type_reduce, BLOCK_SIZE> BlockReduceT;

  // Shared memory
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  typedef cub::BlockLoad<_type_reduce, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_DIRECT> BlockLoadT;

  __shared__ typename BlockLoadT::TempStorage temp_storage_load;

  // Declare reduction operation
  _reduction_op op;

  // input & output raw pointers
  const type_mid_in* _in0 = static_cast<const type_mid_in*>(_raw_in0);
  type_mid_out* _out0 = static_cast<type_mid_out*>(_raw_out0);

  // Per-thread tile data
  _type_reduce _sdata[ITEMS_PER_THREAD] = {_type_reduce(${identity})};

  // each block handles the reduction of 1 segment
  const type_mid_in* segment_head = _in0 + blockIdx.x * _segment_size;  // TODO(leofang): auto-gen this
  _type_reduce aggregate = _type_reduce(${identity});
  size_t i = 0;  // tile head within the segment
  int tile_size = (BLOCK_SIZE * ITEMS_PER_THREAD < _segment_size ? BLOCK_SIZE * ITEMS_PER_THREAD : _segment_size);

  // loop over tiles within 1 segment
  for (i = 0; i < _segment_size; i += BLOCK_SIZE * ITEMS_PER_THREAD) {
      // TODO: try splitting the for loop into full tiles and partil tiles to utilize
      // LoadDirectBlockedVectorized? See, for example,
      // https://github.com/NVlabs/cub/blob/c3cceac115c072fb63df1836ff46d8c60d9eb304/cub/agent/agent_reduce.cuh#L311-L346

      if (_segment_size - i < tile_size)  // for the last tile
          tile_size = _segment_size - i;
'''

    if pre_map_expr == 'in0':
        module_code += '''

      //if (_bid == 0) {
      //    printf("i: %d\\n", i);
      //    printf("_segment_size: %d\\n", _segment_size);
      //    printf("tile_size: %d\\n\\n", tile_size);
      //}

      // load a tile
      // This is equivalent to cub::BlockLoad<_type_reduce, BLOCK_SIZE, ITEMS_PER_THREAD, BLOCK_LOAD_DIRECT>::Load
      //cub::LoadDirectBlocked(_tid, segment_head + i, _sdata, tile_size, _type_reduce(${identity}));
      BlockLoadT(temp_storage_load).Load(segment_head + i, _sdata, tile_size, _type_reduce(${identity}));

      //for (size_t i = 0; i<ITEMS_PER_THREAD; i++)
      //    printf("_bid: %d, local items: %f\\n", _bid, _sdata[i]); 
'''
    else:  # pre_map_expr could be something like "in0 != type_in0_raw(0)"
        module_code += '''
      // load a tile
      #pragma unroll
      for (int j = 0; j < ITEMS_PER_THREAD; j++) {
          _sdata[j] = _type_reduce(${identity});
          // some pre_map_expr uses _J internally...
          #if defined FIRST_PASS
          int _J = (blockIdx.x * _segment_size + i + _tid * ITEMS_PER_THREAD + j);
          #elif defined SECOND_PASS
          //do nothing
          #else
          int _J = (blockIdx.x * _segment_size + i + _tid * ITEMS_PER_THREAD + j) % _segment_size;
          #endif

          if ((_tid * ITEMS_PER_THREAD) + j < tile_size)
          {
              const type_mid_in in0 = *(segment_head + i + _tid * ITEMS_PER_THREAD + j);
              #ifndef SECOND_PASS
              _sdata[j] = static_cast<_type_reduce>(${pre_map_expr});
              #else
              _sdata[j] = in0;
              #endif
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
      //printf("_bid: %d (blockIdx.x: %d), block out: %f\\n", _bid, blockIdx.x, aggregate);

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
        #params=_kernel._get_kernel_params(params, arginfos),  # used
        params=_reduction._get_cub_kernel_params(params, arginfos),  # used
        identity=identity,  # used
        reduce_expr=reduce_expr,  # used
        pre_map_expr=pre_map_expr,  # used
        post_map_expr=post_map_expr,  # used
        type_preamble=type_map.get_typedef_code(),  # used
        input_expr=input_expr,  # used
        output_expr=output_expr,  # used
        preamble=preamble)  # used
    #print('\n', module_code, '\n')

    return module_code
