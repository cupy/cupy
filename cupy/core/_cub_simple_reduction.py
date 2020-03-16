
import string

from cupy.core import _kernel
from cupy.core import _reduction


def _get_cub_reduction_function_code(
        name, block_size, segment_size, items_per_thread,
        reduce_type, params, arginfos, identity,
        pre_map_expr, reduce_expr, post_map_expr,
        type_map, input_expr, output_expr, preamble, options):
    # TODO: clean up
    module_code = '''
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_load.cuh>

${type_preamble}
${preamble}

#define ITEMS_PER_THREAD ${items_per_thread}
#define POST_MAP(a) (${post_map_expr})
#define SEGMENT_SIZE ${segment_size}

typedef ${reduce_type} _type_reduce;

struct _reduction_op
{
    __device__ __forceinline__ _type_reduce operator()(const _type_reduce &a, const _type_reduce &b) const
    {
        return ${reduce_expr};
    }
};

extern "C" {
__global__ void ${name}(${params}) {
  unsigned int _tid = threadIdx.x;
  unsigned int _bid = blockIdx.x * blockDim.x + _tid;
  //printf("%d\\n", _bid);

  // Specialize BlockReduce type for our thread block
  typedef cub::BlockReduce<_type_reduce, ${block_size}> BlockReduceT;

  // Shared memory
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  // Declare reduction operation
  _reduction_op op;

  // input & output raw pointers
  // TODO(leofang): auto-gen these
  const type_in0_raw* _in0 = static_cast<const type_in0_raw*>(_raw_in0);
  type_out0_raw* _out0 = static_cast<type_out0_raw*>(_raw_out0);

  // Per-thread tile data
  _type_reduce _sdata[ITEMS_PER_THREAD] = {${identity}};

  // each block handles the reduction of 1 segment
  const type_in0_raw* segment_head = _in0 + blockIdx.x * SEGMENT_SIZE;  // TODO(leofang): auto-gen this
  _type_reduce aggregate = ${identity};
  size_t i = 0;  // tile head within the segment
  int tile_size = (blockDim.x * ITEMS_PER_THREAD < SEGMENT_SIZE ? blockDim.x * ITEMS_PER_THREAD : SEGMENT_SIZE); 

  // loop over tiles within 1 segment
  for (i = 0; i < SEGMENT_SIZE; i += blockDim.x * ITEMS_PER_THREAD) {
      // TODO: try splitting the for loop into full tiles and partil tiles to utilize
      // LoadDirectBlockedVectorized? See, for example,
      // https://github.com/NVlabs/cub/blob/c3cceac115c072fb63df1836ff46d8c60d9eb304/cub/agent/agent_reduce.cuh#L311-L346

      if (SEGMENT_SIZE - i < tile_size)  // for the last tile
          tile_size = SEGMENT_SIZE - i;
'''

    if pre_map_expr == 'in0':
        module_code += '''

      //if (_bid == 0) {
      //    printf("i: %d\\n", i);
      //    printf("SEGMENT_SIZE: %d\\n", SEGMENT_SIZE);
      //    printf("tile_size: %d\\n\\n", tile_size);
      //}

      // load a tile
      // This is equivalent to cub::BlockLoad<_type_reduce, ${block_size}, ITEMS_PER_THREAD, BLOCK_LOAD_DIRECT>::Load
      cub::LoadDirectBlocked(_tid, segment_head + i, _sdata, tile_size, ${identity});

      //for (size_t i = 0; i<ITEMS_PER_THREAD; i++)
      //    printf("_bid: %d, local items: %f\\n", _bid, _sdata[i]); 
'''
    else:  # pre_map_expr could be something like "in0 != type_in0_raw(0)"
        module_code += '''
      // load a tile
      #pragma unroll
      for (int j = 0; j < ITEMS_PER_THREAD; j++) {
          _sdata[j] = ${identity};

          if ((_tid * ITEMS_PER_THREAD) + j < tile_size)
          {
              const type_in0_raw in0 = *(segment_head + i + _tid * ITEMS_PER_THREAD + j);
              _sdata[j] = static_cast<_type_reduce>(${pre_map_expr});
          }
      }
'''        

    module_code += '''
      // Compute block reduction
      // Note that the output is only meaningful for thread 0
      aggregate += BlockReduceT(temp_storage).Reduce(_sdata, op);

      __syncthreads();  // for reusing temp_storage
  }

  if (_tid == 0) {
      //printf("_bid: %d (blockIdx.x: %d), block out: %f\\n", _bid, blockIdx.x, aggregate);

      type_out0_raw& out0 = *(_out0 + blockIdx.x);
      POST_MAP(aggregate);
  }
} // kernel end
} // extern C
'''
    module_code = string.Template(module_code).substitute(
        name=name,  # used
        block_size=block_size,  # used
        segment_size=segment_size,  # used
        items_per_thread=items_per_thread,  # used
        reduce_type=reduce_type,  # used
        #params=_kernel._get_kernel_params(params, arginfos),  # used
        params=_reduction._get_cub_kernel_params(params, arginfos),  # used
        identity=identity,  # used
        reduce_expr=reduce_expr,  # used
        pre_map_expr=pre_map_expr,  # used
        post_map_expr=post_map_expr,  # used
        type_preamble=type_map.get_typedef_code(),  # used
        input_expr=input_expr,
        output_expr=output_expr,
        preamble=preamble)  # used
    print('\n', module_code, '\n')
    return module_code
