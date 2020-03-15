
import string

from cupy.core import _kernel


def _get_cub_reduction_function_code(
        name, block_size, segment_size, items_per_thread,
        reduce_type, params, arginfos, identity,
        pre_map_expr, reduce_expr, post_map_expr,
        type_map, input_expr, output_expr, preamble, options):
    # TODO: implement for-loop load
    # TODO: see if we can do, say, 4 segments per block? (to increase write throughput)
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

//  const int in_ndim = _raw_in0.ndim;
//  const int out_ndim = _raw_out0.ndim;
//  const ptrdiff_t* in_shape = _raw_in0.shape();
//  const ptrdiff_t* out_shape = _raw_out0.shape();
//  const ptrdiff_t* in_strides = _raw_in0.strides();
//  const ptrdiff_t* out_strides = _raw_out0.strides();
//  if (_bid == 0) {
//    printf("%i %i\\n(", in_ndim, out_ndim);
//    for(int i = 0; i<in_ndim; i++) {
//       printf("%d, ", in_shape[i]);
//    }
//    printf("\\b\\b)\\n(");
//
//    for(int i = 0; i<in_ndim; i++) {
//       printf("%d, ", in_strides[i]);
//    }
//    printf("\\b\\b)\\n\\n(");
//
//    for(int i = 0; i<out_ndim; i++) {
//       printf("%d, ", out_shape[i]);
//    }
//    printf("\\b\\b)\\n(");
//
//    for(int i = 0; i<out_ndim; i++) {
//       printf("%d, ", out_strides[i]);
//    }
//    printf("\\b\\b)\\n");
//  }

  // Declare reduction operation
  _reduction_op op;

  // input & output raw pointers
  // TODO: relax the type requriments?
  _type_reduce* _in0 = (_type_reduce*)&(_raw_in0[0]);
  _type_reduce* _out0 = (_type_reduce*)&(_raw_out0[0]);

  // Per-thread tile data
  _type_reduce _sdata[ITEMS_PER_THREAD] = {${identity}};

  // each block handles the reduction of 1 segment
  _type_reduce _block_out = ${identity};
  size_t i = 0;  // tile head within the segment
  int tile_size = (blockDim.x * ITEMS_PER_THREAD < SEGMENT_SIZE ? blockDim.x * ITEMS_PER_THREAD : SEGMENT_SIZE); 
  for (i = 0; i < SEGMENT_SIZE; i += blockDim.x * ITEMS_PER_THREAD) {
'''

    # TODO: if map_expr == 'in0', use CUB load, else use for loop
    if pre_map_expr == 'in0':
        module_code += '''

      // TODO: try splitting the for loop into full tiles and partil tiles to utilize
      // LoadDirectBlockedVectorized? See, for example,
      // https://github.com/NVlabs/cub/blob/c3cceac115c072fb63df1836ff46d8c60d9eb304/cub/agent/agent_reduce.cuh#L311-L346

      if (SEGMENT_SIZE - i < tile_size)  // for the last tile
          tile_size = SEGMENT_SIZE - i;
      //if (_bid == 0) {
      //    printf("i: %d\\n", i);
      //    printf("SEGMENT_SIZE: %d\\n", SEGMENT_SIZE);
      //    printf("tile_size: %d\\n\\n", tile_size);
      //}
      cub::LoadDirectBlocked(_tid, _in0 + blockIdx.x * SEGMENT_SIZE + i, _sdata, tile_size, ${identity});

      //for (size_t i = 0; i<ITEMS_PER_THREAD; i++)
      //    printf("_bid: %d, local items: %f\\n", _bid, _sdata[i]); 
'''
    else:  # TODO
        raise NotImplementedError
        module_code += '''
'''        

    module_code += '''
      // Compute block reduction
      _type_reduce aggregate = BlockReduceT(temp_storage).Reduce(_sdata, op);

      if (_tid == 0)
          _block_out = op(_block_out, aggregate);

      __syncthreads();  // for reusing temp_storage
  }
'''

#    # TODO: if map_expr == 'in0', use CUB load, else use for loop
#    if pre_map_expr == 'in0':
#        module_code += '''
#//  if (i  // last tile
#//  cub::LoadDirectBlocked(_tid, _in0 + blockIdx.x * SEGMENT_SIZE + i, _sdata, SEGMENT_SIZE - i);
#//
#//  //for (size_t i = 0; i<ITEMS_PER_THREAD; i++)
#//  //    printf("_bid: %d, local items: %f\\n", _bid, _sdata[i]); 
#'''
#    else:  # TODO
#        raise NotImplementedError
#        module_code += '''
#'''        

    module_code += '''
//      // Compute block reduction
//      _type_reduce aggregate = BlockReduceT(temp_storage).Reduce(_sdata, op);
//
//      if (_tid == 0)
//          _block_out = op(_block_out, aggregate);
//  }

  if (_tid == 0) {
      //printf("_bid: %d (blockIdx.x: %d), block out: %f\\n", _bid, blockIdx.x, _block_out);

      type_out0_raw& out0 = *(_out0 + blockIdx.x);
      POST_MAP(_block_out);
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
        params=_kernel._get_kernel_params(params, arginfos),  # used
        identity=identity,  # used
        reduce_expr=reduce_expr,  # used
        pre_map_expr=pre_map_expr,
        post_map_expr=post_map_expr,  # used
        type_preamble=type_map.get_typedef_code(),  # used
        input_expr=input_expr,
        output_expr=output_expr,
        preamble=preamble)  # used
    print('\n', module_code, '\n')
    return module_code
