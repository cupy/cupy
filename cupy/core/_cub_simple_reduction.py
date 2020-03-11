
import string

from cupy.core import _kernel


def _get_cub_reduction_function_code(
        name, block_size, reduce_type, params, arginfos, identity,
        pre_map_expr, reduce_expr, post_map_expr,
        type_map, input_expr, output_expr, preamble, options):
    print("**************** I AM HERE ******************")
    module_code = string.Template('''
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_load.cuh>
#include <inttypes.h>
${type_preamble}
${preamble}

//TODO(leofang): this should be auto-tuned based on CUDA arch?
#define ITEMS_PER_THREAD 4

#define POST_MAP(a) (${post_map_expr})

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

  unsigned int _bid = blockIdx.x;
  unsigned int _id = blockIdx.x * blockDim.x + threadIdx.x;

  // Specialize BlockReduce type for our thread block
  typedef cub::BlockReduce<_type_reduce, ${block_size}> BlockReduceT;

  // Shared memory
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  // Attempt: ignore CIndexer _in_ind and _out_ind?
  const int in_ndim = _raw_in0.ndim;
  const int out_ndim = _raw_out0.ndim;
  const ptrdiff_t* in_shape = _raw_in0.shape();
  const ptrdiff_t* out_shape = _raw_out0.shape();
  const ptrdiff_t* in_strides = _raw_in0.strides();
  const ptrdiff_t* out_strides = _raw_out0.strides();
  if (_id == 0) {
    printf("%i %i\\n", in_ndim, out_ndim);
    for(int i = 0; i<in_ndim; i++) {
       printf("%d\\n", in_shape[i]);
    }

    for(int i = 0; i<in_ndim; i++) {
       printf("%d\\n", in_strides[i]);
    }
    for(int i = 0; i<out_ndim; i++) {
       printf("%d\\n", out_shape[i]);
    }

    for(int i = 0; i<out_ndim; i++) {
       printf("%d\\n", out_strides[i]);
    }
  }

  // Per-thread tile data
  _type_reduce _sdata[ITEMS_PER_THREAD];
  cub::LoadDirectStriped<${block_size}>(threadIdx.x, (_type_reduce*)&(_raw_in0[0]), _sdata);

  // Declare reduction operation
  _reduction_op op;

  // Compute reduction
  _type_reduce aggregate = BlockReduceT(temp_storage).Reduce(_sdata, op);

//  if (_tid < _block_stride && _i < _out_ind.size()) {
//    _out_ind.set(static_cast<ptrdiff_t>(_i));
//    ${output_expr}
//    POST_MAP(_s);
//  }
}
} // extern C''').substitute(
        name=name,  # used
        block_size=block_size,  # used
        reduce_type=reduce_type,  # used
        params=_kernel._get_kernel_params(params, arginfos),  # used
        identity=identity,
        reduce_expr=reduce_expr,  # used
        pre_map_expr=pre_map_expr,
        post_map_expr=post_map_expr,  # used
        type_preamble=type_map.get_typedef_code(),  # used
        input_expr=input_expr,
        output_expr=output_expr,
        preamble=preamble)  # used
    print('\n', module_code, '\n')
    return module_code
