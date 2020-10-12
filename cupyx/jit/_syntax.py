import cupy.core
from cupyx.jit import _compile
from cupyx.jit import _types


class CudaObject:
    def __init__(self, s):
        self._s = s


threadIdx = CudaObject('threadIdx')
blockDim = CudaObject('blockDim')
blockIdx = CudaObject('blockIdx')
gridDim = CudaObject('gridDim')
syncthreads = CudaObject('__syncthreads')




def shared_memory(shape, dtype):
    raise NotImplementedError('')


class CudaFunction:
    def __init__(self, func, attributes):
        self.func = func
        self.attributes = attributes
        self._kernels = {}

    def __call__(self, grid, block, args, **kwargs):
        # TODO(asi1024): Cache kernel
        in_types = [_types._type_from_obj(x) for x in args]
        code = self.emit_code_from_types(in_types, _types.Void())
        # print(code)
        kern = cupy.core.RawKernel(code, 'kernel')
        kern(grid, block, args, **kwargs)

    def emit_code_from_types(self, in_types, out_type):
        return _compile.transpile(
            self.func, self.attributes, in_types, out_type)


def cuda_function(device=False, inline=False):
    def wrapper(func):
        attributes = []

        if device:
            attributes.append('__device__')
        else:
            attributes.append('__global__')

        if inline:
            attributes.append('inline')

        return CudaFunction(func, attributes)

    return wrapper
