class CudaObject:
    def __init__(self, s):
        self.s = s


threadIdx = CudaObject('threadIdx')
blockDim = CudaObject('blockDim')
blockIdx = CudaObject('blockIdx')
gridDim = CudaObject('gridDim')
sqrt = CudaObject('std::sqrt')
power = CudaObject('std::pow')
expr = CudaObject('std::expr')
log = CudaObject('std::log')

syncthreads = CudaObject('__syncthreads')


def shared_memory(shape, dtype):
    raise NotImplementedError('')


class CudaFunction:
    def __init__(self, func, attributes):
        self.func = func
        self.attributes = attributes


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
