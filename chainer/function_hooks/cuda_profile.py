from chainer import cuda
from chainer import function


class CUDAProfileHook(function.FunctionHook):

    name = 'CUDAProfileHook'

    def __init__(self):
        cuda.check_cuda_available()
        if not cuda.cupy.cuda.nvtx_enabled:
            raise RuntimeError('nvtx is required for CUDAProfileHook')

    def forward_preprocess(self, function, in_data):
        cuda.cupy.cuda.nvtx.RangePush(function.label + '.forward')

    def forward_postprocess(self, function, in_data):
        cuda.cupy.cuda.nvtx.RangePop()

    def backward_preprocess(self, function, in_data, out_grad):
        cuda.cupy.cuda.nvtx.RangePush(function.label + '.backward')

    def backward_postprocess(self, function, in_data, out_grad):
        cuda.cupy.cuda.nvtx.RangePop()
