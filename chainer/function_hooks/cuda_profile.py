from chainer import cuda
from chainer import function

if cuda.available:
    from cupy.cuda import nvtx


class CUDAProfileHook(function.FunctionHook):

    name = 'CUDAProfileHook'

    def __init__(self):
        cuda.check_cuda_available()

    def forward_preprocess(self, function, in_data):
        nvtx.RangePush(function.label + '.forward')

    def forward_postprocess(self, function, in_data):
        nvtx.RangePop()

    def backward_preprocess(self, function, in_data, out_grad):
        nvtx.RangePush(function.label + '.backward')

    def backward_postprocess(self, function, in_data, out_grad):
        nvtx.RangePop()
