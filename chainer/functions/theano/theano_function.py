import numpy
import six

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


class TheanoFunction(function.Function):

    def __init__(self, forward_func, backward_func):
        utils.experimental('chainer.functions.TheanoFunction')
        self.forward_func = forward_func
        self.backward_func = backward_func

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == len(self.forward_func.indices))

        for actual_type, input_info in six.moves.zip(
                in_types, self.forward_func.indices):
            expect_type = input_info[0].variable.type
            # Theano cannot check shapes of variables
            type_check.expect(
                actual_type.ndim == expect_type.ndim,
                actual_type.dtype == expect_type.numpy_dtype,
            )

    def forward(self, inputs):
        gpu = cuda.get_array_module(*inputs) is not numpy
        inputs = [cuda.to_cpu(x) for x in inputs]

        outputs = self.forward_func(*inputs)

        if gpu:
            # TODO(unno): We can remove redundant gpu-cpu copy using
            # theano.sandbox.cuda.CudaNdarray.gpudata
            device = cuda.get_device(inputs)
            outputs = [cuda.to_gpu(x, device) for x in outputs]

        return tuple(outputs)

    def backward(self, inputs, grads):
        gpu = cuda.get_array_module(*inputs) is not numpy

        # TODO(unno): We can remove redundant gpu-cpu copy using
        # theano.sandbox.cuda.basic_ops.gpu_from_host
        args = [cuda.to_cpu(x) for x in inputs + grads]

        outputs = self.backward_func(*args)
        assert len(outputs) == len(inputs)

        if gpu:
            # TODO(unno): We can remove redundant gpu-cpu copy using
            # theano.sandbox.cuda.CudaNdarray.gpudata
            device = cuda.get_device(inputs)
            outputs = [cuda.to_gpu(x, device) for x in outputs]

        results = []
        for o, i in zip(outputs, inputs):
            if i.dtype.kind != 'f':
                o = None
            elif o.dtype != i.dtype:
                o = o.astype(i.dtype)
            results.append(o)
        return tuple(results)


def theano_function(forward_func, backward_func, *inputs):
    return TheanoFunction(forward_func, backward_func)(*inputs)
