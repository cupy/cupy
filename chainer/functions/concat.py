import numpy
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from pytools import memoize
from chainer import Function

_args = 'const float* x, float* y, int cdimx, int cdimy, int rdim, int coffset'
_preamble = '''
#define COPY(statement) \
    int l   = i / (rdim * cdimx);  \
    int c   = i / rdim % cdimx + coffset;  \
    int r   = i % rdim;  \
    int idx = r + rdim * (c + cdimy * l);  \
    statement;
'''

@memoize
def _copy_x_to_y_kernel():
    return ElementwiseKernel(_args, 'COPY(y[idx] = x[i])', preamble=_preamble)

@memoize
def _copy_y_to_x_kernel():
    return ElementwiseKernel(_args, 'COPY(x[i] = y[idx])', preamble=_preamble)

class Concat(Function):
    """Concatenate multiple tensors towards specified axis."""

    def __init__(self, axis=1):  # concat along the channel dimension by default
        self.axis = axis

    def forward_cpu(self, xs):
        return numpy.concatenate(xs, axis=self.axis),

    def forward_gpu(self, xs):
        # TODO(beam2d): Unify the process into a single kernel.
        shape = list(xs[0].shape)
        for x in xs[1:]:
            shape[self.axis] += x.shape[self.axis]
        self.shape = shape

        y = gpuarray.empty(shape, dtype=xs[0].dtype)
        self.cdimy = y.shape[self.axis]
        self.rdim  = numpy.prod(shape[self.axis + 1:])

        coffset = 0
        kernel  = _copy_x_to_y_kernel()
        for x in xs:
            cdimx = x.shape[self.axis]
            kernel(x, y, cdimx, self.cdimy, self.rdim, coffset)
            coffset += cdimx

        return y,

    def backward_cpu(self, xs, gy):
        sizes = numpy.array([x.shape[self.axis] for x in xs[1:]]).cumsum()
        return numpy.split(gy, sizes, axis=self.axis)

    def backward_gpu(self, xs, gy):
        gxs = (gpuarray.empty_like(x) for x in xs)

        coffset = 0
        kernel  = _copy_y_to_x_kernel()
        for gx in gxs:
            cdimx = gx.shape[self.axis]
            kernel(gx, gy[0], cdimx, self.cdimy, self.rdim, coffset)
            coffset += cdimx

        return gxs


def concat(xs, axis=1):
    return Concat(axis=axis)(*xs)
