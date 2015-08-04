import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

_args = 'const float* x, float* y, int cdimx, int cdimy, int rdim, int coffset'
_preamble = '''
#define COPY(statement) \
    int l   = i / (rdim * cdimx);  \
    int c   = i / rdim % cdimx + coffset;  \
    int r   = i % rdim;  \
    int idx = r + rdim * (c + cdimy * l);  \
    statement;
'''


class Concat(function.Function):

    """Concatenate multiple tensors towards specified axis."""

    # concat along the channel dimension by default
    def __init__(self, axis=1):
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)
        type_check.expect(in_types[0].ndim >
                          type_check.Variable(self.axis, 'axis'))

        ndim = in_types[0].ndim.eval()
        for i in range(1, in_types.size().eval()):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            for d in range(0, ndim):
                if d == self.axis:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def check_type_backward(self, in_types, out_types):
        type_check.expect(
            in_types.size() > 0,
            out_types.size() == 1,
        )
        y_type, = out_types

        type_check.expect(y_type.dtype == in_types[0].dtype)
        ndim = in_types[0].ndim.eval()
        concat_size = sum(typ.shape[self.axis] for typ in in_types)
        type_check.expect(concat_size == y_type.shape[self.axis])

        for d in range(0, ndim):
            if d == self.axis:
                continue
            type_check.expect(y_type.shape[d] == in_types[0].shape[d])

    def forward_cpu(self, xs):
        return numpy.concatenate(xs, axis=self.axis),

    def forward_gpu(self, xs):
        # TODO(beam2d): Unify the process into a single kernel.
        shape = list(xs[0].shape)
        for x in xs[1:]:
            shape[self.axis] += x.shape[self.axis]
        shape = tuple(shape)
        self.shape = shape

        y = cuda.empty(shape, dtype=xs[0].dtype)
        self.cdimy = y.shape[self.axis]
        self.rdim = numpy.prod(shape[self.axis + 1:], dtype=int)

        coffset = 0
        kernel = cuda.elementwise(
            _args, 'COPY(y[idx] = x[i])', 'concat_fwd', preamble=_preamble)
        for x in xs:
            cdimx = x.shape[self.axis]
            kernel(x, y, cdimx, self.cdimy, self.rdim, coffset)
            coffset += cdimx

        return y,

    def backward_cpu(self, xs, gy):
        sizes = numpy.array([x.shape[self.axis] for x in xs[:-1]]).cumsum()
        return numpy.split(gy[0], sizes, axis=self.axis)

    def backward_gpu(self, xs, gy):
        gxs = tuple(cuda.empty_like(x) for x in xs)

        coffset = 0
        kernel = cuda.elementwise(
            _args, 'COPY(x[i] = y[idx])', 'concat_bwd', preamble=_preamble)
        for gx in gxs:
            cdimx = gx.shape[self.axis]
            kernel(gx, gy[0], cdimx, self.cdimy, self.rdim, coffset)
            coffset += cdimx

        return gxs


def concat(xs, axis=1):
    """Concatenates given variables along an axis.

    Args:
        xs (tuple of Variables): Variables to be concatenated.
        axis (int): Axis that the input arrays are concatenated along.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Concat(axis=axis)(*xs)
