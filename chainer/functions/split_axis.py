import collections

import numpy
import six

from chainer import cuda
from chainer import function

_args = 'float* y, float* x, int cdimy, int cdimx, int rdim, int coffset'
_preamble = '''
#define COPY(statement) \
    int l   = i / (rdim * cdimy);  \
    int c   = i / rdim % cdimy + coffset;  \
    int r   = i % rdim;  \
    int idx = r + rdim * (c + cdimx * l);  \
    statement;
'''


class SplitAxis(function.Function):

    """Function that splits multiple arrays towards the specified axis."""

    def __init__(self, indices_or_sections, axis):
        self.indices_or_sections = indices_or_sections
        self.axis = axis

    def forward_cpu(self, x):
        return tuple(numpy.split(x[0], self.indices_or_sections, self.axis))

    def forward_gpu(self, x):
        xshape = x[0].shape
        self.cdimx = xshape[self.axis]
        self.rdim = numpy.prod(xshape[self.axis + 1:])

        indices = self.indices_or_sections
        if isinstance(indices, collections.Iterable):
            indices = list(indices)
            indices.append(xshape[self.axis])
        else:
            if xshape[self.axis] % indices:
                raise ValueError(
                    'array split does not result in an equal division')
            indices = six.moves.range(
                indices, xshape[self.axis] + indices, indices)
        ys = []
        kernel = cuda.elementwise(
            _args, 'COPY(y[i] = x[idx])', 'split_fwd', preamble=_preamble)
        bi = 0
        for i in indices:
            i = min(i, xshape[self.axis])
            cdimy = max(0, i - bi)
            s = list(xshape)
            s[self.axis] = cdimy
            y = cuda.empty(s, dtype=x[0].dtype)
            if cdimy != 0:
                kernel(y, x[0], cdimy, self.cdimx, self.rdim, bi)
            bi = i
            ys.append(y)
        return tuple(ys)

    def backward_cpu(self, x, gys):
        return numpy.concatenate(gys, axis=self.axis),

    def backward_gpu(self, x, gys):
        gx = cuda.empty_like(x[0])
        coffset = 0
        kernel = cuda.elementwise(
            _args, 'COPY(x[idx] = y[i])', 'split_bwd', preamble=_preamble)
        for gy in gys:
            cdimy = gy.shape[self.axis]
            if cdimy != 0:
                kernel(gy, gx, cdimy, self.cdimx, self.rdim, coffset)
            coffset += cdimy
        return gx,


def split_axis(x, indices_or_sections, axis):
    """Splits given variables along an axis.

    Args:
        x (tuple of Variables): Variables to be split.
        indices_or_sections (int or 1-Darray): If this argument is an integer,
            N, the array will be divided into N equal arrays along axis.
            If it is a 1-D array of sorted integers, the entries
            indicate where along which axis the array is split.
        axis (int): Axis that the input arrays are split along.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return SplitAxis(indices_or_sections, axis)(x)
