import collections

import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


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
        if not isinstance(indices_or_sections, (int, collections.Iterable)):
            raise TypeError('indices_or_sections must be integer or 1-D array')
        self.indices_or_sections = indices_or_sections
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].ndim >= self.axis)

        if isinstance(self.indices_or_sections, collections.Iterable):
            max_index = type_check.Variable(
                self.indices_or_sections[-1], 'max_index')
            type_check.expect(in_types[0].shape[self.axis] > max_index)
        else:
            sections = type_check.Variable(
                self.indices_or_sections, 'sections')
            type_check.expect(in_types[0].shape[self.axis] % sections == 0)

    def forward_cpu(self, x):
        if isinstance(self.indices_or_sections, collections.Iterable):
            cdimx = x[0].shape[self.axis]
            ind = list(self.indices_or_sections)
            ind.append(cdimx)
            prev_i = 0
            for i in ind:
                cdimy = max(0, min(i, cdimx) - prev_i)
                if cdimy == 0:
                    raise ValueError('Not support if shape contains 0')
                prev_i = i
        return tuple(numpy.split(x[0], self.indices_or_sections, self.axis))

    def forward_gpu(self, x):
        xshape = x[0].shape
        self.cdimx = xshape[self.axis]
        self.rdim = numpy.prod(xshape[self.axis + 1:], dtype=int)

        if isinstance(self.indices_or_sections, collections.Iterable):
            ind = list(self.indices_or_sections)
            ind.append(self.cdimx)
        else:
            sec = self.indices_or_sections
            if self.cdimx % sec:
                raise ValueError(
                    'array split does not result in an equal division')
            ind = numpy.arange(1, sec + 1) * (self.cdimx // sec)
        ys = []
        kernel = cuda.elementwise(
            _args, 'COPY(y[i] = x[idx])', 'split_fwd', preamble=_preamble)
        prev_i = 0
        for i in ind:
            cdimy = max(0, min(i, self.cdimx) - prev_i)
            s = list(xshape)
            s[self.axis] = cdimy
            y = cuda.empty(s, dtype=x[0].dtype)
            if cdimy == 0:
                raise ValueError('Not support if shape contains 0')
            kernel(y, x[0], cdimy, self.cdimx, self.rdim, prev_i)
            prev_i = i
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
        indices_or_sections (int or 1-D array): If this argument is an integer,
            N, the array will be divided into N equal arrays along axis.
            If it is a 1-D array of sorted integers, it
            indicates the positions where the array is split.
        axis (int): Axis that the input array is split along.

    Returns:
        ``tuple`` or ``Variable``: Tuple of :class:`~chainer.Variable` objects
             if the number of outputs is more than 1 or
             :class:`~chainer.Variable` otherwise.

    .. note::
        This function raises ``ValueError`` if at least
        one of the outputs is splitted to zero-size
        (i.e. `axis`-th value of its shape is zero).

    """
    return SplitAxis(indices_or_sections, axis)(x)
