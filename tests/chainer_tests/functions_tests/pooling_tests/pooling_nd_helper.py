import itertools
import nose.tools
import six

from chainer import testing


@nose.tools.nottest
def pooling_patches(dims, ksize, stride, pad, cover_all):
    """Return tuples of slices that indicate pooling patches."""
    # Left-top indexes of each pooling patch.
    if cover_all:
        xss = itertools.product(
            *[six.moves.range(-p, d + p - k + s, s)
              for (d, k, s, p) in six.moves.zip(dims, ksize, stride, pad)])
    else:
        xss = itertools.product(
            *[six.moves.range(-p, d + p - k + 1, s)
              for (d, k, s, p) in six.moves.zip(dims, ksize, stride, pad)])
    # Tuples of slices for pooling patches.
    return [tuple(slice(max(x, 0), min(x + k, d))
                  for (x, d, k) in six.moves.zip(xs, dims, ksize))
            for xs in xss]


testing.run_module(__name__, __file__)
