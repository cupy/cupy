import numpy as np
import cupy
from chainer import cuda


def statistics(x, functions=('min', 'max', 'mean', 'std')):

    """Compute statisticts for the given array.

    Args:
        x (array): Target array for which statistics are computed.
        functions (iterable): Statistics to collect, mapping directly to NumPy
            and CuPy functions.

    Returns:
        dict: Mappings from functions keys to statistic values.
    """

    stats = {}
    for f in functions:
        try:
            stats[f] = getattr(x, f)()
        except ValueError:
            stats[f] = float('NaN')
    return stats


def percentiles(x, sigmas=(0.13, 2.28, 15.87, 50, 84.13, 97.72, 99.87)):

    """Compute percentiles for the given array.

    Args:
        x (array): Target array for which percentiles are computed.
        sigmas (iterable): Percentile sigma values.

    Returns:
        array: List of percentiles. The list has the same length as the given
            ``sigma``.
    """

    def _percentiles(_x):
        try:
            return np.percentile(_x, sigmas)
        except IndexError:
            return np.array((float('NaN'),) * 7)

    # TODO(hvy): Make percentile computation faster for GPUs
    if isinstance(x, cupy.ndarray):
        x = cupy.asnumpy(x)
        return cupy.asarray(_percentiles(x))
    return _percentiles(x)


def sparsity(x):
    """Count the number of zeros in the given array.

    Args:
        x (array): Target array for which sparsity is computed.

    Returns:
        int: Number of zeros.
    """

    return x.size - cuda.get_array_module(x).count_nonzero(x)
