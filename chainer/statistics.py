import numpy as np
import cupy
from functools import reduce


def _iterable(x):
    if isinstance(x, (list, tuple)):
        return x
    return x,


def _key_base(link, param_names, attr_names):

    """Create a statistic dictionary key."""

    link_name = 'None' if not hasattr(link, 'name') else link.name
    param_name = '-'.join(param_names)
    attr_name = '-'.join(attr_names)

    return '{}/{}/{}'.format(link_name, param_name, attr_name)


def get_statistics(link, param_names, attr_names,
                   statistics=('min', 'max', 'mean', 'std'),
                   percentile_sigmas=(0.13, 2.28, 15.87, 50, 84.13, 97.72,
                                      99.87)):

    """Compute the percentiles and various statistics for a link.

    Specified parameters and attributes in the given link will be flattened and
    statistics will be computed before being returned in a dictionary.

    Args:
        link (~chainer.Link): Link for which statistics are computed.
        param_names (str or iterable): Parameter names, e.g. ``('W', 'b')``
            or ``'W'``.
        attr_names (str or iterable): Attributes names, e.g.
            ``('data', 'grad')`` or ``'data'``.
        statistics (iterable): Statistics to collect, mapping directly to NumPy
            and CuPy functions.
        percentile_sigmas (iterable): Sigma values for which percentiles are
            computed.

    Returns:
        dict: Dictionary of statistics.
    """

    param_names = _iterable(param_names)
    attr_names = _iterable(attr_names)

    params = flatten_link(link, param_names, attr_names)

    key_base = _key_base(link, param_names, attr_names)

    stats = {}

    if percentile_sigmas:
        percentiles = param_percentiles(params, sigma=percentile_sigmas)
        for i, percentile in enumerate(percentiles):
            stats['{}/percentile/{}'.format(key_base, i)] = percentile

    for s in statistics:
        try:
            stats['{}/{}'.format(key_base, s)] = getattr(params, s)()
        except ValueError:
            # If data is missing from uninitialized model parameters, add
            # NaN placeholders instead of skipping the measurements completely
            # or registering zeros
            stats['{}/{}'.format(key_base, s)] = float('NaN')

    return stats


def get_sparsity(link, param_names, attr_names):

    """Compute the number of zero elements in a link.

    Only specified parameters and attributes are taken into account. It is for
    instance possible to ignore the biases and only count the number of zero
    elements among the weights by setting ``param_names`` to ``'W'``.

    Args:
        link (~chainer.Link): Link for which sparsity is computed.
        param_names (str or iterable): Parameter names, e.g. ``('W', 'b')``
            or ``'W'``.
        attr_names (str or iterable): Attributes names, e.g.
            ``('data', 'grad')`` or ``'data'``.

    Returns:
        dict: Dictionary of statistics.
    """

    param_names = _iterable(param_names)
    attr_names = _iterable(attr_names)

    params = flatten_link(link, param_names, attr_names)

    n_zeros = params.size - link.xp.count_nonzero(params)

    key_base = _key_base(link, param_names, attr_names)
    key = '{}/zeros'.format(key_base)

    return { key: n_zeros }


def flatten_link(link, param_names, attr_names):

    """Flatten link parameters and return a 1-dimensional array located on the
    same device as the link itself.

    Args:
        link (~chainer.Link): Link to flatten.
        param_names (iterable): Parameter names to flatten.
        attr_names (iterable): Attributes names to flatten.

    Returns:
        array: Array of flattened parameters.
    """

    params = []
    for param in link.params():
        if param.name in param_names:
            for attr_name in attr_names:
                p = getattr(param, attr_name)
                p = p.flatten()
                params.append(p)

    return link.xp.concatenate(params)


def param_percentiles(params, sigma):

    """Compute percentiles for given parameters and return an array with the
    same length as the number of elements in ``sigma``.

    Args:
        params (array): 1-dimensional NumPy or CuPy arryay.
        sigma (tuple): Sigma values for which percentiles are computed.

    Returns:
        array: Array of percentiles.
    """

    def _percentiles(_params, _sigma):
        try:
            return np.percentile(_params, _sigma)
        except IndexError:  # Handle uninitialized model parameters
            return np.array((float('NaN'),) * 7)

    # TODO(hvy): Make percentile computation faster for GPUs
    if isinstance(params, cupy.ndarray):
        params = cupy.asnumpy(params)
        return cupy.asarray(_percentiles(params, sigma))

    return _percentiles(params, sigma)
