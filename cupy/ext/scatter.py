import warnings

import cupyx


def scatter_add(a, slices, value):
    """Adds given values to specified elements of an array.

    .. deprecated:: 4.0
       Use :func:`cupyx.scatter_add` instead.

    """
    warnings.warn(
        'cupy.scatter_add is deprecated. Use cupyx.scatter_add instead.',
        DeprecationWarning)
    cupyx.scatter_add(a, slices, value)
