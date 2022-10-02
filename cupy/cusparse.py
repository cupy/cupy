import warnings

from cupyx.cutensor import *  # NOQA


warnings.warn(
    'cupy.cusparse is deprecated. Use cupyx.cusparse instead',
    DeprecationWarning)
