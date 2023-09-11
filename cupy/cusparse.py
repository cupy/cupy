import warnings

from cupyx.cusparse import *  # NOQA


warnings.warn(
    'cupy.cusparse is deprecated. Use cupyx.cusparse instead',
    DeprecationWarning)
