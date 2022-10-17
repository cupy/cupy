import warnings

from cupyx.cutensor import *  # NOQA


warnings.warn(
    'cupy.cutensor is deprecated. Use cupyx.cutensor instead',
    DeprecationWarning)
