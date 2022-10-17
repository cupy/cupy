import warnings

from cupyx.cudnn import *  # NOQA


warnings.warn(
    'cupy.cudnn is deprecated. Use cupyx.cudnn instead',
    DeprecationWarning)
