from __future__ import annotations
import warnings

from cupyx.cutensor import *  # NOQA


warnings.warn(
    'cupy.cutensor is deprecated. Use cupyx.cutensor instead',
    DeprecationWarning)
