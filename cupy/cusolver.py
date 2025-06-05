from __future__ import annotations

import warnings

from cupyx.cusolver import *  # NOQA


warnings.warn(
    'cupy.cusolver is deprecated. Use cupyx.cusolver instead',
    DeprecationWarning)
