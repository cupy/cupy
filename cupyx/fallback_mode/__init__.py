from cupy import _util

# Attributes and Methods for fallback_mode
# Auto-execute numpy method when corresponding cupy method is not found

from cupyx.fallback_mode.fallback import numpy  # NOQA


_util.experimental('cupyx.fallback_mode.numpy')
