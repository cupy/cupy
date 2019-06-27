# Attributes and Methods for fallback_mode
# Auto-execute numpy method when corresponding cupy method is not found

# "NOQA" to suppress flake8 warning
from cupyx.fallback_mode.fallback import numpy  # NOQA

from cupyx.fallback_mode import utils
from cupyx.fallback_mode.utils import seterr  # NOQA
from cupyx.fallback_mode.utils import geterr  # NOQA
from cupyx.fallback_mode.utils import errstate  # NOQA

setlogger = utils.FallbackLogger.setlogger
