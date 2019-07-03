# Attributes and Methods for fallback_mode
# Auto-execute numpy method when corresponding cupy method is not found

# "NOQA" to suppress flake8 warning
from cupyx.fallback_mode.fallback import numpy  # NOQA

from cupyx.fallback_mode import notification

from cupyx.fallback_mode.notification import seterr  # NOQA
from cupyx.fallback_mode.notification import geterr  # NOQA
from cupyx.fallback_mode.notification import errstate  # NOQA

setlogger = notification.FallbackLogger.setlogger
