# Attributes and Methods for fallback_mode
# Auto-execute numpy method when corresponding cupy method is not found

# "NOQA" to suppress flake8 warning
from cupy.fallback_mode import fallback # NOQA

from cupy.fallback_mode.fallback import numpy # NOQA

from cupy.fallback_mode.utils.FallbackUtil import notification_status # NOQA
from cupy.fallback_mode.utils.FallbackUtil import set_notification_status # NOQA
