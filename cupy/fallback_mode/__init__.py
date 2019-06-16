# Attributes and Methods for fallback_mode
# Auto-execute numpy method when corresponding cupy method is not found

# "NOQA" to suppress flake8 warning
from cupy.fallback_mode.fallback import numpy  # NOQA
from cupy.fallback_mode.fallback import _RecursiveAttr  # NOQA

notifications = _RecursiveAttr.notifications
set_notifications = _RecursiveAttr.set_notifications
