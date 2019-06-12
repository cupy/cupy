# Attributes and Methods for fallback_mode
# Auto-execute numpy method when corresponding cupy method is not found

# "NOQA" to suppress flake8 warning
from cupy.fallback_mode.fallback import numpy # NOQA

notification_status = numpy.notification_status
set_notification_status = numpy.set_notification_status
