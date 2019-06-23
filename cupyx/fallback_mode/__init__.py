# Attributes and Methods for fallback_mode
# Auto-execute numpy method when corresponding cupy method is not found

# "NOQA" to suppress flake8 warning
from cupyx.fallback_mode.fallback import numpy  # NOQA

from cupyx.fallback_mode import ndarray


ndarray._create_magic_methods()