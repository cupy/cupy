# mypy: ignore-errors
from __future__ import annotations
import numpy

if numpy.__version__ < '2':
    from numpy import AxisError  # NOQA
    from numpy import ComplexWarning  # NOQA
    from numpy import ModuleDeprecationWarning  # NOQA
    from numpy import TooHardError  # NOQA
    from numpy import VisibleDeprecationWarning  # NOQA
    from numpy import RankWarning   # NOQA
else:
    from numpy.exceptions import AxisError  # NOQA
    from numpy.exceptions import ComplexWarning  # NOQA
    from numpy.exceptions import ModuleDeprecationWarning  # NOQA
    from numpy.exceptions import TooHardError  # NOQA
    from numpy.exceptions import VisibleDeprecationWarning  # NOQA
    from numpy.exceptions import RankWarning   # NOQA
