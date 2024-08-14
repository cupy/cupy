# mypy: ignore-errors
import numpy

if numpy.__version__ < "2":
    from numpy import (
        AxisError,
        ComplexWarning,
        ModuleDeprecationWarning,
        RankWarning,
        TooHardError,
        VisibleDeprecationWarning,
    )
else:
    from numpy.exceptions import (
        AxisError,
        ComplexWarning,
        ModuleDeprecationWarning,
        RankWarning,
        TooHardError,
        VisibleDeprecationWarning,
    )
