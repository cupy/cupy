# mypy: ignore-errors
import numpy

if numpy.__version__ < '2':
    from numpy import AxisError
    from numpy import ComplexWarning
    from numpy import ModuleDeprecationWarning
    from numpy import TooHardError
    from numpy import VisibleDeprecationWarning
    from numpy import RankWarning
else:
    from numpy.exceptions import AxisError
    from numpy.exceptions import ComplexWarning
    from numpy.exceptions import ModuleDeprecationWarning
    from numpy.exceptions import TooHardError
    from numpy.exceptions import VisibleDeprecationWarning
    from numpy.exceptions import RankWarning
