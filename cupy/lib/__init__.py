from cupy.lib import stride_tricks  # NOQA

# Polynomial routines (mirrors numpy.lib's polynomial re-exports).
from cupy.lib._polynomial import poly1d  # NOQA
from cupy.lib._routines_poly import poly  # NOQA
from cupy.lib._routines_poly import polyadd  # NOQA
from cupy.lib._routines_poly import polyder  # NOQA
from cupy.lib._routines_poly import polydiv  # NOQA
from cupy.lib._routines_poly import polyfit  # NOQA
from cupy.lib._routines_poly import polyint  # NOQA
from cupy.lib._routines_poly import polymul  # NOQA
from cupy.lib._routines_poly import polysub  # NOQA
from cupy.lib._routines_poly import polyval  # NOQA
from cupy.lib._routines_poly import roots  # NOQA
