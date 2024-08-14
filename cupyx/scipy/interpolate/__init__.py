# Univariate Interpolation
# 1-D Splines
from cupyx.scipy.interpolate._bspline import BSpline, splantider, splder
from cupyx.scipy.interpolate._bspline2 import (
    make_interp_spline,
    make_lsq_spline,
)
from cupyx.scipy.interpolate._cubic import (
    Akima1DInterpolator,
    CubicHermiteSpline,
    CubicSpline,
    PchipInterpolator,
    pchip_interpolate,
)

# Multivariate interpolation
from cupyx.scipy.interpolate._interpnd import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
)
from cupyx.scipy.interpolate._interpolate import BPoly, NdPPoly, PPoly

# Multivariate interpolation
from cupyx.scipy.interpolate._ndbspline import NdBSpline
from cupyx.scipy.interpolate._ndgriddata import NearestNDInterpolator
from cupyx.scipy.interpolate._polyint import (
    BarycentricInterpolator,
    KroghInterpolator,
    barycentric_interpolate,
    interp1d,
    krogh_interpolate,
)

# Radial basis functions
from cupyx.scipy.interpolate._rbfinterp import RBFInterpolator
from cupyx.scipy.interpolate._rgi import RegularGridInterpolator, interpn

# Backward compatibility
pchip = PchipInterpolator

# FITPACK smoothing splines
from cupyx.scipy.interpolate._fitpack_repro import (
    InterpolatedUnivariateSpline,
    LSQUnivariateSpline,
    UnivariateSpline,
)
