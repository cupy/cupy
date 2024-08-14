# Bessel Functions
from cupy._math.rounding import round
from cupy._math.special import sinc

# Convenience functions
from cupyx.scipy.special._basic import (
    cbrt,
    cosdg,
    cosm1,
    cotdg,
    exp2,
    exp10,
    expm1,
    exprel,
    log1p,
    radian,
    sindg,
    tandg,
)
from cupyx.scipy.special._bessel import (
    i0,
    i0e,
    i1,
    i1e,
    j0,
    j1,
    k0,
    k0e,
    k1,
    k1e,
    y0,
    y1,
    yn,
)
from cupyx.scipy.special._beta import beta, betainc, betaincinv, betaln

# Other special functions
from cupyx.scipy.special._binom import binom

# Information Theory functions
from cupyx.scipy.special._convex_analysis import (
    entr,
    huber,
    kl_div,
    pseudo_huber,
    rel_entr,
)
from cupyx.scipy.special._digamma import digamma
from cupyx.scipy.special._digamma import digamma as psi

# Elliptic functions
from cupyx.scipy.special._ellip import (
    ellipeinc,
    ellipj,
    ellipk,
    ellipkinc,
    ellipkm1,
)

# Error function and Fresnel integrals
from cupyx.scipy.special._erf import erf, erfc, erfcinv, erfcx, erfinv
from cupyx.scipy.special._exp1 import exp1
from cupyx.scipy.special._expi import expi
from cupyx.scipy.special._expn import expn

# Gamma and related functions
from cupyx.scipy.special._gamma import gamma, rgamma
from cupyx.scipy.special._gammainc import (
    gammainc,
    gammaincc,
    gammainccinv,
    gammaincinv,
)
from cupyx.scipy.special._gammaln import gammaln, multigammaln
from cupyx.scipy.special._gammasgn import gammasgn
from cupyx.scipy.special._lambertw import lambertw
from cupyx.scipy.special._loggamma import loggamma
from cupyx.scipy.special._logsoftmax import log_softmax
from cupyx.scipy.special._logsumexp import logsumexp

# Legendre functions
from cupyx.scipy.special._lpmv import lpmv
from cupyx.scipy.special._poch import poch
from cupyx.scipy.special._polygamma import polygamma
from cupyx.scipy.special._softmax import softmax
from cupyx.scipy.special._sph_harm import sph_harm
from cupyx.scipy.special._spherical_bessel import spherical_yn
from cupyx.scipy.special._statistics import (
    boxcox,
    boxcox1p,
    expit,
    inv_boxcox,
    inv_boxcox1p,
    log_expit,
    logit,
)

# Raw statistical functions
from cupyx.scipy.special._stats_distributions import (
    bdtr,
    bdtrc,
    bdtri,
    btdtr,
    btdtri,
    chdtr,
    chdtrc,
    chdtri,
    fdtr,
    fdtrc,
    fdtri,
    gdtr,
    gdtrc,
    log_ndtr,
    nbdtr,
    nbdtrc,
    nbdtri,
    ndtr,
    ndtri,
    pdtr,
    pdtrc,
    pdtri,
)
from cupyx.scipy.special._wright_bessel import wright_bessel
from cupyx.scipy.special._xlogy import xlog1py, xlogy
from cupyx.scipy.special._zeta import zeta
from cupyx.scipy.special._zetac import zetac
