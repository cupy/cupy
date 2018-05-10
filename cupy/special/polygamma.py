import cupy
from cupy import special


def polygamma(n, x):
    n, x = cupy.asarray(n), cupy.asarray(x)
    fac2 = (-1.0)**(n+1) * special.gamma(n+1.0) * special.zeta(n+1.0, x)
    return cupy.where(n == 0, special.digamma(x), fac2)
