import cupy
from cupyx.scipy.special import _digamma
from cupyx.scipy.special import _gamma
from cupyx.scipy.special import _zeta


def polygamma(n, x):
    """Polygamma function n.

    Args:
        n (cupy.ndarray): The order of the derivative of `psi`.
        x (cupy.ndarray): Where to evaluate the polygamma function.

    Returns:
        cupy.ndarray: The result.

    .. seealso:: :data:`scipy.special.polygamma`

    """
    n, x = cupy.broadcast_arrays(n, x)
    fac2 = (-1.0)**(n+1) * _gamma.gamma(n+1.0) * _zeta.zeta(n+1.0, x)
    return cupy.where(n == 0, _digamma.digamma(x), fac2)
