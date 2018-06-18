import cupy
from cupyx.scipy.special import digamma
from cupyx.scipy.special import gamma
from cupyx.scipy.special import zeta


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
    fac2 = (-1.0)**(n+1) * gamma.gamma(n+1.0) * zeta.zeta(n+1.0, x)
    return cupy.where(n == 0, digamma.digamma(x), fac2)
