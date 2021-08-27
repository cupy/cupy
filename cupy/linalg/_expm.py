
import cupy
from cupy_backends.cuda.libs import cublas
from cupy_backends.cuda.libs import cusolver
from cupy.cuda import device
from cupy.linalg import _util

def expm(a):
    """
    This method calculates the exponential of a square matrix, `` \exp{a}``.

    Args:
        a (cupy.ndarray): A symmetric 2-D square matrix ``(M, M)`` or a batch
            of symmetric 2-D square matrices ``(..., M, M)``.
        
    Returns:
        scalar :class:`~cupy.ndarray`:
        
    .. seealso:: :func:`numpy.linalg.expm`
    """
    pass
