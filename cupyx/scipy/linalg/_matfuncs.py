import cupy
from cupy.linalg import _util


def khatri_rao(a, b):
    r"""
    Khatri-rao product

    A column-wise Kronecker product of two matrices

    Parameters
    ----------
    a : (n, k) array_like
        Input array
    b : (m, k) array_like
        Input array

    Returns
    -------
    c:  (n*m, k) ndarray
        Khatri-rao product of `a` and `b`.

    See Also
    --------
    .. seealso:: :func:`scipy.linalg.khatri_rao`

    """

    _util._assert_2d(a)
    _util._assert_2d(b)

    if a.shape[1] != b.shape[1]:
        raise ValueError("The number of columns for both arrays "
                         "should be equal.")

    c = a[..., :, cupy.newaxis, :] * b[..., cupy.newaxis, :, :]
    return c.reshape((-1,) + c.shape[2:])
