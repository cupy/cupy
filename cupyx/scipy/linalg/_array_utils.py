import cupy
from cupy.linalg import _util

# Find the "bandwise position" of a nonzero cell
_kernel_cupy_band_pos_c = cupy.ElementwiseKernel(
    'T A, int64 r, int64 c',
    'int64 out',
    'out = A != 0 ? r - c : 0',
    'cupyx_scipy_linalg_band_pos'
)


def bandwidth(a):

    """Return the lower and upper bandwidth of a 2D numeric array.
    Parameters
    ----------
    a : ndarray
        Input array of size (M, N)
    Returns
    -------
    lu : tuple
        2-tuple of ints indicating the lower and upper bandwith. A zero
        denotes no sub- or super-diagonal on that side (triangular), and,
        say for M rows (M-1) means that side is full. Same example applies
        to the upper triangular part with (N-1).

    .. seealso:: :func:`scipy.linalg.bandwidth`
    """

    a = cupy.asarray(a)

    if a.size == 0:
        return (0, 0)
    _util._assert_2d(a)

    # If a is F contiguous then transpose to C contiguous
    if a.flags['F_CONTIGUOUS']:
        a = a.T

    # row_num and col_num contain info on the row and column number of A
    m, n = a.shape
    row_num, col_num = cupy.mgrid[0:m, 0:n]

    # If F contigious, transpose
    if a.flags['F_CONTIGUOUS']:
        bandpts = _kernel_cupy_band_pos_c(a.T, col_num, row_num)
        upper_band = int(cupy.amax(bandpts))
        lower_band = -int(cupy.amin(bandpts))
    else:
        bandpts = _kernel_cupy_band_pos_c(a, row_num, col_num)
        lower_band = int(cupy.amax(bandpts))
        upper_band = -int(cupy.amin(bandpts))

    return lower_band, upper_band
