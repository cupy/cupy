import math
import cupy
from cupy import _core


__all__ = ['tri', 'tril', 'triu', 'toeplitz', 'circulant', 'hankel',
           'hadamard', 'leslie', 'kron', 'block_diag', 'companion',
           'helmert', 'hilbert', 'dft',
           'fiedler', 'fiedler_companion', 'convolution_matrix']


def tri(N, M=None, k=0, dtype=None):
    """ Construct (``N``, ``M``) matrix filled with ones at and below the
    ``k``-th diagonal. The matrix has ``A[i,j] == 1`` for ``i <= j + k``.

    Args:
        N (int): The size of the first dimension of the matrix.
        M (int, optional): The size of the second dimension of the matrix. If
            ``M`` is None, ``M = N`` is assumed.
        k (int, optional):  Number of subdiagonal below which matrix is filled
            with ones. ``k = 0`` is the main diagonal, ``k < 0`` subdiagonal
            and ``k > 0`` superdiagonal.
        dtype (dtype, optional): Data type of the matrix.

    Returns:
        cupy.ndarray: Tri matrix.

    .. seealso:: :func:`scipy.linalg.tri`
    """
    if M is None:
        M = N
    elif isinstance(M, str):
        # handle legacy interface
        M, dtype = N, M
    # TODO: no outer method
    # m = cupy.greater_equal.outer(cupy.arange(k, N+k), cupy.arange(M))
    # return m if dtype is None else m.astype(dtype, copy=False)
    return cupy.tri(N, M, k, bool if dtype is None else dtype)


def tril(m, k=0):
    """Make a copy of a matrix with elements above the ``k``-th diagonal
    zeroed.

    Args:
        m (cupy.ndarray): Matrix whose elements to return
        k (int, optional): Diagonal above which to zero elements.
            ``k == 0`` is the main diagonal, ``k < 0`` subdiagonal and
            ``k > 0`` superdiagonal.

    Returns:
        (cupy.ndarray): Return is the same shape and type as ``m``.

    .. seealso:: :func:`scipy.linalg.tril`
    """
    # this is ~2x faster than cupy.tril for a 500x500 float64 matrix
    t = tri(m.shape[0], m.shape[1], k=k, dtype=m.dtype.char)
    t *= m
    return t


def triu(m, k=0):
    """Make a copy of a matrix with elements below the ``k``-th diagonal
    zeroed.

    Args:
        m (cupy.ndarray): Matrix whose elements to return
        k (int, optional): Diagonal above which to zero elements.
            ``k == 0`` is the main diagonal, ``k < 0`` subdiagonal and
            ``k > 0`` superdiagonal.

    Returns:
        (cupy.ndarray): Return matrix with zeroed elements below the kth
        diagonal and has same shape and type as ``m``.

    .. seealso:: :func:`scipy.linalg.triu`
    """
    # this is ~25% faster than cupy.tril for a 500x500 float64 matrix
    t = tri(m.shape[0], m.shape[1], k - 1, m.dtype.char)
    cupy.subtract(1, t, out=t)
    t *= m
    return t


def toeplitz(c, r=None):
    """Construct a Toeplitz matrix.

    The Toeplitz matrix has constant diagonals, with ``c`` as its first column
    and ``r`` as its first row. If ``r`` is not given, ``r == conjugate(c)`` is
    assumed.

    Args:
        c (cupy.ndarray): First column of the matrix. Whatever the actual shape
            of ``c``, it will be converted to a 1-D array.
        r (cupy.ndarray, optional): First row of the matrix. If None,
            ``r = conjugate(c)`` is assumed; in this case, if ``c[0]`` is real,
            the result is a Hermitian matrix. r[0] is ignored; the first row of
            the returned matrix is ``[c[0], r[1:]]``. Whatever the actual shape
            of ``r``, it will be converted to a 1-D array.

    Returns:
        cupy.ndarray: The Toeplitz matrix. Dtype is the same as
            ``(c[0] + r[0]).dtype``.

    .. seealso:: :func:`cupyx.scipy.linalg.circulant`
    .. seealso:: :func:`cupyx.scipy.linalg.hankel`
    .. seealso:: :func:`cupyx.scipy.linalg.solve_toeplitz`
    .. seealso:: :func:`cupyx.scipy.linalg.fiedler`
    .. seealso:: :func:`scipy.linalg.toeplitz`
    """
    c = c.ravel()
    r = c.conjugate() if r is None else r.ravel()
    return _create_toeplitz_matrix(c[::-1], r[1:])


def circulant(c):
    """Construct a circulant matrix.

    Args:
        c (cupy.ndarray): 1-D array, the first column of the matrix.

    Returns:
        cupy.ndarray: A circulant matrix whose first column is ``c``.

    .. seealso:: :func:`cupyx.scipy.linalg.toeplitz`
    .. seealso:: :func:`cupyx.scipy.linalg.hankel`
    .. seealso:: :func:`cupyx.scipy.linalg.solve_circulant`
    .. seealso:: :func:`cupyx.scipy.linalg.fiedler`
    .. seealso:: :func:`scipy.linalg.circulant`
    """
    c = c.ravel()
    return _create_toeplitz_matrix(c[::-1], c[:0:-1])


def hankel(c, r=None):
    """Construct a Hankel matrix.

    The Hankel matrix has constant anti-diagonals, with ``c`` as its first
    column and ``r`` as its last row. If ``r`` is not given, then
    ``r = zeros_like(c)`` is assumed.

    Args:
        c (cupy.ndarray): First column of the matrix. Whatever the actual shape
            of ``c``, it will be converted to a 1-D array.
        r (cupy.ndarray, optionnal): Last row of the matrix. If None,
            ``r = zeros_like(c)`` is assumed. ``r[0]`` is ignored; the last row
            of the returned matrix is ``[c[-1], r[1:]]``. Whatever the actual
            shape of ``r``, it will be converted to a 1-D array.

    Returns:
        cupy.ndarray: The Hankel matrix. Dtype is the same as
            ``(c[0] + r[0]).dtype``.

    .. seealso:: :func:`cupyx.scipy.linalg.toeplitz`
    .. seealso:: :func:`cupyx.scipy.linalg.circulant`
    .. seealso:: :func:`scipy.linalg.hankel`
    """
    c = c.ravel()
    r = cupy.zeros_like(c) if r is None else r.ravel()
    return _create_toeplitz_matrix(c, r[1:], True)


def _create_toeplitz_matrix(c, r, hankel=False):
    vals = cupy.concatenate((c, r))
    n = vals.strides[0]
    return cupy.lib.stride_tricks.as_strided(
        vals if hankel else vals[c.size-1:],
        shape=(c.size, r.size+1),
        strides=(n if hankel else -n, n)).copy()


def hadamard(n, dtype=int):
    """Construct an Hadamard matrix.

    Constructs an n-by-n Hadamard matrix, using Sylvester's construction. ``n``
    must be a power of 2.

    Args:
        n (int): The order of the matrix. ``n`` must be a power of 2.
        dtype (dtype, optional): The data type of the array to be constructed.

    Returns:
        (cupy.ndarray): The Hadamard matrix.

    .. seealso:: :func:`scipy.linalg.hadamard`
    """
    lg2 = 0 if n < 1 else (int(n).bit_length() - 1)
    if 2 ** lg2 != n:
        raise ValueError('n must be an positive a power of 2 integer')
    H = cupy.empty((n, n), dtype)
    return _hadamard_kernel(H, H)


_hadamard_kernel = _core.ElementwiseKernel(
    'T in', 'T out',
    'out = (__popc(_ind.get()[0] & _ind.get()[1]) & 1) ? -1 : 1;',
    'hadamard', reduce_dims=False)


def leslie(f, s):
    """Create a Leslie matrix.

    Given the length n array of fecundity coefficients ``f`` and the length n-1
    array of survival coefficients ``s``, return the associated Leslie matrix.

    Args:
        f (cupy.ndarray): The "fecundity" coefficients.
        s (cupy.ndarray): The "survival" coefficients, has to be 1-D.  The
            length of ``s`` must be one less than the length of ``f``, and it
            must be at least 1.

    Returns:
        cupy.ndarray: The array is zero except for the first row, which is
            ``f``, and the first sub-diagonal, which is ``s``. The data-type of
            the array will be the data-type of ``f[0]+s[0]``.

    .. seealso:: :func:`scipy.linalg.leslie`
    """
    if f.ndim != 1:
        raise ValueError('Incorrect shape for f. f must be 1D')
    if s.ndim != 1:
        raise ValueError('Incorrect shape for s. s must be 1D')
    n = f.size
    if n != s.size + 1:
        raise ValueError('Length of s must be one less than length of f')
    if s.size == 0:
        raise ValueError('The length of s must be at least 1.')
    a = cupy.zeros((n, n), dtype=cupy.result_type(f, s))
    a[0] = f
    cupy.fill_diagonal(a[1:], s)
    return a


def kron(a, b):
    """Kronecker product.

    The result is the block matrix::
        a[0,0]*b    a[0,1]*b  ... a[0,-1]*b
        a[1,0]*b    a[1,1]*b  ... a[1,-1]*b
        ...
        a[-1,0]*b   a[-1,1]*b ... a[-1,-1]*b

    Args:
        a (cupy.ndarray): Input array
        b (cupy.ndarray): Input array

    Returns:
        cupy.ndarray: Kronecker product of ``a`` and ``b``.

    .. seealso:: :func:`scipy.linalg.kron`
    """
    o = cupy.outer(a, b)
    o = o.reshape(a.shape + b.shape)
    return cupy.concatenate(cupy.concatenate(o, axis=1), axis=1)


def block_diag(*arrs):
    """Create a block diagonal matrix from provided arrays.

    Given the inputs ``A``, ``B``, and ``C``, the output will have these
    arrays arranged on the diagonal::

        [A, 0, 0]
        [0, B, 0]
        [0, 0, C]

    Args:
        A, B, C, ... (cupy.ndarray): Input arrays. A 1-D array of length ``n``
            is treated as a 2-D array with shape ``(1,n)``.

    Returns:
        (cupy.ndarray): Array with ``A``, ``B``, ``C``, ... on the diagonal.
            Output has the same dtype as ``A``.

    .. seealso:: :func:`scipy.linalg.block_diag`
    """
    if not arrs:
        return cupy.empty((1, 0))

    # Convert to 2D and check
    if len(arrs) == 1:
        arrs = (cupy.atleast_2d(*arrs),)
    else:
        arrs = cupy.atleast_2d(*arrs)
    if any(a.ndim != 2 for a in arrs):
        bad = [k for k in range(len(arrs)) if arrs[k].ndim != 2]
        raise ValueError('arguments in the following positions have dimension '
                         'greater than 2: {}'.format(bad))

    shapes = tuple(a.shape for a in arrs)
    shape = tuple(sum(x) for x in zip(*shapes))
    dtype = cupy.find_common_type([a.dtype for a in arrs], [])
    out = cupy.zeros(shape, dtype=dtype)
    r, c = 0, 0
    for arr in arrs:
        rr, cc = arr.shape
        out[r:r + rr, c:c + cc] = arr
        r += rr
        c += cc
    return out


def companion(a):
    """Create a companion matrix.

    Create the companion matrix associated with the polynomial whose
    coefficients are given in ``a``.

    Args:
        a (cupy.ndarray): 1-D array of polynomial coefficients. The length of
            ``a`` must be at least two, and ``a[0]`` must not be zero.

    Returns:
        (cupy.ndarray): The first row of the output is ``-a[1:]/a[0]``, and the
            first sub-diagonal is all ones. The data-type of the array is the
            same as the data-type of ``-a[1:]/a[0]``.

    .. seealso:: :func:`cupyx.scipy.linalg.fiedler_companion`
    .. seealso:: :func:`scipy.linalg.companion`
    """
    n = a.size
    if a.ndim != 1:
        raise ValueError('`a` must be one-dimensional.')
    if n < 2:
        raise ValueError('The length of `a` must be at least 2.')
    # Following check requires device-to-host synchronization so will we not
    # raise an error this situation
    # if a[0] == 0:
    #     raise ValueError('The first coefficient in `a` must not be zero.')
    first_row = -a[1:] / a[0]
    c = cupy.zeros((n - 1, n - 1), dtype=first_row.dtype)
    c[0] = first_row
    cupy.fill_diagonal(c[1:], 1)
    return c


def helmert(n, full=False):
    """Create an Helmert matrix of order ``n``.

    This has applications in statistics, compositional or simplicial analysis,
    and in Aitchison geometry.

    Args:
        n (int): The size of the array to create.
        full (bool, optional): If True the (n, n) ndarray will be returned.
            Otherwise, the default, the submatrix that does not include the
            first row will be returned.

    Returns:
        cupy.ndarray: The Helmert matrix. The shape is (n, n) or (n-1, n)
            depending on the ``full`` argument.

    .. seealso:: :func:`scipy.linalg.helmert`
    """
    d = cupy.arange(n)
    H = cupy.tri(n, n, -1)
    H.diagonal()[:] -= d
    d *= cupy.arange(1, n+1)
    H[0] = 1
    d[0] = n
    H /= cupy.sqrt(d)[:, None]
    return H if full else H[1:]


def hilbert(n):
    """Create a Hilbert matrix of order ``n``.

    Returns the ``n`` by ``n`` array with entries ``h[i,j] = 1 / (i + j + 1)``.

    Args:
        n (int): The size of the array to create.

    Returns:
        cupy.ndarray: The Hilbert matrix.

    .. seealso:: :func:`scipy.linalg.hilbert`
    """
    values = cupy.arange(1, 2*n, dtype=cupy.float64)
    cupy.reciprocal(values, values)
    return hankel(values[:n], r=values[n-1:])


# TODO: invhilbert(n, exact=False)
# TODO: pascal(n, kind='symmetric', exact=True)
# TODO: invpascal(n, kind='symmetric', exact=True)


def dft(n, scale=None):
    """Discrete Fourier transform matrix.

    Create the matrix that computes the discrete Fourier transform of a
    sequence. The nth primitive root of unity used to generate the matrix is
    exp(-2*pi*i/n), where i = sqrt(-1).

    Args:
        n (int): Size the matrix to create.
        scale (str, optional): Must be None, 'sqrtn', or 'n'.
            If ``scale`` is 'sqrtn', the matrix is divided by ``sqrt(n)``.
            If ``scale`` is 'n', the matrix is divided by ``n``.
            If ``scale`` is None (default), the matrix is not normalized, and
            the return value is simply the Vandermonde matrix of the roots of
            unity.

    Returns:
        (cupy.ndarray): The DFT matrix.

    Notes:
        When ``scale`` is None, multiplying a vector by the matrix returned by
        ``dft`` is mathematically equivalent to (but much less efficient than)
        the calculation performed by ``scipy.fft.fft``.

    .. seealso:: :func:`scipy.linalg.dft`
    """
    if scale not in (None, 'sqrtn', 'n'):
        raise ValueError('scale must be None, \'sqrtn\', or \'n\'; '
                         '%r is not valid.' % (scale,))
    r = cupy.arange(n, dtype='complex128')
    r *= -2j*cupy.pi/n
    omegas = cupy.exp(r, out=r)[:, None]
    m = omegas ** cupy.arange(n)
    if scale is not None:
        m *= (1/math.sqrt(n)) if scale == 'sqrtn' else (1/n)
    return m


def fiedler(a):
    """Returns a symmetric Fiedler matrix

    Given an sequence of numbers ``a``, Fiedler matrices have the structure
    ``F[i, j] = np.abs(a[i] - a[j])``, and hence zero diagonals and nonnegative
    entries. A Fiedler matrix has a dominant positive eigenvalue and other
    eigenvalues are negative. Although not valid generally, for certain inputs,
    the inverse and the determinant can be derived explicitly.

    Args:
        a (cupy.ndarray): coefficient array

    Returns:
        cupy.ndarray: the symmetric Fiedler matrix

    .. seealso:: :func:`cupyx.scipy.linalg.circulant`
    .. seealso:: :func:`cupyx.scipy.linalg.toeplitz`
    .. seealso:: :func:`scipy.linalg.fiedler`
    """
    if a.ndim != 1:
        raise ValueError('Input `a` must be a 1D array.')
    if a.size == 0:
        return cupy.zeros(0)
    if a.size == 1:
        return cupy.zeros((1, 1))
    a = a[:, None] - a
    return cupy.abs(a, out=a)


def fiedler_companion(a):
    """Returns a Fiedler companion matrix

    Given a polynomial coefficient array ``a``, this function forms a
    pentadiagonal matrix with a special structure whose eigenvalues coincides
    with the roots of ``a``.

    Args:
        a (cupy.ndarray): 1-D array of polynomial coefficients in descending
            order with a nonzero leading coefficient. For ``N < 2``, an empty
            array is returned.

    Returns:
        cupy.ndarray: Resulting companion matrix

    Notes:
        Similar to ``companion`` the leading coefficient should be nonzero. In
        the case the leading coefficient is not 1, other coefficients are
        rescaled before the array generation. To avoid numerical issues, it is
        best to provide a monic polynomial.

    .. seealso:: :func:`cupyx.scipy.linalg.companion`
    .. seealso:: :func:`scipy.linalg.fiedler_companion`
    """
    if a.ndim != 1:
        raise ValueError('Input `a` must be a 1-D array.')
    if a.size < 2:
        return cupy.zeros((0,), a.dtype)
    if a.size == 2:
        return (-a[1]/a[0])[None, None]
    # Following check requires device-to-host synchronization so will we not
    # raise an error this situation
    # if a[0] == 0.:
    #     raise ValueError('Leading coefficient is zero.')
    a = a/a[0]
    n = a.size - 1
    c = cupy.zeros((n, n), dtype=a.dtype)
    # subdiagonals
    cupy.fill_diagonal(c[3::2, 1::2], 1)
    cupy.fill_diagonal(c[2::2, 1::2], -a[3::2])
    # superdiagonals
    cupy.fill_diagonal(c[::2, 2::2], 1)
    cupy.fill_diagonal(c[::2, 1::2], -a[2::2])
    c[0, 0] = -a[1]
    c[1, 0] = 1
    return c


def convolution_matrix(a, n, mode='full'):
    """Construct a convolution matrix.

    Constructs the Toeplitz matrix representing one-dimensional convolution.

    Args:
        a (cupy.ndarray): The 1-D array to convolve.
        n (int): The number of columns in the resulting matrix. It gives the
            length of the input to be convolved with ``a``. This is analogous
            to the length of ``v`` in ``numpy.convolve(a, v)``.
        mode (str): This must be one of (``'full'``, ``'valid'``, ``'same'``).
            This is analogous to ``mode`` in ``numpy.convolve(v, a, mode)``.

    Returns:
        cupy.ndarray: The convolution matrix whose row count depends on
            ``mode``:
                ``'full'   m + n -1``
                ``'same'   max(m, n)``
                ``'valid'  max(m, n) - min(m, n) + 1``

    .. seealso:: :func:`cupyx.scipy.linalg.toeplitz`
    .. seealso:: :func:`scipy.linalg.convolution_matrix`
    """
    if n <= 0:
        raise ValueError('n must be a positive integer.')
    if a.ndim != 1:
        raise ValueError('convolution_matrix expects a one-dimensional '
                         'array as input')
    if a.size == 0:
        raise ValueError('len(a) must be at least 1.')
    if mode not in ('full', 'valid', 'same'):
        raise ValueError(
            '`mode` argument must be one of (\'full\', \'valid\', \'same\')')

    # create zero padded versions of the array
    az = cupy.pad(a, (0, n-1), 'constant')
    raz = cupy.pad(a[::-1], (0, n-1), 'constant')
    if mode == 'same':
        trim = min(n, a.size) - 1
        tb = trim//2
        te = trim - tb
        col0 = az[tb:az.size-te]
        row0 = raz[-n-tb:raz.size-tb]
    elif mode == 'valid':
        tb = min(n, a.size) - 1
        te = tb
        col0 = az[tb:az.size-te]
        row0 = raz[-n-tb:raz.size-tb]
    else:  # 'full'
        col0 = az
        row0 = raz[-n:]
    return toeplitz(col0, row0)
