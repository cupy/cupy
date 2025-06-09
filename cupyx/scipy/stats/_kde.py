
import cupy
from cupy._core._scalar import get_typename

from cupyx.scipy.linalg import solve_triangular


GAUSSIAN_MODULE = cupy.RawModule(code=r"""

#define CUDART_PI               3.1415926535897931e+0

template<typename T>
__global__ void gaussian_estimate_inner(
        const T* points, const T* values, const T* xi, T* estimate,
        const T* cho_cov, int n, int m, int d, int p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= n * m) {
        return;
    }

    int row = idx / m;
    int col = idx % m;

    T norm = pow((2 * CUDART_PI), -d / 2.0);
    for(int i = 0; i < d; i++) {
        norm /= cho_cov[d * i + i];
    }

    T arg = 0;
    for(int k = 0; k < d; k++) {
        T residual = points[d * row + k] - xi[d * col + k];
        arg += residual * residual;
    }

    arg = exp(-arg / 2.0) * norm;
    for(int k = 0; k < p; k++) {
        estimate[p * col + k] += values[p * row + k] * arg;
    }
}

""", options=('-std=c++11',), name_expressions=[
    'gaussian_estimate_inner<float>', 'gaussian_estimate_inner<double>'])


def gaussian_kernel_estimate(points, values, xi, cho_cov, dtype, c_dtype):
    """
    Evaluate a multivariate Gaussian kernel estimate.

    Parameters
    ----------
    points : array_like with shape (n, d)
        Data points to estimate from in d dimensions.
    values : real[:, :] with shape (n, p)
        Multivariate values associated with the data points.
    xi : array_like with shape (m, d)
        Coordinates to evaluate the estimate at in d dimensions.
    cho_cov : array_like with shape (d, d)
        (Lower) Cholesky factor of the covariance.

    Returns
    -------
    estimate : double[:, :] with shape (m, p)
        Multivariate Gaussian kernel estimate evaluated at the input
        coordinates.
    """
    n = points.shape[0]
    d = points.shape[1]
    m = xi.shape[0]
    p = values.shape[1]

    if xi.shape[1] != d:
        raise ValueError("points and xi must have same trailing dim")
    if cho_cov.shape[0] != d or cho_cov.shape[1] != d:
        raise ValueError("Covariance matrix must match data dims")

    # Rescale the data
    cho_cov_ = cho_cov.astype(dtype, copy=False)
    points_ = cupy.asarray(solve_triangular(cho_cov, points.T, lower=True).T,
                           dtype=dtype)
    xi_ = cupy.asarray(solve_triangular(cho_cov, xi.T, lower=True).T,
                       dtype=dtype)
    values_ = values.astype(dtype, copy=False)

    # Create the result array and evaluate the weighted sum
    estimate = cupy.zeros((m, p), dtype)

    total = n * m
    block_sz = 128
    n_blocks = ((total + block_sz - 1) // block_sz)

    gaussian_kernel_estimate_inner = GAUSSIAN_MODULE.get_function(
        f'gaussian_estimate_inner<{c_dtype}>')

    gaussian_kernel_estimate_inner(
        (n_blocks,), (block_sz,), (points_, values_, xi_,
                                   estimate, cho_cov_, n, m, d, p))


class gaussian_kde:
    """Representation of a kernel-density estimate using Gaussian kernels.

    Kernel density estimation is a way to estimate the probability density
    function (PDF) of a random variable in a non-parametric way.
    `gaussian_kde` works for both uni-variate and multi-variate data.   It
    includes automatic bandwidth determination.  The estimation works best for
    a unimodal distribution; bimodal or multi-modal distributions tend to be
    oversmoothed.

    Parameters
    ----------
    dataset : array_like
        Datapoints to estimate from. In case of univariate data this is a 1-D
        array, otherwise a 2-D array with shape (# of dims, # of data).
    bw_method : str, scalar or callable, optional
        The method used to calculate the bandwidth factor.  This can be
        'scott', 'silverman', a scalar constant or a callable.  If a scalar,
        this will be used directly as `factor`.  If a callable, it should
        take a `gaussian_kde` instance as only parameter and return a scalar.
        If None (default), 'scott' is used.  See Notes for more details.
    weights : array_like, optional
        weights of datapoints. This must be the same shape as dataset.
        If None (default), the samples are assumed to be equally weighted

    Attributes
    ----------
    dataset : ndarray
        The dataset with which `gaussian_kde` was initialized.
    d : int
        Number of dimensions.
    n : int
        Number of datapoints.
    neff : int
        Effective number of datapoints.

        .. versionadded:: 1.2.0
    factor : float
        The bandwidth factor obtained from `covariance_factor`.
    covariance : ndarray
        The kernel covariance matrix; this is the data covariance matrix
        multiplied by the square of the bandwidth factor, e.g.
        ``np.cov(dataset) * factor**2``.
    inv_cov : ndarray
        The inverse of `covariance`.

    Methods
    -------
    evaluate
    __call__
    integrate_gaussian
    integrate_box_1d
    integrate_box
    integrate_kde
    pdf
    logpdf
    resample
    set_bandwidth
    covariance_factor

    Notes
    -----
    Bandwidth selection strongly influences the estimate obtained from the KDE
    (much more so than the actual shape of the kernel).  Bandwidth selection
    can be done by a "rule of thumb", by cross-validation, by "plug-in
    methods" or by other means; see [3]_, [4]_ for reviews.  `gaussian_kde`
    uses a rule of thumb, the default is Scott's Rule.

    Scott's Rule [1]_, implemented as `scotts_factor`, is::

        n**(-1./(d+4)),

    with ``n`` the number of data points and ``d`` the number of dimensions.
    In the case of unequally weighted points, `scotts_factor` becomes::

        neff**(-1./(d+4)),

    with ``neff`` the effective number of datapoints.
    Silverman's suggestion for *multivariate* data [2]_, implemented as
    `silverman_factor`, is::

        (n * (d + 2) / 4.)**(-1. / (d + 4)).

    or in the case of unequally weighted points::

        (neff * (d + 2) / 4.)**(-1. / (d + 4)).

    Note that this is not the same as "Silverman's rule of thumb" [6]_, which
    may be more robust in the univariate case; see documentation of the
    ``set_bandwidth`` method for implementing a custom bandwidth rule.

    Good general descriptions of kernel density estimation can be found in [1]_
    and [2]_, the mathematics for this multi-dimensional implementation can be
    found in [1]_.

    With a set of weighted samples, the effective number of datapoints ``neff``
    is defined by::

        neff = sum(weights)^2 / sum(weights^2)

    as detailed in [5]_.

    `gaussian_kde` does not currently support data that lies in a
    lower-dimensional subspace of the space in which it is expressed. For such
    data, consider performing principal component analysis / dimensionality
    reduction and using `gaussian_kde` with the transformed data.

    References
    ----------
    .. [1] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
           Visualization", John Wiley & Sons, New York, Chicester, 1992.
    .. [2] B.W. Silverman, "Density Estimation for Statistics and Data
           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
           Chapman and Hall, London, 1986.
    .. [3] B.A. Turlach, "Bandwidth Selection in Kernel Density Estimation: A
           Review", CORE and Institut de Statistique, Vol. 19, pp. 1-33, 1993.
    .. [4] D.M. Bashtannyk and R.J. Hyndman, "Bandwidth selection for kernel
           conditional density estimation", Computational Statistics & Data
           Analysis, Vol. 36, pp. 279-298, 2001.
    .. [5] Gray P. G., 1969, Journal of the Royal Statistical Society.
           Series A (General), 132, 272
    .. [6] Kernel density estimation. *Wikipedia.*
           https://en.wikipedia.org/wiki/Kernel_density_estimation

    Examples
    --------
    Generate some random two-dimensional data:

    >>> import numpy as np
    >>> from scipy import stats
    >>> def measure(n):
    ...     "Measurement model, return two coupled measurements."
    ...     m1 = np.random.normal(size=n)
    ...     m2 = np.random.normal(scale=0.5, size=n)
    ...     return m1+m2, m1-m2

    >>> m1, m2 = measure(2000)
    >>> xmin = m1.min()
    >>> xmax = m1.max()
    >>> ymin = m2.min()
    >>> ymax = m2.max()

    Perform a kernel density estimate on the data:

    >>> X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    >>> positions = np.vstack([X.ravel(), Y.ravel()])
    >>> values = np.vstack([m1, m2])
    >>> kernel = stats.gaussian_kde(values)
    >>> Z = np.reshape(kernel(positions).T, X.shape)

    Plot the results:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
    ...           extent=[xmin, xmax, ymin, ymax])
    >>> ax.plot(m1, m2, 'k.', markersize=2)
    >>> ax.set_xlim([xmin, xmax])
    >>> ax.set_ylim([ymin, ymax])
    >>> plt.show()

    Compare against manual KDE at a point:

    >>> point = [1, 2]
    >>> mean = values.T
    >>> cov = kernel.factor**2 * np.cov(values)
    >>> X = stats.multivariate_normal(cov=cov)
    >>> res = kernel.pdf(point)
    >>> ref = X.pdf(point - mean).sum() / len(mean)
    >>> np.allclose(res, ref)
    True
    """  # NOQA

    def __init__(self, dataset, bw_method=None, weights=None):
        self.dataset = cupy.atleast_2d(cupy.asarray(dataset))
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.d, self.n = self.dataset.shape

        if weights is not None:
            self._weights = cupy.atleast_1d(weights).astype(float)
            self._weights /= cupy.sum(self._weights)
            if self.weights.ndim != 1:
                raise ValueError("`weights` input should be one-dimensional.")
            if len(self._weights) != self.n:
                raise ValueError("`weights` input should be of length n")
            self._neff = 1 / cupy.sum(self._weights * self._weights, axis=-1)

        # This can be converted to a warning once scipy/gh-10205 is resolved
        if self.d > self.n:
            msg = ("Number of dimensions is greater than number of samples. "
                   "This results in a singular data covariance matrix, which "
                   "cannot be treated using the algorithms implemented in "
                   "`gaussian_kde`. Note that `gaussian_kde` interprets each "
                   "*column* of `dataset` to be a point; consider transposing "
                   "the input to `dataset`.")
            raise ValueError(msg)

        try:
            self.set_bandwidth(bw_method=bw_method)
        except cupy.linalg.LinAlgError as e:
            msg = ("The data appears to lie in a lower-dimensional subspace "
                   "of the space in which it is expressed. This has resulted "
                   "in a singular data covariance matrix, which cannot be "
                   "treated using the algorithms implemented in "
                   "`gaussian_kde`. Consider performing principal component "
                   "analysis / dimensionality reduction and using "
                   "`gaussian_kde` with the transformed data.")
            raise cupy.linalg.LinAlgError(msg) from e

    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different
                     than the dimensionality of the KDE.

        """
        points = cupy.atleast_2d(cupy.asarray(points))

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = cupy.reshape(points, (self.d, 1))
                m = 1
            else:
                msg = (f"points have dimension {d}, "
                       f"dataset has dimension {self.d}")
                raise ValueError(msg)

        output_dtype, spec = _get_output_dtype(self.covariance, points)
        result = gaussian_kernel_estimate(
            self.dataset.T, self.weights[:, None],
            points.T, self.cho_cov, output_dtype, spec)

        return result[:, 0]

    __call__ = evaluate

    def set_bandwidth(self, bw_method=None):
        """Compute the bandwidth factor with given method.

        The new bandwidth calculated after a call to `set_bandwidth` is used
        for subsequent evaluations of the estimated density.

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the bandwidth factor.  This can be
            'scott', 'silverman', a scalar constant or a callable.  If a
            scalar, this will be used directly as `factor`.  If a callable,
            it should take a `gaussian_kde` instance as only parameter and
            return a scalar.  If None (default), nothing happens; the current
            `covariance_factor` method is kept.

        Notes
        -----
        .. versionadded:: 0.11

        Examples
        --------
        >>> import numpy as np
        >>> import scipy.stats as stats
        >>> x1 = np.array([-7, -5, 1, 4, 5.])
        >>> kde = stats.gaussian_kde(x1)
        >>> xs = np.linspace(-10, 10, num=50)
        >>> y1 = kde(xs)
        >>> kde.set_bandwidth(bw_method='silverman')
        >>> y2 = kde(xs)
        >>> kde.set_bandwidth(bw_method=kde.factor / 3.)
        >>> y3 = kde(xs)

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot(x1, np.full(x1.shape, 1 / (4. * x1.size)), 'bo',
        ...         label='Data points (rescaled)')
        >>> ax.plot(xs, y1, label='Scott (default)')
        >>> ax.plot(xs, y2, label='Silverman')
        >>> ax.plot(xs, y3, label='Const (1/3 * Silverman)')
        >>> ax.legend()
        >>> plt.show()

        """
        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        elif cupy.isscalar(bw_method) and not isinstance(bw_method, str):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        self._compute_covariance()

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and Cholesky decomp of covariance
        if not hasattr(self, '_data_cho_cov'):
            self._data_covariance = cupy.atleast_2d(cupy.cov(
                self.dataset, rowvar=1, bias=False,
                aweights=self.weights))
            self._data_cho_cov = cupy.linalg.cholesky(self._data_covariance)

        self.covariance = self._data_covariance * self.factor ** 2
        self.cho_cov = (self._data_cho_cov * self.factor).astype(cupy.float64)
        self.log_det = 2 * cupy.log(
            cupy.diag(self.cho_cov * cupy.sqrt(2 * cupy.pi))).sum()


def _get_output_dtype(covariance, points):
    """
    Calculates the output dtype and the "spec" (=C type name).

    This was necessary in order to deal with the fused types in the Cython
    routine `gaussian_kernel_estimate`. See gh-10824 for details.
    """
    output_dtype = cupy.common_type(covariance, points)
    spec = get_typename(output_dtype)

    return output_dtype, spec
