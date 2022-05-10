import cupy
import cupyx.scipy.sparse
try:
    from pylibraft.distance import pairwise_distance
    pylibraft_available = True
except ModuleNotFoundError:
    pylibraft_available = False


def _convert_to_type(X, out_type):
    return cupy.ascontiguousarray(X, dtype=out_type)


def _validate_pdist_input(X, m, n, metric_info, **kwargs):
    # get supported types
    types = metric_info.types
    # choose best type
    typ = types[types.index(X.dtype)] if X.dtype in types else types[0]
    # validate data
    X = _convert_to_type(X, out_type=typ)

    # validate kwargs
    _validate_kwargs = metric_info.validator
    if _validate_kwargs:
        kwargs = _validate_kwargs(X, m, n, **kwargs)
    return X, typ, kwargs


class MetricInfo:

    def __init__(self, canonical_name=None, aka=None, validator=None, types=None):
        self.canonical_name_ = canonical_name
        self.aka_ = aka
        self.validator_ = validator
        self.types_ = types


_METRIC_INFOS = [
    MetricInfo(
        canonical_name="canberra",
        aka={'canberra'}
    ),
    MetricInfo(
        canonical_name="chebyshev",
        aka={"chebychev", "chebyshev", "cheby", "cheb", "ch"}
    ),
    MetricInfo(
        canonical_name="cityblock",
        aka={"cityblock", "cblock", "cb", "c"}
    ),
    MetricInfo(
        canonical_name="correlation",
        aka={"correlation", "co"}
    ),
    MetricInfo(
        canonical_name="cosine",
        aka={"cosine", "cos"}
    ),
    MetricInfo(
        canonical_name="hamming",
        aka={"matching", "hamming", "hamm", "ha", "h"},
        types=["double", "bool"]
    ),
    MetricInfo(
        canonical_name="euclidean",
        aka={"euclidean", "euclid", "eu", "e"},
    ),
    MetricInfo(
        canonical_name="jensenshannon",
        aka={"jensenshannon", "js"}
    ),
    MetricInfo(
        canonical_name="minkowski",
        aka={"minkowski", "mi", "m", "pnorm"}
    ),
    MetricInfo(
        canonical_name="russellrao",
        aka={"russellrao"},
        types=["bool"]
    ),
    MetricInfo(
        canonical_name="sqeuclidean",
        aka={"sqeuclidean", "sqe", "sqeuclid"}
    ),
    MetricInfo(
        canonical_name="hellinger",
        aka={"hellinger"}
    ),
    MetricInfo(
        canonical_name="kl_divergence",
        aka={"kl_divergence", "kl_div", "kld"}
    )


]

_METRICS = {info.canonical_name_: info for info in _METRIC_INFOS}
_METRIC_ALIAS = dict((alias, info)
                     for info in _METRIC_INFOS
                     for alias in info.aka_)

_METRICS_NAMES = list(_METRICS.keys())


def minkowski(u, v, p):
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    return pairwise_distance(u, v, output_arr, "minkowski", p)


def pdist(X, metric='euclidean', *, out=None, **kwargs):
    """
    Pairwise distances between observations in n-dimensional space.
    See Notes for common calling conventions.
    Parameters
    ----------
    X : array_like
        An m by n array of m original observations in an
        n-dimensional space.
    metric : str or function, optional
        The distance metric to use. The distance function can
        be 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'euclidean', 'hamming', 'hellinger',
        'jensenshannon', 'kl_divergence', 'matching', 'minkowski','russellrao',
        'sqeuclidean'.
    **kwargs : dict, optional
        Extra arguments to `metric`: refer to each metric documentation for a
        list of all possible arguments.
        Some possible arguments:
        p : scalar
        The p-norm to apply for Minkowski, weighted and unweighted.
        Default: 2.
        w : ndarray
        The weight vector for metrics that support weights (e.g., Minkowski).
        V : ndarray
        The variance vector for standardized Euclidean.
        Default: var(X, axis=0, ddof=1)
        VI : ndarray
        The inverse of the covariance matrix for Mahalanobis.
        Default: inv(cov(X.T)).T
        out : ndarray.
        The output array
        If not None, condensed distance matrix Y is stored in this array.
    Returns
    -------
    Y : ndarray
        Returns a condensed distance matrix Y. For each :math:`i` and :math:`j`
        (where :math:`i<j<m`),where m is the number of original observations.
        The metric ``dist(u=X[i], v=X[j])`` is computed and stored in entry ``m
        * i + j - ((i + 2) * (i + 1)) // 2``.
    See Also
    --------
    squareform : converts between condensed distance matrices and
                 square distance matrices.
    Notes
    -----
    The following are common calling conventions.
    1. ``Y = pdist(X, 'euclidean')``
       Computes the distance between m points using Euclidean distance
       (2-norm) as the distance metric between the points. The points
       are arranged as m n-dimensional row vectors in the matrix X.
    2. ``Y = pdist(X, 'minkowski', p=2.)``
       Computes the distances using the Minkowski distance
       :math:`\\|u-v\\|_p` (:math:`p`-norm) where :math:`p > 0` (note
       that this is only a quasi-metric if :math:`0 < p < 1`).
    3. ``Y = pdist(X, 'cityblock')``
       Computes the city block or Manhattan distance between the
       points.
    4. ``Y = pdist(X, 'sqeuclidean')``
       Computes the squared Euclidean distance :math:`\\|u-v\\|_2^2` between
       the vectors.
    5. ``Y = pdist(X, 'cosine')``
       Computes the cosine distance between vectors u and v,
       .. math::
          1 - \\frac{u \\cdot v}
                   {{\\|u\\|}_2 {\\|v\\|}_2}
       where :math:`\\|*\\|_2` is the 2-norm of its argument ``*``, and
       :math:`u \\cdot v` is the dot product of ``u`` and ``v``.
    6. ``Y = pdist(X, 'correlation')``
       Computes the correlation distance between vectors u and v. This is
       .. math::
          1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                   {{\\|(u - \\bar{u})\\|}_2 {\\|(v - \\bar{v})\\|}_2}
       where :math:`\\bar{v}` is the mean of the elements of vector v,
       and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.
    7. ``Y = pdist(X, 'hamming')``
       Computes the normalized Hamming distance, or the proportion of
       those vector elements between two n-vectors ``u`` and ``v``
       which disagree. To save memory, the matrix ``X`` can be of type
       boolean.
    8. ``Y = pdist(X, 'jensenshannon')``
        Computes the Jensen-Shannon distance between two probability arrays.
        Given two probability vectors, :math:`p` and :math:`q`, the
        Jensen-Shannon distance is
        .. math::
           \\sqrt{\\frac{D(p \\parallel m) + D(q \\parallel m)}{2}}
        where :math:`m` is the pointwise mean of :math:`p` and :math:`q`
        and :math:`D` is the Kullback-Leibler divergence.
    9. ``Y = pdist(X, 'chebyshev')``
        Computes the Chebyshev distance between the points. The
        Chebyshev distance between two n-vectors ``u`` and ``v`` is the
        maximum norm-1 distance between their respective elements. More
        precisely, the distance is given by
        .. math::
           d(u,v) = \\max_i {|u_i-v_i|}
    10. ``Y = pdist(X, 'canberra')``
        Computes the Canberra distance between the points. The
        Canberra distance between two points ``u`` and ``v`` is
        .. math::
          d(u,v) = \\sum_i \\frac{|u_i-v_i|}
                               {|u_i|+|v_i|}
    11. ``Y = pdist(X, 'russellrao')``
        Computes the Russell-Rao distance between each pair of
        boolean vectors. (see russellrao function documentation)
    """

    # TODO(cjnolet): We won't accept objects. Need to look at how to support masks
    # X = _asarray_validated(X, sparse_ok=False, objects_ok=True, mask_ok=True,
    #                        check_finite=False)

    s = X.shape
    if len(s) != 2:
        raise ValueError('A 2-dimensional array must be passed.')

    m, n = s

    if isinstance(metric, str):
        mstr = metric.lower()
        metric_info = _METRIC_ALIAS.get(mstr, None)
        if metric_info is not None:

            output_arr = out if out is not None else cupy.zeros((m, m), dtype=X.dtype)
            pairwise_distance(X, X, output_arr, metric)
            return output_arr
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier')


def cdist(XA, XB, metric='euclidean', *, out=None, **kwargs):
    """
    Compute distance between each pair of the two collections of inputs.
    See Notes for common calling conventions.
    Parameters
    ----------
    XA : array_like
        An :math:`m_A` by :math:`n` array of :math:`m_A`
        original observations in an :math:`n`-dimensional space.
        Inputs are converted to float type.
    XB : array_like
        An :math:`m_B` by :math:`n` array of :math:`m_B`
        original observations in an :math:`n`-dimensional space.
        Inputs are converted to float type.
    metric : str or callable, optional
        The distance metric to use. If a string, the distance function can be
        'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'euclidean', 'hamming', 'hellinger', 'jensenshannon',
        'kl_divergence', 'matching', 'minkowski', 'russellrao', 'sqeuclidean'.
    **kwargs : dict, optional
        Extra arguments to `metric`: refer to each metric documentation for a
        list of all possible arguments.
        Some possible arguments:
        p : scalar
        The p-norm to apply for Minkowski, weighted and unweighted.
        Default: 2.
        w : array_like
        The weight vector for metrics that support weights (e.g., Minkowski).
        V : array_like
        The variance vector for standardized Euclidean.
        Default: var(vstack([XA, XB]), axis=0, ddof=1)
        VI : array_like
        The inverse of the covariance matrix for Mahalanobis.
        Default: inv(cov(vstack([XA, XB].T))).T
        out : ndarray
        The output array
        If not None, the distance matrix Y is stored in this array.
    Returns
    -------
    Y : ndarray
        A :math:`m_A` by :math:`m_B` distance matrix is returned.
        For each :math:`i` and :math:`j`, the metric
        ``dist(u=XA[i], v=XB[j])`` is computed and stored in the
        :math:`ij` th entry.
    Raises
    ------
    ValueError
        An exception is thrown if `XA` and `XB` do not have
        the same number of columns.
    Notes
    -----
    The following are common calling conventions:
    1. ``Y = cdist(XA, XB, 'euclidean')``
       Computes the distance between :math:`m` points using
       Euclidean distance (2-norm) as the distance metric between the
       points. The points are arranged as :math:`m`
       :math:`n`-dimensional row vectors in the matrix X.
    2. ``Y = cdist(XA, XB, 'minkowski', p=2.)``
       Computes the distances using the Minkowski distance
       :math:`\\|u-v\\|_p` (:math:`p`-norm) where :math:`p > 0` (note
       that this is only a quasi-metric if :math:`0 < p < 1`).
    3. ``Y = cdist(XA, XB, 'cityblock')``
       Computes the city block or Manhattan distance between the
       points.
    4. ``Y = cdist(XA, XB, 'sqeuclidean')``
       Computes the squared Euclidean distance :math:`\\|u-v\\|_2^2` between
       the vectors.
    5. ``Y = cdist(XA, XB, 'cosine')``
       Computes the cosine distance between vectors u and v,
       .. math::
          1 - \\frac{u \\cdot v}
                   {{\\|u\\|}_2 {\\|v\\|}_2}
       where :math:`\\|*\\|_2` is the 2-norm of its argument ``*``, and
       :math:`u \\cdot v` is the dot product of :math:`u` and :math:`v`.
    6. ``Y = cdist(XA, XB, 'correlation')``
       Computes the correlation distance between vectors u and v. This is
       .. math::
          1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                   {{\\|(u - \\bar{u})\\|}_2 {\\|(v - \\bar{v})\\|}_2}
       where :math:`\\bar{v}` is the mean of the elements of vector v,
       and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.
    7. ``Y = cdist(XA, XB, 'hamming')``
       Computes the normalized Hamming distance, or the proportion of
       those vector elements between two n-vectors ``u`` and ``v``
       which disagree. To save memory, the matrix ``X`` can be of type
       boolean.
    8. ``Y = cdist(XA, XB, 'jensenshannon')``
        Computes the Jensen-Shannon distance between two probability arrays.
        Given two probability vectors, :math:`p` and :math:`q`, the
        Jensen-Shannon distance is
        .. math::
           \\sqrt{\\frac{D(p \\parallel m) + D(q \\parallel m)}{2}}
        where :math:`m` is the pointwise mean of :math:`p` and :math:`q`
        and :math:`D` is the Kullback-Leibler divergence.
    9. ``Y = cdist(XA, XB, 'chebyshev')``
        Computes the Chebyshev distance between the points. The
        Chebyshev distance between two n-vectors ``u`` and ``v`` is the
        maximum norm-1 distance between their respective elements. More
        precisely, the distance is given by
        .. math::
           d(u,v) = \\max_i {|u_i-v_i|}.
    10. ``Y = cdist(XA, XB, 'canberra')``
        Computes the Canberra distance between the points. The
        Canberra distance between two points ``u`` and ``v`` is
        .. math::
          d(u,v) = \\sum_i \\frac{|u_i-v_i|}
                               {|u_i|+|v_i|}.
    11. ``Y = cdist(XA, XB, 'matching')``
        Synonym for 'hamming'.
    12. ``Y = cdist(XA, XB, 'russellrao')``
        Computes the Russell-Rao distance between the boolean
        vectors. (see `russellrao` function documentation)

    Examples
    --------
    Find the Euclidean distances between four 2-D coordinates:
    >>> from cupyx.scipy.spatial import distance
    >>> coords = [(35.0456, -85.2672),
    ...           (35.1174, -89.9711),
    ...           (35.9728, -83.9422),
    ...           (36.1667, -86.7833)]
    >>> distance.cdist(coords, coords, 'euclidean')
    array([[ 0.    ,  4.7044,  1.6172,  1.8856],
           [ 4.7044,  0.    ,  6.0893,  3.3561],
           [ 1.6172,  6.0893,  0.    ,  2.8477],
           [ 1.8856,  3.3561,  2.8477,  0.    ]])
    Find the Manhattan distance from a 3-D point to the corners of the unit
    cube:
    >>> a = cp.array([[0, 0, 0],
    ...               [0, 0, 1],
    ...               [0, 1, 0],
    ...               [0, 1, 1],
    ...               [1, 0, 0],
    ...               [1, 0, 1],
    ...               [1, 1, 0],
    ...               [1, 1, 1]])
    >>> b = cp.array([[ 0.1,  0.2,  0.4]])
    >>> distance.cdist(a, b, 'cityblock')
    array([[ 0.7],
           [ 0.9],
           [ 1.3],
           [ 1.5],
           [ 1.5],
           [ 1.7],
           [ 2.1],
           [ 2.3]])
    """
    XA = cupy.asarray(XA, dtype='float32')
    XB = cupy.asarray(XB, dtype='float32')

    s = XA.shape
    sB = XB.shape

    if len(s) != 2:
        raise ValueError('XA must be a 2-dimensional array.')
    if len(sB) != 2:
        raise ValueError('XB must be a 2-dimensional array.')
    if s[1] != sB[1]:
        raise ValueError('XA and XB must have the same number of columns '
                         '(i.e. feature dimension.)')

    mA = s[0]
    mB = sB[0]

    if isinstance(metric, str):
        mstr = metric.lower()
        metric_info = _METRIC_ALIAS.get(mstr, None)
        if metric_info is not None:
            output_arr = out if out is not None else cupy.zeros((mA, mB), dtype=XA.dtype)
            pairwise_distance(XA, XB, output_arr, metric)
            return output_arr
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier')


def distance_matrix(x, y, p=2):
    """Compute the distance matrix.
    Returns the matrix of all pair-wise distances.
    Parameters
    ----------
    x : (M, K) array_like
        Matrix of M vectors in K dimensions.
    y : (N, K) array_like
        Matrix of N vectors in K dimensions.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    Returns
    -------
    result : (M, N) ndarray
        Matrix containing the distance from every vector in `x` to every vector
        in `y`.
    Examples
    --------
    >>> from cupyx.scipy.spatial import distance_matrix
    >>> distance_matrix([[0,0],[0,1]], [[1,0],[1,1]])
    array([[ 1.        ,  1.41421356],
           [ 1.41421356,  1.        ]])
    """

    x = cupy.asarray(x)
    m, k = x.shape
    y = cupy.asarray(y)
    n, kk = y.shape

    if k != kk:
        raise ValueError("x contains %d-dimensional vectors but y contains %d-dimensional vectors" % (k, kk))

    return cdist(x, y, metric="minkowski", p=p)