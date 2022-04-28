import cupy
import cupyx.scipy.sparse
try:
    import pylibraft
    pylibraft_available = True
except ModuleNotFoundError:
    pylibraft_available = False


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
        be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
        'jaccard', 'jensenshannon', 'kulsinski', 'kulczynski1',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
        'sqeuclidean', 'yule'.
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
    4. ``Y = pdist(X, 'seuclidean', V=None)``
       Computes the standardized Euclidean distance. The standardized
       Euclidean distance between two n-vectors ``u`` and ``v`` is
       .. math::
          \\sqrt{\\sum {(u_i-v_i)^2 / V[x_i]}}
       V is the variance vector; V[i] is the variance computed over all
       the i'th components of the points.  If not passed, it is
       automatically computed.
    5. ``Y = pdist(X, 'sqeuclidean')``
       Computes the squared Euclidean distance :math:`\\|u-v\\|_2^2` between
       the vectors.
    6. ``Y = pdist(X, 'cosine')``
       Computes the cosine distance between vectors u and v,
       .. math::
          1 - \\frac{u \\cdot v}
                   {{\\|u\\|}_2 {\\|v\\|}_2}
       where :math:`\\|*\\|_2` is the 2-norm of its argument ``*``, and
       :math:`u \\cdot v` is the dot product of ``u`` and ``v``.
    7. ``Y = pdist(X, 'correlation')``
       Computes the correlation distance between vectors u and v. This is
       .. math::
          1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                   {{\\|(u - \\bar{u})\\|}_2 {\\|(v - \\bar{v})\\|}_2}
       where :math:`\\bar{v}` is the mean of the elements of vector v,
       and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.
    8. ``Y = pdist(X, 'hamming')``
       Computes the normalized Hamming distance, or the proportion of
       those vector elements between two n-vectors ``u`` and ``v``
       which disagree. To save memory, the matrix ``X`` can be of type
       boolean.
    9. ``Y = pdist(X, 'jaccard')``
       Computes the Jaccard distance between the points. Given two
       vectors, ``u`` and ``v``, the Jaccard distance is the
       proportion of those elements ``u[i]`` and ``v[i]`` that
       disagree.
    10. ``Y = pdist(X, 'jensenshannon')``
        Computes the Jensen-Shannon distance between two probability arrays.
        Given two probability vectors, :math:`p` and :math:`q`, the
        Jensen-Shannon distance is
        .. math::
           \\sqrt{\\frac{D(p \\parallel m) + D(q \\parallel m)}{2}}
        where :math:`m` is the pointwise mean of :math:`p` and :math:`q`
        and :math:`D` is the Kullback-Leibler divergence.
    11. ``Y = pdist(X, 'chebyshev')``
        Computes the Chebyshev distance between the points. The
        Chebyshev distance between two n-vectors ``u`` and ``v`` is the
        maximum norm-1 distance between their respective elements. More
        precisely, the distance is given by
        .. math::
           d(u,v) = \\max_i {|u_i-v_i|}
    12. ``Y = pdist(X, 'canberra')``
        Computes the Canberra distance between the points. The
        Canberra distance between two points ``u`` and ``v`` is
        .. math::
          d(u,v) = \\sum_i \\frac{|u_i-v_i|}
                               {|u_i|+|v_i|}
    13. ``Y = pdist(X, 'braycurtis')``
        Computes the Bray-Curtis distance between the points. The
        Bray-Curtis distance between two points ``u`` and ``v`` is
        .. math::
             d(u,v) = \\frac{\\sum_i {|u_i-v_i|}}
                            {\\sum_i {|u_i+v_i|}}
    17. ``Y = pdist(X, 'dice')``
        Computes the Dice distance between each pair of boolean
        vectors. (see dice function documentation)
    20. ``Y = pdist(X, 'russellrao')``
        Computes the Russell-Rao distance between each pair of
        boolean vectors. (see russellrao function documentation)
    24. ``Y = pdist(X, f)``
        Computes the distance between all pairs of vectors in X
        using the user supplied 2-arity function f. For example,
        Euclidean distance between the vectors could be computed
        as follows::
          dm = pdist(X, lambda u, v: np.sqrt(((u-v)**2).sum()))
        Note that you should avoid passing a reference to one of
        the distance functions defined in this library. For example,::
          dm = pdist(X, sokalsneath)
        would calculate the pair-wise distances between the vectors in
        X using the Python function sokalsneath. This would result in
        sokalsneath being called :math:`{n \\choose 2}` times, which
        is inefficient. Instead, the optimized C version is more
        efficient, and we call it using the following syntax.::
          dm = pdist(X, 'sokalsneath')
    """
    # You can also call this as:
    #     Y = pdist(X, 'test_abc')
    # where 'abc' is the metric being tested.  This computes the distance
    # between all pairs of vectors in X using the distance metric 'abc' but
    # with a more succinct, verifiable, but less efficient implementation.

    X = _asarray_validated(X, sparse_ok=False, objects_ok=True, mask_ok=True,
                           check_finite=False)

    s = X.shape
    if len(s) != 2:
        raise ValueError('A 2-dimensional array must be passed.')

    m, n = s

    if callable(metric):
        mstr = getattr(metric, '__name__', 'UnknownCustomMetric')
        metric_info = _METRIC_ALIAS.get(mstr, None)

        if metric_info is not None:
            X, typ, kwargs = _validate_pdist_input(
                X, m, n, metric_info, **kwargs)

        return _pdist_callable(X, metric=metric, out=out, **kwargs)
    elif isinstance(metric, str):
        mstr = metric.lower()
        metric_info = _METRIC_ALIAS.get(mstr, None)

        if metric_info is not None:
            pdist_fn = metric_info.pdist_func
            return pdist_fn(X, out=out, **kwargs)
        elif mstr.startswith("test_"):
            metric_info = _TEST_METRICS.get(mstr, None)
            if metric_info is None:
                raise ValueError(f'Unknown "Test" Distance Metric: {mstr[5:]}')
            X, typ, kwargs = _validate_pdist_input(
                X, m, n, metric_info, **kwargs)
            return _pdist_callable(
                X, metric=metric_info.dist_func, out=out, **kwargs)
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')
