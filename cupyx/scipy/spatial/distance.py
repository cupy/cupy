import cupy
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

    def __init__(self, canonical_name=None, aka=None,
                 validator=None, types=None):
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
    """Compute the Minkowski distance between two 1-D arrays.

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)
        p (float): The order of the norm of the difference
            :math:`{\\|u-v\\|}_p`. Note that for :math:`0 < p < 1`,
            the triangle inequality only holds with an additional
            multiplicative factor, i.e. it is only a quasi-metric.

    Returns:
        minkowski (double): The Minkowski distance between vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, "minkowski", p)

    return output_arr[0]


def canberra(u, v):
    """Compute the Canberra distance between two 1-D arrays.

    The Canberra distance is defined as

    .. math::
        d(u, v) = \\sum_{i} \\frac{| u_i - v_i |}{|u_i| + |v_i|}

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        canberra (double): The Canberra distance between vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, "canberra")

    return output_arr[0]


def chebyshev(u, v):
    """Compute the Chebyshev distance between two 1-D arrays.

    The Chebyshev distance is defined as

    .. math::
        d(u, v) = \\max_{i} |u_i - v_i|

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        chebyshev (double): The Chebyshev distance between vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, "chebyshev")

    return output_arr[0]


def cityblock(u, v):
    """Compute the City Block (Manhattan) distance between two 1-D arrays.

    The City Block distance is defined as

    .. math::
        d(u, v) = \\sum_{i} |u_i - v_i|

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        cityblock (double): The City Block distance between
        vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, "cityblock")

    return output_arr[0]


def correlation(u, v):
    """Compute the correlation distance between two 1-D arrays.

    The correlation distance is defined as

    .. math::
        d(u, v) = 1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}{
        \\|(u - \\bar{u})\\|_2 \\|(v - \\bar{v})\\|_2}

    where :math:`\\bar{u}` is the mean of the elements of :math:`u` and
    :math:`x \\cdot y` is the dot product.

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        correlation (double): The correlation distance between
        vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, "correlation")

    return output_arr[0]


def cosine(u, v):
    """Compute the Cosine distance between two 1-D arrays.

    The Cosine distance is defined as

    .. math::
        d(u, v) = 1 - \\frac{u \\cdot v}{\\|u\\|_2 \\|v\\|_2}

    where :math:`x \\cdot y` is the dot product.

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        cosine (double): The Cosine distance between vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, "cosine")

    return output_arr[0]


def hamming(u, v):
    """Compute the Hamming distance between two 1-D arrays.

    The Hamming distance is defined as the proportion of elements
    in both `u` and `v` that are not in the exact same position:

    .. math::
        d(u, v) = \\frac{1}{n} \\sum_{k=0}^n u_i \\neq v_i

    where :math:`x \\neq y` is one if :math:`x` is different from :math:`y`
    and zero otherwise.

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        hamming (double): The Hamming distance between vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, "hamming")

    return output_arr[0]


def euclidean(u, v):
    """Compute the Euclidean distance between two 1-D arrays.

    The Euclidean distance is defined as

    .. math::
        d(u, v) = \\left(\\sum_{i} (u_i - v_i)^2\\right)^{\\sfrac{1}{2}}

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        euclidean (double): The Euclidean distance between vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, "euclidean")

    return output_arr[0]


def jensenshannon(u, v):
    """Compute the Jensen-Shannon distance between two 1-D arrays.

    The Jensen-Shannon distance is defined as

    .. math::
        d(u, v) = \\sqrt{\\frac{KL(u \\| m) + KL(v \\| m)}{2}}

    where :math:`KL` is the Kullback-Leibler divergence and :math:`m` is the
    pointwise mean of `u` and `v`.

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        jensenshannon (double): The Jensen-Shannon distance between
        vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, "jensenshannon")

    return output_arr[0]


def russellrao(u, v):
    """Compute the Russell-Rao distance between two 1-D arrays.

    The Russell-Rao distance is defined as the proportion of elements
    in both `u` and `v` that are in the exact same position:

    .. math::
        d(u, v) = \\frac{1}{n} \\sum_{k=0}^n u_i = v_i

    where :math:`x = y` is one if :math:`x` is different from :math:`y`
    and zero otherwise.

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        hamming (double): The Hamming distance between vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, "russellrao")

    return output_arr[0]


def sqeuclidean(u, v):
    """Compute the squared Euclidean distance between two 1-D arrays.

    The squared Euclidean distance is defined as

    .. math::
        d(u, v) = \\sum_{i} (u_i - v_i)^2

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        sqeuclidean (double): The squared Euclidean distance between
        vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed.')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, "sqeuclidean")

    return output_arr[0]


def hellinger(u, v):
    """Compute the Hellinger distance between two 1-D arrays.

    The Hellinger distance is defined as

    .. math::
        d(u, v) = \\frac{1}{\\sqrt{2}} \\sqrt{
            \\sum_{i} (\\sqrt{u_i} - \\sqrt{v_i})^2}

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        hellinger (double): The Hellinger distance between
        vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, "hellinger")

    return output_arr[0]


def kl_divergence(u, v):
    """Compute the Kullback-Leibler divergence between two 1-D arrays.

    The Kullback-Leibler divergence is defined as

    .. math::
        KL(U \\| V) = \\sum_{i} U_i \\log{\\left(\\frac{U_i}{V_i}\\right)}

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        kl_divergence (double): The Kullback-Leibler divergence between
        vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, "kl_divergence")

    return output_arr[0]


def cdist(XA, XB, metric='euclidean', out=None, **kwargs):
    """Compute distance between each pair of the two collections of inputs.

    Args:
        XA (array_like): An :math:`m_A` by :math:`n` array of :math:`m_A`
            original observations in an :math:`n`-dimensional space.
            Inputs are converted to float type.
        XB (array_like): An :math:`m_B` by :math:`n` array of :math:`m_B`
            original observations in an :math:`n`-dimensional space.
            Inputs are converted to float type.
        metric (str, optional): The distance metric to use.
            The distance function can be 'canberra', 'chebyshev',
            'cityblock', 'correlation', 'cosine', 'euclidean', 'hamming',
            'hellinger', 'jensenshannon', 'kl_divergence', 'matching',
            'minkowski', 'russellrao', 'sqeuclidean'.
        out (cupy.ndarray, optional): The output array. If not None, the
            distance matrix Y is stored in this array.
        **kwargs (dict, optional): Extra arguments to `metric`: refer to each
            metric documentation for a list of all possible arguments.
            Some possible arguments:
            p (float): The p-norm to apply for Minkowski, weighted and
            unweighted. Default: 2.0

    Returns:
        Y (cupy.ndarray): A :math:`m_A` by :math:`m_B` distance matrix is
            returned. For each :math:`i` and :math:`j`, the metric
            ``dist(u=XA[i], v=XB[j])`` is computed and stored in the
            :math:`ij` th entry.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
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

    p = kwargs["p"] if "p" in kwargs else 2.0

    if out is not None:
        if out.dtype != 'float32':
            out = out.astype('float32', copy=False)
        if out.shape != (mA, mB):
            cupy.resize(out, (mA, mB))
        out[:] = 0.0

    if isinstance(metric, str):
        mstr = metric.lower()
        metric_info = _METRIC_ALIAS.get(mstr, None)
        if metric_info is not None:
            output_arr = out if out is not None else cupy.zeros((mA, mB),
                                                                dtype=XA.dtype)
            pairwise_distance(XA, XB, output_arr, metric, p=p)
            return output_arr
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier')


def pdist(X, metric='euclidean', *, out=None, **kwargs):
    """Compute distance between observations in n-dimensional space.

    Args:
        X (array_like): An :math:`m` by :math:`n` array of :math:`m`
            original observations in an :math:`n`-dimensional space.
            Inputs are converted to float type.
        metric (str, optional): The distance metric to use.
            The distance function can be 'canberra', 'chebyshev',
            'cityblock', 'correlation', 'cosine', 'euclidean', 'hamming',
            'hellinger', 'jensenshannon', 'kl_divergence', 'matching',
            'minkowski', 'russellrao', 'sqeuclidean'.
        out (cupy.ndarray, optional): The output array. If not None, the
            distance matrix Y is stored in this array.
        **kwargs (dict, optional): Extra arguments to `metric`: refer to each
            metric documentation for a list of all possible arguments.
            Some possible arguments:
            p (float): The p-norm to apply for Minkowski, weighted and
            unweighted. Default: 2.0

    Returns:
        Y (cupy.ndarray):
            A :math:`m` by :math:`m` distance matrix is
            returned. For each :math:`i` and :math:`j`, the metric
            ``dist(u=X[i], v=X[j])`` is computed and stored in the
            :math:`ij` th entry.
    """
    all_dist = cdist(X, X, metric=metric, out=out, **kwargs)
    up_idx = cupy.triu_indices_from(all_dist, 1)
    return all_dist[up_idx]


def distance_matrix(x, y, p=2.0):
    """Compute the distance matrix.

    Returns the matrix of all pair-wise distances.

    Args:
        x (array_like): Matrix of M vectors in K dimensions.
        y (array_like): Matrix of N vectors in K dimensions.
        p (float): Which Minkowski p-norm to use (1 <= p <= infinity).
            Default=2.0
    Returns:
        result (cupy.ndarray): Matrix containing the distance from every
            vector in `x` to every vector in `y`, (size M, N).
    """
    x = cupy.asarray(x)
    m, k = x.shape
    y = cupy.asarray(y)
    n, kk = y.shape

    if k != kk:
        raise ValueError("x contains %d-dimensional vectors but y "
                         "contains %d-dimensional vectors" % (k, kk))

    return cdist(x, y, metric="minkowski", p=p)
