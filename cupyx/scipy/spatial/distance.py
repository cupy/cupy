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
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, "minkowski", p)

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
