import cupy

from cupy.linalg import _util
import cupyx.scipy.sparse
try:
    from cupy_backends.cuda.libs import cugraph
    _cugraph_available = True
except ImportError:
    _cugraph_available = False


def connected_components(csgraph, directed=True, connection='weak',
                         return_labels=True):
    """Analyzes the connected components of a sparse graph
    Args:
        csgraph (cupy.ndarray of cupyx.scipy.sparse.csr_matrix): The adjacency
            matrix representing connectivity among nodes.
        directed (bool): If ``True``, it operates on a directed graph. If
            ``False``, it operates on an undirected graph.
        connection (str): ``'weak'`` or ``'strong'``. For directed graphs, the
            type of connection to use. Nodes i and j are "strongly" connected
            only when a path exists both from i to j and from j to i.
            If ``directed`` is ``False``, this argument is ignored.
        return_labels (bool): If ``True``, it returns the labels for each of
            the connected components.
    Returns:
        tuple of int and cupy.ndarray, or int:
            If ``return_labels`` == ``True``, returns a tuple ``(n, labels)``,
            where ``n`` is the number of connected components and ``labels`` is
            labels of each connected components. Otherwise, returns ``n``.
    .. seealso:: :func:`scipy.sparse.csgraph.connected_components`
    """
    if not _cugraph_available:
        raise RuntimeError('cugraph is not available')

    connection = connection.lower()
    if connection not in ('weak', 'strong'):
        raise ValueError("connection must be 'weak' or 'strong'")

    if not directed:
        connection = 'weak'

    if not cupyx.scipy.sparse.isspmatrix_csr(csgraph):
        csgraph = cupyx.scipy.sparse.csr_matrix(csgraph)
    _util._assert_nd_squareness(csgraph)
    m = csgraph.shape[0]
    if csgraph.nnz == 0:
        return m, cupy.arange(m, dtype=csgraph.indices.dtype)
    labels = cupy.empty(m, dtype=csgraph.indices.dtype)

    if connection == 'strong':
        cugraph.strongly_connected_components(csgraph, labels)
    else:
        csgraph += csgraph.T
        if not cupyx.scipy.sparse.isspmatrix_csr(csgraph):
            csgraph = cupyx.scipy.sparse.csr_matrix(csgraph)
        cugraph.weakly_connected_components(csgraph, labels)
        # Note: In the case of weak connection, cuGraph creates labels with a
        # start number of 1, so decrement the label number.
        labels -= 1

    count = cupy.zeros((1,), dtype=csgraph.indices.dtype)
    root_labels = cupy.empty((m,), dtype=csgraph.indices.dtype)
    _cupy_count_components(labels, count, root_labels, size=m)
    n = int(count[0])
    if not return_labels:
        return n
    _cupy_adjust_labels(n, cupy.sort(root_labels[:n]), labels)
    return n, labels


_cupy_count_components = cupy.ElementwiseKernel(
    '',
    'raw I labels, raw int32 count, raw int32 root_labels',
    '''
    int j = i;
    while (j != labels[j]) { j = labels[j]; }
    if (j != i) {
        labels[i] = j;
    } else {
        int k = atomicAdd(&count[0], 1);
        root_labels[k] = i;
    }
    ''',
    '_cupy_count_components')


_cupy_adjust_labels = cupy.ElementwiseKernel(
    'int32 n_root_labels, raw I root_labels',
    'I labels',
    '''
    int cur_label = labels;
    int j_min = 0;
    int j_max = n_root_labels - 1;
    int j = (j_min + j_max) / 2;
    while (j_min < j_max) {
        if (cur_label == root_labels[j]) break;
        if (cur_label < root_labels[j]) {
            j_max = j - 1;
        } else {
            j_min = j + 1;
        }
        j = (j_min + j_max) / 2;
    }
    labels = j;
    ''',
    '_cupy_adjust_labels')
