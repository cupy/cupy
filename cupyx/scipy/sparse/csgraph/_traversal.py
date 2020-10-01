import numpy

import cupy

from cupy.linalg import util
import cupyx.scipy.sparse


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
    if connection.lower() not in ('weak', 'strong'):
        raise ValueError("connection must be 'weak' or 'strong'")

    if connection.lower() == 'weak':
        directed = False

    if not cupyx.scipy.sparse.isspmatrix_csr(csgraph):
        csgraph = cupyx.scipy.sparse.csr_matrix(csgraph)
    util._assert_nd_squareness(csgraph)

    return _connected_components(csgraph, directed, return_labels)


def _connected_components(csgraph, directed, return_labels):
    m = csgraph.shape[0]
    labels = cupy.arange(m, dtype=numpy.int32)
    count = cupy.zeros((2,), dtype=numpy.int32)
    if directed:
        # Note: The following implementation is naive and may require a lot of
        # memory to run.
        nnz = csgraph.nnz
        data = cupy.ones((nnz,), dtype=numpy.float32)
        csgraph = cupyx.scipy.sparse.csr_matrix(
            (data, csgraph.indices, csgraph.indptr), shape=csgraph.shape)
        csgraph += csgraph * csgraph
        while csgraph.nnz > nnz:
            nnz = csgraph.nnz
            csgraph.data[:] = 1
            csgraph += csgraph * csgraph
        _cupy_connect_directed(csgraph.indices, csgraph.indptr, m, labels,
                               size=csgraph.nnz)
    else:
        _cupy_connect_undirected(csgraph.indices, csgraph.indptr, m, labels,
                                 size=csgraph.nnz)
    _cupy_count_components(labels, count, size=m)
    n = int(count[0])
    if not return_labels:
        return n
    root_labels = cupy.empty((n,), dtype=numpy.int32)
    _cupy_get_root_labels(labels, count, root_labels, size=labels.size)
    _cupy_adjust_labels(n, cupy.sort(root_labels), labels)
    return n, labels


_GET_ROW_ID_ = '''
__device__ inline int get_row_id(int i, int min, int max, const int *indptr) {
    int row = (min + max) / 2;
    while (min < max) {
        if (i < indptr[row]) {
            max = row - 1;
        } else if (i >= indptr[row + 1]) {
            min = row + 1;
        } else {
            break;
        }
        row = (min + max) / 2;
    }
    return row;
}
'''

_CONNECT_NODES_ = '''
__device__ inline void connect_nodes(int i, int j, int *labels) {
    while (true) {
        while (i != labels[i]) { i = labels[i]; }
        while (j != labels[j]) { j = labels[j]; }
        if (i == j) break;
        if (i < j) {
            int old = atomicCAS( &labels[j], j, i );
            if (old == j) break;
            j = old;
        } else {
            int old = atomicCAS( &labels[i], i, j );
            if (old == i) break;
            i = old;
        }
    }
}
'''

_cupy_connect_directed = cupy.core.ElementwiseKernel(
    'raw I INDICES, raw I INDPTR, int32 M',
    'raw O LABELS',
    '''
    int row = get_row_id(i, 0, M - 1, &(INDPTR[0]));
    int col = INDICES[i];
    if (row <= col) continue;

    // check if there is a connection from "col" to "row"
    int r = -1;
    int p_min = INDPTR[col];
    int p_max = INDPTR[col + 1] - 1;
    while (p_min <= p_max) {
        int p = (p_min + p_max) / 2;
        r = INDICES[p];
        if (r == row) {
            connect_nodes(row, col, &(LABELS[0]));
            break;
        }
        if (r < row) {
            p_min = p + 1;
        } else {
            p_max = p - 1;
        }
    }
    ''',
    '_cupy_connect_directed',
    preamble=_GET_ROW_ID_ + _CONNECT_NODES_
)

_cupy_connect_undirected = cupy.core.ElementwiseKernel(
    'raw I INDICES, raw I INDPTR, int32 M',
    'raw O LABELS',
    '''
    int row = get_row_id(i, 0, M - 1, &(INDPTR[0]));
    int col = INDICES[i];
    if (row == col) continue;
    connect_nodes(row, col, &(LABELS[0]));
    ''',
    '_cupy_connect_undirected',
    preamble=_GET_ROW_ID_ + _CONNECT_NODES_
)

_cupy_count_components = cupy.core.ElementwiseKernel(
    '',
    'raw I LABELS, raw int32 count',
    '''
    int j = i;
    while (j != LABELS[j]) { j = LABELS[j]; }
    if (j != i) LABELS[i] = j;
    else atomicAdd(&count[0], 1);
    ''',
    '_cupy_count_components')

_cupy_get_root_labels = cupy.core.ElementwiseKernel(
    '',
    'raw I LABELS, raw int32 count, raw int32 ROOT_LABELS',
    '''
    if (LABELS[i] == i) {
        int j = atomicAdd(&count[1], 1);
        ROOT_LABELS[j] = i;
    }
    ''',
    '_cupy_get_root_labels')

_cupy_adjust_labels = cupy.core.ElementwiseKernel(
    'int32 N, raw I ROOT_LABELS',
    'I LABELS',
    '''
    int cur_label = LABELS;
    int j_min = 0;
    int j_max = N - 1;
    int j = (j_min + j_max) / 2;
    while (j_min < j_max) {
        if (cur_label == ROOT_LABELS[j]) break;
        if (cur_label < ROOT_LABELS[j]) {
            j_max = j - 1;
        } else {
            j_min = j + 1;
        }
        j = (j_min + j_max) / 2;
    }
    LABELS = j;
    ''',
    '_cupy_adjust_labels')
