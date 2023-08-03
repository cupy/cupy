
import warnings

import cupy
from cupyx.scipy.spatial._kdtree_utils import (
    asm_kd_tree, compute_knn, compute_tree_bounds)


def broadcast_contiguous(x, shape, dtype):
    """Broadcast ``x`` to ``shape`` and make contiguous, possibly by copying"""
    # Avoid copying if possible
    try:
        if x.shape == shape:
            return cupy.ascontiguousarray(x, dtype)
    except AttributeError:
        pass
    # Assignment will broadcast automatically
    ret = cupy.empty(shape, dtype)
    ret[...] = x
    return ret


class KDTree:
    """
    KDTree(data, leafsize=16, compact_nodes=True, copy_data=False,
            balanced_tree=True, boxsize=None)

    kd-tree for quick nearest-neighbor lookup

    This class provides an index into a set of k-dimensional points
    which can be used to rapidly look up the nearest neighbors of any
    point.

    Parameters
    ----------
    data : array_like, shape (n,m)
        The n data points of dimension m to be indexed. This array is
        not copied unless this is necessary to produce a contiguous
        array of doubles, and so modifying this data will result in
        bogus results. The data are also copied if the kd-tree is built
        with copy_data=True.
    leafsize : positive int, optional
        The number of points at which the algorithm switches over to
        brute-force. Default: 16.
    compact_nodes : bool, optional
        If True, the kd-tree is built to shrink the hyperrectangles to
        the actual data range. This usually gives a more compact tree that
        is robust against degenerated input data and gives faster queries
        at the expense of longer build time. Default: True.
    copy_data : bool, optional
        If True the data is always copied to protect the kd-tree against
        data corruption. Default: False.
    balanced_tree : bool, optional
        If True, the median is used to split the hyperrectangles instead of
        the midpoint. This usually gives a more compact tree and
        faster queries at the expense of longer build time. Default: True.
    boxsize : array_like or scalar, optional
        Apply a m-d toroidal topology to the KDTree.. The topology is generated
        by :math:`x_i + n_i L_i` where :math:`n_i` are integers and :math:`L_i`
        is the boxsize along i-th dimension. The input data shall be wrapped
        into :math:`[0, L_i)`. A ValueError is raised if any of the data is
        outside of this bound.

    Notes
    -----
    The algorithm used is described in Wald, I. 2022 [1]_.
    The general idea is that the kd-tree is a binary tree, each of whose
    nodes represents an axis-aligned hyperrectangle. Each node specifies
    an axis and splits the set of points based on whether their coordinate
    along that axis is greater than or less than a particular value.

    The tree can be queried for the r closest neighbors of any given point
    (optionally returning only those within some maximum distance of the
    point). It can also be queried, with a substantial gain in efficiency,
    for the r approximate closest neighbors. See [2]_ for more information
    regarding the implementation.

    For large dimensions (20 is already large) do not expect this to run
    significantly faster than brute force. High-dimensional nearest-neighbor
    queries are a substantial open problem in computer science.

    Attributes
    ----------
    data : ndarray, shape (n,m)
        The n data points of dimension m to be indexed. This array is
        not copied unless this is necessary to produce a contiguous
        array of doubles. The data are also copied if the kd-tree is built
        with `copy_data=True`.
    leafsize : positive int
        The number of points at which the algorithm switches over to
        brute-force.
    m : int
        The dimension of a single data-point.
    n : int
        The number of data points.
    maxes : ndarray, shape (m,)
        The maximum value in each dimension of the n data points.
    mins : ndarray, shape (m,)
        The minimum value in each dimension of the n data points.
    tree : ndarray
        This attribute exposes the array representation of the tree.
    size : int
        The number of nodes in the tree.

    References
    ----------
    .. [1] Wald, I., GPU-friendly, Parallel, and (Almost-)In-Place
           Construction of Left-Balanced k-d Trees, 2022.
           doi:10.48550/arXiv.2211.00120.
    .. [2] Wald, I., A Stack-Free Traversal Algorithm for Left-Balanced
           k-d Trees, 2022. doi:10.48550/arXiv.2210.12859.
    """

    def __init__(self, data, leafsize=10, compact_nodes=True, copy_data=False,
                 balanced_tree=True, boxsize=None):
        self.data = data
        if copy_data:
            self.data = self.data.copy()

        if not balanced_tree:
            warnings.warn('balanced_tree=False is not supported by the GPU '
                          'implementation of KDTree, skipping.')

        self.copy_query_points = False
        self.n, self.m = self.data.shape

        self.boxsize = cupy.full(self.m, cupy.inf, dtype=cupy.float64)
        # self.boxsize_data = None

        if boxsize is not None:
            # self.boxsize_data = cupy.empty(self.m, dtype=data.dtype)
            self.copy_query_points = True
            boxsize = broadcast_contiguous(boxsize, shape=(self.m,),
                                           dtype=cupy.float64)
            # self.boxsize_data[:self.m] = boxsize
            # self.boxsize_data[self.m:] = 0.5 * boxsize

            self.boxsize = boxsize
            periodic_mask = self.boxsize > 0
            if ((self.data >= self.boxsize[None, :])[:, periodic_mask]).any():
                raise ValueError(
                    "Some input data are greater than the size of the "
                    "periodic box.")
            if ((self.data < 0)[:, periodic_mask]).any():
                raise ValueError("Negative input data are outside of the "
                                 "periodic box.")

        self.tree, self.index = asm_kd_tree(self.data)
        self.bounds = compute_tree_bounds(self.tree)
        self.mins = self.bounds[0, :, 0]
        self.maxes = self.bounds[0, :, 1]

    def query(self, x, k=1, eps=0.0, p=2.0, distance_upper_bound=cupy.inf):
        """
        Query the kd-tree for nearest neighbors

        Parameters
        ----------
        x : array_like, last dimension self.m
            An array of points to query.
        k : list of integer or integer
            The list of k-th nearest neighbors to return. If k is an
            integer it is treated as a list of [1, ... k] (range(1, k+1)).
            Note that the counting starts from 1.
        eps : non-negative float
            Return approximate nearest neighbors; the k-th returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real k-th nearest neighbor.
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use.
            1 is the sum-of-absolute-values "Manhattan" distance
            2 is the usual Euclidean distance
            infinity is the maximum-coordinate-difference distance
            A finite large p may cause a ValueError if overflow can occur.
        distance_upper_bound : nonnegative float
            Return only neighbors within this distance.  This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point.

        Returns
        -------
        d : array of floats
            The distances to the nearest neighbors.
            If ``x`` has shape ``tuple+(self.m,)``, then ``d`` has shape
            ``tuple+(k,)``. When k == 1, the last dimension of the output is
            squeezed. Missing neighbors are indicated with infinite distances.
        i : ndarray of ints
            The index of each neighbor in ``self.data``.
            If ``x`` has shape ``tuple+(self.m,)``, then ``i`` has shape
            ``tuple+(k,)``. When k == 1, the last dimension of the output is
            squeezed. Missing neighbors are indicated with ``self.n``.

        Notes
        -----
        If the KD-Tree is periodic, the position ``x`` is wrapped into the
        box.

        When the input k is a list, a query for arange(max(k)) is performed,
        but only columns that store the requested values of k are preserved.
        This is implemented in a manner that reduces memory usage.

        Examples
        --------

        >>> import cupy as cp
        >>> from cupyx.scipy.spatial import KDTree
        >>> x, y = cp.mgrid[0:5, 2:8]
        >>> tree = KDTree(cp.c_[x.ravel(), y.ravel()])

        To query the nearest neighbours and return squeezed result, use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=1)
        >>> print(dd, ii, sep='\n')
        [2.         0.2236068]
        [ 0 13]

        To query the nearest neighbours and return unsqueezed result, use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[1])
        >>> print(dd, ii, sep='\n')
        [[2.        ]
         [0.2236068]]
        [[ 0]
         [13]]

        To query the second nearest neighbours and return unsqueezed result,
        use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[2])
        >>> print(dd, ii, sep='\n')
        [[2.23606798]
         [0.80622577]]
        [[ 6]
         [19]]

        To query the first and second nearest neighbours, use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=2)
        >>> print(dd, ii, sep='\n')
        [[2.         2.23606798]
         [0.2236068  0.80622577]]
        [[ 0  6]
         [13 19]]

        or, be more specific

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[1, 2])
        >>> print(dd, ii, sep='\n')
        [[2.         2.23606798]
         [0.2236068  0.80622577]]
        [[ 0  6]
         [13 19]]
        """
        if self.copy_query_points:
            if x.dtype != cupy.float64:
                raise ValueError('periodic KDTree is only available '
                                 'on float64')
            x = x.copy()

        common_dtype = cupy.result_type(self.tree.dtype, x.dtype)
        tree = self.tree
        if cupy.dtype(self.tree.dtype) is not common_dtype:
            tree = self.tree.astype(common_dtype)
        if cupy.dtype(x.dtype) is not common_dtype:
            x = x.astype(common_dtype)

        if not isinstance(k, list):
            try:
                k = int(k)
            except TypeError:
                raise ValueError('k must be an integer or list of integers')

        return compute_knn(
            x, tree, self.index, self.boxsize, self.bounds, k=k,
            eps=float(eps), p=float(p),
            distance_upper_bound=distance_upper_bound,
            adjust_to_box=self.copy_query_points)
