
import warnings

import cupy
from cupyx.scipy.spatial._kdtree_utils import (
    asm_kd_tree, compute_knn, compute_tree_bounds, find_nodes_in_radius)
from cupyx.scipy.spatial.distance import distance_matrix
from cupyx.scipy.sparse import coo_matrix


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
        self.bounds = cupy.empty((0,))
        if self.copy_query_points:
            if self.data.dtype != cupy.float64:
                raise ValueError('periodic KDTree is only available '
                                 'on float64')
            self.bounds = compute_tree_bounds(self.tree)

        self.mins = cupy.min(self.tree, axis=0)
        self.maxes = cupy.max(self.tree, axis=0)

    def query(self, x, k=1, eps=0.0, p=2.0, distance_upper_bound=cupy.inf):
        r"""
        Query the kd-tree for nearest neighbors.

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

    def query_ball_point(self, x, r, p=2., eps=0, return_sorted=None,
                         return_length=False):
        """
        Find all points within distance r of point(s) x.

        Parameters
        ----------
        x : array_like, shape tuple + (self.m,)
            The point or points to search for neighbors of.
        r : array_like, float
            The radius of points to return, shall broadcast to the length of x.
        p : float, optional
            Which Minkowski p-norm to use.  Should be in the range [1, inf].
            A finite large p may cause a ValueError if overflow can occur.
        eps : nonnegative float, optional
            Approximate search. Branches of the tree are not explored if their
            nearest points are further than ``r / (1 + eps)``, and branches are
            added in bulk if their furthest points are nearer than
            ``r * (1 + eps)``.
        return_sorted : bool, optional
            Sorts returned indices if True and does not sort them if False. If
            None, does not sort single point queries, but does sort
            multi-point queries which was the behavior before this option
            was added in SciPy.
        return_length: bool, optional
            Return the number of points inside the radius instead of a list
            of the indices.

        Returns
        -------
        results : list or array of lists
            If `x` is a single point, returns a list of the indices of the
            neighbors of `x`. If `x` is an array of points, returns an object
            array of shape tuple containing lists of neighbors.

        Notes
        -----
        If you have many points whose neighbors you want to find, you may save
        substantial amounts of time by putting them in a KDTree and using
        query_ball_tree.

        Examples
        --------
        >>> import cupy as cp
        >>> from cupyx.scipy import spatial
        >>> x, y = cp.mgrid[0:4, 0:4]
        >>> points = cp.c_[x.ravel(), y.ravel()]
        >>> tree = spatial.KDTree(points)
        >>> tree.query_ball_point([2, 0], 1)
        [4, 8, 9, 12]
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

        return find_nodes_in_radius(
            x, tree, self.index, self.boxsize, self.bounds,
            r, p=p, eps=eps, return_sorted=return_sorted,
            return_length=return_length, adjust_to_box=self.copy_query_points)

    def query_ball_tree(self, other, r, p=2.0, eps=0.0):
        """
        Find all pairs of points between `self` and `other` whose distance
        is at most r.

        Parameters
        ----------
        other : KDTree instance
            The tree containing points to search against.
        r : float
            The maximum distance, has to be positive.
        p : float, optional
            Which Minkowski norm to use.  `p` has to meet the condition
            ``1 <= p <= infinity``.
            A finite large p may cause a ValueError if overflow can occur.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.

        Returns
        -------
        results : list of ndarrays
            For each element ``self.data[i]`` of this tree, ``results[i]`` is a
            list of the indices of its neighbors in ``other.data``.

        Examples
        --------
        You can search all pairs of points between two kd-trees within a
        distance:

        >>> import matplotlib.pyplot as plt
        >>> import cupy as cp
        >>> from cupyx.scipy.spatial import KDTree
        >>> points1 = cp.random.rand((15, 2))
        >>> points2 = cp.random.rand((15, 2))
        >>> plt.figure(figsize=(6, 6))
        >>> plt.plot(points1[:, 0], points1[:, 1], "xk", markersize=14)
        >>> plt.plot(points2[:, 0], points2[:, 1], "og", markersize=14)
        >>> kd_tree1 = KDTree(points1)
        >>> kd_tree2 = KDTree(points2)
        >>> indexes = kd_tree1.query_ball_tree(kd_tree2, r=0.2)
        >>> for i in range(len(indexes)):
        ...     for j in indexes[i]:
        ...         plt.plot([points1[i, 0], points2[j, 0]],
        ...             [points1[i, 1], points2[j, 1]], "-r")
        >>> plt.show()

        """
        return other.query_ball_point(
            self.data, r, p=p, eps=eps, return_sorted=True)

    def query_pairs(self, r, p=2.0, eps=0, output_type='ndarray'):
        """
        Find all pairs of points in `self` whose distance is at most r.

        Parameters
        ----------
        r : positive float
            The maximum distance.
        p : float, optional
            Which Minkowski norm to use.  ``p`` has to meet the condition
            ``1 <= p <= infinity``.
            A finite large p may cause a ValueError if overflow can occur.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.
        output_type : string, optional
            Choose the output container, 'set' or 'ndarray'. Default: 'ndarray'
            Note: 'set' output is not supported.

        Returns
        -------
        results : ndarray
            An ndarray of size ``(total_pairs, 2)``, containing each pair
            ``(i,j)``, with ``i < j``, for which the corresponding
            positions are close.

        Notes
        -----
        This method does not support the `set` output type.

        Examples
        --------
        You can search all pairs of points in a kd-tree within a distance:

        >>> import matplotlib.pyplot as plt
        >>> import cupy as cp
        >>> from cupyx.scipy.spatial import KDTree
        >>> points = cp.random.rand((20, 2))
        >>> plt.figure(figsize=(6, 6))
        >>> plt.plot(points[:, 0], points[:, 1], "xk", markersize=14)
        >>> kd_tree = KDTree(points)
        >>> pairs = kd_tree.query_pairs(r=0.2)
        >>> for (i, j) in pairs:
        ...     plt.plot([points[i, 0], points[j, 0]],
        ...             [points[i, 1], points[j, 1]], "-r")
        >>> plt.show()

        """
        if output_type == 'set':
            warnings.warn("output_type='set' is not supported by the GPU "
                          "implementation of KDTree, resorting back to "
                          "'ndarray'.")

        x = self.data
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

        return find_nodes_in_radius(
            x, tree, self.index, self.boxsize, self.bounds,
            r, p=p, eps=eps, return_sorted=True, return_tuples=True,
            adjust_to_box=self.copy_query_points)

    def count_neighbors(self, other, r, p=2.0, weights=None, cumulative=True):
        """
        Count how many nearby pairs can be formed.

        Count the number of pairs ``(x1,x2)`` can be formed, with ``x1`` drawn
        from ``self`` and ``x2`` drawn from ``other``, and where
        ``distance(x1, x2, p) <= r``.

        Data points on ``self`` and ``other`` are optionally weighted by the
        ``weights`` argument. (See below)

        This is adapted from the "two-point correlation" algorithm described by
        Gray and Moore [1]_.  See notes for further discussion.

        Parameters
        ----------
        other : KDTree instance
            The other tree to draw points from, can be the same tree as self.
        r : float or one-dimensional array of floats
            The radius to produce a count for. Multiple radii are searched with
            a single tree traversal.
            If the count is non-cumulative(``cumulative=False``), ``r`` defines
            the edges of the bins, and must be non-decreasing.
        p : float, optional
            1<=p<=infinity.
            Which Minkowski p-norm to use.
            Default 2.0.
            A finite large p may cause a ValueError if overflow can occur.
        weights : tuple, array_like, or None, optional
            If None, the pair-counting is unweighted.
            If given as a tuple, weights[0] is the weights of points in
            ``self``, and weights[1] is the weights of points in ``other``;
            either can be None to indicate the points are unweighted.
            If given as an array_like, weights is the weights of points in
            ``self`` and ``other``. For this to make sense, ``self`` and
            ``other`` must be the same tree. If ``self`` and ``other`` are two
            different trees, a ``ValueError`` is raised.
            Default: None
        cumulative : bool, optional
            Whether the returned counts are cumulative. When cumulative is set
            to ``False`` the algorithm is optimized to work with a large number
            of bins (>10) specified by ``r``. When ``cumulative`` is set to
            True, the algorithm is optimized to work with a small number of
            ``r``. Default: True

        Returns
        -------
        result : scalar or 1-D array
            The number of pairs. For unweighted counts, the result is integer.
            For weighted counts, the result is float.
            If cumulative is False, ``result[i]`` contains the counts with
            ``(-inf if i == 0 else r[i-1]) < R <= r[i]``

        """
        raise NotImplementedError('count_neighbors is not available on CuPy')

    def sparse_distance_matrix(self, other, max_distance, p=2.0,
                               output_type='coo_matrix'):
        """
        Compute a sparse distance matrix

        Computes a distance matrix between two KDTrees, leaving as zero
        any distance greater than max_distance.

        Parameters
        ----------
        other : KDTree
        max_distance : positive float
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use.
            A finite large p may cause a ValueError if overflow can occur.
        output_type : string, optional
            Which container to use for output data. Options:
            'coo_matrix' or 'ndarray'. Default: 'coo_matrix'.

        Returns
        -------
        result : coo_matrix or ndarray
            Sparse matrix representing the results in "dictionary of keys"
            format. If output_type is 'ndarray' an NxM distance matrix will be
            returned.

        Examples
        --------
        You can compute a sparse distance matrix between two kd-trees:

        >>> import cupy
        >>> from cupyx.scipy.spatial import KDTree
        >>> points1 = cupy.random.rand((5, 2))
        >>> points2 = cupy.random.rand((5, 2))
        >>> kd_tree1 = KDTree(points1)
        >>> kd_tree2 = KDTree(points2)
        >>> sdm = kd_tree1.sparse_distance_matrix(kd_tree2, 0.3)
        >>> sdm.toarray()
        array([[0.        , 0.        , 0.12295571, 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.28942611, 0.        , 0.        , 0.2333084 , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.24617575, 0.29571802, 0.26836782, 0.        , 0.        ]])

        You can check distances above the `max_distance` are zeros:

        >>> from cupyx.scipy.spatial import distance_matrix
        >>> distance_matrix(points1, points2)
        array([[0.56906522, 0.39923701, 0.12295571, 0.8658745 , 0.79428925],
           [0.37327919, 0.7225693 , 0.87665969, 0.32580855, 0.75679479],
           [0.28942611, 0.30088013, 0.6395831 , 0.2333084 , 0.33630734],
           [0.31994999, 0.72658602, 0.71124834, 0.55396483, 0.90785663],
           [0.24617575, 0.29571802, 0.26836782, 0.57714465, 0.6473269 ]])
        """
        if output_type not in {'coo_matrix', 'ndarray'}:
            raise ValueError(
                "sparse_distance_matrix only supports 'coo_matrix' and "
                "'ndarray' outputs")

        dist = distance_matrix(self.data, other.data, p)
        dist[dist > max_distance] = 0

        if output_type == 'coo_matrix':
            return coo_matrix(dist)
        return dist
