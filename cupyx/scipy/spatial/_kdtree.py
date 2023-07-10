
import warnings

from cupyx.scipy.spatial._kdtree_utils import asm_kd_tree


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

    During construction, the axis and splitting point are chosen by the
    "sliding midpoint" rule, which ensures that the cells do not all
    become long and thin.

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

        self.m, self.n = self.data.shape
        self.size = self.m
        self.tree, self.index = asm_kd_tree(self.data)
