
import cupy
from cupyx.scipy.spatial.delaunay_2d._tri import GDel2D


class Delaunay:
    """
    Delaunay tessellation in 2 dimensions.

    Parameters
    ----------
    points : ndarray of floats, shape (npoints, ndim)
        Coordinates of points to triangulate
    furthest_site : bool, optional
        Whether to compute a furthest-site Delaunay triangulation. This option
        will be ignored, since it is not supported by CuPy
        Default: False
    incremental : bool, optional
        Allow adding new points incrementally. This takes up some additional
        resources. This option will be ignored, since it is not supported by
        CuPy. Default: False

    Attributes
    ----------
    points : ndarray of double, shape (npoints, ndim)
        Coordinates of input points.
    simplices : ndarray of ints, shape (nsimplex, ndim+1)
        Indices of the points forming the simplices in the triangulation.
        For 2-D, the points are oriented counterclockwise.
    neighbors : ndarray of ints, shape (nsimplex, ndim+1)
        Indices of neighbor simplices for each simplex.
        The kth neighbor is opposite to the kth vertex.
        For simplices at the boundary, -1 denotes no neighbor.0
    vertex_neighbor_vertices : tuple of two ndarrays of int; (indptr, indices)
        Neighboring vertices of vertices. The indices of neighboring
        vertices of vertex `k` are ``indices[indptr[k]:indptr[k+1]]``.

    Notes
    -----
    This implementation makes use of GDel2D to perform the triangulation in 2D.
    See [1]_ for more information.

    References
    ----------
    .. [1] A GPU accelerated algorithm for 3D Delaunay triangulation (2014).
        Thanh-Tung Cao, Ashwin Nanjappa, Mingcen Gao, Tiow-Seng Tan.
        Proc. 18th ACM SIGGRAPH Symp. Interactive 3D Graphics and Games, 47-55.
    """

    def __init__(self, points, furthest_site=False, incremental=False):
        if points.shape[-1] != 2:
            raise ValueError('Delaunay only supports 2D inputs at the moment.')

        if furthest_site:
            raise ValueError(
                'furthest_site argument is not supported by CuPy.')

        if incremental:
            raise ValueError(
                'incremental argument is not supported by CuPy.')

        self.points = points
        self._triangulator = GDel2D(self.points)
        self.simplices, self.neighbors = self._triangulator.compute()

    def _find_simplex_coordinates(self, xi, eps, find_coords=False):
        """
        Find the simplices containing the given points.

        Parameters
        ----------
        xi : ndarray of double, shape (..., ndim)
            Points to locate
        eps: float
            Tolerance allowed in the inside-triangle check.
        find_coords: bool, optional
            Whether to return the barycentric coordinates of `xi`
            with respect to the found simplices.

        Returns
        -------
        i : ndarray of int, same shape as `xi`
            Indices of simplices containing each point.
            Points outside the triangulation get the value -1.
        c : ndarray of float64, same shape as `xi`, optional
            Barycentric coordinates of `xi` with respect to the enclosing
            simplices. Returned only when `find_coords` is True.
        """
        out = cupy.empty((xi.shape[0],), dtype=cupy.int32)
        c = cupy.empty(0, dtype=cupy.float64)
        if find_coords:
            c = cupy.empty((xi.shape[0], xi.shape[-1] + 1), dtype=cupy.float64)

        out, c = self._triangulator.find_point_in_triangulation(
            xi, eps, find_coords)

        if find_coords:
            return out, c
        return out

    def find_simplex(self, xi, bruteforce=False, tol=None):
        """
        Find the simplices containing the given points.

        Parameters
        ----------
        xi : ndarray of double, shape (..., ndim)
            Points to locate
        bruteforce : bool, optional
            Whether to only perform a brute-force search. Not used by CuPy
        tol : float, optional
            Tolerance allowed in the inside-triangle check.
            Default is ``100*eps``.

        Returns
        -------
        i : ndarray of int, same shape as `xi`
            Indices of simplices containing each point.
            Points outside the triangulation get the value -1.
        """
        if tol is None:
            eps = 100 * cupy.finfo(cupy.double).eps
        else:
            eps = float(tol)

        if xi.shape[-1] != 2:
            raise ValueError('Delaunay only supports 2D inputs at the moment.')

        return self._find_simplex_coordinates(xi, eps)

    def vertex_neighbor_vertices(self):
        """
        Neighboring vertices of vertices.

        Tuple of two ndarrays of int: (indptr, indices). The indices of
        neighboring vertices of vertex `k` are
        ``indices[indptr[k]:indptr[k+1]]``.
        """
        return self._triangulator.vertex_neighbor_vertices()
