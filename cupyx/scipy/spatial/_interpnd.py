
import cupy
from cupyx.scipy.spatial._delaunay import Delaunay


def _ndim_coords_from_arrays(points, ndim=None):
    """Convert a tuple of coordinate arrays to a (..., ndim)-shaped array."""

    if isinstance(points, tuple) and len(points) == 1:
        # handle argument tuple
        points = points[0]
    if isinstance(points, tuple):
        p = cupy.broadcast_arrays(*points)
        n = len(p)
        for j in range(1, n):
            if p[j].shape != p[0].shape:
                raise ValueError(
                    "coordinate arrays do not have the same shape")
        points = cupy.empty(p[0].shape + (len(points),), dtype=float)
        for j, item in enumerate(p):
            points[..., j] = item
    else:
        points = cupy.asanyarray(points)
        if points.ndim == 1:
            if ndim is None:
                points = points.reshape(-1, 1)
            else:
                points = points.reshape(-1, ndim)
    return points


def _check_init_shape(points, values, ndim=None):
    """
    Check shape of points and values arrays

    """
    if values.shape[0] != points.shape[0]:
        raise ValueError("different number of values and points")
    if points.ndim != 2:
        raise ValueError("invalid shape for input data points")
    if points.shape[1] < 2:
        raise ValueError("input data must be at least 2-D")
    if ndim is not None and points.shape[1] != ndim:
        raise ValueError("this mode of interpolation available only for "
                         "%d-D data" % ndim)


class NDInterpolatorBase:
    """Common routines for interpolators."""

    def __init__(self, points, values, fill_value=cupy.nan, ndim=None,
                 rescale=False, need_contiguous=True, need_values=True):
        """
        Check shape of points and values arrays, and reshape values to
        (npoints, nvalues).  Ensure the `points` and values arrays are
        C-contiguous, and of correct type.
        """

        if isinstance(points, Delaunay):
            # Precomputed triangulation was passed in
            if rescale:
                raise ValueError("Rescaling is not supported when passing "
                                 "a Delaunay triangulation as ``points``.")
            self.tri = points
            points = points.points
        else:
            self.tri = None

        points = _ndim_coords_from_arrays(points)

        if need_contiguous:
            points = cupy.ascontiguousarray(points, dtype=cupy.float64)

        if not rescale:
            self.scale = None
            self.points = points
        else:
            # scale to unit cube centered at 0
            self.offset = cupy.mean(points, axis=0)
            self.points = points - self.offset
            self.scale = cupy.ptp(points, axis=0)
            self.scale[~(self.scale > 0)] = 1.0  # avoid division by 0
            self.points /= self.scale

        self._calculate_triangulation(self.points)

        if need_values or values is not None:
            self._set_values(values, fill_value, need_contiguous, ndim)
        else:
            self.values = None

    def _calculate_triangulation(self, points):
        pass

    def _set_values(self, values, fill_value=cupy.nan,
                    need_contiguous=True, ndim=None):
        values = cupy.asarray(values)
        _check_init_shape(self.points, values, ndim=ndim)

        self.values_shape = values.shape[1:]
        if values.ndim == 1:
            self.values = values[:, None]
        elif values.ndim == 2:
            self.values = values
        else:
            self.values = values.reshape(values.shape[0],
                                         cupy.prod(values.shape[1:]))

        # Complex or real?
        self.is_complex = cupy.issubdtype(
            self.values.dtype, cupy.complexfloating)
        if self.is_complex:
            if need_contiguous:
                self.values = cupy.ascontiguousarray(self.values,
                                                     dtype=cupy.complex128)
            self.fill_value = complex(fill_value)
        else:
            if need_contiguous:
                self.values = cupy.ascontiguousarray(
                    self.values, dtype=cupy.float64
                )
            self.fill_value = float(fill_value)

    def _check_call_shape(self, xi):
        xi = cupy.asanyarray(xi)
        if xi.shape[-1] != self.points.shape[1]:
            raise ValueError("number of dimensions in xi does not match x")
        return xi

    def _scale_x(self, xi):
        if self.scale is None:
            return xi
        else:
            return (xi - self.offset) / self.scale

    def _preprocess_xi(self, *args):
        xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
        xi = self._check_call_shape(xi)
        interpolation_points_shape = xi.shape
        xi = xi.reshape(-1, xi.shape[-1])
        xi = cupy.ascontiguousarray(xi, dtype=cupy.float64)
        return self._scale_x(xi), interpolation_points_shape

    def _find_simplicies(self, xi):
        return self.tri._find_simplex_coordinates(xi, find_coords=True)

    def __call__(self, *args):
        """
        interpolator(xi)

        Evaluate interpolator at given points.

        Parameters
        ----------
        x1, x2, ... xn: array-like of float
            Points where to interpolate data at.
            x1, x2, ... xn can be array-like of float with broadcastable shape.
            or x1 can be array-like of float with shape ``(..., ndim)``
        """
        xi, interpolation_points_shape = self._preprocess_xi(*args)

        if self.is_complex:
            r = self._evaluate_complex(xi)
        else:
            r = self._evaluate_double(xi)

        return cupy.asarray(r).reshape(
            interpolation_points_shape[:-1] + self.values_shape)


# ------------------------------------------------------------------------------
# Linear interpolation in N-D
# ------------------------------------------------------------------------------

class LinearNDInterpolator(NDInterpolatorBase):
    """
    LinearNDInterpolator(points, values, fill_value=cupy.nan, rescale=False)

    Piecewise linear interpolant in N > 1 dimensions.

    Methods
    -------
    __call__

    Parameters
    ----------
    points : ndarray of floats, shape (npoints, ndims); or :class:`Delaunay`
        2-D array of data point coordinates, or a precomputed
        Delaunay triangulation.
    values : ndarray of float or complex, shape (npoints, ...), optional
        N-D array of data values at `points`.  The length of `values` along the
        first axis must be equal to the length of `points`. Unlike some
        interpolators, the interpolation axis cannot be changed.
    fill_value : float, optional
        Value used to fill in for requested points outside of the
        convex hull of the input points.  If not provided, then
        the default is ``nan``.
    rescale : bool, optional
        Rescale points to unit cube before performing interpolation.
        This is useful if some of the input dimensions have
        incommensurable units and differ by many orders of magnitude.

    Notes
    -----
    The interpolant is constructed by triangulating the input data
    with GDel2D [1]_, and on each triangle performing linear
    barycentric interpolation.

    .. note:: For data on a regular grid use `interpn` instead.

    Examples
    --------
    We can interpolate values on a 2D plane:

    >>> from scipy.interpolate import LinearNDInterpolator
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.random(10) - 0.5
    >>> y = rng.random(10) - 0.5
    >>> z = np.hypot(x, y)
    >>> X = np.linspace(min(x), max(x))
    >>> Y = np.linspace(min(y), max(y))
    >>> X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    >>> interp = LinearNDInterpolator(list(zip(x, y)), z)
    >>> Z = interp(X, Y)
    >>> plt.pcolormesh(X, Y, Z, shading='auto')
    >>> plt.plot(x, y, "ok", label="input point")
    >>> plt.legend()
    >>> plt.colorbar()
    >>> plt.axis("equal")
    >>> plt.show()

    See also
    --------
    griddata :
        Interpolate unstructured D-D data.
    NearestNDInterpolator :
        Nearest-neighbor interpolation in N dimensions.
    CloughTocher2DInterpolator :
        Piecewise cubic, C1 smooth, curvature-minimizing interpolant in 2D.
    interpn : Interpolation on a regular grid or rectilinear grid.
    RegularGridInterpolator : Interpolation on a regular or rectilinear grid
                              in arbitrary dimensions (`interpn` wraps this
                              class).

    References
    ----------
    .. [1] A GPU accelerated algorithm for 3D Delaunay triangulation (2014).
    Thanh-Tung Cao, Ashwin Nanjappa, Mingcen Gao, Tiow-Seng Tan.
    Proc. 18th ACM SIGGRAPH Symp. Interactive 3D Graphics and Games, 47-55.

    """

    def __init__(self, points, values, fill_value=cupy.nan, rescale=False):
        NDInterpolatorBase.__init__(
            self, points, values, fill_value=fill_value, rescale=rescale)

    def _calculate_triangulation(self, points):
        self.tri = Delaunay(points)

    def _evaluate_double(self, xi):
        return self._do_evaluate(xi, 1.0)

    def _evaluate_complex(self, xi):
        return self._do_evaluate(xi, 1.0j)
