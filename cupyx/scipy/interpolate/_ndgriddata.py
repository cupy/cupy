"""
Convenience interface to N-D interpolation
"""
from __future__ import annotations


import cupy
from cupyx.scipy.interpolate._interpnd import (
    LinearNDInterpolator, NDInterpolatorBase,
    CloughTocher2DInterpolator, _ndim_coords_from_arrays)
from cupyx.scipy.spatial import KDTree


# -----------------------------------------------------------------------------
# Nearest-neighbor interpolation
# -----------------------------------------------------------------------------


class NearestNDInterpolator(NDInterpolatorBase):
    """NearestNDInterpolator(x, y).

    Nearest-neighbor interpolator in N > 1 dimensions.

    Parameters
    ----------
    x : (npoints, ndims) 2-D ndarray of floats
        Data point coordinates.
    y : (npoints, ) 1-D ndarray of float or complex
        Data values.
    rescale : boolean, optional
        Rescale points to unit cube before performing interpolation.
        This is useful if some of the input dimensions have
        incommensurable units and differ by many orders of magnitude.
    tree_options : dict, optional
        Options passed to the underlying ``cKDTree``.

    See Also
    --------
    griddata :
        Interpolate unstructured D-D data.
    LinearNDInterpolator :
        Piecewise linear interpolator in N dimensions.
    CloughTocher2DInterpolator :
        Piecewise cubic, C1 smooth, curvature-minimizing interpolator in 2D.
    interpn : Interpolation on a regular grid or rectilinear grid.
    RegularGridInterpolator : Interpolator on a regular or rectilinear grid
                              in arbitrary dimensions (`interpn` wraps this
                              class).

    Notes
    -----
    Uses ``cupyx.scipy.spatial.KDTree``

    .. note:: For data on a regular grid use `interpn` instead.

    Examples
    --------
    We can interpolate values on a 2D plane:

    >>> from scipy.interpolate import NearestNDInterpolator
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> rng = cupy.random.default_rng()
    >>> x = rng.random(10) - 0.5
    >>> y = rng.random(10) - 0.5
    >>> z = cupy.hypot(x, y)
    >>> X = cupy.linspace(min(x), max(x))
    >>> Y = cupy.linspace(min(y), max(y))
    >>> X, Y = cupy.meshgrid(X, Y)  # 2D grid for interpolation
    >>> interp = NearestNDInterpolator(list(zip(x, y)), z)
    >>> Z = interp(X, Y)
    >>> plt.pcolormesh(X, Y, Z, shading='auto')
    >>> plt.plot(x, y, "ok", label="input point")
    >>> plt.legend()
    >>> plt.colorbar()
    >>> plt.axis("equal")
    >>> plt.show()

    """

    def __init__(self, x, y, rescale=False, tree_options=None):
        NDInterpolatorBase.__init__(self, x, y, rescale=rescale,
                                    need_contiguous=False,
                                    need_values=False)
        if tree_options is None:
            tree_options = dict()
        self.tree = KDTree(self.points, **tree_options)
        self.values = cupy.asarray(y)

    def __call__(self, *args, **query_options):
        """
        Evaluate interpolator at given points.

        Parameters
        ----------
        x1, x2, ... xn : array-like of float
            Points where to interpolate data at.
            x1, x2, ... xn can be array-like of float with broadcastable shape.
            or x1 can be array-like of float with shape ``(..., ndim)``
        **query_options
            This allows ``eps``, ``p`` and ``distance_upper_bound``
            being passed to the KDTree's query function to be explicitly set.
            See `cupyx.scipy.spatial.KDTree.query` for an overview of
            the different options.

            .. versionadded:: 1.12.0

        """
        # For the sake of enabling subclassing, NDInterpolatorBase._set_xi
        # performs some operations which are not required by
        # NearestNDInterpolator.__call__, hence here we operate on xi directly,
        # without calling a parent class function.
        xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
        xi = self._check_call_shape(xi)
        xi = self._scale_x(xi)

        # We need to handle two important cases:
        # (1) the case where xi has trailing dimensions (..., ndim), and
        # (2) the case where y has trailing dimensions
        # We will first flatten xi to deal with case (1),
        # do the computation in flattened array while retaining y's
        # dimensionality, and then reshape the interpolated values back
        # to match xi's shape.

        # Flatten xi for the query
        xi_flat = xi.reshape(-1, xi.shape[-1])
        original_shape = xi.shape
        flattened_shape = xi_flat.shape

        # if distance_upper_bound is set to not be infinite,
        # then we need to consider the case where cKDtree
        # does not find any points within distance_upper_bound to return.
        # It marks those points as having infinte distance, which is what
        # will be used below to mask the array and return only the points
        # that were deemed to have a close enough neighbor to return
        # something useful.
        dist, i = self.tree.query(xi_flat, **query_options)
        valid_mask = cupy.isfinite(dist)

        # create a holder interp_values array and fill with nans.
        if self.values.ndim > 1:
            interp_shape = flattened_shape[:-1] + self.values.shape[1:]
        else:
            interp_shape = flattened_shape[:-1]

        if cupy.issubdtype(self.values.dtype, cupy.complexfloating):
            interp_values = cupy.full(
                interp_shape, cupy.nan, dtype=self.values.dtype)
        else:
            interp_values = cupy.full(interp_shape, cupy.nan)

        interp_values[valid_mask] = self.values[i[valid_mask], ...]

        if self.values.ndim > 1:
            new_shape = original_shape[:-1] + self.values.shape[1:]
        else:
            new_shape = original_shape[:-1]
        interp_values = interp_values.reshape(new_shape)

        return interp_values


def griddata(points, values, xi,
             method='linear', fill_value=cupy.nan, rescale=False):
    """
    Interpolate unstructured N-dimensional data using GPU-accelerated CuPy.

    This function performs interpolation of scattered data points onto a set of
    query points using GPU-accelerated algorithms, leveraging CuPy for
    high-performance computation.
    It supports nearest-neighbor, linear, and cubic interpolation methods
    for 1D and multi-dimensional data.

    Parameters
    ----------
    points : 2-D ndarray of floats with shape (n, D),
    or length D tuple of 1-D ndarrays with shape (n,)
        Data point coordinates. Must be a CuPy array
        or convertible to one.
    values : ndarray of float or complex, shape (n,)
        Data values corresponding to `points`.
        Must be a CuPy array or convertible to one.
    xi : 2-D ndarray of floats with shape (m, D),
    or length D tuple of ndarrays broadcastable to the same shape
        Points at which to interpolate data.
        Must be a CuPy array or convertible to one.
    method : {'linear', 'nearest', 'cubic'}, optional
        Method of interpolation. One of:

        ``nearest``
          Returns the value at the data point closest
          to the point of interpolation.
          See `NearestNDInterpolator` for more details.

        ``linear``
          Tessellates the input point set to N-D simplices
          and interpolates linearly on each simplex.
          See `LinearNDInterpolator` for more details.

        ``cubic``
          For 1D data, uses a cubic spline (see `CubicSpline`).
          For 2D data, uses a piecewise cubic,
          continuously differentiable (C1),
          and approximately curvature-minimizing polynomial surface
          (see `CloughTocher2DInterpolator`).
          Not supported for dimensions > 2.

    fill_value : float, optional
        Value used to fill in for requested points
        outside the convex hull of the input points.
        If not provided, the default is ``nan``.
        This option has no effect for the 'nearest' method.
    rescale : bool, optional
        Rescale points to the unit cube before performing interpolation.
        This is useful if some of the input dimensions
        have incommensurable units and differ by many orders of magnitude.

    Returns
    -------
    ndarray
        Array of interpolated values with shape determined by `xi`.

    Notes
    -----
    - This implementation is fully GPU-accelerated using CuPy,
        ensuring high performance for large datasets.
    - For 'nearest' and 'linear' methods, the function uses Delaunay
        triangulation on the GPU for multi-dimensional data.
    - For 'cubic' interpolation in 1D, it employs `CubicSpline`.
        In 2D, it uses `CloughTocher2DInterpolator`.
    - The `rescale` option normalizes input coordinates
        to a unit cube on the GPU, which may improve numerical stability.
    - All inputs (`points`, `values`, `xi`) are converted to CuPy arrays
        to ensure GPU compatibility.
    - For optimal performance, ensure inputs are already CuPy arrays
        to avoid CPU-GPU data transfers.

    Examples
    --------
    Suppose we want to interpolate the 2-D function:

    >>> def func(x, y):
    ...     return x * (1 - x) * cupy.cos(4 * cupy.pi * x) *
            cupy.sin(4 * cupy.pi * y**2)**2

    on a grid in [0, 1]x[0, 1]:

    >>> grid_x, grid_y = cupy.mgrid[0:1:100j, 0:1:200j]
    >>> rng = cupy.random.default_rng()
    >>> points = rng.random((1000, 2))
    >>> values = func(points[:, 0], points[:, 1])

    Interpolate using different methods:

    >>> from cupyx.scipy.interpolate import griddata
    >>> grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
    >>> grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
    >>> grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

    Visualize the results (assuming matplotlib with CuPy backend):

    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(221)
    >>> plt.imshow(func(grid_x, grid_y).get().T,
        extent=(0, 1, 0, 1), origin='lower')
    >>> plt.plot(points[:, 0].get(), points[:, 1].get(), 'k.', ms=1)
    >>> plt.title('Original')
    >>> plt.subplot(222)
    >>> plt.imshow(grid_z0.get().T, extent=(0, 1, 0, 1), origin='lower')
    >>> plt.title('Nearest')
    >>> plt.subplot(223)
    >>> plt.imshow(grid_z1.get().T, extent=(0, 1, 0, 1), origin='lower')
    >>> plt.title('Linear')
    >>> plt.subplot(224)
    >>> plt.imshow(grid_z2.get().T, extent=(0, 1, 0, 1), origin='lower')
    >>> plt.title('Cubic')
    >>> plt.gcf().set_size_inches(6, 6)
    >>> plt.show()

    See Also
    --------
    LinearNDInterpolator : Piecewise linear interpolator in N dimensions.
    NearestNDInterpolator : Nearest-neighbor interpolator in N dimensions.
    CloughTocher2DInterpolator : Piecewise cubic, C1 smooth,
    curvature-minimizing interpolator in 2D.
    CubicSpline : Cubic spline interpolator for 1D data.
    RegularGridInterpolator : Interpolator on a regular or rectilinear grid
    in arbitrary dimensions.
    interpn : Interpolation on a regular grid or rectilinear grid.
    """
    points = _ndim_coords_from_arrays(points)
    if points.ndim < 2:
        ndim = points.ndim
    else:
        ndim = points.shape[-1]

    if ndim == 1 and method in ('nearest', 'linear', 'cubic'):
        from ._polyint import interp1d
        points = points.ravel()
        if isinstance(xi, tuple):
            if len(xi) != 1:
                raise ValueError("invalid number of dimensions in xi")
            xi, = xi
        # Sort points/values together, necessary as input for interp1d
        idx = cupy.argsort(points)
        points = points[idx]
        values = values[idx]
        if method == 'nearest':
            fill_value = 'extrapolate'
        ip = interp1d(points, values, kind=method, axis=0, bounds_error=False,
                      fill_value=fill_value)
        return ip(xi)
    elif method == 'nearest':
        ip = NearestNDInterpolator(points, values, rescale=rescale)
        return ip(xi)
    elif method == 'linear':
        ip = LinearNDInterpolator(points, values, fill_value=fill_value,
                                  rescale=rescale)
        return ip(xi)
    elif method == 'cubic' and ndim == 2:
        ip = CloughTocher2DInterpolator(points, values, fill_value=fill_value,
                                        rescale=rescale)
        return ip(xi)
    else:
        raise ValueError(
            f"Unknown interpolation method {method!r} "
            f"for {ndim} dimensional data"
        )
