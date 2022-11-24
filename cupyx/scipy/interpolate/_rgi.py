__all__ = ['RegularGridInterpolator', 'interpn']

import itertools
import cupy as cp


def _ndim_coords_from_arrays(points, ndim=None):
    """
    Convert a tuple of coordinate arrays to a (..., ndim)-shaped array.
    """
    if isinstance(points, tuple) and len(points) == 1:
        # handle argument tuple
        points = points[0]
    if isinstance(points, tuple):
        p = cp.broadcast_arrays(*points)
        n = len(p)
        for j in range(1, n):
            if p[j].shape != p[0].shape:
                raise ValueError(
                    "coordinate arrays do not have the same shape")
        points = cp.empty(p[0].shape + (len(points),), dtype=float)
        for j, item in enumerate(p):
            points[..., j] = item
    else:
        points = cp.asanyarray(points)
        if points.ndim == 1:
            if ndim is None:
                points = points.reshape(-1, 1)
            else:
                points = points.reshape(-1, ndim)
    return points


class RegularGridInterpolator:
    _ALL_METHODS = ["linear", "nearest"]

    def __init__(self, points, values, method="linear", bounds_error=True,
                 fill_value=cp.nan):
        if method not in self._ALL_METHODS:
            raise ValueError("Method '%s' is not defined" % method)

        self.method = method
        self.bounds_error = bounds_error

        self.grid, self._descending_dimensions = self._check_points(points)
        self.values = self._check_values(values)
        self._check_dimensionality(self.grid, self.values)
        self.fill_value = self._check_fill_value(self.values, fill_value)
        if self._descending_dimensions:
            self.values = cp.flip(values, axis=self._descending_dimensions)

    def _check_dimensionality(self, points, values):
        if len(points) > values.ndim:
            raise ValueError("There are %d point arrays, but values has %d "
                             "dimensions" % (len(points), values.ndim))
        for i, p in enumerate(points):
            if not cp.asarray(p).ndim == 1:
                raise ValueError("The points in dimension %d must be "
                                 "1-dimensional" % i)
            if not values.shape[i] == len(p):
                raise ValueError("There are %d points and %d values in "
                                 "dimension %d" % (len(p), values.shape[i], i))

    def _check_values(self, values):
        if not hasattr(values, 'ndim'):
            # allow reasonable duck-typed values
            values = cp.asarray(values)

        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not cp.issubdtype(values.dtype, cp.inexact):
                values = values.astype(float)

        return values

    def _check_fill_value(self, values, fill_value):
        if fill_value is not None:
            fill_value_dtype = cp.asarray(fill_value).dtype
            if (hasattr(values, 'dtype') and not
                    cp.can_cast(fill_value_dtype, values.dtype,
                                casting='same_kind')):
                raise ValueError("fill_value must be either 'None' or "
                                 "of a type compatible with values")
        return fill_value

    def _check_points(self, points):
        descending_dimensions = []
        grid = []
        for i, p in enumerate(points):
            # early make points float
            # see https://github.com/scipy/scipy/pull/17230
            p = cp.asarray(p, dtype=float)
            if not cp.all(p[1:] > p[:-1]):
                if cp.all(p[1:] < p[:-1]):
                    # input is descending, so make it ascending
                    descending_dimensions.append(i)
                    p = cp.flip(p)
                else:
                    raise ValueError(
                        "The points in dimension %d must be strictly "
                        "ascending or descending" % i)
            grid.append(p)

        return tuple(grid), tuple(descending_dimensions)

    def __call__(self, xi, method=None):
        """
        Interpolation at coordinates.

        Parameters
        ----------
        xi : cupy.ndarray of shape (..., ndim)
            The coordinates to evaluate the interpolator at.

        method : str
            The method of interpolation to perform. Supported are "linear" and
            "nearest".  Default is "linear".

        Returns
        -------
        values_x : ndarray, shape xi.shape[:-1] + values.shape[ndim:]
            Interpolated values at `xi`. See notes for behaviour when
            ``xi.ndim == 1``.
        Notes
        -----
        In the case that ``xi.ndim == 1`` a new axis is inserted into
        the 0 position of the returned array, values_x, so its shape is
        instead ``(1,) + values.shape[ndim:]``.

        If the input data is such that dimensions have incommensurate
        units and differ by many orders of magnitude, the interpolant may have
        numerical artifacts. Consider rescaling the data before interpolating.

        Examples
        --------
        Here we define a nearest-neighbor interpolator of a simple function

        >>> import cupy as cp
        >>> x, y = cp.array([0, 1, 2]), cp.array([1, 3, 7])
        >>> def f(x, y):
        ...     return x**2 + y**2
        >>> data = f(*cp.meshgrid(x, y, indexing='ij', sparse=True))
        >>> from cupyx.scipy.interpolate import RegularGridInterpolator
        >>> interp = RegularGridInterpolator((x, y), data, method='nearest')

        By construction, the interpolator uses the nearest-neighbor
        interpolation

        >>> interp([[1.5, 1.3], [0.3, 4.5]])
        array([2., 9.])

        We can however evaluate the linear interpolant by overriding the
        `method` parameter

        >>> interp([[1.5, 1.3], [0.3, 4.5]], method='linear')
        array([ 4.7, 24.3])
        """
        method = self.method if method is None else method
        if method not in self._ALL_METHODS:
            raise ValueError("Method '%s' is not defined" % method)
        xi, xi_shape, ndim, nans, out_of_bounds = self._prepare_xi(xi)

        if method == "linear":
            indices, norm_distances = self._find_indices(xi.T)
            result = self._evaluate_linear(indices, norm_distances)
        elif method == "nearest":
            indices, norm_distances = self._find_indices(xi.T)
            result = self._evaluate_nearest(indices, norm_distances)

        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value

        # f(nan) = nan, if any
        if cp.any(nans):
            result[nans] = cp.nan
        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _prepare_xi(self, xi):
        ndim = len(self.grid)
        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             "%d, but this RegularGridInterpolator has "
                             "dimension %d" % (xi.shape[-1], ndim))

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        # find nans in input
        nans = cp.any(cp.isnan(xi), axis=-1)

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not cp.logical_and(cp.all(self.grid[i][0] <= p),
                                      cp.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of "
                                     "bounds in dimension %d" % i)
            out_of_bounds = None
        else:
            out_of_bounds = self._find_out_of_bounds(xi.T)

        return xi, xi_shape, ndim, nans, out_of_bounds

    def _evaluate_linear(self, indices, norm_distances):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))

        # Compute shifting up front before zipping everything together
        shift_norm_distances = [1 - yi for yi in norm_distances]
        shift_indices = [i + 1 for i in indices]

        # The formula for linear interpolation in 2d takes the form:
        # values = self.values[(i0, i1)] * (1 - y0) * (1 - y1) + \
        #          self.values[(i0, i1 + 1)] * (1 - y0) * y1 + \
        #          self.values[(i0 + 1, i1)] * y0 * (1 - y1) + \
        #          self.values[(i0 + 1, i1 + 1)] * y0 * y1
        # We pair i with 1 - yi (zipped1) and i + 1 with yi (zipped2)
        zipped1 = zip(indices, shift_norm_distances)
        zipped2 = zip(shift_indices, norm_distances)

        # Take all products of zipped1 and zipped2 and iterate over them
        # to get the terms in the above formula. This corresponds to iterating
        # over the vertices of a hypercube.
        hypercube = itertools.product(*zip(zipped1, zipped2))
        values = 0.
        for h in hypercube:
            edge_indices, weights = zip(*h)
            weight = 1.
            for w in weights:
                weight *= w
            values += cp.asarray(self.values[edge_indices]) * weight[vslice]
        return values

    def _evaluate_nearest(self, indices, norm_distances):
        idx_res = [cp.where(yi <= .5, i, i + 1)
                   for i, yi in zip(indices, norm_distances)]
        return self.values[tuple(idx_res)]

    def _find_indices(self, xi):
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            i = cp.searchsorted(grid, x) - 1
            i[i < 0] = 0
            i[i > grid.size - 2] = grid.size - 2
            indices.append(i)

            # compute norm_distances, incl length-1 grids,
            # where `grid[i+1] == grid[i]`
            denom = grid[i + 1] - grid[i]
            norm_dist = cp.where(denom != 0, (x - grid[i]) / denom, 0)
            norm_distances.append(norm_dist)

        return indices, norm_distances

    def _find_out_of_bounds(self, xi):
        # check for out of bounds xi
        out_of_bounds = cp.zeros((xi.shape[1]), dtype=bool)
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            out_of_bounds += x < grid[0]
            out_of_bounds += x > grid[-1]
        return out_of_bounds


def interpn(points, values, xi, method="linear", bounds_error=True,
            fill_value=cp.nan):
    """
    Multidimensional interpolation on regular or rectilinear grids.

    Strictly speaking, not all regular grids are supported - this function
    works on *rectilinear* grids, that is, a rectangular grid with even or
    uneven spacing.

    Parameters
    ----------
    points : tuple of cupy.ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions. The points in
        each dimension (i.e. every elements of the points tuple) must be
        strictly ascending or descending.

    values : cupy.ndarray, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions. Complex data can be
        acceptable.

    xi : cupy.ndarray of shape (..., ndim)
        The coordinates to sample the gridded data at

    method : str, optional
        The method of interpolation to perform. Supported are "linear" and
        "nearest".  Default is "linear".

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.

    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.

    Returns
    -------
    values_x : ndarray, shape xi.shape[:-1] + values.shape[ndim:]
        Interpolated values at `xi`.  See notes for behaviour when
        ``xi.ndim == 1``.

    Notes
    -----
    In the case that ``xi.ndim == 1`` a new axis is inserted into
    the 0 position of the returned array, values_x, so its shape is
    instead ``(1,) + values.shape[ndim:]``.

    If the input data is such that dimensions have incommensurate
    units and differ by many orders of magnitude, the interpolant may have
    numerical artifacts. Consider rescaling the data before interpolating.

    Examples
    --------
    Evaluate a simple example function on the points of a regular 3-D grid:

    >>> import cupy as cp
    >>> from cupyx.scipy.interpolate import interpn
    >>> def value_func_3d(x, y, z):
    ...     return 2 * x + 3 * y - z
    >>> x = cp.linspace(0, 4, 5)
    >>> y = cp.linspace(0, 5, 6)
    >>> z = cp.linspace(0, 6, 7)
    >>> points = (x, y, z)
    >>> values = value_func_3d(*cp.meshgrid(*points, indexing='ij'))

    Evaluate the interpolating function at a point

    >>> point = cp.array([2.21, 3.12, 1.15])
    >>> print(interpn(points, values, point))
    [12.63]

    See Also
    --------
    RegularGridInterpolator : interpolation on a regular or rectilinear grid
                              in arbitrary dimensions (`interpn` wraps this
                              class).

    cupyx.scipy.ndimage.map_coordinates : interpolation on grids with equal
                                          spacing (suitable for e.g., N-D image
                                          resampling)

    """
    # sanity check 'method' kwarg
    if method not in ["linear", "nearest"]:
        raise ValueError(
            "interpn only understands the methods 'linear' and 'nearest'. "
            "You provided %s." % method)

    if not hasattr(values, 'ndim'):
        values = cp.asarray(values)

    ndim = values.ndim

    # sanity check consistency of input dimensions
    if len(points) > ndim:
        raise ValueError("There are %d point arrays, but values has %d "
                         "dimensions" % (len(points), ndim))

    points = list(points)
    # sanity check input grid
    for i, p in enumerate(points):
        diff_p = cp.diff(p)
        if not cp.all(diff_p > 0.):
            if cp.all(diff_p < 0.):
                # input is descending, so make it ascending
                points[i] = points[i][::-1]
                values = cp.flip(values, axis=i)
            else:
                raise ValueError("The points in dimension %d must be strictly "
                                 "ascending or descending" % i)
        if not cp.asarray(p).ndim == 1:
            raise ValueError("The points in dimension %d must be "
                             "1-dimensional" % i)
        if not values.shape[i] == len(p):
            raise ValueError("There are %d points and %d values in "
                             "dimension %d" % (len(p), values.shape[i], i))
    grid = tuple([cp.asarray(p) for p in points])

    # sanity check requested xi
    xi = _ndim_coords_from_arrays(xi, ndim=len(grid))
    if xi.shape[-1] != len(grid):
        raise ValueError("The requested sample points xi have dimension "
                         f"{xi.shape[-1]} but this "
                         f"RegularGridInterpolator has dimension {ndim}")

    if bounds_error:
        for i, p in enumerate(xi.T):
            if not cp.logical_and(cp.all(grid[i][0] <= p),
                                  cp.all(p <= grid[i][-1])):
                raise ValueError("One of the requested xi is out of bounds "
                                 "in dimension %d" % i)

    # perform interpolation
    interp = RegularGridInterpolator(points, values, method=method,
                                     bounds_error=bounds_error,
                                     fill_value=fill_value)
    return interp(xi)
