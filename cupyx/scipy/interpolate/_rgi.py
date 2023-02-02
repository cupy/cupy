__all__ = ['RegularGridInterpolator', 'interpn']

import itertools
import cupy as cp
from cupyx.scipy.interpolate._bspline2 import make_interp_spline
from cupyx.scipy.interpolate._cubic import PchipInterpolator


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


def _check_points(points):
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
                p = cp.ascontiguousarray(p)
            else:
                raise ValueError(
                    "The points in dimension %d must be strictly "
                    "ascending or descending" % i)
        grid.append(p)
    return tuple(grid), tuple(descending_dimensions)


def _check_dimensionality(points, values):
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


class RegularGridInterpolator:
    """
    Interpolation on a regular or rectilinear grid in arbitrary dimensions.

    The data must be defined on a rectilinear grid; that is, a rectangular
    grid with even or uneven spacing. Linear and nearest-neighbor
    interpolations are supported. After setting up the interpolator object,
    the interpolation method may be chosen at each evaluation.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions. The points in
        each dimension (i.e. every elements of the points tuple) must be
        strictly ascending or descending.

    values : ndarray, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions. Complex data can be
        acceptable.

    method : str, optional
        The method of interpolation to perform. Supported are "linear",
        "nearest", "slinear", "cubic", "quintic" and "pchip".
        This parameter will become the default for the object's
        ``__call__`` method. Default is "linear".

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.
        Default is True.

    fill_value : float or None, optional
        The value to use for points outside of the interpolation domain.
        If None, values outside the domain are extrapolated.
        Default is ``cp.nan``.

    Notes
    -----
    Contrary to scipy's `LinearNDInterpolator` and `NearestNDInterpolator`,
    this class avoids expensive triangulation of the input data by taking
    advantage of the regular grid structure.

    In other words, this class assumes that the data is defined on a
    *rectilinear* grid.

    If the input data is such that dimensions have incommensurate
    units and differ by many orders of magnitude, the interpolant may have
    numerical artifacts. Consider rescaling the data before interpolating.

    Examples
    --------
    **Evaluate a function on the points of a 3-D grid**

    As a first example, we evaluate a simple example function on the points of
    a 3-D grid:

    >>> from cupyx.scipy.interpolate import RegularGridInterpolator
    >>> import cupy as cp
    >>> def f(x, y, z):
    ...     return 2 * x**3 + 3 * y**2 - z
    >>> x = cp.linspace(1, 4, 11)
    >>> y = cp.linspace(4, 7, 22)
    >>> z = cp.linspace(7, 9, 33)
    >>> xg, yg ,zg = cp.meshgrid(x, y, z, indexing='ij', sparse=True)
    >>> data = f(xg, yg, zg)

    ``data`` is now a 3-D array with ``data[i, j, k] = f(x[i], y[j], z[k])``.
    Next, define an interpolating function from this data:

    >>> interp = RegularGridInterpolator((x, y, z), data)

    Evaluate the interpolating function at the two points
    ``(x,y,z) = (2.1, 6.2, 8.3)`` and ``(3.3, 5.2, 7.1)``:

    >>> pts = cp.array([[2.1, 6.2, 8.3],
    ...                 [3.3, 5.2, 7.1]])
    >>> interp(pts)
    array([ 125.80469388,  146.30069388])

    which is indeed a close approximation to

    >>> f(2.1, 6.2, 8.3), f(3.3, 5.2, 7.1)
    (125.54200000000002, 145.894)

    **Interpolate and extrapolate a 2D dataset**

    As a second example, we interpolate and extrapolate a 2D data set:

    >>> x, y = cp.array([-2, 0, 4]), cp.array([-2, 0, 2, 5])
    >>> def ff(x, y):
    ...     return x**2 + y**2

    >>> xg, yg = cp.meshgrid(x, y, indexing='ij')
    >>> data = ff(xg, yg)
    >>> interp = RegularGridInterpolator((x, y), data,
    ...                                  bounds_error=False, fill_value=None)

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(projection='3d')
    >>> ax.scatter(xg.ravel().get(), yg.ravel().get(), data.ravel().get(),
    ...            s=60, c='k', label='data')

    Evaluate and plot the interpolator on a finer grid

    >>> xx = cp.linspace(-4, 9, 31)
    >>> yy = cp.linspace(-4, 9, 31)
    >>> X, Y = cp.meshgrid(xx, yy, indexing='ij')

    >>> # interpolator
    >>> ax.plot_wireframe(X.get(), Y.get(), interp((X, Y)).get(),
                          rstride=3, cstride=3, alpha=0.4, color='m',
                          label='linear interp')

    >>> # ground truth
    >>> ax.plot_wireframe(X.get(), Y.get(), ff(X, Y).get(),
                          rstride=3, cstride=3,
    ...                   alpha=0.4, label='ground truth')
    >>> plt.legend()
    >>> plt.show()

    See Also
    --------
    interpn : a convenience function which wraps `RegularGridInterpolator`

    scipy.ndimage.map_coordinates : interpolation on grids with equal spacing
                                    (suitable for e.g., N-D image resampling)

    References
    ----------
    [1] Python package *regulargrid* by Johannes Buchner, see
        https://pypi.python.org/pypi/regulargrid/
    [2] Wikipedia, "Trilinear interpolation",
        https://en.wikipedia.org/wiki/Trilinear_interpolation
    [3] Weiser, Alan, and Sergio E. Zarantonello. "A note on piecewise
        linear and multilinear table interpolation in many dimensions."
        MATH. COMPUT. 50.181 (1988): 189-196.
        https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf
    """
    # this class is based on code originally programmed by Johannes Buchner,
    # see https://github.com/JohannesBuchner/regulargrid

    _SPLINE_DEGREE_MAP = {"slinear": 1, "cubic": 3, "quintic": 5, 'pchip': 3}
    _SPLINE_METHODS = list(_SPLINE_DEGREE_MAP.keys())
    _ALL_METHODS = ["linear", "nearest"] + _SPLINE_METHODS

    def __init__(self, points, values, method="linear", bounds_error=True,
                 fill_value=cp.nan):
        if method not in self._ALL_METHODS:
            raise ValueError("Method '%s' is not defined" % method)
        elif method in self._SPLINE_METHODS:
            self._validate_grid_dimensions(points, method)

        self.method = method
        self.bounds_error = bounds_error
        self.grid, self._descending_dimensions = _check_points(points)
        self.values = self._check_values(values)
        self._check_dimensionality(self.grid, self.values)
        self.fill_value = self._check_fill_value(self.values, fill_value)
        if self._descending_dimensions:
            self.values = cp.flip(values, axis=self._descending_dimensions)

    def _check_dimensionality(self, grid, values):
        _check_dimensionality(grid, values)

    def _validate_grid_dimensions(self, points, method):
        k = self._SPLINE_DEGREE_MAP[method]
        for i, point in enumerate(points):
            ndim = len(cp.atleast_1d(point))
            if ndim <= k:
                raise ValueError(f"There are {ndim} points in dimension {i},"
                                 f" but method {method} requires at least "
                                 f" {k+1} points per dimension.")

    def _check_points(self, points):
        return _check_points(points)

    def _check_values(self, values):
        if not cp.issubdtype(values.dtype, cp.inexact):
            values = values.astype(float)

        return values

    def _check_fill_value(self, values, fill_value):
        if fill_value is not None:
            fill_value_dtype = cp.asarray(fill_value).dtype
            if (hasattr(values, 'dtype') and
                not cp.can_cast(fill_value_dtype, values.dtype,
                                casting='same_kind')):
                raise ValueError("fill_value must be either 'None' or "
                                 "of a type compatible with values")
        return fill_value

    def __call__(self, xi, method=None):
        """
        Interpolation at coordinates.

        Parameters
        ----------
        xi : cupy.ndarray of shape (..., ndim)
            The coordinates to evaluate the interpolator at.

        method : str, optional
            The method of interpolation to perform. Supported are "linear" and
            "nearest".  Default is the method chosen when the interpolator was
            created.

        Returns
        -------
        values_x : cupy.ndarray, shape xi.shape[:-1] + values.shape[ndim:]
            Interpolated values at `xi`. See notes for behaviour when
            ``xi.ndim == 1``.

        Notes
        -----
        In the case that ``xi.ndim == 1`` a new axis is inserted into
        the 0 position of the returned array, values_x, so its shape is
        instead ``(1,) + values.shape[ndim:]``.

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
        is_method_changed = self.method != method
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
        elif method in self._SPLINE_METHODS:
            if is_method_changed:
                self._validate_grid_dimensions(self.grid, method)
            result = self._evaluate_spline(xi, method)

        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value

        if nans.ndim < result.ndim:
            nans = nans[..., None]
        result = cp.where(nans, cp.nan, result)
        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _prepare_xi(self, xi):
        ndim = len(self.grid)
        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             f"{xi.shape[-1]} but this "
                             f"RegularGridInterpolator has dimension {ndim}")

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])
        xi = cp.asarray(xi, dtype=float)

        # find nans in input
        is_nans = cp.isnan(xi).T
        nans = is_nans[0].copy()
        for is_nan in is_nans[1:]:
            cp.logical_or(nans, is_nan, nans)

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not cp.logical_and(cp.all(self.grid[i][0] <= p),
                                      cp.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of bounds"
                                     " in dimension %d" % i)
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
        value = cp.array([0.])
        for h in hypercube:
            edge_indices, weights = zip(*h)
            term = cp.asarray(self.values[edge_indices])
            for w in weights:
                term *= w[vslice]
            value = value + term   # cannot use += because broadcasting
        return value

    def _evaluate_nearest(self, indices, norm_distances):
        idx_res = [cp.where(yi <= .5, i, i + 1)
                   for i, yi in zip(indices, norm_distances)]
        return self.values[tuple(idx_res)]

    def _evaluate_spline(self, xi, method):
        # ensure xi is 2D list of points to evaluate (`m` is the number of
        # points and `n` is the number of interpolation dimensions,
        # ``n == len(self.grid)``.)
        if xi.ndim == 1:
            xi = xi.reshape((1, xi.size))
        m, n = xi.shape

        # Reorder the axes: n-dimensional process iterates over the
        # interpolation axes from the last axis downwards: E.g. for a 4D grid
        # the order of axes is 3, 2, 1, 0. Each 1D interpolation works along
        # the 0th axis of its argument array (for 1D routine it's its ``y``
        # array). Thus permute the interpolation axes of `values` *and keep
        # trailing dimensions trailing*.
        axes = tuple(range(self.values.ndim))
        axx = axes[:n][::-1] + axes[n:]
        values = self.values.transpose(axx)

        if method == 'pchip':
            _eval_func = self._do_pchip
        else:
            _eval_func = self._do_spline_fit
        k = self._SPLINE_DEGREE_MAP[method]

        # Non-stationary procedure: difficult to vectorize this part entirely
        # into numpy-level operations. Unfortunately this requires explicit
        # looping over each point in xi.

        # can at least vectorize the first pass across all points in the
        # last variable of xi.
        last_dim = n - 1
        first_values = _eval_func(self.grid[last_dim],
                                  values,
                                  xi[:, last_dim],
                                  k)

        # the rest of the dimensions have to be on a per point-in-xi basis
        shape = (m, *self.values.shape[n:])
        result = cp.empty(shape, dtype=self.values.dtype)
        for j in range(m):
            # Main process: Apply 1D interpolate in each dimension
            # sequentially, starting with the last dimension.
            # These are then "folded" into the next dimension in-place.
            folded_values = first_values[j, ...]
            for i in range(last_dim-1, -1, -1):
                # Interpolate for each 1D from the last dimensions.
                # This collapses each 1D sequence into a scalar.
                folded_values = _eval_func(self.grid[i],
                                           folded_values,
                                           xi[j, i],
                                           k)
            result[j, ...] = folded_values

        return result

    @staticmethod
    def _do_spline_fit(x, y, pt, k):
        local_interp = make_interp_spline(x, y, k=k, axis=0)
        values = local_interp(pt)
        return values

    @staticmethod
    def _do_pchip(x, y, pt, k):
        local_interp = PchipInterpolator(x, y, axis=0)
        values = local_interp(pt)
        return values

    def _find_indices(self, xi):
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            i = cp.searchsorted(grid, x) - 1
            cp.clip(i, 0, grid.size - 2, i)
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

    values : cupy.ndarray of shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions. Complex data can be
        acceptable.

    xi : cupy.ndarray of shape (..., ndim)
        The coordinates to sample the gridded data at

    method : str, optional
        The method of interpolation to perform. Supported are "linear",
        "nearest", "slinear", "cubic", "quintic" and "pchip".

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
        Interpolated values at `xi`. See notes for behaviour when
        ``xi.ndim == 1``.

    Notes
    -----

    In the case that ``xi.ndim == 1`` a new axis is inserted into
    the 0 position of the returned array, values_x, so its shape is
    instead ``(1,) + values.shape[ndim:]``.

    If the input data is such that input dimensions have incommensurate
    units and differ by many orders of magnitude, the interpolant may have
    numerical artifacts. Consider rescaling the data before interpolation.

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
    if method not in ["linear", "nearest", "slinear", "cubic",
                      "quintic", "pchip"]:
        raise ValueError(
            "interpn only understands the methods 'linear', 'nearest', "
            "'slinear', 'cubic', 'quintic' and 'pchip'. "
            "You provided {method}.")

    ndim = values.ndim

    # sanity check consistency of input dimensions
    if len(points) > ndim:
        raise ValueError("There are %d point arrays, but values has %d "
                         "dimensions" % (len(points), ndim))

    grid, descending_dimensions = _check_points(points)
    _check_dimensionality(grid, values)

    # sanity check requested xi
    xi = _ndim_coords_from_arrays(xi, ndim=len(grid))
    if xi.shape[-1] != len(grid):
        raise ValueError("The requested sample points xi have dimension "
                         "%d, but this RegularGridInterpolator has "
                         "dimension %d" % (xi.shape[-1], len(grid)))

    if bounds_error:
        for i, p in enumerate(xi.T):
            if not cp.logical_and(cp.all(grid[i][0] <= p),
                                  cp.all(p <= grid[i][-1])):
                raise ValueError("One of the requested xi is out of bounds "
                                 "in dimension %d" % i)

    # perform interpolation
    if method in ["linear", "nearest", "slinear", "cubic", "quintic", "pchip"]:
        interp = RegularGridInterpolator(points, values, method=method,
                                         bounds_error=bounds_error,
                                         fill_value=fill_value)
        return interp(xi)
