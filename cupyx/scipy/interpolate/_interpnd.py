
import cupy
from cupy._core._scalar import get_typename
from cupyx.scipy.spatial._delaunay import Delaunay


TYPES = ['double', 'thrust::complex<double>']


def _get_module_func(module, func_name, *template_args):
    def _get_typename(dtype):
        typename = get_typename(dtype)
        if dtype.kind == 'c':
            typename = 'thrust::' + typename
        return typename
    args_dtypes = [_get_typename(arg.dtype) for arg in template_args]
    template = ', '.join(args_dtypes)
    kernel_name = f'{func_name}<{template}>' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel


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
            self.fill_value = cupy.asarray(
                complex(fill_value), dtype=cupy.complex128)
        else:
            if need_contiguous:
                self.values = cupy.ascontiguousarray(
                    self.values, dtype=cupy.float64
                )
            self.fill_value = cupy.asarray(
                float(fill_value), dtype=cupy.float64)

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
        return self.tri._find_simplex_coordinates(xi, 0.0, find_coords=True)

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

LINEAR_INTERP_ND_DEF = r"""
#include <cupy/complex.cuh>
#include <cupy/math_constants.h>

template<typename T>
__global__ void evaluate_linear_nd_interp(
        const int num_x, const int ndim, const int values_sz,
        const T* fill_value, const int* enc_simplices, const int* simplices,
        const double* coords, const T* values, T* out) {

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= num_x) {
        return;
    }

    const int ch_simplex = enc_simplices[idx];
    if(ch_simplex == -1) {
        for(int k = 0; k < values_sz; k++) {
            out[idx * values_sz + k] = fill_value[0];
        }
        return;
    }

    const int simplex_sz = ndim + 1;
    const int* simplex = simplices + simplex_sz * ch_simplex;

    for(int k = 0; k < values_sz; k++) {
        out[idx * values_sz + k] = 0;
    }

    for(int j = 0; j < ndim + 1; j++) {
        const int m = simplex[j];
        double coord = coords[simplex_sz * idx + j];

        for(int k = 0; k < values_sz; k++) {
            out[idx * values_sz + k] += coord * values[values_sz * m + k];
        }
    }
}
"""

LINEAR_INTERP_ND_MODULE = cupy.RawModule(
    code=LINEAR_INTERP_ND_DEF, options=('-std=c++11',),
    name_expressions=[f'evaluate_linear_nd_interp<{t}>' for t in TYPES])


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

    def _do_evaluate(self, xi, dummy):
        isimplices, c = self._find_simplicies(xi)
        ndim = xi.shape[1]
        fill_value = self.fill_value

        out = cupy.empty((xi.shape[0], self.values.shape[1]),
                         dtype=self.values.dtype)
        nvalues = out.shape[1]

        _eval_linear_nd_interp = _get_module_func(
            LINEAR_INTERP_ND_MODULE, 'evaluate_linear_nd_interp', out)

        block_sz = 128
        n_blocks = (xi.shape[0] + block_sz - 1) // block_sz
        _eval_linear_nd_interp(
            (n_blocks,), (block_sz,),
            (int(xi.shape[0]), int(ndim), int(nvalues), fill_value,
             isimplices, self.tri.simplices, c, self.values, out))

        return out


def estimate_gradients_2d_global(tri, y, maxiter=400, tol=1e-6):
    indptr, indices = tri.vertex_neighbor_vertices()  # NOQA


class CloughTocher2DInterpolator(NDInterpolatorBase):
    """CloughTocher2DInterpolator(points, values, tol=1e-6).

    Piecewise cubic, C1 smooth, curvature-minimizing interpolator in 2D.

    Methods
    -------
    __call__

    Parameters
    ----------
    points : ndarray of floats, shape (npoints, ndims); or Delaunay
        2-D array of data point coordinates, or a precomputed Delaunay
        triangulation.
    values : ndarray of float or complex, shape (npoints, ...)
        N-D array of data values at `points`. The length of `values` along the
        first axis must be equal to the length of `points`. Unlike some
        interpolators, the interpolation axis cannot be changed.
    fill_value : float, optional
        Value used to fill in for requested points outside of the
        convex hull of the input points.  If not provided, then
        the default is ``nan``.
    tol : float, optional
        Absolute/relative tolerance for gradient estimation.
    maxiter : int, optional
        Maximum number of iterations in gradient estimation.
    rescale : bool, optional
        Rescale points to unit cube before performing interpolation.
        This is useful if some of the input dimensions have
        incommensurable units and differ by many orders of magnitude.

    Notes
    -----
    The interpolant is constructed by triangulating the input data
    with GDel2D [1]_, and constructing a piecewise cubic
    interpolating Bezier polynomial on each triangle, using a
    Clough-Tocher scheme [CT]_.  The interpolant is guaranteed to be
    continuously differentiable.

    The gradients of the interpolant are chosen so that the curvature
    of the interpolating surface is approximatively minimized. The
    gradients necessary for this are estimated using the global
    algorithm described in [Nielson83]_ and [Renka84]_.

    .. note:: For data on a regular grid use `interpn` instead.

    Examples
    --------
    We can interpolate values on a 2D plane:

    >>> from scipy.interpolate import CloughTocher2DInterpolator
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.random(10) - 0.5
    >>> y = rng.random(10) - 0.5
    >>> z = np.hypot(x, y)
    >>> X = np.linspace(min(x), max(x))
    >>> Y = np.linspace(min(y), max(y))
    >>> X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    >>> interp = CloughTocher2DInterpolator(list(zip(x, y)), z)
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
    LinearNDInterpolator :
        Piecewise linear interpolator in N > 1 dimensions.
    NearestNDInterpolator :
        Nearest-neighbor interpolator in N > 1 dimensions.
    interpn : Interpolation on a regular grid or rectilinear grid.
    RegularGridInterpolator : Interpolator on a regular or rectilinear grid
                              in arbitrary dimensions (`interpn` wraps this
                              class).

    References
    ----------
    .. [1] A GPU accelerated algorithm for 3D Delaunay triangulation (2014).
    Thanh-Tung Cao, Ashwin Nanjappa, Mingcen Gao, Tiow-Seng Tan.
    Proc. 18th ACM SIGGRAPH Symp. Interactive 3D Graphics and Games, 47-55.

    .. [CT] See, for example,
       P. Alfeld,
       ''A trivariate Clough-Tocher scheme for tetrahedral data''.
       Computer Aided Geometric Design, 1, 169 (1984);
       G. Farin,
       ''Triangular Bernstein-Bezier patches''.
       Computer Aided Geometric Design, 3, 83 (1986).

    .. [Nielson83] G. Nielson,
       ''A method for interpolating scattered data based upon a minimum norm
       network''.
       Math. Comp., 40, 253 (1983).

    .. [Renka84] R. J. Renka and A. K. Cline.
       ''A Triangle-based C1 interpolation method.'',
       Rocky Mountain J. Math., 14, 223 (1984).

    """

    def __init__(self, points, values, fill_value=cupy.nan,
                 tol=1e-6, maxiter=400, rescale=False):
        self._tol = tol
        self._maxiter = maxiter
        NDInterpolatorBase.__init__(self, points, values, ndim=2,
                                    fill_value=fill_value, rescale=rescale,
                                    need_values=False)

    def _set_values(self, values, fill_value=cupy.nan,
                    need_contiguous=True, ndim=None):
        """
        Sets the values of the interpolation points.

        Parameters
        ----------
        values : ndarray of float or complex, shape (npoints, ...)
            Data values.
        """
        NDInterpolatorBase._set_values(
            self, values, fill_value=fill_value,
            need_contiguous=need_contiguous, ndim=ndim)

        if self.values is not None:
            self.grad = estimate_gradients_2d_global(
                self.tri, self.values, tol=self._tol, maxiter=self._maxiter)

    def _calculate_triangulation(self, points):
        self.tri = Delaunay(points)

    def _evaluate_double(self, xi):
        return self._do_evaluate(xi, 1.0)

    def _evaluate_complex(self, xi):
        return self._do_evaluate(xi, 1.0j)
