import cupy
import cupy.core.internal

from .filters import _generate_boundary_condition_ops
_prod = cupy.core.internal.prod


def _get_coord_map(ndim):
    """Extract target coordinate from coords array (for map_coordinates).

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        coords (ndarray): array of shape (ncoords, ndim) containing the target
            coordinates.
        c_j: variables to hold the target coordinates

    computes::

        c_j = coords[i + j * ncoords];

    ncoords is determined by the size of the output array, y.
    y will be indexed by the CIndexer, _ind.
    Thus ncoords = _ind.size();

    """
    ops = []
    ops.append('ptrdiff_t ncoords = _ind.size();')
    for j in range(ndim):
        ops.append(
            """
    W c_{j} = coords[i + {j} * ncoords];
            """.format(j=j))
    return ops


def _get_coord_zoom_and_shift(ndim):
    """Compute target coordinate based on a shift followed by a zoom.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        zoom[ndim]: array containing the zoom for each axis
        shift[ndim]: array containing the zoom for each axis

    computes::

        c_j = zoom[j] * (in_coord[j] - shift[j])

    """
    ops = []
    for j in range(ndim):
        ops.append(
            """
    W c_{j} = zoom[{j}] * ((W)in_coord[{j}] - shift[{j}]);
            """.format(j=j))
    return ops


def _get_coord_zoom(ndim):
    """Compute target coordinate based on a zoom.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        zoom[ndim]: array containing the zoom for each axis

    computes::

        c_j = zoom[j] * in_coord[j]

    """
    ops = []
    for j in range(ndim):
        ops.append(
            """
    W c_{j} = zoom[{j}] * (W)in_coord[{j}];
            """.format(j=j))
    return ops


def _get_coord_shift(ndim):
    """Compute target coordinate based on a shift.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        shift[ndim]: array containing the zoom for each axis

    computes::

        c_j = in_coord[j] - shift[j]

    """
    ops = []
    for j in range(ndim):
        ops.append(
            """
    W c_{j} = (W)in_coord[{j}] - shift[{j}];
            """.format(j=j))
    return ops


def _get_coord_affine(ndim):
    """Compute target coordinate based on a homogeneous transformation matrix.

    The homogeneous matrix has shape (ndim, ndim + 1). It corresponds to
    affine matrix where the last row of the affine is assumed to be:
    ``[0] * ndim + [1]``.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        mat(array): array containing the (ndim, ndim + 1) transform matrix.
        in_coords(array): coordinates of the input

    For example, in 2D:

        c_0 = mat[0] * in_coords[0] + mat[1] * in_coords[1] + aff[2];
        c_1 = mat[3] * in_coords[0] + mat[4] * in_coords[1] + aff[5];

    """
    ops = []
    ncol = ndim + 1
    for j in range(ndim):
        ops.append("""
            W c_{j} = (W)0.0;
            """.format(j=j))
        for k in range(ndim):
            m_index = ncol * j + k
            ops.append(
                """
            c_{j} += mat[{m_index}] * (W)in_coord[{k}];
                """.format(j=j, k=k, m_index=m_index))
        ops.append(
            """
            c_{j} += mat[{m_index}];
            """.format(j=j, m_index=ncol * j + ndim))
    return ops


def _generate_interp_custom(coord_func, xshape, yshape, mode, cval, order,
                            name='', integer_output=False):
    """
    Args:
        coord_func (function): generates code to do the coordinate
            transformation. See for example, `_get_coord_shift`.
        xshape (tuple): Shape of the array to be transformed.
        yshape (tuple): Shape of the output array.
        mode (str): Signal extension mode to use at the array boundaries
        cval (float): constant value used when `mode == 'constant'`.
        name (str): base name for the interpolation kernel
        integer_output (bool): boolean indicating whether the output has an
            integer type.

    Returns:
        operation (str): code body for the ElementwiseKernel
        name (str): name for the ElementwiseKernel
    """

    ndim = len(xshape)

    ops = []
    ops.append('double out = 0.0;')

    size = max(_prod(xshape), _prod(yshape))
    if (size > 1 << 31):
        uint_t = 'size_t'
        int_t = 'ptrdiff_t'
    else:
        uint_t = 'unsigned int'
        int_t = 'int'
    ops.append('{uint_t} in_coord[{ndim}];'.format(
        uint_t=uint_t, ndim=ndim))

    # determine strides for x along each axis
    ops.append(
        'const {uint_t} sx_{j} = 1;'.format(uint_t=uint_t, j=ndim - 1))
    for j in range(ndim - 1, 0, -1):
        ops.append(
            'const {uint_t} sx_{jm} = sx_{j} * {xsize_j};'.format(
                uint_t=uint_t, jm=j - 1, j=j, xsize_j=xshape[j]
            )
        )

    # determine nd coordinate in x corresponding to a given raveled coordinate,
    # i, in y.
    ops.append("""
        {uint_t} idx = i;
        {uint_t} s, t;
        """.format(uint_t=uint_t))
    for j in range(ndim - 1, 0, -1):
        ops.append("""
        s = {zsize_j};
        t = idx / s;
        in_coord[{j}] = idx - t * s;
        idx = t;
        """.format(j=j, zsize_j=yshape[j]))
    ops.append("in_coord[0] = idx;")

    # compute the transformed (target) coordinates, c_j
    ops = ops + coord_func(ndim)

    if mode == 'constant':
        # use cval if coordinate is outside the bounds of x
        _cond = ' || '.join(
            ['(c_{j} < 0) || (c_{j} > {cmax})'.format(j=j, cmax=xshape[j] - 1)
             for j in range(ndim)])
        ops.append("""
        if ({cond})
        {{
            out = (double){cval};
        }}
        else
        {{""".format(cond=_cond, cval=cval))

    if order == 0:
        for j in range(ndim):
            # determine nearest neighbor
            ops.append("""
            {int_t} cf_{j} = ({int_t})lrint((double)c_{j});
            """.format(int_t=int_t, j=j))

            # handle boundary
            if mode != 'constant':
                ixvar = 'cf_{j}'.format(j=j)
                ops.append(
                    _generate_boundary_condition_ops(
                        mode, ixvar, xshape[j]))

            # sum over ic_j will give the raveled coordinate in the input
            ops.append("""
            {int_t} ic_{j} = cf_{j} * sx_{j};
            """.format(int_t=int_t, j=j))
        _coord_idx = ' + '.join(['ic_{}'.format(j) for j in range(ndim)])
        ops.append("""
            out = x[{coord_idx}];
            """.format(coord_idx=_coord_idx))

    elif order == 1:
        for j in range(ndim):
            # get coordinates for linear interpolation along axis j
            ops.append("""
            {int_t} cf_{j} = ({int_t})floor((double)c_{j});
            {int_t} cc_{j} = cf_{j} + 1;
            {int_t} n_{j} = (c_{j} == cf_{j}) ? 1 : 2;  // points needed
            """.format(int_t=int_t, j=j))

            # handle boundaries for extension modes.
            ops.append("""
            {int_t} cf_bounded_{j} = cf_{j};
            {int_t} cc_bounded_{j} = cc_{j};
            """.format(int_t=int_t, j=j))
            if mode != 'constant':
                ixvar = 'cf_bounded_{j}'.format(j=j)
                ops.append(
                    _generate_boundary_condition_ops(
                        mode, ixvar, xshape[j]))
                ixvar = 'cc_bounded_{j}'.format(j=j)
                ops.append(
                    _generate_boundary_condition_ops(
                        mode, ixvar, xshape[j]))

            ops.append("""
            for (int s_{j} = 0; s_{j} < n_{j}; s_{j}++)
                {{
                    W w_{j};
                    {int_t} ic_{j};
                    if (s_{j} == 0)
                    {{
                        w_{j} = (W)cc_{j} - c_{j};
                        ic_{j} = cf_bounded_{j} * sx_{j};
                    }} else
                    {{
                        w_{j} = c_{j} - (W)cf_{j};
                        ic_{j} = cc_bounded_{j} * sx_{j};
                    }}""".format(int_t=int_t, j=j))

        _weight = ' * '.join(['w_{j}'.format(j=j) for j in range(ndim)])
        _coord_idx = ' + '.join(['ic_{j}'.format(j=j) for j in range(ndim)])
        ops.append("""
        X val = x[{coord_idx}];
        out += val * ({weight});
        """.format(coord_idx=_coord_idx, weight=_weight))
        ops.append('}' * ndim)

    if mode == 'constant':
        ops.append('}')

    if integer_output:
        ops.append('y = (Y)rint((double)out);')
    else:
        ops.append('y = (Y)out;')
    operation = '\n'.join(ops)

    name = 'interpolate_{}_order{}_{}_x{}_y{}'.format(
        name,
        order,
        mode,
        '_'.join(['{}'.format(j) for j in xshape]),
        '_'.join(['{}'.format(j) for j in yshape]),
    )
    return operation, name


@cupy.util.memoize()
def _get_map_kernel(xshape, mode, cval=0.0, order=1, integer_output=False):
    in_params = 'raw X x, raw W coords'
    out_params = 'Y y'
    operation, name = _generate_interp_custom(
        coord_func=_get_coord_map,
        xshape=xshape,
        yshape=xshape,
        mode=mode,
        cval=cval,
        order=order,
        name='shift',
        integer_output=integer_output,
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


@cupy.util.memoize()
def _get_shift_kernel(xshape, yshape, mode, cval=0.0, order=1,
                      integer_output=False):
    in_params = 'raw X x, raw W shift'
    out_params = 'Y y'
    operation, name = _generate_interp_custom(
        coord_func=_get_coord_shift,
        xshape=xshape,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name='shift',
        integer_output=integer_output,
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


@cupy.util.memoize()
def _get_zoom_shift_kernel(xshape, yshape, mode, cval=0.0, order=1,
                           integer_output=False):
    in_params = 'raw X x, raw W shift, raw W zoom'
    out_params = 'Y y'
    operation, name = _generate_interp_custom(
        coord_func=_get_coord_zoom_and_shift,
        xshape=xshape,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name='zoom_shift',
        integer_output=integer_output,
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


@cupy.util.memoize()
def _get_zoom_kernel(xshape, yshape, mode, cval=0.0, order=1,
                     integer_output=False):
    in_params = 'raw X x, raw W zoom'
    out_params = 'Y y'
    operation, name = _generate_interp_custom(
        coord_func=_get_coord_zoom,
        xshape=xshape,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name='zoom',
        integer_output=integer_output,
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


@cupy.util.memoize()
def _get_affine_kernel(xshape, yshape, mode, cval=0.0, order=1,
                       integer_output=False):
    in_params = 'raw X x, raw W mat'
    out_params = 'Y y'
    operation, name = _generate_interp_custom(
        coord_func=_get_coord_affine,
        xshape=xshape,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name='affine',
        integer_output=integer_output,
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)
