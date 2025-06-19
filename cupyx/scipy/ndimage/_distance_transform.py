import numbers

from ._pba_2d import _pba_2d
from ._pba_3d import _pba_3d


def distance_transform_edt(image, sampling=None, return_distances=True,
                           return_indices=False, distances=None, indices=None,
                           *, block_params=None, float64_distances=True):
    r"""Exact Euclidean distance transform.

    This function calculates the distance transform of the `input`, by
    replacing each foreground (non-zero) element, with its shortest distance to
    the background (any zero-valued element).

    In addition to the distance transform, the feature transform can be
    calculated. In this case the index of the closest background element to
    each foreground element is returned in a separate array.

    Parameters
    ----------
    image : array_like
        Input data to transform. Can be any type but will be converted into
        binary: 1 wherever image equates to True, 0 elsewhere.
    sampling : float, or sequence of float, optional
        Spacing of elements along each dimension. If a sequence, must be of
        length equal to the image rank; if a single number, this is used for
        all axes. If not specified, a grid spacing of unity is implied.
    return_distances : bool, optional
        Whether to calculate the distance transform.
    return_indices : bool, optional
        Whether to calculate the feature transform.
    distances : cupy.ndarray, optional
        An output array to store the calculated distance transform, instead of
        returning it. `return_distances` must be ``True``. It must be the same
        shape as `image`. Should have dtype ``cp.float32`` if
        `float64_distances` is ``False``, otherwise it should be
        ``cp.float64``.
    indices : cupy.ndarray, optional
        An output array to store the calculated feature transform, instead of
        returning it. `return_indicies` must be ``True``. Its shape must be
        ``(image.ndim,) + image.shape``. Its dtype must be a signed or unsigned
        integer type of at least 16-bits in 2D or 32-bits in 3D.

    Other Parameters
    ----------------
    block_params : 3-tuple of int
        The m1, m2, m3 algorithm parameters as described in [2]_. If None,
        suitable defaults will be chosen. Note: This parameter is specific to
        cuCIM and does not exist in SciPy.
    float64_distances : bool, optional
        If True, use double precision in the distance computation (to match
        SciPy behavior). Otherwise, single precision will be used for
        efficiency. Note: This parameter is specific to cuCIM and does not
        exist in SciPy.

    Returns
    -------
    distances : cupy.ndarray, optional
        The calculated distance transform. Returned only when
        `return_distances` is ``True`` and `distances` is not supplied. It will
        have the same shape as `image`. Will have dtype `cp.float64` if
        `float64_distances` is ``True``, otherwise it will have dtype
        ``cp.float32``.
    indices : ndarray, optional
        The calculated feature transform. It has an image-shaped array for each
        dimension of the image. See example below. Returned only when
        `return_indices` is ``True`` and `indices` is not supplied.

    Notes
    -----
    The Euclidean distance transform gives values of the Euclidean distance.

    .. math::

      y_i = \sqrt{\sum_{i}^{n} (x[i] - b[i])^2}

    where :math:`b[i]` is the background point (value 0) with the smallest
    Euclidean distance to input points :math:`x[i]`, and :math:`n` is the
    number of dimensions.

    Note that the `indices` output may differ from the one given by
    :func:`scipy.ndimage.distance_transform_edt` in the case of input pixels
    that are equidistant from multiple background points.

    The parallel banding algorithm implemented here was originally described in
    [1]_. The kernels used here correspond to the revised PBA+ implementation
    that is described on the author's website [2]_. The source code of the
    author's PBA+ implementation is available at [3]_.

    References
    ----------
    .. [1] Thanh-Tung Cao, Ke Tang, Anis Mohamed, and Tiow-Seng Tan. 2010.
        Parallel Banding Algorithm to compute exact distance transform with the
        GPU. In Proceedings of the 2010 ACM SIGGRAPH symposium on Interactive
        3D Graphics and Games (I3D ’10). Association for Computing Machinery,
        New York, NY, USA, 83–90.
        DOI:https://doi.org/10.1145/1730804.1730818
    .. [2] https://www.comp.nus.edu.sg/~tants/pba.html
    .. [3] https://github.com/orzzzjq/Parallel-Banding-Algorithm-plus

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.core.operations import morphology
    >>> a = cp.array(([0,1,1,1,1],
    ...               [0,0,1,1,1],
    ...               [0,1,1,1,1],
    ...               [0,1,1,1,0],
    ...               [0,1,1,0,0]))
    >>> morphology.distance_transform_edt(a)
    array([[ 0.    ,  1.    ,  1.4142,  2.2361,  3.    ],
           [ 0.    ,  0.    ,  1.    ,  2.    ,  2.    ],
           [ 0.    ,  1.    ,  1.4142,  1.4142,  1.    ],
           [ 0.    ,  1.    ,  1.4142,  1.    ,  0.    ],
           [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])

    With a sampling of 2 units along x, 1 along y:

    >>> morphology.distance_transform_edt(a, sampling=[2,1])
    array([[ 0.    ,  1.    ,  2.    ,  2.8284,  3.6056],
           [ 0.    ,  0.    ,  1.    ,  2.    ,  3.    ],
           [ 0.    ,  1.    ,  2.    ,  2.2361,  2.    ],
           [ 0.    ,  1.    ,  2.    ,  1.    ,  0.    ],
           [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])

    Asking for indices as well:

    >>> edt, inds = morphology.distance_transform_edt(a, return_indices=True)
    >>> inds
    array([[[0, 0, 1, 1, 3],
            [1, 1, 1, 1, 3],
            [2, 2, 1, 3, 3],
            [3, 3, 4, 4, 3],
            [4, 4, 4, 4, 4]],
           [[0, 0, 1, 1, 4],
            [0, 1, 1, 1, 4],
            [0, 0, 1, 4, 4],
            [0, 0, 3, 3, 4],
            [0, 0, 3, 3, 4]]])

    """
    scalar_sampling = None
    if sampling is not None:
        if isinstance(sampling, numbers.Number):
            sampling = (sampling,)
        if len(set(sampling)) == 1:
            # In the isotropic case, can use the kernels without sample scaling
            # and just adjust the final distance accordingly.
            scalar_sampling = float(sampling[0])
            sampling = None

    if image.ndim == 3:
        pba_func = _pba_3d
    elif image.ndim == 2:
        pba_func = _pba_2d
    else:
        raise NotImplementedError(
            "Only 2D and 3D distance transforms are supported.")

    vals = pba_func(
        image,
        sampling=sampling,
        return_distances=return_distances,
        return_indices=return_indices,
        block_params=block_params,
        distances=distances,
        indices=indices,
        float64_distances=float64_distances,
    )

    if return_distances and scalar_sampling is not None:
        # inplace multiply in case distance != None
        vals = list(vals)
        vals[0] *= scalar_sampling
        vals = tuple(vals)

    if len(vals) == 1:
        vals = vals[0]

    return vals
