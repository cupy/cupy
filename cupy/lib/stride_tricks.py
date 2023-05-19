import cupy as _cupy
import numpy as np
from ..core.internal import _normalize_axis_indices as normalize_axis_tuple

def as_strided(x, shape=None, strides=None):
    """
    Create a view into the array with the given shape and strides.

    .. warning:: This function has to be used with extreme care, see notes.

    Parameters
    ----------
    x : ndarray
        Array to create a new.
    shape : sequence of int, optional
        The shape of the new array. Defaults to ``x.shape``.
    strides : sequence of int, optional
        The strides of the new array. Defaults to ``x.strides``.

    Returns
    -------
    view : ndarray

    See also
    --------
    numpy.lib.stride_tricks.as_strided
    reshape : reshape an array.

    Notes
    -----
    ``as_strided`` creates a view into the array given the exact strides
    and shape. This means it manipulates the internal data structure of
    ndarray and, if done incorrectly, the array elements can point to
    invalid memory and can corrupt results or crash your program.
    """
    shape = x.shape if shape is None else tuple(shape)
    strides = x.strides if strides is None else tuple(strides)

    return _cupy.ndarray(shape=shape, dtype=x.dtype,
                         memptr=x.data, strides=strides)

def sliding_window_view(x, window_shape, axis=None, *,
                        subok=False, writeable=False):

    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))

    # -- because normalize_axis_tuple does not handle lists:
    # -- with Xarray it causes error : 
    # -- TypeError: 'list' object cannot be interpreted as an integer
    print ('axis :', axis)
    axis = (
        axis[0]
        if np.iterable(axis) and len(axis) == 1
        else axis)

    # first convert input to array, possibly keeping subclass
    x = _cupy.array(x, copy=False, subok=subok)

    window_shape_array = _cupy.array(window_shape)
    if _cupy.any(window_shape_array < 0):
        raise ValueError('`window_shape` cannot contain negative values')

    if axis is None:
        axis = tuple(range(x.ndim))
        #print ('axis after None:', axis)
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide '
                             f'window_shape for all dimensions of `x`; '
                             f'got {len(window_shape)} window_shape elements '
                             f'and `x.ndim` is {x.ndim}.')
    else:
        #print ('axis:', axis)
        #print ('x.ndim:',x.ndim)
        axis = normalize_axis_tuple(axis, x.ndim)
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and '
                             f'axis; got {len(window_shape)} window_shape '
                             f'elements and {len(axis)} axes elements.')

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError(
                'window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(x, strides=out_strides, shape=out_shape)







