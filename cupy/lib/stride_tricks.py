def as_strided(x, shape=None, strides=None, subok=False):
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
    interface = dict(x.__cuda_array_interface__)
    if shape is not None:
        interface['shape'] = tuple(shape)
    if strides is not None:
        interface['strides'] = tuple(strides)

    array = cp.ndarray(shape=interface['shape'], dtype=x.dtype,
                       memptr=x.data, strides=interface['strides'])

    return array
