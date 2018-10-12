"""
Utilities that manipulate strides to achieve desirable effects.
An explanation of strides can be found in the "ndarray.rst" file in the
NumPy reference guide.
"""
from __future__ import division, absolute_import, print_function

import cupy as cp


class DummyArray(object):
    """Dummy object that just exists to hang __cuda_array_interface__
    dictionaries and possibly keep alive a reference to a base array.
    """

    def __init__(self, interface, base=None):
        self.__cuda_array_interface__ = interface
        self.base = base


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
    subok : bool, optional
        .. versionadded:: 1.10
        If True, subclasses are preserved.
    Returns
    -------
    view : ndarray
    See also
    --------
    broadcast_to: broadcast an array to a given shape.
    reshape : reshape an array.
    Notes
    -----
    ``as_strided`` creates a view into the array given the exact strides
    and shape. This means it manipulates the internal data structure of
    ndarray and, if done incorrectly, the array elements can point to
    invalid memory and can corrupt results or crash your program.
    It is advisable to always use the original ``x.strides`` when
    calculating new strides to avoid reliance on a contiguous memory
    layout.
    Furthermore, arrays created with this function often contain self
    overlapping memory, so that two elements are identical.
    Vectorized write operations on such arrays will typically be
    unpredictable. They may even give different results for small, large,
    or transposed arrays.
    Since writing to these arrays has to be tested and done with great
    care, you may want to use ``writeable=False`` to avoid accidental write
    operations.
    For these reasons it is advisable to avoid ``as_strided`` when
    possible.
    """
    interface = dict(x.__cuda_array_interface__)
    if shape is not None:
        interface['shape'] = tuple(shape)
    if strides is not None:
        interface['strides'] = tuple(strides)

    array = cp.ndarray(shape=interface['shape'], dtype=x.dtype,
                       memptr=x.data, strides=interface['strides'])

    return array
