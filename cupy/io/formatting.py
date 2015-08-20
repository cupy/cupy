import numpy

import cupy


def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    """Returns the string representation of an array.

    Args:
        arr (array_like): Input array. It should be able to feed to
            :func:`cupy.asnumpy`.
        max_line_width (int): The maximum number of line lengths.
        precision (int): Floating point precision. It uses the current printing
            precision of NumPy.
        suppress_small (bool): If True, very small numbers are printed as
            zeros

    Returns:
        str: The string representation of ``arr``.

    .. seealso:: :func:`numpy.array_repr`

    """
    return numpy.array_repr(cupy.asnumpy(arr), max_line_width, precision,
                            suppress_small)


def array_str(arr, max_line_width=None, precision=None, suppress_small=None):
    """Returns the string representation of the content of an array.

    Args:
        arr (array_like): Input array. It should be able to feed to
            :func:`cupy.asnumpy`.
        max_line_width (int): The maximum number of line lengths.
        precision (int): Floating point precision. It uses the current printing
            precision of NumPy.
        suppress_small (bool): If True, very small number are printed as
            zeros.

    .. seealso:: :func:`numpy.array_str`

    """
    return numpy.array_str(cupy.asnumpy(arr), max_line_width, precision,
                           suppress_small)
