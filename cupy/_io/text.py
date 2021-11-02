import numpy
import cupy


def savetxt(fname, X, *args, **kwargs):
    """Save an array to a text file.

    .. note::
        Uses NumPy's ``savetxt``.

    .. seealso:: :func:`numpy.savetxt`
    """
    numpy.savetxt(fname, cupy.asnumpy(X), *args, **kwargs)
