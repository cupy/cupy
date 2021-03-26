import numpy


def sinhm(arr):
    """Hyperbolic Sine calculation for a given nXn matrix

    Args:
        arr(cupy.ndarray) :
                            Square matrix with dimension nXn.

    Returns:
        (cupy.ndarray):

                        Hyperbolic sine of given square matrix as input.

    ..seealso:: :func: 'scipy.linlag.matfuncs.py'                
    """

    arr = numpy.asanyarray(arr)

    # Checking whether the input is a 2D matrix or not
    if(len(arr.shape) != 2):
        raise ValueError("Dimensions of matrix should be 2")

    # Checking whether the input matrix is square matrix or not
    if(arr.shape[0] != arr.shape[1]):
        raise ValueError("Input matrix should be a square matrix")

    # Checking whether the input matrix elements are nan or not
    if(numpy.isnan(numpy.sum(arr))):
        raise ValueError("Input matrix elements cannot be nan")

    # Checking whether the input matrix elements are infinity or not
    if(numpy.isinf(numpy.sum(arr))):
        raise ValueError("Input matrix elements cannot be infinity")

    return 0.5 * (numpy.exp(arr) - numpy.exp(-1 * arr))
