import cupy


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

    arr = cupy.array(arr)

    # Checking whether the input is a 2D matrix or not
    if(len(arr.shape) != 2):
        raise ValueError("Dimensions of matrix should be 2")

    # Checking whether the input matrix is square matrix or not
    if(arr.shape[0] != arr.shape[1]):
        raise ValueError("Input matrix should be a square matrix")

    # Checking whether the input matrix elements are nan or not
    if(cupy.isnan(cupy.cumsum(arr)[2*arr.shape[0]-1])):
        raise ValueError("Input matrix elements cannot be nan")

    # Checking whether the input matrix elements are infinity or not
    if(cupy.isinf(cupy.cumsum(arr)[2*arr.shape[0]-1])):
        raise ValueError("Input matrix elements cannot be infinity")

    return 0.5 * (cupy.exp(arr) - cupy.exp(-1 * arr))
