from numpy import exp


def sinhm(arr):
    """Returns hyperbolic sine of the given square matrix.
    
    Args: 
    arr(cupy.ndarray) : Square matrix whose hyperbolic sine has to be calculated .

    Returns: 
    
    (cupy.ndarray):  Hyperbolic sine of given square matrix as input.
    """
    return 0.5 * (exp(arr) - exp(-1 * arr))
