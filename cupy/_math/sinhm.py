from numpy import exp


def sinhm(arr):
    """Returns hyperbolic sine of the given square matrix.
Args: arr: square matrix .
Returns: Hyperbolic sine of given square matrix as input.
    """
    return 0.5*(exp(arr)-exp(-1*arr))
