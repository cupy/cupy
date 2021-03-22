from numpy import shape


def alen(arr):
    """Returns the first dimension of an array

Args:
a (array_like): Input array

Returns:
Single Int value : it gives the length of the first
dimension of the array.

    """

    return arr.shape[0]
