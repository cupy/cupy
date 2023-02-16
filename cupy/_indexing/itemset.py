from typing import Union
import cupy


def itemset(arr: cupy.array, index: Union[int, tuple[int]], value: int) -> cupy.array:
    """description
    Insert scalar into an array (scalar is cast to array's dtype, if possible)
    There must be at least 1 argument, and define the last argument as item. 
    Then, a.itemset(*args) is equivalent to but faster than a[args] = item. 
    The item should be a scalar value and args must select a single item in the array a.
    """
    dtype: cupy.dtype = arr.dtype
    if isinstance(index, tuple):
        for num in index:
            if not isinstance(num, int):
                raise ValueError("invalid index value. index value is only int.")

        arr[index] = cupy.array(value, dtype=dtype)
    elif isinstance(index, int):
        shape: tuple = arr.shape
        arr = arr.ravel()
        arr[index] = cupy.array(value, dtype=dtype)
        arr = arr.reshape(shape)
    else:
        raise ValueError("invalid method of specifying index. only int or tuple can be use.")

    return cupy.array(arr, dtype=dtype)


def asfarray(a, dtype=cupy.float_):
    """Converts array elements to float type.

    Args:
        a (cupy.ndarray): Source array.
        dtype: str or dtype object, optional

    Returns:
        cupy.ndarray: The input array ``a`` as a float ndarray.

    .. seealso:: :func:`numpy.asfarray`

    """
    if not cupy.issubdtype(dtype, cupy.inexact):
        dtype = cupy.float_
    return cupy.asarray(a, dtype=dtype)

