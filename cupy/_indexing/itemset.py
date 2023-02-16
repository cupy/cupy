from typing import Union
import cupy


def itemset(self: cupy.array,
            index: Union[int, tuple[int]],
            value: int) -> None:
    """description
    Insert scalar into an array (scalar is cast to array's dtype, if possible)
    There must be at least 1 argument, and define the last argument as item.
    Then, a.itemset(*args) is equivalent to but faster than a[args] = item.
    The item should be a scalar value and args must select a single item in
    the array a.
    """
    dtype: cupy.dtype = self.dtype
    if isinstance(index, tuple):
        for num in index:
            if not isinstance(num, int):
                raise ValueError("invalid index value. \
                    index value is only int.")

        self[index] = cupy.array(value, dtype=dtype)
    elif isinstance(index, int):
        shape: tuple = self.shape
        self = self.ravel()
        self[index] = cupy.array(value, dtype=dtype)
        self = self.reshape(shape)
    else:
        raise ValueError("invalid method of specifying index. \
            only int or tuple can be use.")


setattr(cupy.ndarray, "itemset", itemset)
