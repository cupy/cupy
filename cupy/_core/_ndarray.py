from cupy._core.core import _ndarray_base

class ndarray(_ndarray_base):
    """
    __init__(self, shape, dtype=float, memptr=None, strides=None, order='C')

    Multi-dimensional array on a CUDA device.

    This class implements a subset of methods of :class:`numpy.ndarray`.
    The difference is that this class allocates the array content on the
    current GPU device.

    Args:
        shape (tuple of ints): Length of axes.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        memptr (cupy.xpu.MemoryPointer): Pointer to the array content head.
        strides (tuple of ints or None): Strides of data in memory.
        order ({'C', 'F'}): Row-major (C-style) or column-major
            (Fortran-style) order.

    Attributes:
        base (None or cupy.ndarray): Base array from which this array is
            created as a view.
        data (cupy.xpu.MemoryPointer): Pointer to the array content head.
        ~ndarray.dtype(numpy.dtype): Dtype object of element type.

            .. seealso::
               `Data type objects (dtype) \
               <https://numpy.org/doc/stable/reference/arrays.dtypes.html>`_
        ~ndarray.size (int): Number of elements this array holds.

            This is equivalent to product over the shape tuple.

            .. seealso:: :attr:`numpy.ndarray.size`

    """

    __module__ = 'cupy'
    __slots__ = []

    def __new__(cls, *args, _obj=None, _no_init=False, **kwargs):
        x = super().__new__(cls, *args, **kwargs)
        if _no_init:
            return x
        x._init(*args, **kwargs)
        if cls is not ndarray:
            x.__array_finalize__(_obj)
        return x

    def __init__(self, *args, **kwargs):
        # Prevent from calling the super class `_ndarray_base.__init__()` as
        # it is used to check accidental direct instantiation of underlaying
        # `_ndarray_base` extention.
        pass

    def __array_finalize__(self, obj):
        pass

    # We provide the Python-level wrapper of `view` method to follow NumPy's
    # API signature, as it seems that Cython's `cpdef`d methods does not take
    # an argument named `type`. Cython also does not take starargs
    # (`*args` and `**kwargs`) for `cpdef`d methods so we can not interpret the
    # arguments `dtype` and `type` from them.
    def view(self, dtype=None, type=None):
        """Returns a view of the array.

        Args:
            dtype: If this is different from the data type of the array, the
                returned view reinterpret the memory sequence as an array of
                this type.

        Returns:
            cupy.ndarray: A view of the array. A reference to the original
            array is stored at the :attr:`~ndarray.base` attribute.

        .. seealso:: :meth:`numpy.ndarray.view`

        """
        return super(ndarray, self).view(dtype=dtype, array_class=type)
