import numpy

import cupy


def _mapped_malloc(size):
    mem = cupy.cuda.PinnedMemory(size, cupy.cuda.runtime.hostAllocMapped)
    return cupy.cuda.PinnedMemoryPointer(mem, 0)


# Return shared memory array - similar to np.empty
def get_shared_mem(
    shape,
    dtype=cupy.float32,
    strides=None,
    order="C",
    stream=0,
    portable=False,
    wc=True,
):
    """Return shared memory between GPU and CPU. Similar to numpy.zeros

    Parameters
    ----------
    shape : ndarray.shape
        Size of shared memory allocation
    dtype : cupy.dtype or numpy.dtype
        Data type of allocation
    strides: int or None
    order: char
    stream : int
        Stream number (0 for default)
    portable : bool
    wc : bool
    """
    # Check https://github.com/cupy/cupy/issues/3452#issuecomment-903212530
    pinned_mem_pool = cupy.cuda.PinnedMemoryPool(_mapped_malloc)
    old_alloc = cupy.cuda.get_pinned_memory_allocator()
    try:
        cupy.cuda.set_pinned_memory_allocator(pinned_mem_pool.malloc)
        data = numpy.empty(shape, dtype, order=order)
        if strides is not None:
            numpy.lib.stride_tricks.as_strided(
                data, shape=shape, strides=strides
            )
        mem = cupy.cuda.alloc_pinned_memory(data.nbytes)
        ret = numpy.frombuffer(
            mem, data.dtype, data.size
        ).reshape(shape).astype(dtype)
        ret[...] = data
        return ret
    finally:
        cupy.cuda.set_pinned_memory_allocator(old_alloc)
    return None


def get_pinned_mem(shape, dtype):
    """
    Create a pinned memory allocation.

    Parameters
    ----------
    shape : int or tuple of ints
        Output shape.
    dtype : data-type
        Output data type.

    Returns
    -------
    ret : ndarray
        Pinned memory numpy array.

    """

    from math import prod

    count = prod(shape) if isinstance(shape, tuple) else shape
    mem = cupy.cuda.alloc_pinned_memory(count * cupy.dtype(dtype).itemsize)
    ret = numpy.frombuffer(mem, dtype, count).reshape(shape)

    return ret
