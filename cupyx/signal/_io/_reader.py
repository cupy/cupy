import cupy
import numpy


def read_bin(file, buffer=None, dtype=cupy.uint8, num_samples=None, offset=0):
    """
    Reads binary file into GPU memory.
    Can be used as a building blocks for custom unpack/pack
    data readers/writers.

    Parameters
    ----------
    file : str
        A string of filename to be read to GPU.
    buffer : ndarray, optional
        Pinned memory buffer to use when copying data from GPU.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type.
    num_samples : int, optional
        Number of samples to be loaded to GPU. If set to 0,
        read in all samples.
    offset : int, optional
        In the file, array data starts at this offset.
        Since offset is measured in bytes, it should normally
        be a multiple of the byte-size of dtype.
    Returns
    -------
    out : ndarray
        An 1-dimensional array containing binary data.

    """

    # Get current stream, default or not.
    stream = cupy.cuda.get_current_stream()

    # prioritize dtype of buffer if provided
    if buffer is not None:
        dtype = buffer.dtype

    # offset is measured in bytes
    offset *= cupy.dtype(dtype).itemsize

    fp = numpy.memmap(
        file, mode="r", offset=offset, shape=num_samples, dtype=dtype
    )

    if buffer is not None:
        buffer[:] = fp[:]
        out = cupy.empty(buffer.shape, buffer.dtype)
        out.set(buffer)
    else:
        out = cupy.asarray(fp)

    stream.synchronize()

    del fp

    return out
