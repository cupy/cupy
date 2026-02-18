import cupy


def write_bin(file, binary, buffer=None, append=True):
    """
    Writes binary array to file.

    Parameters
    ----------
    file : str
        A string of filename to store output.
    binary : ndarray
        Binary array to be written to file.
    buffer : ndarray, optional
        Pinned memory buffer to use when copying data from GPU.
    append : bool, optional
        Append to file if created.

    Returns
    -------
    out : ndarray
        An 1-dimensional array containing binary data.

    """

    # Get current stream, default or not.
    stream = cupy.cuda.get_current_stream()

    if buffer is None:
        buffer = cupy.asnumpy(binary)
    else:
        binary.get(out=buffer)

    if append is True:
        mode = "ab"
    else:
        mode = "wb"

    with open(file, mode) as f:
        stream.synchronize()
        buffer.tofile(f)
