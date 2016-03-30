import numpy

import cupy


class NpzFile(object):

    def __init__(self, npz_file):
        self.npz_file = npz_file

    def __enter__(self):
        self.npz_file.__enter__()
        return self

    def __exit__(self, typ, val, traceback):
        self.npz_file.__exit__(typ, val, traceback)

    def __getitem__(self, key):
        arr = self.npz_file[key]
        return cupy.array(arr)

    def close(self):
        self.npz_file.close()


def load(file, mmap_mode=None):
    """Loads arrays or pickled objects from ``.npy``, ``.npz`` or pickled file.

    This function just calls ``numpy.load`` and then sends the arrays to the
    current device. NPZ file is converted to NpzFile object, which defers the
    transfer to the time of accessing the items.

    Args:
        file (file-like object or string): The file to read.
        mmap_mode (None, 'r+', 'r', 'w+', 'c'): If not ``None``, memory-map the
            file to construct an intermediate :class:`numpy.ndarray` object and
            transfer it to the current device.

    Returns:
        CuPy array or NpzFile object depending on the type of the file. NpzFile
        object is a dictionary-like object with the context manager protocol
        (which enables us to use *with* statement on it).

    .. seealso:: :func:`numpy.load`

    """
    obj = numpy.load(file, mmap_mode)
    if isinstance(obj, numpy.ndarray):
        return cupy.array(obj)
    elif isinstance(obj, numpy.lib.npyio.NpzFile):
        return NpzFile(obj)
    else:
        return obj


def save(file, arr):
    """Saves an array to a binary file in ``.npy`` format.

    Args:
        file (file or str): File or filename to save.
        arr (array_like): Array to save. It should be able to feed to
            :func:`cupy.asnumpy`.

    .. seealso:: :func:`numpy.save`

    """
    numpy.save(file, cupy.asnumpy(arr))


def savez(file, *args, **kwds):
    """Saves one or more arrays into a file in uncompressed ``.npz`` format.

    Arguments without keys are treated as arguments with automatic keys named
    ``arr_0``, ``arr_1``, etc. corresponding to the positions in the argument
    list. The keys of arguments are used as keys in the ``.npz`` file, which
    are used for accessing NpzFile object when the file is read by
    :func:`cupy.load` function.

    Args:
        file (file or str): File or filename to save.
        *args: Arrays with implicit keys.
        **kwds: Arrays with explicit keys.

    .. seealso:: :func:`numpy.savez`

    """
    args = map(cupy.asnumpy, args)
    for key in kwds:
        kwds[key] = cupy.asnumpy(kwds[key])
    numpy.savez(file, *args, **kwds)


def savez_compressed(file, *args, **kwds):
    """Saves one or more arrays into a file in compressed ``.npz`` format.

    It is equivalent to :func:`cupy.savez` function except the output file is
    compressed.

    .. seealso::
       :func:`cupy.savez` for more detail,
       :func:`numpy.savez_compressed`

    """
    args = map(cupy.asnumpy, args)
    for key in kwds:
        kwds[key] = cupy.asnumpy(kwds[key])
    numpy.savez_compressed(file, *args, **kwds)
