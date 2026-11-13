from __future__ import annotations

import numpy

import cupy
import cupyx


# Make loading safe vs. malicious input
PICKLE_KWARGS = dict(allow_pickle=False)

# SciPy formats that CuPy does not implement.  These are reported separately
# from a genuinely unknown format: telling a user that a perfectly valid BSR
# file has an "Unknown format" sends them hunting for a corrupt archive
# instead of telling them the format is not supported on the GPU.
_UNSUPPORTED_FORMATS = ('bsr', 'dok', 'lil')


def load_npz(file):
    """Load a sparse array/matrix from a file using ``.npz`` format.

    The file must have been produced by :func:`scipy.sparse.save_npz`.  The
    ``.npz`` archive is read on the host with :func:`numpy.load` -- the same
    wrapper pattern :func:`cupy.save` and :func:`cupy.load` use -- and the
    index/data arrays are then copied to the current device, so the returned
    object is a CuPy sparse array/matrix.

    Args:
        file (str or file-like object):
            Either the file name (string) or an open file (file-like object)
            where the data will be loaded.

    Returns:
        cupyx.scipy.sparse.sparray or cupyx.scipy.sparse.spmatrix:
            A sparse array/matrix containing the loaded data.  A file written
            by :func:`scipy.sparse.save_npz` records whether it holds a sparse
            array or a sparse matrix, and the matching CuPy class is built:
            ``csc_array``, ``csr_array``, ``dia_array`` or ``coo_array`` for
            an array, and the ``*_matrix`` counterpart for a matrix.

    Raises:
        OSError:
            If the input file does not exist or cannot be read.
        ValueError:
            If the file does not hold a sparse array/matrix, names a format
            that CuPy does not implement (BSR, DOK and LIL are not supported),
            or holds a data dtype CuPy sparse cannot represent.

    Notes:
        The data dtype must be one CuPy sparse supports: ``bool``,
        ``float32``, ``float64``, ``complex64`` or ``complex128``.

    .. seealso::

        :func:`scipy.sparse.load_npz`, :func:`numpy.load`

    Examples:
        Store a sparse array on disk with SciPy and load it onto the GPU::

            import scipy.sparse
            import cupyx.scipy.sparse

            scipy.sparse.save_npz(
                '/tmp/sparse_array.npz',
                scipy.sparse.csc_array([[0, 0, 3], [4, 0, 0]]))
            sparse_array = cupyx.scipy.sparse.load_npz('/tmp/sparse_array.npz')

    """
    with numpy.load(file, **PICKLE_KWARGS) as loaded:
        sparse_format = loaded.get('format')
        if sparse_format is None:
            raise ValueError(f'The file {file} does not contain '
                             'a sparse array or matrix.')
        sparse_format = sparse_format.item()

        if not isinstance(sparse_format, str):
            # Play safe with Python 2 vs 3 backward compatibility;
            # files saved with SciPy < 1.0.0 may contain unicode or bytes.
            sparse_format = sparse_format.decode('ascii')

        if sparse_format in _UNSUPPORTED_FORMATS:
            raise ValueError(
                f'CuPy does not support the "{sparse_format}" sparse '
                f'format. Convert the matrix to CSR, CSC, DIA or COO '
                f'before saving it.')

        if loaded.get('_is_array'):
            sparse_type = sparse_format + '_array'
        else:
            sparse_type = sparse_format + '_matrix'

        try:
            cls = getattr(cupyx.scipy.sparse, sparse_type)
        except AttributeError as e:
            raise ValueError(f'Unknown format "{sparse_type}"') from e

        # CuPy sparse objects keep every array on the device.
        def dev(key):
            return cupy.asarray(loaded[key])

        if sparse_format in ('csc', 'csr', 'bsr'):
            return cls((dev('data'), dev('indices'), dev('indptr')),
                       shape=loaded['shape'])
        elif sparse_format == 'dia':
            return cls((dev('data'), dev('offsets')), shape=loaded['shape'])
        elif sparse_format == 'coo':
            if 'coords' in loaded:
                return cls((dev('data'), dev('coords')),
                           shape=loaded['shape'])
            return cls((dev('data'), (dev('row'), dev('col'))),
                       shape=loaded['shape'])
        else:
            raise NotImplementedError(
                'Load is not implemented for sparse matrix of format '
                f'{sparse_format}.')
