from __future__ import annotations

import os
import pathlib
import tempfile
import unittest

import numpy
import pytest
try:
    import scipy.sparse
    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
from cupy import testing
from cupyx.scipy import sparse


# The formats ``scipy.sparse.save_npz`` writes that CuPy can build.  ``bsr``
# is deliberately absent: CuPy has no BSR matrix class, so loading a BSR file
# raises ``ValueError`` (see ``TestLoadNpzErrors``).
_HOST_CLASSES = {
    'csr': scipy.sparse.csr_matrix if scipy_available else None,
    'csc': scipy.sparse.csc_matrix if scipy_available else None,
    'coo': scipy.sparse.coo_matrix if scipy_available else None,
    'dia': scipy.sparse.dia_matrix if scipy_available else None,
}
# ``{sparse_format: cupy class name}`` for the formats above.
_CUPY_CLASSES = {
    'csr': sparse.csr_matrix,
    'csc': sparse.csc_matrix,
    'coo': sparse.coo_matrix,
    'dia': sparse.dia_matrix,
}

# dtypes the CuPy sparse classes accept (see docs/source/reference/
# scipy_sparse.rst); float16 and the integer dtypes are not supported.
DTYPES = ['bool', 'float32', 'float64', 'complex64', 'complex128']


def _save(matrix, compressed=True):
    """Write ``matrix`` with the real ``save_npz``; return the file path."""
    fd, path = tempfile.mkstemp(suffix='.npz')
    os.close(fd)
    scipy.sparse.save_npz(path, matrix, compressed=compressed)
    return path


def _numpy_load_error(path):
    """Return the exception *type* :func:`numpy.load` raises for ``path``.

    Only the type is kept: holding the exception would pin the traceback and,
    with it, the file handle NumPy leaks on a bad archive, which then blocks
    cleanup on Windows.
    """
    try:
        with numpy.load(path, allow_pickle=False) as loaded:
            loaded['format']
    except Exception as e:
        return type(e)
    raise AssertionError(f'numpy.load unexpectedly succeeded for {path}')


def _assert_matches(loaded, original):
    """Assert ``loaded`` is a device-backed CuPy sparse object equal to
    ``original`` (a SciPy sparse object)."""
    assert isinstance(loaded, sparse.spmatrix)
    assert isinstance(loaded, _CUPY_CLASSES[original.format])
    assert loaded.shape == original.shape
    assert loaded.dtype == original.dtype
    # Every array the result owns must live on the device.
    for name in ('data', 'indices', 'indptr', 'offsets', 'row', 'col'):
        arr = getattr(loaded, name, None)
        if arr is not None:
            # ``isinstance`` is the real device-residency check: it fails if
            # any array came back as a NumPy array.
            assert isinstance(arr, cupy.ndarray)
    testing.assert_allclose(loaded.toarray(), original.toarray())


@unittest.skipUnless(scipy_available, 'requires scipy')
class TestLoadNpz(unittest.TestCase):

    def _roundtrip(self, host_matrix, compressed=True):
        path = _save(host_matrix, compressed=compressed)
        try:
            return sparse.load_npz(path)
        finally:
            os.remove(path)

    def test_dtypes(self):
        for dtype in DTYPES:
            for fmt in ('csr', 'csc', 'coo', 'dia'):
                host = _HOST_CLASSES[fmt](
                    numpy.array([[1, 0, 2], [0, 0, 3], [0, 0, 0]],
                                dtype=dtype))
                with self.subTest(dtype=dtype, format=fmt):
                    _assert_matches(self._roundtrip(host), host)

    def test_compressed_and_uncompressed(self):
        for dtype in DTYPES:
            for fmt in ('csr', 'csc', 'coo', 'dia'):
                host = _HOST_CLASSES[fmt](
                    numpy.array([[1.5, 0, 2.5], [0, 0, 3.5], [0, 0, 0]],
                                dtype=dtype))
                for compressed in (True, False):
                    with self.subTest(dtype=dtype, format=fmt,
                                      compressed=compressed):
                        _assert_matches(
                            self._roundtrip(host, compressed), host)

    def test_sparse_array_vs_sparse_matrix(self):
        dense = numpy.array([[1.2, 0, 0.9], [0, 0.3, 0]])
        path = _save(scipy.sparse.csr_matrix(dense))
        try:
            loaded_matrix = sparse.load_npz(path)
        finally:
            os.remove(path)
        path = _save(scipy.sparse.csr_array(dense))
        try:
            loaded_array = sparse.load_npz(path)
        finally:
            os.remove(path)

        assert not isinstance(loaded_matrix, sparse.sparray)
        assert isinstance(loaded_array, sparse.sparray)
        assert loaded_matrix.dtype == loaded_array.dtype
        testing.assert_allclose(loaded_matrix.toarray(), dense)
        testing.assert_allclose(loaded_array.toarray(), dense)

    def test_empty(self):
        # All zeros, non-square.
        dense = numpy.zeros((4, 6))
        for fmt in ('csr', 'csc', 'coo', 'dia'):
            host = _HOST_CLASSES[fmt](dense)
            with self.subTest(format=fmt):
                loaded = self._roundtrip(host)
                _assert_matches(loaded, host)
                assert loaded.nnz == 0

    def test_zero_rows(self):
        dense = numpy.zeros((0, 6))
        for fmt in ('csr', 'csc', 'coo', 'dia'):
            host = _HOST_CLASSES[fmt](dense)
            with self.subTest(format=fmt):
                loaded = self._roundtrip(host)
                _assert_matches(loaded, host)
                assert loaded.shape == (0, 6)

    def test_zero_cols(self):
        dense = numpy.zeros((4, 0))
        for fmt in ('csr', 'csc', 'coo'):
            host = _HOST_CLASSES[fmt](dense)
            with self.subTest(format=fmt):
                _assert_matches(self._roundtrip(host), host)

    def test_non_square(self):
        dense = numpy.array([[1., 0, 2, 3], [0, 0, 0, 4]])
        for fmt in ('csr', 'csc', 'coo', 'dia'):
            host = _HOST_CLASSES[fmt](dense)
            with self.subTest(format=fmt):
                _assert_matches(self._roundtrip(host), host)

    def test_one_entry(self):
        dense = numpy.zeros((4, 6))
        dense[1, 2] = 1
        for fmt in ('csr', 'csc', 'coo'):
            host = _HOST_CLASSES[fmt](dense)
            with self.subTest(format=fmt):
                loaded = self._roundtrip(host)
                _assert_matches(loaded, host)
                assert loaded.nnz == 1

    def test_random(self):
        rng = numpy.random.RandomState(0)
        dense = rng.random_sample((10, 10))
        dense[dense > 0.7] = 0
        for fmt in ('csr', 'csc', 'coo', 'dia'):
            host = _HOST_CLASSES[fmt](dense)
            with self.subTest(format=fmt):
                _assert_matches(self._roundtrip(host), host)

    def test_duplicate_entries(self):
        # coo with repeated coordinates must survive the round trip as-is.
        data = numpy.array([1.0, 2.0, 3.0])
        row = numpy.array([0, 0, 1])
        col = numpy.array([1, 1, 2])
        host = scipy.sparse.coo_matrix((data, (row, col)), shape=(2, 3))
        assert host.nnz == 3
        loaded = self._roundtrip(host)
        _assert_matches(loaded, host)
        assert loaded.nnz == 3

    def test_unsorted_indices(self):
        # csr/csc with column indices out of order inside a row/column.
        data = numpy.array([2.0, 1.0, 3.0])
        indices = numpy.array([2, 0, 1])
        indptr = numpy.array([0, 2, 3])
        host = scipy.sparse.csr_matrix((data, indices, indptr), shape=(2, 3))
        path = _save(host)
        try:
            loaded = sparse.load_npz(path)
        finally:
            os.remove(path)
        assert loaded.format == 'csr'
        testing.assert_array_equal(loaded.indices, indices)
        testing.assert_array_equal(loaded.data, data)
        # Still a valid, fully usable sparse object after sorting.
        testing.assert_allclose(loaded.toarray(), host.toarray())
        assert loaded.nnz == 3

    def test_file_like_object(self):
        dense = numpy.array([[1., 0, 2.], [0, 3., 0]])
        path = _save(scipy.sparse.csr_matrix(dense))
        try:
            with open(path, 'rb') as f:
                loaded = sparse.load_npz(f)
            _assert_matches(loaded, scipy.sparse.csr_matrix(dense))
        finally:
            os.remove(path)

    def test_pathlike(self):
        dense = numpy.array([[1., 0, 2.], [0, 3., 0]])
        path = _save(scipy.sparse.csr_matrix(dense))
        try:
            _assert_matches(
                sparse.load_npz(pathlib.Path(path)),
                scipy.sparse.csr_matrix(dense))
        finally:
            os.remove(path)

    def test_int64_indices(self):
        data = numpy.array([2.0, 1.0])
        indices = numpy.array([1, 0], dtype=numpy.int64)
        indptr = numpy.array([0, 1, 2], dtype=numpy.int64)
        host = scipy.sparse.csr_matrix((data, indices, indptr), shape=(2, 2))
        loaded = self._roundtrip(host)
        _assert_matches(loaded, host)


@unittest.skipUnless(scipy_available, 'requires scipy')
class TestLoadNpzErrors(unittest.TestCase):

    def test_bsr_not_supported(self):
        # CuPy has no bsr_matrix/bsr_array class.  The file is a perfectly
        # valid SciPy archive, so the error says the format is unsupported
        # rather than claiming the format is unknown.
        if not hasattr(scipy.sparse, 'bsr_matrix'):  # pragma: no cover
            self.skipTest('scipy.sparse has no bsr_matrix')
        dense = numpy.array([[1., 0, 2.], [0, 3., 0], [0, 0, 4.]])
        path = _save(scipy.sparse.bsr_matrix(dense))
        try:
            with pytest.raises(ValueError, match='does not support the "bsr"'):
                sparse.load_npz(path)
        finally:
            os.remove(path)

    def test_unsupported_dtype(self):
        # cuSPARSE-backed formats only accept bool/float/complex dtypes, so an
        # integer matrix saved by SciPy cannot be built as CSR.
        host = scipy.sparse.csr_matrix(numpy.eye(3, dtype='int32'))
        path = _save(host)
        try:
            with pytest.raises(ValueError, match='supported'):
                sparse.load_npz(path)
        finally:
            os.remove(path)

    def test_unknown_format(self):
        path = _save(scipy.sparse.csr_matrix(numpy.eye(2)))
        try:
            with numpy.load(path, allow_pickle=False) as loaded:
                arrays = dict(loaded)
            del arrays['format']
            numpy.savez(path + '.2.npz', format=numpy.array(b'wat'),
                        **arrays)
            with pytest.raises(ValueError, match='Unknown format'):
                sparse.load_npz(path + '.2.npz')
        finally:
            os.remove(path)
            if os.path.exists(path + '.2.npz'):
                os.remove(path + '.2.npz')

    def test_not_a_sparse_file(self):
        fd, path = tempfile.mkstemp(suffix='.npz')
        os.close(fd)
        try:
            numpy.savez(path, junk=numpy.arange(3))
            with pytest.raises(ValueError, match='does not contain'):
                sparse.load_npz(path)
        finally:
            os.remove(path)

    def test_missing_file(self):
        with pytest.raises(OSError):
            sparse.load_npz(os.path.join(tempfile.gettempdir(),
                                         'no_such_sparse_file.npz'))

    def test_garbage_file(self):
        fd, path = tempfile.mkstemp(suffix='.npz')
        try:
            os.write(fd, b'not a zip archive at all')
        finally:
            os.close(fd)
        try:
            # cupy must surface exactly the error numpy.load raises for the
            # same file, not some other (or no) error.
            with pytest.raises(_numpy_load_error(path)):
                sparse.load_npz(path)
        finally:
            os.remove(path)

    def test_truncated_file(self):
        dense = numpy.array([[1., 0, 2.], [0, 3., 0]])
        path = _save(scipy.sparse.csr_matrix(dense))
        with open(path, 'rb') as f:
            blob = f.read()
        os.remove(path)
        # Chop the tail off the archive, including the zip central directory.
        truncated = path + '.trunc'
        with open(truncated, 'wb') as f:
            f.write(blob[:len(blob) // 2])
        try:
            with pytest.raises(_numpy_load_error(truncated)):
                sparse.load_npz(truncated)
        finally:
            os.remove(truncated)

    def test_malicious_load(self):
        # Pickled objects must not be executed: allow_pickle=False makes
        # numpy.load raise instead.
        class Executor:
            def __reduce__(self):
                return (numpy.testing.assert_,
                        (False, 'unexpected code execution'))

        fd, path = tempfile.mkstemp(suffix='.npz')
        os.close(fd)
        try:
            numpy.savez(path, format=Executor())
            with pytest.raises(ValueError):
                sparse.load_npz(path)
        finally:
            os.remove(path)
