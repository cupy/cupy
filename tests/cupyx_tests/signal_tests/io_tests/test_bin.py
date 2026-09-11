import os
import tempfile

import cupy
import cupyx.signal


def test_write_paged():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_fname = os.path.join(tmpdir, "test_read.sigmf-data")
        actual = cupy.random.rand(100).astype(cupy.complex64)
        cupyx.signal.write_bin(str(data_fname), actual)
        expect = cupyx.signal.read_bin(str(data_fname), dtype=cupy.complex64)
        cupy.testing.assert_array_equal(actual, expect)


def test_write_pinned_buffer():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_fname = os.path.join(tmpdir, "test_read.sigmf-data")
        actual = cupy.random.rand(100).astype(cupy.complex64)
        cupyx.signal.write_bin(str(data_fname), actual)
        binary = cupyx.signal.read_bin(str(data_fname), dtype=cupy.complex64)
        buffer = cupyx.signal.get_pinned_mem(binary.shape, cupy.complex64)
        cupyx.signal.write_bin(
            str(data_fname), actual, buffer=buffer, append=False
        )
        expect = cupyx.signal.read_bin(str(data_fname), buffer=buffer)
        cupy.testing.assert_array_equal(actual, expect)


def test_write_shared_buffer():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_fname = os.path.join(tmpdir, "test_read.sigmf-data")
        actual = cupy.random.rand(100).astype(cupy.complex64)
        cupyx.signal.write_bin(str(data_fname), actual)
        binary = cupyx.signal.read_bin(str(data_fname), dtype=cupy.complex64)
        buffer = cupyx.signal.get_shared_mem(binary.shape, cupy.complex64)
        cupyx.signal.write_bin(
            str(data_fname), actual, buffer=buffer, append=False
        )
        expect = cupyx.signal.read_bin(str(data_fname), buffer=buffer)
        cupy.testing.assert_array_equal(actual, expect)
