from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import cuda, testing
from cupy._core import _host_transfer


@pytest.fixture
def pipelined(monkeypatch):
    # Force the pipelined paths on small arrays, with small chunks and only
    # two slots so that slot reuse (waiting on earlier DMAs) is exercised.
    monkeypatch.setattr(_host_transfer, 'H2D_THRESHOLD', 0)
    monkeypatch.setattr(_host_transfer, 'H2D_REORDER_THRESHOLD', 0)
    monkeypatch.setattr(_host_transfer, 'D2H_THRESHOLD', 0)
    monkeypatch.setattr(_host_transfer, 'CHUNK_BYTES', 4096)
    monkeypatch.setattr(_host_transfer, 'NUM_SLOTS', 2)
    # Several chunks per reorder block, and several blocks per array.
    monkeypatch.setattr(_host_transfer, 'REORDER_BLOCK_BYTES', 16384)


_shapes = [(1,), (1000,), (1025, 3), (3, 1025), (17, 33, 9), (100003,)]
_dtypes = [numpy.float32, numpy.int8, numpy.complex128]


@pytest.mark.usefixtures('pipelined')
@pytest.mark.parametrize('shape', _shapes)
@pytest.mark.parametrize('dtype', _dtypes)
@pytest.mark.parametrize('src_order', ['C', 'F'])
@pytest.mark.parametrize('dst_order', ['C', 'F', 'K', None])
def test_host_to_device(shape, dtype, src_order, dst_order):
    a = numpy.asarray(testing.shaped_random(shape, numpy, dtype),
                      order=src_order)
    b = cupy.asarray(a, order=dst_order) if dst_order else cupy.asarray(a)
    # asarray defaults to order='K'
    expected = 'F' if (dst_order == 'F' or (
        dst_order in ('K', None) and a.flags.f_contiguous
        and not a.flags.c_contiguous)) else 'C'
    assert b.flags.f_contiguous if expected == 'F' else b.flags.c_contiguous
    testing.assert_array_equal(b, a)


@pytest.mark.usefixtures('pipelined')
@pytest.mark.parametrize('shape', _shapes)
@pytest.mark.parametrize('dtype', _dtypes)
@pytest.mark.parametrize('order', ['C', 'F'])
def test_device_to_host(shape, dtype, order):
    a = testing.shaped_random(shape, cupy, dtype, order=order)
    expected = a.get(blocking=False)
    cuda.Device().synchronize()
    testing.assert_array_equal(a.get(), expected)
    for o in ('C', 'F', 'A'):
        testing.assert_array_equal(a.get(order=o), expected)
    testing.assert_array_equal(cupy.asnumpy(a), expected)
    out = numpy.empty(shape, dtype, order='F')
    a.get(out=out)
    testing.assert_array_equal(out, expected)


@pytest.mark.usefixtures('pipelined')
def test_stream():
    a = testing.shaped_random((5000, 7), numpy, numpy.float64)
    stream = cuda.Stream()
    with stream:
        b = cupy.asarray(a, order='F')
        c = b.get()
    testing.assert_array_equal(c, a)


@pytest.mark.usefixtures('pipelined')
def test_source_reuse_after_asarray():
    # The staging copy must be complete when asarray returns, so the caller
    # may overwrite the source immediately.
    srcs = [testing.shaped_random((20000,), numpy, numpy.float32, seed=i)
            for i in range(4)]
    expected = [s.copy() for s in srcs]
    dsts = [cupy.asarray(s) for s in srcs]
    for s in srcs:
        s[...] = -1
    for d, e in zip(dsts, expected):
        testing.assert_array_equal(d, e)


def test_noncontiguous_input_unchanged_path():
    a = testing.shaped_random((3000, 2000), numpy, numpy.float32)[::2, ::3]
    testing.assert_array_equal(cupy.asarray(a), a)
