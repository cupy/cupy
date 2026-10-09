from __future__ import annotations

import numpy as np
import pytest

import cupy
from cupy import testing
from cupyx.scipy import interpolate, ndimage, signal, sparse, spatial


@pytest.fixture(autouse=True)
def release_memory():
    cupy.get_default_memory_pool().free_all_blocks()
    yield
    cupy.get_default_memory_pool().free_all_blocks()


def _require_memory(nbytes):
    free, _ = cupy.cuda.runtime.memGetInfo()
    if free < nbytes:
        pytest.skip('not enough GPU memory for the large-array regression')


@testing.slow
@pytest.mark.thread_unsafe(reason='Large GPU allocations.')
class TestLargeKernels:
    @pytest.mark.parametrize('kind',
                             ['BSpline', 'NdBSpline', 'PPoly', 'BPoly'])
    def test_spline_large_output(self, kind):
        _require_memory(24 << 30)
        # All individual dimensions fit int32, but the output product does not.
        nvalues = 65536
        x = cupy.full(32769, 0.25)
        c = cupy.arange(nvalues, dtype=cupy.float64)
        t = cupy.array([0., 0., 1., 1.])
        if kind == 'BSpline':
            spline = interpolate.BSpline(t, cupy.stack([c, c]), 1)
        elif kind == 'NdBSpline':
            spline = interpolate.NdBSpline((t,),
                                           cupy.stack([c, c]), 1)
            x = x[:, None]
        else:
            spline = getattr(interpolate, kind)(c[None, None, :], t[1:3])
        out = spline(x)
        rows = cupy.array([0, 32767, 32768])
        cols = cupy.array([0, 123, nvalues - 1])
        expected = np.broadcast_to([0, 123, nvalues - 1], (3, 3))
        testing.assert_array_equal(out[rows[:, None], cols], expected)

    @pytest.mark.parametrize('kind', ['BSpline', 'NdBSpline'])
    def test_spline_large_design_matrix(self, kind):
        _require_memory(90 << 30)
        n = 2**29 + 1
        x = cupy.full(n, 0.5)
        t = cupy.array([0., 0., 0., 0., 1., 1., 1., 1.])
        if kind == 'BSpline':
            out = interpolate.BSpline.design_matrix(x, t, 3, extrapolate=True)
        else:
            out = interpolate.NdBSpline.design_matrix(x[:, None], (t,), 3)
        assert out.nnz > 2**31
        testing.assert_array_equal(out.indptr[-2:], [4 * (n - 1), 4 * n])
        testing.assert_array_equal(out.indices[-4:], [0, 1, 2, 3])
        testing.assert_allclose(out.data[-4:], [0.125, 0.375, 0.375, 0.125])

    @pytest.mark.parametrize('kind', ['LinearNDInterpolator',
                                      'CloughTocher2DInterpolator'])
    def test_interpnd_large_output(self, kind):
        _require_memory(24 << 30)
        points = cupy.array([[0., 0.], [1., 0.], [0., 1.]])
        values = cupy.tile(cupy.arange(65536, dtype=cupy.float64), (3, 1))
        spline = getattr(interpolate, kind)(points, values)
        out = spline(cupy.full((32769, 2), 0.25))
        assert out.size > 2**31
        testing.assert_allclose(out[[0, -1], -1], [65535, 65535])

    def test_ppoly_large_coefficient_offsets(self):
        _require_memory(40 << 30)
        n = 2**30 + 1
        c = cupy.zeros((3, n, 1))
        c[1] = 2
        c[2] = 3
        x = cupy.arange(n + 1, dtype=cupy.float64)
        spline = interpolate.PPoly.construct_fast(c, x)
        testing.assert_allclose(spline(cupy.array([n - 0.5])), [[4]])
        testing.assert_allclose(spline.integrate(n - 1, n), [4])

    @pytest.mark.parametrize('shape, axis', [
        ((2**31 + 17,), 0),
        ((1, 2**31 + 17), 1),
        ((2**31 + 17, 1), 0),
        ((1, 2**31 + 17, 1), 1),
    ])
    def test_upfirdn_large_input(self, shape, axis):
        _require_memory(24 << 30)
        x = cupy.ones(shape, dtype=cupy.float32)
        last = [slice(None)] * x.ndim
        last[axis] = -1
        x[tuple(last)] = 2
        out = signal.upfirdn([1], x, down=shape[axis] - 1, axis=axis)
        testing.assert_array_equal(out.ravel(), [1, 2])

    def test_upfirdn_large_output(self):
        _require_memory(16 << 30)
        x = cupy.ones((65537, 2), dtype=cupy.float32)
        out = signal.upfirdn(cupy.ones(1, dtype=cupy.float32), x,
                             up=32769, axis=1)
        assert out.size > 2**31
        testing.assert_array_equal(out[[0, -1], 0], [1, 1])
        testing.assert_array_equal(out[[0, -1], -1], [1, 1])
        testing.assert_array_equal(out[[0, -1], -2], [0, 0])

    def test_peak_prominences_large_input(self):
        _require_memory(4 << 30)
        x = cupy.zeros(2**31 + 5, dtype=cupy.int8)
        peak = 2**31 + 2
        x[peak] = 2
        peaks = cupy.array([peak], dtype=cupy.int64)
        prominence, left, right = signal.peak_prominences(x, peaks, wlen=3)
        testing.assert_array_equal(prominence, [2])
        testing.assert_array_equal(left, [peak - 1])
        testing.assert_array_equal(right, [peak + 1])
        widths, heights, left_ip, right_ip = signal.peak_widths(
            x, peaks, prominence_data=(prominence, left, right))
        testing.assert_array_equal(widths, [1])
        testing.assert_array_equal(heights, [1])
        testing.assert_array_equal(left_ip, [peak - 0.5])
        testing.assert_array_equal(right_ip, [peak + 0.5])

    def test_argrelmax_large_flat_offsets(self):
        _require_memory(8 << 30)
        x = cupy.zeros((65537, 32769), dtype=cupy.int8)
        x[-1, -2] = 1
        rows, cols = signal.argrelmax(x, axis=1)
        testing.assert_array_equal(rows, [65536])
        testing.assert_array_equal(cols, [32767])

    @pytest.mark.parametrize('periodic', [False, True])
    def test_kdtree_large_query_output(self, periodic):
        _require_memory(64 << 30)
        tree = spatial.KDTree(cupy.array([[0.], [0.5]]),
                              boxsize=1 if periodic else None)
        distances, indices = tree.query(cupy.zeros((32769, 1)), k=65537)
        assert distances.size > 2**31
        testing.assert_array_equal(distances[[0, -1], 0], [0, 0])
        testing.assert_array_equal(indices[[0, -1], 0], [0, 0])
        testing.assert_array_equal(distances[[0, -1], -1], [np.inf, np.inf])
        testing.assert_array_equal(indices[[0, -1], -1], [2, 2])

    def test_sparse_large_major_axis(self):
        _require_memory(64 << 30)
        n = 2**31 + 1
        indptr = cupy.zeros(n + 1, dtype=cupy.int64)
        indptr[-1] = 1
        a = sparse.csr_matrix((cupy.array([7.]),
                               cupy.zeros(1, dtype=cupy.int64), indptr),
                              shape=(n, 2))
        out = a.max(axis=1).tocoo()
        assert out.shape == (n, 1)
        testing.assert_array_equal(out.data, [7])
        testing.assert_array_equal(out.row, [n - 1])

    @pytest.mark.parametrize('sos', [False, True])
    def test_iir_large_flat_offsets(self, sos):
        _require_memory(24 << 30)
        x = cupy.ones((2**21 + 1, 1024), dtype=cupy.float32)
        if sos:
            coeff = cupy.array([[1., 0., 0., 1., -0.5, 0.]],
                               dtype=cupy.float32)
            out = signal.sosfilt(coeff, x)
        else:
            out = signal.lfilter(
                cupy.ones(1, dtype=cupy.float32),
                cupy.array([1., -0.5], dtype=cupy.float32), x)
        testing.assert_allclose(out[[0, -1], :3],
                                [[1, 1.5, 1.75], [1, 1.5, 1.75]])
        testing.assert_allclose(out[[0, -1], -1], [2, 2])

    def test_symiirorder2_large_axis(self):
        _require_memory(64 << 30)
        x = cupy.ones(2**31 + 1, dtype=cupy.float32)
        out = signal.symiirorder2(x, 0.1, 0.0)
        testing.assert_allclose(out[[0, -1]], [1, 1], atol=1e-5)

    @pytest.mark.parametrize('ndim', [2, 3])
    @pytest.mark.parametrize('sampling', [None, 2])
    def test_distance_transform_large_flat_offsets(self, ndim, sampling):
        _require_memory(80 << 30)
        side = 46341 if ndim == 2 else 1291
        x = cupy.ones((side,) * ndim, dtype=cupy.uint8)
        x[..., 0] = 0
        spacing = None if sampling is None else (sampling,) * ndim
        out = ndimage.distance_transform_edt(
            x, sampling=spacing, float64_distances=False)
        pos = (side - 1,) * (ndim - 1)
        cols = cupy.array([0, side // 2, side - 1])
        testing.assert_allclose(out[pos][cols],
                                cupy.asnumpy(cols) * (sampling or 1))


def test_argrelmax_many_rows():
    # Cross gridDim.y's limit with a modest allocation.
    x = cupy.zeros((2**20 + 1, 3), dtype=cupy.int8)
    x[-1, 1] = 1
    rows, cols = signal.argrelmax(x, axis=1)
    testing.assert_array_equal(rows, [2**20])
    testing.assert_array_equal(cols, [1])


@pytest.mark.parametrize('nvalues', [1, 129])
@testing.with_requires('scipy')
def test_clough_tocher_partial_block(nvalues):
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.3, 0.4]])
    values = points[:, 0, None] + 2 * points[:, 1, None] + np.arange(nvalues)
    queries = np.array([[0.2, 0.3], [0.7, 0.8]])
    spline = interpolate.CloughTocher2DInterpolator(
        cupy.asarray(points), cupy.asarray(values))
    expected = (queries[:, 0, None] + 2 * queries[:, 1, None] +
                np.arange(nvalues))
    testing.assert_allclose(spline(cupy.asarray(queries)), expected, atol=1e-6)


def test_clough_tocher_grid_stride_fill_value():
    points = cupy.array([[0, 0], [1, 0], [0, 1]], dtype=cupy.float64)
    spline = interpolate.CloughTocher2DInterpolator(points, cupy.ones(3))
    queries = cupy.full((65537, 2), 0.25)
    queries[0] = 2  # Same thread later handles the last, valid query.
    out = spline(queries)
    assert cupy.isnan(out[0])
    testing.assert_allclose(out[[1, -1]], [1, 1])


def test_delaunay_coordinate_limit():
    points = cupy.broadcast_to(cupy.zeros((1, 2)), (2**26, 2))
    with pytest.raises(ValueError, match='Too many points'):
        spatial.Delaunay(points)


@pytest.mark.parametrize('index_dtype', [cupy.int32, cupy.int64])
def test_sparse_reduction_multiple_grid_rounds(index_dtype):
    # More major entries than blocks: exercise both sparse-index variants.
    n = 65537
    data = cupy.array([2., -3.])
    indices = cupy.zeros(2, dtype=index_dtype)
    indptr = cupy.zeros(n + 1, dtype=index_dtype)
    indptr[1:-1] = 1
    indptr[-1] = 2
    a = sparse.csr_matrix((data, indices, indptr), shape=(n, 2))
    testing.assert_array_equal(a.max(axis=1).toarray()[[0, -1], 0], [2, 0])
    testing.assert_array_equal(a.min(axis=1).toarray()[[0, -1], 0], [0, -3])
    testing.assert_array_equal(a.argmax(axis=1)[[0, -1], 0], [0, 1])
    testing.assert_array_equal(a.argmin(axis=1)[[0, -1], 0], [1, 0])
