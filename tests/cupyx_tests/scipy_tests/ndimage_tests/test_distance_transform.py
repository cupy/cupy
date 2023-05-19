import copy

import numpy
import pytest

import cupy
from cupy import testing
import cupyx.scipy.ndimage  # NOQA

try:
    import scipy.ndimage  # NOQA
except ImportError:
    pass


@testing.with_requires('scipy')
class TestDistanceTransform:

    def _binary_image(self, shape, xp=cupy, pct_true=50):
        # cupy and numpy random streams don't match so generate with CuPy only
        rng = cupy.random.default_rng(123)
        x = rng.integers(0, 100, size=shape, dtype=xp.uint8)
        img = x >= pct_true
        if xp == numpy:
            img = cupy.asnumpy(img)
        return img

    def _assert_percentile_equal(self, arr1, arr2, pct=95):
        """Assert a target percentage of arr1 and arr2 are equal."""
        pct_mismatch = (100 - pct) / 100
        arr1 = cupy.asnumpy(arr1)
        arr2 = cupy.asnumpy(arr2)
        mismatch = numpy.sum(arr1 != arr2) / arr1.size
        assert mismatch < pct_mismatch

    @pytest.mark.parametrize('return_indices', [False, True])
    @pytest.mark.parametrize('return_distances', [False, True])
    @pytest.mark.parametrize(
        'shape, sampling',
        [
            ((256, 128), None),
            ((384, 256), (1.5, 1.5)),
            ((384, 256), (3, 2)),  # integer-valued anisotropic
            ((384, 256), (2.25, .85)),
            ((14, 32, 50), None),
            ((50, 32, 24), (2., 2., 2.)),
            ((50, 32, 24), (3, 1, 2)),  # integer-valued anisotropic
        ],
    )
    @pytest.mark.parametrize('density', ['single_point', 5, 50, 95])
    @pytest.mark.parametrize('block_params', [None, (1, 1, 1)])
    def test_distance_transform_edt_basic(
        self, shape, sampling, return_distances, return_indices, density,
        block_params
    ):
        # Note: Not using @numpy_cupy_allclose because indices array
        #       comparisons need a custom function to check.
        if not (return_indices or return_distances):
            return

        kwargs_scipy = dict(
            sampling=sampling,
            return_distances=return_distances,
            return_indices=return_indices,
        )
        kwargs_cucim = copy.copy(kwargs_scipy)
        kwargs_cucim['block_params'] = block_params
        if density == 'single_point':
            img = cupy.ones(shape, dtype=bool)
            img[tuple(s // 2 for s in shape)] = 0
        else:
            img = self._binary_image(shape, pct_true=density)

        dt_gpu = cupyx.scipy.ndimage.distance_transform_edt
        dt_cpu = scipy.ndimage.distance_transform_edt
        out = dt_gpu(img, **kwargs_cucim)
        expected = dt_cpu(cupy.asnumpy(img), **kwargs_scipy)
        if sampling is None:
            target_pct = 95
        else:
            target_pct = 90
        if return_indices and return_distances:
            assert len(out) == 2
            cupy.testing.assert_allclose(out[0], expected[0], rtol=1e-6)
            # May differ at a small % of coordinates where multiple points were
            # equidistant.
            self._assert_percentile_equal(out[1], expected[1], pct=target_pct)
        elif return_distances:
            cupy.testing.assert_allclose(out, expected, rtol=1e-6)
        elif return_indices:
            self._assert_percentile_equal(out, expected, pct=target_pct)

    @pytest.mark.parametrize(
        # Fine sampling of shapes to make sure kernels are robust to all shapes
        'shape',
        (
            [(s,) * 2 for s in range(512, 512 + 32)]
            + [(s,) * 2 for s in range(1024, 1024 + 16)]
            + [(s,) * 2 for s in range(2050, 2050)]
            + [(s,) * 2 for s in range(4100, 4100)]
        ),
    )
    @pytest.mark.parametrize('density', [2, 98])
    @pytest.mark.parametrize('float64_distances', [False, True])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_distance_transform_edt_additional_shapes(
        self, xp, scp, shape, density, float64_distances
    ):
        kwargs = dict(return_distances=True, return_indices=False)
        if xp == cupy:
            # test CuPy-specific behavior allowing float32 distances
            kwargs['float64_distances'] = float64_distances
        img = self._binary_image(shape, xp=xp, pct_true=density)
        distances = scp.ndimage.distance_transform_edt(img, **kwargs)
        if not float64_distances and xp == cupy:
            assert distances.dtype == cupy.float32
            # cast to same dtype as numpy for numpy_cupy_allclose comparison
            distances = distances.astype(cupy.float64)
        return distances

    @pytest.mark.parametrize(
        'shape', [(s,) * 2 for s in range(1024, 1024 + 4)],
    )
    @pytest.mark.parametrize(
        'block_params',
        [(1, 1, 1), (5, 4, 2), (3, 8, 4), (7, 16, 1), (11, 32, 3), (1, 1, 16)]
    )
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_distance_transform_edt_block_params(self, xp, scp, shape,
                                                 block_params):

        kwargs = dict(return_distances=True, return_indices=False)
        if xp == cupy:
            # testing different block size parameters for the raw kernels
            kwargs['block_params'] = block_params
        img = self._binary_image(shape, xp=xp, pct_true=4)
        return scp.ndimage.distance_transform_edt(img, **kwargs)

    @pytest.mark.parametrize(
        'block_params', [
            (0, 1, 1), (1, 0, 1), (1, 1, 0),  # no elements can be < 1
            (1, 3, 1), (1, 5, 1), (1, 7, 1),  # 2nd element must be power of 2
            (128, 1, 1),  # m1 too large for the array size
            (1, 128, 1),  # m2 too large for the array size
            (1, 1, 128),  # m3 too large for the array size
        ]
    )
    def test_distance_transform_edt_block_params_invalid(self, block_params):
        img = self._binary_image((512, 512), xp=cupy, pct_true=4)
        with pytest.raises(ValueError):
            cupyx.scipy.ndimage.distance_transform_edt(
                img, block_params=block_params
            )

    @pytest.mark.parametrize('value', [0, 1, 3])
    @pytest.mark.parametrize('ndim', [2, 3])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_distance_transform_edt_uniform_valued(self, xp, scp, value, ndim):
        """ensure default block_params is robust to anisotropic shape."""
        img = xp.full((48, ) * ndim, value, dtype=cupy.uint8)
        # ensure there is at least 1 pixel at background intensity
        img[(slice(24, 25),) * ndim] = 0
        return scp.ndimage.distance_transform_edt(img)

    @pytest.mark.parametrize('sx', list(range(16)))
    @pytest.mark.parametrize('sy', list(range(16)))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_distance_transform_edt_2d_aniso(self, xp, scp, sx, sy):
        """ensure default block_params is robust to anisotropic shape."""
        shape = (128 + sy, 128 + sx)
        img = self._binary_image(shape, xp=xp, pct_true=80)
        return scp.ndimage.distance_transform_edt(img)

    @pytest.mark.parametrize('sx', list(range(4)))
    @pytest.mark.parametrize('sy', list(range(4)))
    @pytest.mark.parametrize('sz', list(range(4)))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_distance_transform_edt_3d_aniso(self, xp, scp, sx, sy, sz):
        """ensure default block_params is robust to anisotropic shape."""
        shape = (16 + sz, 32 + sy, 48 + sx)
        img = self._binary_image(shape, xp=xp, pct_true=80)
        return scp.ndimage.distance_transform_edt(img)

    @pytest.mark.parametrize('ndim', [2, 3])
    @pytest.mark.parametrize('sampling', [None, 'iso', 'aniso'])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_distance_transform_inplace_distance(self, xp, scp, ndim,
                                                 sampling):
        img = self._binary_image((32, ) * ndim, xp=xp, pct_true=80)
        distances = xp.empty(img.shape, dtype=xp.float64)
        if sampling == 'iso':
            sampling = (1.5,) * ndim
        elif sampling == 'aniso':
            sampling = tuple(range(1, ndim + 1))
        scp.ndimage.distance_transform_edt(img, sampling=sampling,
                                           distances=distances)
        return distances

    @pytest.mark.parametrize('ndim', [2, 3])
    def test_distance_transform_inplace_distance_errors(self, ndim):
        img = self._binary_image((32, ) * ndim, xp=cupy, pct_true=80)

        dt_func = cupyx.scipy.ndimage.distance_transform_edt

        # for binary input, distances output is float32. Other dtypes raise
        with pytest.raises(RuntimeError):
            distances = cupy.empty(img.shape, dtype=cupy.float64)
            dt_func(img, distances=distances, float64_distances=False)
        with pytest.raises(RuntimeError):
            distances = cupy.empty(img.shape, dtype=cupy.int32)
            dt_func(img, distances=distances)

        # wrong shape
        with pytest.raises(RuntimeError):
            distances = cupy.empty(img.shape + (2,), dtype=cupy.float64)
            dt_func(img, distances=distances)

        # can't provide indices array when return_indices is False
        with pytest.raises(RuntimeError):
            distances = cupy.empty(img.shape, dtype=cupy.float64)
            dt_func(img, distances=distances, return_distances=False,
                    return_indices=True)

    @pytest.mark.parametrize('ndim', [2, 3])
    @pytest.mark.parametrize('sampling', [None, 'iso', 'aniso'])
    @pytest.mark.parametrize('dtype', [cupy.int16, cupy.uint16, cupy.uint32,
                                       cupy.int32, cupy.uint64, cupy.int64])
    @pytest.mark.parametrize('return_distances', [False, True])
    def test_distance_transform_inplace_indices(
        self, ndim, sampling, dtype, return_distances
    ):
        img = self._binary_image((32, ) * ndim, xp=cupy, pct_true=80)
        if ndim == 3 and dtype in [cupy.int16, cupy.uint16]:
            pytest.skip(reason="3D requires at least 32-bit integer output")
        if sampling == 'iso':
            sampling = (1.5,) * ndim
        elif sampling == 'aniso':
            sampling = tuple(range(1, ndim + 1))
        common_kwargs = dict(
            sampling=sampling, return_distances=return_distances,
            return_indices=True
        )
        # verify that in-place and out-of-place results agree
        indices = cupy.empty((ndim,) + img.shape, dtype=dtype)
        dt_func = cupyx.scipy.ndimage.distance_transform_edt
        dt_func(img, indices=indices, **common_kwargs)
        expected = dt_func(img, **common_kwargs)
        if return_distances:
            cupy.testing.assert_array_equal(indices, expected[1])
        else:
            cupy.testing.assert_array_equal(indices, expected)

    @pytest.mark.parametrize('ndim', [2, 3])
    def test_distance_transform_inplace_indices_errors(self, ndim):
        img = self._binary_image((32, ) * ndim, pct_true=80)
        common_kwargs = dict(return_distances=False, return_indices=True)

        dt_func = cupyx.scipy.ndimage.distance_transform_edt

        # int8 has itemsize too small
        with pytest.raises(RuntimeError):
            indices = cupy.empty((ndim,) + img.shape, dtype=cupy.int8)
            dt_func(img, indices=indices, **common_kwargs)

        # float not allowed
        with pytest.raises(RuntimeError):
            indices = cupy.empty((ndim,) + img.shape, dtype=cupy.float64)
            dt_func(img, indices=indices, **common_kwargs)

        # wrong shape
        with pytest.raises(RuntimeError):
            indices = cupy.empty((ndim,), dtype=cupy.float32)
            dt_func(img, indices=indices, **common_kwargs)

        # can't provide indices array when return_indices is False
        with pytest.raises(RuntimeError):
            indices = cupy.empty((ndim,) + img.shape, dtype=cupy.int32)
            dt_func(img, indices=indices, return_indices=False)

    @pytest.mark.parametrize('ndim', [1, 4, 5])
    def test_distance_transform_edt_unsupported_ndim(self, ndim):
        with pytest.raises(NotImplementedError):
            cupyx.scipy.ndimage.distance_transform_edt(cupy.zeros((8,) * ndim))

    @pytest.mark.skip(reason="excessive memory requirement (and CPU runtime)")
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_distance_transform_edt_3d_int64(self, xp, scp):
        # Test 3D with shape > 2**10 to test the int64 kernels
        # This takes minutes to run on SciPy, so will need to skip on CI
        shape = (1040, 1040, 1040)
        img = self._binary_image(shape, xp=xp, pct_true=80)
        return scp.ndimage.distance_transform_edt(img)
