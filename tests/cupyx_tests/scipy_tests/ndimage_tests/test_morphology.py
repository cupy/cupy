import cupy
import numpy
import pytest

from cupy import testing
import cupyx.scipy.ndimage  # NOQA

try:
    import scipy.ndimage  # NOQA
except ImportError:
    pass


@testing.parameterize(
    {'rank': 0, 'connectivity': 1},
    {'rank': 1, 'connectivity': 1},
    {'rank': 2, 'connectivity': 1},
    {'rank': 2, 'connectivity': 2},
    {'rank': 3, 'connectivity': 1},
    {'rank': 3, 'connectivity': 2},
    {'rank': 3, 'connectivity': 3},
    # edge cases
    {'rank': -1, 'connectivity': 0},
    {'rank': 3, 'connectivity': 0},
    {'rank': 3, 'connectivity': 500})
@testing.with_requires('scipy')
class TestGenerateBinaryStructure:

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_generate_binary_structure(self, xp, scp):
        return scp.ndimage.generate_binary_structure(self.rank,
                                                     self.connectivity)


@testing.with_requires('scipy')
class TestIterateStructure:

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_iterate_structure1(self, xp, scp):
        struct = xp.asarray([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]])
        return scp.ndimage.iterate_structure(struct, 2)

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_iterate_structure2(self, xp, scp):
        struct = xp.asarray([[0, 1],
                             [1, 1],
                             [0, 1]])
        return scp.ndimage.iterate_structure(struct, 2)

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_iterate_structure3(self, xp, scp):
        struct = xp.asarray([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]])
        out, origin = scp.ndimage.iterate_structure(struct, 2, 1)
        assert origin == [2, 2]
        return out


@testing.parameterize(*(
    testing.product({  # cases with boolean input and output
        'x_dtype': [bool],
        'border_value': [0, 1],
        'structure': [None, [1, 0, 1], [1, 1, 0]],
        'origin': [-1, 0, 1],
        'data': [[1, 1, 0, 1, 1]],
        'filter': ['binary_erosion', 'binary_dilation'],
        'output': [None]}
    ) + testing.product({  # cases with empty data
        'x_dtype': [bool],
        'border_value': [0],
        'structure': [None],
        'origin': [0],
        'data': [[]],
        'filter': ['binary_erosion', 'binary_dilation'],
        'output': [None]}
    ) + testing.product({  # dtype and output combinations
        'x_dtype': [numpy.bool_, numpy.int8, numpy.uint8, numpy.float32,
                    numpy.float64],
        'border_value': [0],
        'structure': [[1, 1, 0]],
        'origin': [0],
        'data': [[1, 1, 0, 1, 1]],
        'filter': ['binary_erosion', 'binary_dilation'],
        'output': [None, numpy.float32, numpy.int8, 'array']}
    )
))
@testing.with_requires('scipy')
class TestBinaryErosionAndDilation1d:
    def _filter(self, xp, scp, x):
        filter = getattr(scp.ndimage, self.filter)
        structure = self.structure
        if structure is not None:
            structure = xp.asarray(structure)
        if self.output == 'array':
            output = xp.zeros_like(x)
            filter(x, structure, iterations=1, mask=None,
                   output=output, border_value=self.border_value,
                   origin=self.origin, brute_force=True)
            return output
        return filter(x, structure, iterations=1, mask=None,
                      output=self.output, border_value=self.border_value,
                      origin=self.origin, brute_force=True)

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_binary_erosion_and_dilation_1d(self, xp, scp):
        if self.x_dtype == self.output:
            pytest.skip('redundant')
        x = xp.asarray(self.data, dtype=self.x_dtype)
        return self._filter(xp, scp, x)


@testing.parameterize(*(
    testing.product({  # cases with boolean input and output
        'x_dtype': [bool],
        'border_value': [0, 1],
        'connectivity': [1, 2],
        'origin': [0, 1],
        'data': [[[0, 1, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 0],
                  [0, 0, 1, 1, 0, 1, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]],

                 [[1, 1, 1, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]],
                 ],
        'filter': ['binary_opening', 'binary_closing'],
        'output': [None]}
    ) + testing.product({  # dtype and output combinations
        'x_dtype': [numpy.bool_, numpy.float64],
        'border_value': [0],
        'connectivity': [1],
        'origin': [0],
        'data': [[[0, 1, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 0],
                  [0, 0, 1, 1, 0, 1, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]],
                 ],
        'filter': ['binary_opening', 'binary_closing'],
        'output': [None, numpy.float32, numpy.int8]}
    )
))
@testing.with_requires('scipy>=1.1.0')
class TestBinaryOpeningAndClosing:
    def _filter(self, xp, scp, x):
        filter = getattr(scp.ndimage, self.filter)
        structure = scp.ndimage.generate_binary_structure(x.ndim,
                                                          self.connectivity)
        return filter(x, structure, iterations=1, output=self.output,
                      origin=self.origin, mask=None,
                      border_value=self.border_value, brute_force=True)

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_binary_opening_and_closing(self, xp, scp):
        if self.x_dtype == self.output:
            pytest.skip('redundant')
        x = xp.asarray(self.data, dtype=self.x_dtype)
        return self._filter(xp, scp, x)


@testing.parameterize(*(
    testing.product({
        'border_value': [0, 1],
        'structure': [2, (3, 1), (1, 3), (3, 3)],
        'data': [[[0, 1, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 0],
                  [0, 0, 1, 1, 0, 1, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]],

                 [[1, 1, 1, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]],
                 ],
        'filter': ['binary_erosion', 'binary_dilation', 'binary_opening',
                   'binary_closing']}
    ))
)
@testing.with_requires('scipy>=1.1.0')
class TestBinaryMorphologyTupleFootprint:
    def _filter(self, x, structure):
        filter = getattr(cupyx.scipy.ndimage, self.filter)
        return filter(x, structure, iterations=1, output=None, origin=0,
                      mask=None, border_value=self.border_value,
                      brute_force=True)

    def test_binary_opening_and_closing(self):
        """Compare tuple-based footprint to a matching ndarray one.

        SciPy doesn't support specifying the footprint as a tuple (the overhead
        of small array allocation is much less on the CPU), so we compare
        CuPy with tuple to CuPy with an equivalent ndarray instead.
        """
        x = cupy.asarray(self.data, dtype=bool)
        if numpy.isscalar(self.structure):
            ndarray_structure = cupy.ones((self.structure, ) * x.ndim,
                                          dtype=bool)
        else:
            ndarray_structure = cupy.ones(self.structure, dtype=bool)
        tuple_result = self._filter(x, self.structure)
        ndarray_result = self._filter(x, ndarray_structure)
        testing.assert_array_equal(tuple_result, ndarray_result)


@testing.parameterize(*(
    testing.product({
        'x_dtype': [bool],
        'connectivity': [1, 2],
        'origin': [-1, 0, 1],
        'data': [[[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 1, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]],

                 [[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]],

                 [[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 1, 1],
                  [0, 1, 0, 1, 0, 1, 0, 1],
                  [0, 1, 0, 1, 0, 1, 0, 1],
                  [0, 0, 1, 0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0]],
                 ],
        'output': [None]}
    ) + testing.product({  # dtype and output combinations
        'x_dtype': [numpy.bool_, numpy.float64],
        'connectivity': [1],
        'origin': [0],
        'data': [[[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]],
                 ],
        'output': [None, numpy.float32, numpy.int8]}
    ))
)
@testing.with_requires('scipy')
class TestBinaryFillHoles:
    def _filter(self, xp, scp, x):
        filter = scp.ndimage.binary_fill_holes
        structure = scp.ndimage.generate_binary_structure(x.ndim,
                                                          self.connectivity)
        return filter(x, structure, output=self.output, origin=self.origin)

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_binary_fill_holes(self, xp, scp):
        if self.x_dtype == self.output:
            pytest.skip('redundant')
        x = xp.asarray(self.data, dtype=self.x_dtype)
        return self._filter(xp, scp, x)


@testing.parameterize(*(
    testing.product({  # cases with boolean input and output
        'x_dtype': [bool],
        'struct': ['same', 'separate'],
        'origins': [((0, 0), (0, 0)),
                    ((0, 1), (-1, 0))],
        'data': [[[0, 1, 0, 0, 0],
                  [1, 1, 1, 0, 0],
                  [0, 1, 0, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0]],

                 [[0, 1, 0, 0, 1, 1, 1, 0],
                  [1, 1, 1, 0, 0, 1, 0, 0],
                  [0, 1, 0, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]],

                 [[0, 1, 0, 0, 1, 1, 1, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 1, 1, 1, 1, 0],
                  [0, 0, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]],
                 ],
        'output': [None]}
    ) + testing.product({  # dtype and output combinations
        'x_dtype': [numpy.bool_, numpy.float64],
        'struct': ['same', 'separate'],
        'origins': [((0, 0), (0, 0))],
        'data': [[[0, 1, 0, 0, 1, 1, 1, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 1, 1, 1, 1, 0],
                  [0, 0, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]],
                 ],
        'output': [None, numpy.float32, numpy.int8]}
    )
))
@testing.with_requires('scipy')
class TestBinaryHitOrMiss:
    def _filter(self, xp, scp, x):
        filter = scp.ndimage.binary_hit_or_miss
        if self.struct == 'same':
            structure1 = scp.ndimage.generate_binary_structure(x.ndim, 1)
            structure2 = structure1
        elif self.struct == 'separate':
            structure1 = xp.asarray([[0, 0, 0],
                                     [1, 1, 1],
                                     [0, 0, 0]])

            structure2 = xp.asarray([[1, 1, 1],
                                     [0, 0, 0],
                                     [1, 1, 1]])
        origin1, origin2 = self.origins
        return filter(x, structure1, structure2, output=self.output,
                      origin1=origin1, origin2=origin2)

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_binary_hit_or_miss(self, xp, scp):
        if self.x_dtype == self.output:
            pytest.skip('redundant')
        x = xp.asarray(self.data, dtype=self.x_dtype)
        return self._filter(xp, scp, x)


@testing.parameterize(*(
    testing.product({  # cases with boolean input and output
        'x_dtype': [bool],
        'border_value': [0, 1],
        'connectivity': [1, 2],
        'origin': [0, 1],
        'mask': [[[0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 0, 0, 0],
                  [0, 1, 1, 0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]],

                 [[0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]],
                 ],
        'data': [[[0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]],

                 [[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]],
                 ],
        'output': [None]}
    ) + testing.product({  # dtype and output combinations
        'x_dtype': [numpy.bool_, numpy.float64],
        'border_value': [0],
        'connectivity': [1],
        'origin': [0],
        'mask': [[[0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]],
                 ],
        'data': [[[0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]],
                 ],
        'output': [None, numpy.float32, numpy.int8]}
    )
))
@testing.with_requires('scipy')
class TestBinaryPropagation:
    def _filter(self, xp, scp, x):
        filter = scp.ndimage.binary_propagation
        structure = scp.ndimage.generate_binary_structure(x.ndim,
                                                          self.connectivity)
        mask = xp.asarray(self.mask)
        return filter(x, structure, mask=mask, output=self.output,
                      border_value=self.border_value, origin=self.origin)

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_binary_propagation(self, xp, scp):
        if self.x_dtype == self.output:
            pytest.skip('redundant')
        x = xp.asarray(self.data, dtype=self.x_dtype)
        return self._filter(xp, scp, x)


@testing.parameterize(*(
    testing.product({  # cases with boolean input and output
        'x_dtype': [bool],
        'border_value': [0, 1],
        'connectivity': [1, 2],
        'origin': [0, -1],
        'shape': [(64,), (16, 15), (5, 7, 9)],
        'density': [0.1, 0.5, 0.9],
        'filter': ['binary_erosion', 'binary_dilation'],
        'iterations': [1],
        'output': [None]}
    ) + testing.product({
        'x_dtype': [numpy.int8, numpy.float32],
        'border_value': [0],
        'connectivity': [1],
        'origin': [0],
        'shape': [(18, 13)],
        'density': [0.2],
        'filter': ['binary_erosion', 'binary_dilation'],
        'iterations': [1, 2, 0],
        'output': [None, numpy.float32, 'array']}
    )
))
@testing.with_requires('scipy')
class TestBinaryErosionAndDilation:
    def _filter(self, xp, scp, x):
        filter = getattr(scp.ndimage, self.filter)
        ndim = len(self.shape)
        structure = scp.ndimage.generate_binary_structure(ndim,
                                                          self.connectivity)
        if self.output == 'array':
            output = xp.zeros_like(x)
            filter(x, structure, iterations=self.iterations, mask=None,
                   output=output, border_value=self.border_value,
                   origin=self.origin, brute_force=True)
            return output
        return filter(x, structure, iterations=self.iterations, mask=None,
                      output=self.output, border_value=self.border_value,
                      origin=self.origin, brute_force=True)

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_binary_erosion_and_dilation(self, xp, scp):
        if self.x_dtype == self.output:
            pytest.skip('redundant')
        rstate = numpy.random.RandomState(5)
        x = rstate.randn(*self.shape) > self.density
        x = xp.asarray(x, dtype=self.x_dtype)
        return self._filter(xp, scp, x)


@testing.parameterize(*(
    testing.product({  # cases with boolean input and output
        'x_dtype': [bool],
        'border_value': [0],
        'connectivity': [1],
        'origin': [(0, 1)],
        'shape': [(16, 15), (5, 7, 9)],
        'axes': [(0, 1), (0, -1), (-2, -1), (1,)],
        'density': [0.1, 0.5, 0.9],
        'filter': ['binary_erosion', 'binary_dilation', 'binary_opening',
                   'binary_closing', 'binary_hit_or_miss',
                   'binary_propagation', 'binary_fill_holes'],
        'output': [None]}
    )
))
@testing.with_requires('scipy>=1.15.0rc1')
class TestBinaryMorphologyAxes:
    def _filter(self, xp, scp, x):
        filter = getattr(scp.ndimage, self.filter)
        ndim = len(self.shape)
        kwargs = {}
        if self.axes is None:
            structure = scp.ndimage.generate_binary_structure(
                ndim, self.connectivity)
        else:
            structure = scp.ndimage.generate_binary_structure(
                len(self.axes), self.connectivity)
        if len(self.origin) > len(self.axes):
            self.origin = [self.origin[ax] for ax in self.axes]

        if self.filter not in ["binary_hit_or_miss", "binary_fill_holes"]:
            kwargs["border_value"] = self.border_value

        if self.filter == "binary_hit_or_miss":
            kwargs["origin1"] = self.origin
            kwargs["origin2"] = self.origin
        else:
            kwargs["origin"] = self.origin
        return filter(x, structure, axes=self.axes, **kwargs)

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_binary_morphology_axes(self, xp, scp):
        if self.x_dtype == self.output:
            pytest.skip('redundant')
        rstate = numpy.random.RandomState(5)
        x = rstate.randn(*self.shape) > self.density
        x = xp.asarray(x, dtype=self.x_dtype)
        return self._filter(xp, scp, x)


@testing.parameterize(*(
    testing.product({
        'x_dtype': [numpy.int8, numpy.float32],
        'shape': [(16, 24)],
        'filter': ['binary_erosion', 'binary_dilation'],
        'iterations': [1, 2],
        'contiguity': ['C', 'F', 'none']}
    ))
)
@testing.with_requires('scipy')
class TestBinaryErosionAndDilationContiguity:
    def _filter(self, xp, scp, x):
        filter = getattr(scp.ndimage, self.filter)
        ndim = len(self.shape)
        structure = scp.ndimage.generate_binary_structure(ndim, 1)
        return filter(x, structure, iterations=self.iterations, mask=None,
                      output=None, border_value=0, origin=0, brute_force=True)

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_binary_erosion_and_dilation_input_contiguity(self, xp, scp):
        rstate = numpy.random.RandomState(5)
        x = rstate.randn(*self.shape) > 0.3
        x = xp.asarray(x, dtype=self.x_dtype)
        if self.contiguity == 'F':
            x = xp.asfortranarray(x)
        elif self.contiguity == 'none':
            x = x[::2, ::3]
        return self._filter(xp, scp, x)


@testing.parameterize(*(
    testing.product({
        'shape': [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
        'size': [3, 4],
        'footprint': [None, 'random'],
        'structure': [None, 'random'],
        'mode': ['reflect'],
        'cval': [0.0],
        'origin': [0, 1, None],
        'x_dtype': [numpy.int8, numpy.int16, numpy.int32,
                    numpy.float32, numpy.float64],
        'output': [None, numpy.int32, numpy.float64],
        'filter': ['grey_erosion', 'grey_dilation']
    }) + testing.product({
        'shape': [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
        'size': [3, 4],
        'footprint': [None, 'random'],
        'structure': [None, 'random'],
        'mode': ['constant'],
        'cval': [-1.0, 0.0, 1.0],
        'origin': [0],
        'x_dtype': [numpy.int32, numpy.float64],
        'output': [None],
        'filter': ['grey_erosion', 'grey_dilation']
    }) + testing.product({
        'shape': [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
        'size': [3, 4],
        'footprint': [None, 'random'],
        'structure': [None, 'random'],
        'mode': ['nearest', 'mirror', 'wrap'],
        'cval': [0.0],
        'origin': [0],
        'x_dtype': [numpy.int32, numpy.float64],
        'output': [None],
        'filter': ['grey_erosion', 'grey_dilation']
    })
))
@testing.with_requires('scipy')
class TestGreyErosionAndDilation:

    def _filter(self, xp, scp, x):
        filter = getattr(scp.ndimage, self.filter)
        if self.origin is None:
            origin = (-1, 1, -1, 1)[:x.ndim]
        else:
            origin = self.origin
        if self.footprint is None:
            footprint = None
        else:
            shape = (self.size, ) * x.ndim
            r = testing.shaped_random(shape, xp, scale=1)
            footprint = xp.where(r < .5, 1, 0)
            if not footprint.any():
                footprint = xp.ones(shape)
        if self.structure is None:
            structure = None
        else:
            shape = (self.size, ) * x.ndim
            structure = testing.shaped_random(shape, xp, dtype=xp.int32)
        return filter(x, size=self.size, footprint=footprint,
                      structure=structure, output=self.output,
                      mode=self.mode, cval=self.cval, origin=origin)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_grey_erosion_and_dilation(self, xp, scp):
        if self.mode == 'mirror' and 1 in self.shape:
            pytest.skip('not testable against scipy')
        if self.x_dtype == self.output:
            pytest.skip('redundant')
        x = testing.shaped_random(self.shape, xp, self.x_dtype)
        return self._filter(xp, scp, x)


@testing.parameterize(*testing.product({
    'size': [3, 4],
    'structure': [None, 'random'],
    'mode': ['reflect', 'constant', 'nearest', 'mirror', 'wrap'],
    'origin': [0, None],
    'x_dtype': [numpy.int32, numpy.float32],
    'output': [None, numpy.float64],
    'filter': ['grey_closing', 'grey_opening']
}))
@testing.with_requires('scipy')
class TestGreyClosingAndOpening:

    shape = (4, 5)
    footprint = None
    cval = 0.0

    def _filter(self, xp, scp, x):
        filter = getattr(scp.ndimage, self.filter)
        if self.origin is None:
            origin = (-1, 1, -1, 1)[:x.ndim]
        else:
            origin = self.origin
        if self.structure is None:
            structure = None
        else:
            shape = (self.size, ) * x.ndim
            structure = testing.shaped_random(shape, xp, dtype=xp.int32)
        return filter(x, size=self.size, footprint=self.footprint,
                      structure=structure, output=self.output,
                      mode=self.mode, cval=self.cval, origin=origin)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_grey_closing_and_opening(self, xp, scp):
        x = testing.shaped_random(self.shape, xp, self.x_dtype)
        return self._filter(xp, scp, x)


@testing.parameterize(*(
    testing.product({
        'x_dtype': [numpy.int32],
        'origin': [-1, 0, 1],
        'filter': ['morphological_gradient', 'morphological_laplace'],
        'mode': ['reflect', 'constant'],
        'output': [None],
        'size': [(3, 3), (4, 3)],
        'footprint': [None, 'random'],
        'structure': [None, 'random']}
    ) + testing.product({
        'x_dtype': [numpy.int32, numpy.float64],
        'origin': [0],
        'filter': ['morphological_gradient', 'morphological_laplace'],
        'mode': ['reflect', 'constant', 'nearest', 'mirror', 'wrap'],
        'output': [None, numpy.float32, 'zeros'],
        'size': [3],
        'footprint': [None, 'random'],
        'structure': [None, 'random']}
    ))
)
@testing.with_requires('scipy')
class TestMorphologicalGradientAndLaplace:

    def _filter(self, xp, scp, x):
        filter = getattr(scp.ndimage, self.filter)
        if xp.isscalar(self.size):
            shape = (self.size,) * x.ndim
        else:
            shape = tuple(self.size)
        if self.footprint is None:
            footprint = None
        else:
            r = testing.shaped_random(shape, xp, scale=1)
            footprint = xp.where(r < .5, 1, 0)
            if not footprint.any():
                footprint = xp.ones(shape)
        if self.structure is None:
            structure = None
        else:
            structure = testing.shaped_random(shape, xp, dtype=xp.int32)
        if self.output == 'zeros':
            output = xp.zeros_like(x)
        else:
            output = self.output
        return filter(x, self.size, footprint, structure,
                      output=output, mode=self.mode, cval=0.0,
                      origin=self.origin)

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_morphological_gradient_and_laplace(self, xp, scp):
        x = xp.zeros((7, 7), dtype=self.x_dtype)
        x[2:5, 2:5] = 1
        x[4, 4] = 2
        x[2, 3] = 3
        if self.x_dtype == self.output:
            pytest.skip('redundant')
        return self._filter(xp, scp, x)


@testing.parameterize(*(
    testing.product({
        'x_dtype': [numpy.int32],
        'shape': [(5, 7)],
        'origin': [-1, 0, 1],
        'filter': ['white_tophat', 'black_tophat'],
        'mode': ['reflect', 'constant'],
        'output': [None],
        'size': [(3, 3), (4, 3)],
        'footprint': [None, 'random'],
        'structure': [None, 'random']}
    ) + testing.product({
        'x_dtype': [numpy.int32, numpy.float64],
        'shape': [(6, 8)],
        'origin': [0],
        'filter': ['white_tophat', 'black_tophat'],
        'mode': ['reflect', 'constant', 'nearest', 'mirror', 'wrap'],
        'output': [None, numpy.float32, 'zeros'],
        'size': [3],
        'footprint': [None, 'random'],
        'structure': [None, 'random']}
    ))
)
@testing.with_requires('scipy')
class TestWhiteTophatAndBlackTopHat:

    def _filter(self, xp, scp, x):
        filter = getattr(scp.ndimage, self.filter)
        if xp.isscalar(self.size):
            shape = (self.size,) * x.ndim
        else:
            shape = tuple(self.size)
        if self.footprint is None:
            footprint = None
        else:
            r = testing.shaped_random(shape, xp, scale=1)
            footprint = xp.where(r < .5, 1, 0)
            if not footprint.any():
                footprint = xp.ones(shape)
        if self.structure is None:
            structure = None
        else:
            structure = testing.shaped_random(shape, xp, dtype=xp.int32)
        if self.output == 'zeros':
            output = xp.zeros_like(x)
        else:
            output = self.output
        return filter(x, self.size, footprint, structure,
                      output=output, mode=self.mode, cval=0.0,
                      origin=self.origin)

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_white_tophat_and_black_tophat(self, xp, scp):
        x = testing.shaped_random(self.shape, xp, self.x_dtype)
        if self.x_dtype == self.output:
            pytest.skip('redundant')
        return self._filter(xp, scp, x)


@testing.parameterize(*(
    testing.product({
        'shape': [(8, 9), (3, 4, 7, 8)],
        'size': [3],
        'axes': [(0, 1), (0, -1), (-2, -1)],
        'footprint': [None, 'random'],
        'structure': [None, 'random'],
        'mode': ['reflect'],
        'cval': [0.0],
        'x_dtype': [numpy.int16, numpy.float32],
        'filter': ['grey_erosion', 'grey_dilation', 'grey_opening',
                   'grey_closing', 'morphological_laplace',
                   'morphological_gradient', 'white_tophat', 'black_tophat']
    })
))
@testing.with_requires('scipy>=1.15.0rc1')
class TestGreyMorphologyAxes:

    def _filter(self, xp, scp, x):
        filter = getattr(scp.ndimage, self.filter)
        num_axes = len(self.axes)
        origin = (-1, 1, -1, 1)[:num_axes]
        if self.footprint is None:
            footprint = None
        else:
            shape = (self.size, ) * num_axes
            r = testing.shaped_random(shape, xp, scale=1)
            footprint = xp.where(r < .5, 1, 0)
            if not footprint.any():
                footprint = xp.ones(shape)
        if self.structure is None:
            structure = None
        else:
            shape = (self.size, ) * num_axes
            structure = testing.shaped_random(shape, xp, dtype=xp.int32)
        return filter(x, size=self.size, footprint=footprint,
                      structure=structure, mode=self.mode, cval=self.cval,
                      origin=origin, axes=self.axes)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_grey_morphology_axes(self, xp, scp):
        x = testing.shaped_random(self.shape, xp, self.x_dtype)
        return self._filter(xp, scp, x)
