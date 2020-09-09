import unittest

import numpy

from cupy import testing
import cupyx.scipy.ndimage  # NOQA

try:
    import scipy.ndimage  # NOQA
except ImportError:
    pass


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
@testing.gpu
@testing.with_requires('scipy')
class TestGreyErosionAndDilation(unittest.TestCase):

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
            raise unittest.SkipTest('not testable against scipy')
        if self.x_dtype == self.output:
            raise unittest.SkipTest('redundant')
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
@testing.gpu
@testing.with_requires('scipy')
class TestGreyClosingAndOpening(unittest.TestCase):

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
@testing.gpu
@testing.with_requires('scipy')
class MorphologicalGradientAndLaplace(unittest.TestCase):

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
            raise unittest.SkipTest('redundant')
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
@testing.gpu
@testing.with_requires('scipy')
class WhiteTophatAndBlackTopHat(unittest.TestCase):

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
            raise unittest.SkipTest('redundant')
        return self._filter(xp, scp, x)
