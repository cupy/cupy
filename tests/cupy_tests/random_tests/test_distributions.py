import unittest

import cupy
from cupy.random import distributions
from cupy import testing

import numpy


_signed_dtypes = tuple(numpy.dtype(i).type for i in 'bhilq')
_unsigned_dtypes = tuple(numpy.dtype(i).type for i in 'BHILQ')
_int_dtypes = _signed_dtypes + _unsigned_dtypes


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (3, 2)],
    'n_shape': [(), (3, 2)],
    'p_shape': [(), (3, 2)],
    'dtype': _int_dtypes,  # to escape timeout
})
)
@testing.gpu
class TestDistributionsBinomial(unittest.TestCase):

    def check_distribution(self, dist_func, n_dtype, p_dtype, dtype):
        n = 5 * cupy.ones(self.n_shape, dtype=n_dtype)
        p = 0.5 * cupy.ones(self.p_shape, dtype=p_dtype)
        out = dist_func(n, p, self.shape, dtype)
        self.assertEqual(self.shape, out.shape)
        self.assertEqual(out.dtype, dtype)

    @cupy.testing.for_int_dtypes('n_dtype')
    @cupy.testing.for_float_dtypes('p_dtype')
    def test_binomial(self, n_dtype, p_dtype):
        self.check_distribution(distributions.binomial,
                                n_dtype, p_dtype, self.dtype)


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (3, 2)],
    'df_shape': [(), (3, 2)],
})
)
@testing.gpu
class TestDistributionsChisquare(unittest.TestCase):

    def check_distribution(self, dist_func, df_dtype, dtype):
        df = 5 * cupy.ones(self.df_shape, dtype=df_dtype)
        out = dist_func(df, self.shape, dtype)
        self.assertEqual(self.shape, out.shape)
        self.assertEqual(out.dtype, dtype)

    @cupy.testing.for_float_dtypes('df_dtype')
    @cupy.testing.for_float_dtypes('dtype')
    def test_chisquare(self, df_dtype, dtype):
        self.check_distribution(distributions.chisquare, df_dtype, dtype)


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (3, 2)],
    'loc_shape': [(), (3, 2)],
    'scale_shape': [(), (3, 2)],
})
)
@testing.gpu
class TestDistributionsGumbel(unittest.TestCase):

    def check_distribution(self, dist_func, loc_dtype, scale_dtype, dtype):
        loc = cupy.ones(self.loc_shape, dtype=loc_dtype)
        scale = cupy.ones(self.scale_shape, dtype=scale_dtype)
        out = dist_func(loc, scale, self.shape, dtype)
        self.assertEqual(self.shape, out.shape)
        self.assertEqual(out.dtype, dtype)

    @cupy.testing.for_float_dtypes('dtype', no_float16=True)
    @cupy.testing.for_float_dtypes('loc_dtype')
    @cupy.testing.for_float_dtypes('scale_dtype')
    def test_gumbel(self, loc_dtype, scale_dtype, dtype):
        self.check_distribution(distributions.gumbel,
                                loc_dtype, scale_dtype, dtype)


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (3, 2)],
    'loc_shape': [(), (3, 2)],
    'scale_shape': [(), (3, 2)],
})
)
@testing.gpu
class TestDistributionsLaplace(unittest.TestCase):

    def check_distribution(self, dist_func, loc_dtype, scale_dtype, dtype):
        loc = cupy.ones(self.loc_shape, dtype=loc_dtype)
        scale = cupy.ones(self.scale_shape, dtype=scale_dtype)
        out = dist_func(loc, scale, self.shape, dtype)
        self.assertEqual(self.shape, out.shape)
        self.assertEqual(out.dtype, dtype)

    @cupy.testing.for_float_dtypes('dtype', no_float16=True)
    @cupy.testing.for_float_dtypes('loc_dtype')
    @cupy.testing.for_float_dtypes('scale_dtype')
    def test_laplace(self, loc_dtype, scale_dtype, dtype):
        self.check_distribution(distributions.laplace,
                                loc_dtype, scale_dtype, dtype)


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (3, 2)],
    'mean_shape': [()],
    'sigma_shape': [()],
})
)
@testing.gpu
class TestDistributionsLognormal(unittest.TestCase):

    def check_distribution(self, dist_func, mean_dtype, sigma_dtype, dtype):
        mean = cupy.ones(self.mean_shape, dtype=mean_dtype)
        sigma = cupy.ones(self.sigma_shape, dtype=sigma_dtype)
        out = dist_func(mean, sigma, self.shape, dtype)
        self.assertEqual(self.shape, out.shape)
        self.assertEqual(out.dtype, dtype)

    @cupy.testing.for_float_dtypes('dtype', no_float16=True)
    @cupy.testing.for_float_dtypes('mean_dtype')
    @cupy.testing.for_float_dtypes('sigma_dtype')
    def test_lognormal(self, mean_dtype, sigma_dtype, dtype):
        self.check_distribution(distributions.lognormal,
                                mean_dtype, sigma_dtype, dtype)


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (3, 2)],
    'loc_shape': [(), (3, 2)],
    'scale_shape': [(), (3, 2)],
})
)
@testing.gpu
class TestDistributionsNormal(unittest.TestCase):

    def check_distribution(self, dist_func, loc_dtype, scale_dtype, dtype):
        loc = cupy.ones(self.loc_shape, dtype=loc_dtype)
        scale = cupy.ones(self.scale_shape, dtype=scale_dtype)
        out = dist_func(loc, scale, self.shape, dtype)
        self.assertEqual(self.shape, out.shape)
        self.assertEqual(out.dtype, dtype)

    @cupy.testing.for_float_dtypes('dtype', no_float16=True)
    @cupy.testing.for_float_dtypes('loc_dtype')
    @cupy.testing.for_float_dtypes('scale_dtype')
    def test_normal(self, loc_dtype, scale_dtype, dtype):
        self.check_distribution(distributions.normal,
                                loc_dtype, scale_dtype, dtype)


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (3, 2)],
})
)
@testing.gpu
class TestDistributionsStandardNormal(unittest.TestCase):

    def check_distribution(self, dist_func, dtype):
        out = dist_func(self.shape, dtype)
        self.assertEqual(self.shape, out.shape)
        self.assertEqual(out.dtype, dtype)

    @cupy.testing.for_float_dtypes('dtype', no_float16=True)
    def test_standardnormal(self, dtype):
        self.check_distribution(distributions.standard_normal, dtype)


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (3, 2)],
    'low_shape': [(), (3, 2)],
    'high_shape': [(), (3, 2)],
})
)
@testing.gpu
class TestDistributionsUniform(unittest.TestCase):

    def check_distribution(self, dist_func, low_dtype, high_dtype, dtype):
        low = cupy.ones(self.low_shape, dtype=low_dtype)
        high = cupy.ones(self.high_shape, dtype=high_dtype) * 2.
        out = dist_func(low, high, self.shape, dtype)
        self.assertEqual(self.shape, out.shape)
        self.assertEqual(out.dtype, dtype)

    @cupy.testing.for_float_dtypes('dtype', no_float16=True)
    @cupy.testing.for_float_dtypes('low_dtype')
    @cupy.testing.for_float_dtypes('high_dtype')
    def test_uniform(self, low_dtype, high_dtype, dtype):
        self.check_distribution(distributions.uniform,
                                low_dtype, high_dtype, dtype)
