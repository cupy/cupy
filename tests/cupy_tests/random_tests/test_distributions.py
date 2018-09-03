import unittest

import cupy
from cupy.random import distributions
from cupy import testing

import numpy


_regular_float_dtypes = (numpy.float64, numpy.float32)
_float_dtypes = _regular_float_dtypes + (numpy.float16,)
_signed_dtypes = tuple(numpy.dtype(i).type for i in 'bhilq')
_unsigned_dtypes = tuple(numpy.dtype(i).type for i in 'BHILQ')
_int_dtypes = _signed_dtypes + _unsigned_dtypes


class RandomDistributionsTestCase(unittest.TestCase):
    def check_distribution(self, dist_name, params, dtype):
        cp_params = {k: cupy.asarray(params[k]) for k in params}
        np_out = getattr(numpy.random, dist_name)(
            size=self.shape, **params).astype(dtype)
        cp_out = getattr(distributions, dist_name)(
            size=self.shape, dtype=dtype, **cp_params)
        self.assertEqual(cp_out.shape, np_out.shape)
        self.assertEqual(cp_out.dtype, np_out.dtype)


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (3, 2)],
    'a_shape': [(), (3, 2)],
    'b_shape': [(), (3, 2)],
    'dtype': _float_dtypes,  # to escape timeout
})
)
@testing.gpu
class TestDistributionsBeta(RandomDistributionsTestCase):

    @cupy.testing.for_float_dtypes('a_dtype')
    @cupy.testing.for_float_dtypes('b_dtype')
    def test_beta(self, a_dtype, b_dtype):
        a = 3. * numpy.ones(self.a_shape, dtype=a_dtype)
        b = 3. * numpy.ones(self.b_shape, dtype=b_dtype)
        self.check_distribution('beta',
                                {'a': a, 'b': b}, self.dtype)


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (3, 2)],
    'n_shape': [(), (3, 2)],
    'p_shape': [(), (3, 2)],
    'dtype': _int_dtypes,  # to escape timeout
})
)
@testing.gpu
class TestDistributionsBinomial(RandomDistributionsTestCase):

    @cupy.testing.for_signed_dtypes('n_dtype')
    @cupy.testing.for_float_dtypes('p_dtype')
    def test_binomial(self, n_dtype, p_dtype):
        n = 5 * numpy.ones(self.n_shape, dtype=n_dtype)
        p = 0.5 * numpy.ones(self.p_shape, dtype=p_dtype)
        self.check_distribution('binomial',
                                {'n': n, 'p': p}, self.dtype)


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2, 3), (3, 2, 3)],
    'alpha_shape': [(3,)],
})
)
@testing.gpu
class TestDistributionsDirichlet(RandomDistributionsTestCase):

    @cupy.testing.for_float_dtypes('alpha_dtype')
    @cupy.testing.for_float_dtypes('dtype')
    def test_dirichlet(self, alpha_dtype, dtype):
        alpha = numpy.ones(self.alpha_shape, dtype=alpha_dtype)
        self.check_distribution('dirichlet',
                                {'alpha': alpha}, dtype)


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (3, 2)],
    'loc_shape': [(), (3, 2)],
    'scale_shape': [(), (3, 2)],
})
)
@testing.gpu
class TestDistributionsGumbel(RandomDistributionsTestCase):

    @cupy.testing.for_float_dtypes('dtype', no_float16=True)
    @cupy.testing.for_float_dtypes('loc_dtype')
    @cupy.testing.for_float_dtypes('scale_dtype')
    def test_gumbel(self, loc_dtype, scale_dtype, dtype):
        loc = numpy.ones(self.loc_shape, dtype=loc_dtype)
        scale = numpy.ones(self.scale_shape, dtype=scale_dtype)
        self.check_distribution('gumbel',
                                {'loc': loc, 'scale': scale}, dtype)


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (3, 2)],
    'loc_shape': [(), (3, 2)],
    'scale_shape': [(), (3, 2)],
})
)
@testing.gpu
class TestDistributionsLaplace(RandomDistributionsTestCase):

    @cupy.testing.for_float_dtypes('dtype', no_float16=True)
    @cupy.testing.for_float_dtypes('loc_dtype')
    @cupy.testing.for_float_dtypes('scale_dtype')
    def test_laplace(self, loc_dtype, scale_dtype, dtype):
        loc = numpy.ones(self.loc_shape, dtype=loc_dtype)
        scale = numpy.ones(self.scale_shape, dtype=scale_dtype)
        self.check_distribution('laplace',
                                {'loc': loc, 'scale': scale}, dtype)


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (3, 2)],
    'mean_shape': [()],
    'sigma_shape': [()],
})
)
@testing.gpu
class TestDistributionsLognormal(RandomDistributionsTestCase):

    @cupy.testing.for_float_dtypes('dtype', no_float16=True)
    @cupy.testing.for_float_dtypes('mean_dtype')
    @cupy.testing.for_float_dtypes('sigma_dtype')
    def test_lognormal(self, mean_dtype, sigma_dtype, dtype):
        mean = numpy.ones(self.mean_shape, dtype=mean_dtype)
        sigma = numpy.ones(self.sigma_shape, dtype=sigma_dtype)
        self.check_distribution('lognormal',
                                {'mean': mean, 'sigma': sigma}, dtype)


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (3, 2)],
    'loc_shape': [(), (3, 2)],
    'scale_shape': [(), (3, 2)],
})
)
@testing.gpu
class TestDistributionsNormal(RandomDistributionsTestCase):

    @cupy.testing.for_float_dtypes('dtype', no_float16=True)
    @cupy.testing.for_float_dtypes('loc_dtype')
    @cupy.testing.for_float_dtypes('scale_dtype')
    def test_normal(self, loc_dtype, scale_dtype, dtype):
        loc = numpy.ones(self.loc_shape, dtype=loc_dtype)
        scale = numpy.ones(self.scale_shape, dtype=scale_dtype)
        self.check_distribution('normal',
                                {'loc': loc, 'scale': scale}, dtype)


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (3, 2)],
})
)
@testing.gpu
class TestDistributionsStandardNormal(RandomDistributionsTestCase):

    @cupy.testing.for_float_dtypes('dtype', no_float16=True)
    def test_standardnormal(self, dtype):
        self.check_distribution('standard_normal', {}, dtype)


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (3, 2)],
    'df_shape': [(), (3, 2)],
})
)
@testing.gpu
class TestDistributionsStandardT(unittest.TestCase):

    def check_distribution(self, dist_func, df_dtype, dtype):
        df = cupy.ones(self.df_shape, dtype=df_dtype)
        out = dist_func(df, self.shape, dtype)
        self.assertEqual(self.shape, out.shape)
        self.assertEqual(out.dtype, dtype)

    @cupy.testing.for_float_dtypes('dtype', no_float16=True)
    @cupy.testing.for_float_dtypes('df_dtype')
    def test_standard_t(self, df_dtype, dtype):
        self.check_distribution(distributions.standard_t, df_dtype, dtype)


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (3, 2)],
    'low_shape': [(), (3, 2)],
    'high_shape': [(), (3, 2)],
})
)
@testing.gpu
class TestDistributionsUniform(RandomDistributionsTestCase):

    @cupy.testing.for_float_dtypes('dtype', no_float16=True)
    @cupy.testing.for_float_dtypes('low_dtype')
    @cupy.testing.for_float_dtypes('high_dtype')
    def test_uniform(self, low_dtype, high_dtype, dtype):
        low = numpy.ones(self.low_shape, dtype=low_dtype)
        high = numpy.ones(self.high_shape, dtype=high_dtype) * 2.
        self.check_distribution('uniform',
                                {'low': low, 'high': high}, dtype)
