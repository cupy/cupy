import unittest

import numpy

import cupy
from cupy import testing

@testing.gpu
class TestGradient(unittest.TestCase):

    
    # basic test
    @testing.numpy_cupy_allclose()
    def test_basic(self, xp):
        a = testing.shaped_arange((4,5), xp)
        return xp.gradient(a)

    # test axis
    @testing.numpy_cupy_allclose()
    def test_with_axis_None(self, xp):
        a = testing.shaped_arange((4,5), xp)
        return xp.gradient(a, axis=None)

    @testing.numpy_cupy_allclose()
    def test_with_axis_positive(self, xp):
        a = testing.shaped_arange((4,5), xp)
        return xp.gradient(a, axis=1)    
        
    @testing.numpy_cupy_allclose()
    def test_with_axis_negative(self, xp):
        a = testing.shaped_arange((4,5), xp)
        return xp.gradient(a, axis=-1)
        
    @testing.numpy_cupy_allclose()
    def test_with_axis_tuple(self, xp):
        a = testing.shaped_arange((4,5), xp)
        return xp.gradient(a, axis=(1, 0))   

    # test arguments
    @testing.numpy_cupy_allclose()
    def test_with_argument_number(self, xp):
        f_2d = testing.shaped_arange((5,), xp)
        return xp.gradient(f_2d, 3.0)  
    
    # TypeError: no implementation found for 'numpy.ndim' on types that implement __array_function__: [<class 'cupy.core.core.ndarray'>]

    # def test_with_argument_array(self):
    #     # cp_dx = cupy.cumsum(cupy.ones(5))
    #     np_dx = numpy.cumsum(numpy.ones(5))
    #     cp_f = cupy.arange(5)
    #     np_f = numpy.arange(5)
    #     return testing.assert_array_list_equal(
    #         cupy.gradient(cp_f, np_dx),
    #         numpy.gradient(np_f, np_dx)
    #     )

    # test dateime
    # Unsupported dtype datetime64[D]
    # @testing.numpy_cupy_allclose()
    # def test_datetime(self, xp):
    #     x = xp.array([
    #             "1910-08-16",
    #             "1910-08-11",
    #             "1910-08-10",
    #             "1910-08-12",
    #             "1910-10-12",
    #             "1910-12-12",
    #             "1912-12-12",
    #         ],
    #         dtype="datetime64[D]",
    #     )
    #     return xp.gradient(x)

    # TypeError: no implementation found for 'numpy.ndim' on types that implement __array_function__: [<class 'cupy.core.core.ndarray'>]
    # def test_with_argument_2d_and_array(self):
    #     # cp_dx = cupy.cumsum(cupy.ones(5))
    #     np_dx = cupy.array(1.15)
    #     cp_f = cupy.arange(25).reshape(5, 5)
    #     np_f = numpy.arange(25).reshape(5, 5)
    #     return testing.assert_array_list_equal(
    #         cupy.gradient(cp_f, np_dx),
    #         numpy.gradient(np_f, np_dx)
    #     )