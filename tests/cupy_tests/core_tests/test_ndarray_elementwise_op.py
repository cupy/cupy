from __future__ import annotations

import operator

import numpy
import pytest

import cupy
from cupy import testing


class TestArrayElementwiseOp:

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_cupy_allclose(rtol=1e-6, accept_error=TypeError)
    def check_array_scalar_op(self, op, xp, x_type, y_type, swap=False,
                              no_bool=False, no_complex=False):
        x_dtype = numpy.dtype(x_type)
        y_dtype = numpy.dtype(y_type)
        if no_bool and x_dtype == '?' and y_dtype == '?':
            return xp.array(True)
        if no_complex and (x_dtype.kind == 'c' or y_dtype.kind == 'c'):
            return xp.array(True)
        a = xp.array([[1, 2, 3], [4, 5, 6]], x_type)
        if swap:
            return op(y_type(3), a)
        else:
            return op(a, y_type(3))

    @testing.with_requires('numpy>=1.25')
    def test_add_scalar(self):
        self.check_array_scalar_op(operator.add)

    @testing.with_requires('numpy>=1.25')
    def test_radd_scalar(self):
        self.check_array_scalar_op(operator.add, swap=True)

    def test_iadd_scalar(self):
        self.check_array_scalar_op(operator.iadd)

    @testing.with_requires('numpy>=1.25')
    def test_sub_scalar(self):
        self.check_array_scalar_op(operator.sub, no_bool=True)

    @testing.with_requires('numpy>=1.25')
    def test_rsub_scalar(self):
        self.check_array_scalar_op(operator.sub, swap=True, no_bool=True)

    def test_isub_scalar(self):
        self.check_array_scalar_op(operator.isub, no_bool=True)

    @testing.with_requires('numpy>=1.25')
    def test_mul_scalar(self):
        self.check_array_scalar_op(operator.mul)

    @testing.with_requires('numpy>=1.25')
    def test_rmul_scalar(self):
        self.check_array_scalar_op(operator.mul, swap=True)

    def test_imul_scalar(self):
        self.check_array_scalar_op(operator.imul)

    @testing.with_requires('numpy>=1.25')
    def test_truediv_scalar(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_scalar_op(operator.truediv)

    @testing.with_requires('numpy>=1.25')
    def test_rtruediv_scalar(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_scalar_op(operator.truediv, swap=True)

    def test_itruediv_scalar(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_scalar_op(operator.itruediv)

    def test_floordiv_scalar(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_scalar_op(operator.floordiv, no_complex=True)

    def test_rfloordiv_scalar(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_scalar_op(operator.floordiv, swap=True,
                                       no_complex=True)

    def test_ifloordiv_scalar(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_scalar_op(operator.ifloordiv, no_complex=True)

    @testing.with_requires('numpy>=1.25')
    def test_pow_scalar(self):
        self.check_array_scalar_op(operator.pow)

    @testing.with_requires('numpy>=1.25')
    def test_rpow_scalar(self):
        self.check_array_scalar_op(operator.pow, swap=True)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_cupy_allclose(atol=1.0, accept_error=TypeError)
    def check_ipow_scalar(self, xp, x_type, y_type):
        a = xp.array([[1, 2, 3], [4, 5, 6]], x_type)
        return operator.ipow(a, y_type(3))

    def test_ipow_scalar(self):
        self.check_ipow_scalar()

    def test_divmod0_scalar(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_scalar_op(lambda x, y: divmod(x, y)[0],
                                       no_complex=True)

    def test_divmod1_scalar(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_scalar_op(lambda x, y: divmod(x, y)[1],
                                       no_complex=True)

    def test_rdivmod0_scalar(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_scalar_op(lambda x, y: divmod(x, y)[0], swap=True,
                                       no_complex=True)

    def test_rdivmod1_scalar(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_scalar_op(lambda x, y: divmod(x, y)[1], swap=True,
                                       no_complex=True)

    def test_lt_scalar(self):
        self.check_array_scalar_op(operator.lt, no_complex=False)

    def test_le_scalar(self):
        self.check_array_scalar_op(operator.le, no_complex=False)

    def test_gt_scalar(self):
        self.check_array_scalar_op(operator.gt, no_complex=False)

    def test_ge_scalar(self):
        self.check_array_scalar_op(operator.ge, no_complex=False)

    def test_eq_scalar(self):
        self.check_array_scalar_op(operator.eq)

    def test_ne_scalar(self):
        self.check_array_scalar_op(operator.ne)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def check_array_array_op(self, op, xp, x_type, y_type,
                             no_bool=False, no_complex=False):
        x_dtype = numpy.dtype(x_type)
        y_dtype = numpy.dtype(y_type)
        if no_bool and x_dtype == '?' and y_dtype == '?':
            return xp.array(True)
        if no_complex and (x_dtype.kind == 'c' or y_dtype.kind == 'c'):
            return xp.array(True)
        a = xp.array([[1, 2, 3], [4, 5, 6]], x_type)
        b = xp.array([[6, 5, 4], [3, 2, 1]], y_type)
        return op(a, b)

    def test_add_array(self):
        self.check_array_array_op(operator.add)

    def test_iadd_array(self):
        self.check_array_array_op(operator.iadd)

    def test_sub_array(self):
        self.check_array_array_op(operator.sub, no_bool=True)

    def test_isub_array(self):
        self.check_array_array_op(operator.isub, no_bool=True)

    def test_mul_array(self):
        self.check_array_array_op(operator.mul)

    def test_imul_array(self):
        self.check_array_array_op(operator.imul)

    def test_truediv_array(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_array_op(operator.truediv)

    def test_itruediv_array(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_array_op(operator.itruediv)

    def test_floordiv_array(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_array_op(operator.floordiv, no_complex=True)

    def test_ifloordiv_array(self):
        if '1.16.1' <= numpy.lib.NumpyVersion(numpy.__version__) < '1.18.0':
            self.skipTest("NumPy Issue #12927")
        with numpy.errstate(divide='ignore'):
            self.check_array_array_op(operator.ifloordiv, no_complex=True)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-6, accept_error=TypeError)
    def check_pow_array(self, xp, x_type, y_type):
        a = xp.array([[1, 2, 3], [4, 5, 6]], x_type)
        b = xp.array([[6, 5, 4], [3, 2, 1]], y_type)
        return operator.pow(a, b)

    def test_pow_array(self):
        # There are some precision issues in HIP that prevent
        # checking with atol=0
        if cupy.cuda.runtime.is_hip:
            self.check_pow_array()
        else:
            self.check_array_array_op(operator.pow)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_cupy_allclose(atol=1.0, accept_error=TypeError)
    def check_ipow_array(self, xp, x_type, y_type):
        a = xp.array([[1, 2, 3], [4, 5, 6]], x_type)
        b = xp.array([[6, 5, 4], [3, 2, 1]], y_type)
        return operator.ipow(a, b)

    def test_ipow_array(self):
        self.check_ipow_array()

    def test_divmod0_array(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_array_op(lambda x, y: divmod(x, y)[0])

    def test_divmod1_array(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_array_op(lambda x, y: divmod(x, y)[1])

    def test_lt_array(self):
        self.check_array_array_op(operator.lt, no_complex=True)

    def test_le_array(self):
        self.check_array_array_op(operator.le, no_complex=True)

    def test_gt_array(self):
        self.check_array_array_op(operator.gt, no_complex=True)

    def test_ge_array(self):
        self.check_array_array_op(operator.ge, no_complex=True)

    def test_eq_array(self):
        self.check_array_array_op(operator.eq)

    def test_ne_array(self):
        self.check_array_array_op(operator.ne)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def check_array_broadcasted_op(self, op, xp, x_type, y_type,
                                   no_bool=False, no_complex=False):
        x_dtype = numpy.dtype(x_type)
        y_dtype = numpy.dtype(y_type)
        if no_bool and x_dtype == '?' and y_dtype == '?':
            return xp.array(True)
        if no_complex and (x_dtype.kind == 'c' or y_dtype.kind == 'c'):
            return xp.array(True)
        a = xp.array([[1, 2, 3], [4, 5, 6]], x_type)
        b = xp.array([[1], [2]], y_type)
        return op(a, b)

    def test_broadcasted_add(self):
        self.check_array_broadcasted_op(operator.add)

    def test_broadcasted_iadd(self):
        self.check_array_broadcasted_op(operator.iadd)

    def test_broadcasted_sub(self):
        # TODO(unno): sub for boolean array is deprecated in numpy>=1.13
        self.check_array_broadcasted_op(operator.sub, no_bool=True)

    def test_broadcasted_isub(self):
        # TODO(unno): sub for boolean array is deprecated in numpy>=1.13
        self.check_array_broadcasted_op(operator.isub, no_bool=True)

    def test_broadcasted_mul(self):
        self.check_array_broadcasted_op(operator.mul)

    def test_broadcasted_imul(self):
        self.check_array_broadcasted_op(operator.imul)

    def test_broadcasted_truediv(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_broadcasted_op(operator.truediv)

    def test_broadcasted_itruediv(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_broadcasted_op(operator.itruediv)

    def test_broadcasted_floordiv(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_broadcasted_op(operator.floordiv, no_complex=True)

    def test_broadcasted_ifloordiv(self):
        if '1.16.1' <= numpy.lib.NumpyVersion(numpy.__version__) < '1.18.0':
            self.skipTest("NumPy Issue #12927")
        with numpy.errstate(divide='ignore'):
            self.check_array_broadcasted_op(operator.ifloordiv,
                                            no_complex=True)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-6, accept_error=TypeError)
    def check_broadcasted_pow(self, xp, x_type, y_type):
        a = xp.array([[1, 2, 3], [4, 5, 6]], x_type)
        b = xp.array([[1], [2]], y_type)
        return operator.pow(a, b)

    def test_broadcasted_pow(self):
        # There are some precision issues in HIP that prevent
        # checking with atol=0
        if cupy.cuda.runtime.is_hip:
            self.check_broadcasted_pow()
        else:
            self.check_array_broadcasted_op(operator.pow)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_cupy_allclose(atol=1.0, accept_error=TypeError)
    def check_broadcasted_ipow(self, xp, x_type, y_type):
        a = xp.array([[1, 2, 3], [4, 5, 6]], x_type)
        b = xp.array([[1], [2]], y_type)
        return operator.ipow(a, b)

    def test_broadcasted_ipow(self):
        self.check_broadcasted_ipow()

    def test_broadcasted_divmod0(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_broadcasted_op(lambda x, y: divmod(x, y)[0],
                                            no_complex=True)

    def test_broadcasted_divmod1(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_broadcasted_op(lambda x, y: divmod(x, y)[1],
                                            no_complex=True)

    def test_broadcasted_lt(self):
        self.check_array_broadcasted_op(operator.lt, no_complex=True)

    def test_broadcasted_le(self):
        self.check_array_broadcasted_op(operator.le, no_complex=True)

    def test_broadcasted_gt(self):
        self.check_array_broadcasted_op(operator.gt, no_complex=True)

    def test_broadcasted_ge(self):
        self.check_array_broadcasted_op(operator.ge, no_complex=True)

    def test_broadcasted_eq(self):
        self.check_array_broadcasted_op(operator.eq)

    def test_broadcasted_ne(self):
        self.check_array_broadcasted_op(operator.ne)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def check_array_doubly_broadcasted_op(self, op, xp, x_type, y_type,
                                          no_bool=False, no_complex=False):
        x_dtype = numpy.dtype(x_type)
        y_dtype = numpy.dtype(y_type)
        if no_bool and x_dtype == '?' and y_dtype == '?':
            return xp.array(True)
        if no_complex and (x_dtype.kind == 'c' or y_dtype.kind == 'c'):
            return xp.array(True)
        a = xp.array([[[1, 2, 3]], [[4, 5, 6]]], x_type)
        b = xp.array([[1], [2], [3]], y_type)
        return op(a, b)

    def test_doubly_broadcasted_add(self):
        self.check_array_doubly_broadcasted_op(operator.add)

    def test_doubly_broadcasted_sub(self):
        self.check_array_doubly_broadcasted_op(operator.sub, no_bool=True)

    def test_doubly_broadcasted_mul(self):
        self.check_array_doubly_broadcasted_op(operator.mul)

    def test_doubly_broadcasted_truediv(self):
        with numpy.errstate(divide='ignore', invalid='ignore'):
            self.check_array_doubly_broadcasted_op(operator.truediv)

    def test_doubly_broadcasted_floordiv(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_doubly_broadcasted_op(operator.floordiv,
                                                   no_complex=True)

    def test_doubly_broadcasted_pow(self):
        self.check_array_doubly_broadcasted_op(operator.pow)

    def test_doubly_broadcasted_divmod0(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_doubly_broadcasted_op(
                lambda x, y: divmod(x, y)[0],
                no_complex=True)

    def test_doubly_broadcasted_divmod1(self):
        with numpy.errstate(divide='ignore'):
            self.check_array_doubly_broadcasted_op(
                lambda x, y: divmod(x, y)[1],
                no_complex=True)

    def test_doubly_broadcasted_lt(self):
        self.check_array_doubly_broadcasted_op(operator.lt, no_complex=True)

    def test_doubly_broadcasted_le(self):
        self.check_array_doubly_broadcasted_op(operator.le, no_complex=True)

    def test_doubly_broadcasted_gt(self):
        self.check_array_doubly_broadcasted_op(operator.gt, no_complex=True)

    def test_doubly_broadcasted_ge(self):
        self.check_array_doubly_broadcasted_op(operator.ge, no_complex=True)

    def test_doubly_broadcasted_eq(self):
        self.check_array_doubly_broadcasted_op(operator.eq)

    def test_doubly_broadcasted_ne(self):
        self.check_array_doubly_broadcasted_op(operator.ne)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_cupy_allclose()
    def check_array_reversed_op(self, op, xp, x_type, y_type, no_bool=False):
        if no_bool and x_type == numpy.bool_ and y_type == numpy.bool_:
            return xp.array(True)
        a = xp.array([1, 2, 3, 4, 5], x_type)
        b = xp.array([1, 2, 3, 4, 5], y_type)
        return op(a, b[::-1])

    def test_array_reversed_add(self):
        self.check_array_reversed_op(operator.add)

    def test_array_reversed_sub(self):
        self.check_array_reversed_op(operator.sub, no_bool=True)

    def test_array_reversed_mul(self):
        self.check_array_reversed_op(operator.mul)

    @pytest.mark.parametrize('val',
                             [True, False,
                              0, -127, 255, -32768, 65535, -2147483648,
                              4294967295,
                              0.0, 100000.0])
    @pytest.mark.parametrize('op', [operator.add, operator.sub,
                                    operator.mul, ])
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(accept_error=OverflowError)
    def test_typecast_(self, xp, op, dtype, val):
        a = op(val, (testing.shaped_arange((5,), xp, dtype) - 2))
        return a

    @pytest.mark.parametrize('val',
                             [True, False,
                              0, -127, 255, -32768, 65535, -2147483648,
                              4294967295,
                              0.0, 100000.0])
    @testing.for_all_dtypes(no_bool=True)
    def test_typecast_2(self, dtype, val):
        op = operator.truediv
        with numpy.errstate(divide='ignore', invalid='ignore'):
            a = op(val, (testing.shaped_arange((5,), numpy, dtype) - 2))
        b = op(val, (testing.shaped_arange((5,), cupy, dtype) - 2))
        assert a.dtype == b.dtype

    # Skip float16 because of NumPy #19514
    @testing.for_all_dtypes(name='x_type', no_float16=True)
    @testing.numpy_cupy_allclose()
    def check_array_boolarray_op(self, op, xp, x_type):
        a = xp.array([[2, 7, 1], [8, 2, 8]], x_type)
        # Cast from np.bool8 array should not read bytes
        b = xp.array([[3, 1, 4], [-1, -5, -9]], numpy.int8).view(bool)
        return op(a, b)

    def test_add_array_boolarray(self):
        self.check_array_boolarray_op(operator.add)

    def test_iadd_array_boolarray(self):
        self.check_array_boolarray_op(operator.iadd)


class TestArrayIntElementwiseOp:

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def check_array_scalar_op(self, op, xp, x_type, y_type, swap=False):
        a = xp.array([[0, 1, 2], [1, 0, 2]], dtype=x_type)
        if swap:
            return op(y_type(2), a)
        else:
            return op(a, y_type(2))

    def test_lshift_scalar(self):
        self.check_array_scalar_op(operator.lshift)

    def test_rlshift_scalar(self):
        self.check_array_scalar_op(operator.lshift, swap=True)

    def test_rshift_scalar(self):
        self.check_array_scalar_op(operator.rshift)

    def test_rrshift_scalar(self):
        self.check_array_scalar_op(operator.rshift, swap=True)

    def test_and_scalar(self):
        self.check_array_scalar_op(operator.and_)

    def test_rand_scalar(self):
        self.check_array_scalar_op(operator.and_, swap=True)

    def test_or_scalar(self):
        self.check_array_scalar_op(operator.or_)

    def test_ror_scalar(self):
        self.check_array_scalar_op(operator.or_, swap=True)

    def test_xor_scalar(self):
        self.check_array_scalar_op(operator.xor)

    def test_rxor_scalar(self):
        self.check_array_scalar_op(operator.xor, swap=True)

    def test_mod_scalar(self):
        with numpy.errstate(divide='ignore', invalid='ignore'):
            self.check_array_scalar_op(operator.mod)

    def test_rmod_scalar(self):
        with numpy.errstate(divide='ignore', invalid='ignore'):
            self.check_array_scalar_op(operator.mod, swap=True)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def check_array_scalarzero_op(self, op, xp, x_type, y_type, swap=False):
        a = xp.array([[0, 1, 2], [1, 0, 2]], dtype=x_type)
        if swap:
            return op(y_type(0), a)
        else:
            return op(a, y_type(0))

    def test_lshift_scalarzero(self):
        self.check_array_scalarzero_op(operator.lshift)

    def test_rlshift_scalarzero(self):
        self.check_array_scalarzero_op(operator.lshift, swap=True)

    def test_rshift_scalarzero(self):
        self.check_array_scalarzero_op(operator.rshift)

    def test_rrshift_scalarzero(self):
        self.check_array_scalarzero_op(operator.rshift, swap=True)

    def test_and_scalarzero(self):
        self.check_array_scalarzero_op(operator.and_)

    def test_rand_scalarzero(self):
        self.check_array_scalarzero_op(operator.and_, swap=True)

    def test_or_scalarzero(self):
        self.check_array_scalarzero_op(operator.or_)

    def test_ror_scalarzero(self):
        self.check_array_scalarzero_op(operator.or_, swap=True)

    def test_xor_scalarzero(self):
        self.check_array_scalarzero_op(operator.xor)

    def test_rxor_scalarzero(self):
        self.check_array_scalarzero_op(operator.xor, swap=True)

    def test_mod_scalarzero(self):
        with numpy.errstate(divide='ignore', invalid='ignore'):
            self.check_array_scalarzero_op(operator.mod)

    def test_rmod_scalarzero(self):
        with numpy.errstate(divide='ignore', invalid='ignore'):
            self.check_array_scalarzero_op(operator.mod, swap=True)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def check_array_array_op(self, op, xp, x_type, y_type):
        a = xp.array([[0, 1, 2], [1, 0, 2]], dtype=x_type)
        b = xp.array([[0, 0, 1], [0, 1, 2]], dtype=y_type)
        return op(a, b)

    def test_lshift_array(self):
        self.check_array_array_op(operator.lshift)

    def test_ilshift_array(self):
        self.check_array_array_op(operator.ilshift)

    def test_rshift_array(self):
        self.check_array_array_op(operator.rshift)

    def test_irshift_array(self):
        self.check_array_array_op(operator.irshift)

    def test_and_array(self):
        self.check_array_array_op(operator.and_)

    def test_iand_array(self):
        self.check_array_array_op(operator.iand)

    def test_or_array(self):
        self.check_array_array_op(operator.or_)

    def test_ior_array(self):
        self.check_array_array_op(operator.ior)

    def test_xor_array(self):
        self.check_array_array_op(operator.xor)

    def test_ixor_array(self):
        self.check_array_array_op(operator.ixor)

    def test_mod_array(self):
        with numpy.errstate(divide='ignore', invalid='ignore'):
            self.check_array_array_op(operator.mod)

    def test_imod_array(self):
        with numpy.errstate(divide='ignore', invalid='ignore'):
            self.check_array_array_op(operator.imod)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def check_array_broadcasted_op(self, op, xp, x_type, y_type):
        a = xp.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]], dtype=x_type)
        b = xp.array([[0, 0, 1]], dtype=y_type)
        return op(a, b)

    def test_broadcasted_lshift(self):
        self.check_array_broadcasted_op(operator.lshift)

    def test_broadcasted_ilshift(self):
        self.check_array_broadcasted_op(operator.ilshift)

    def test_broadcasted_rshift(self):
        self.check_array_broadcasted_op(operator.rshift)

    def test_broadcasted_irshift(self):
        self.check_array_broadcasted_op(operator.irshift)

    def test_broadcasted_and(self):
        self.check_array_broadcasted_op(operator.and_)

    def test_broadcasted_iand(self):
        self.check_array_broadcasted_op(operator.iand)

    def test_broadcasted_or(self):
        self.check_array_broadcasted_op(operator.or_)

    def test_broadcasted_ior(self):
        self.check_array_broadcasted_op(operator.ior)

    def test_broadcasted_xor(self):
        self.check_array_broadcasted_op(operator.xor)

    def test_broadcasted_ixor(self):
        self.check_array_broadcasted_op(operator.ixor)

    def test_broadcasted_mod(self):
        with numpy.errstate(divide='ignore', invalid='ignore'):
            self.check_array_broadcasted_op(operator.mod)

    def test_broadcasted_imod(self):
        with numpy.errstate(divide='ignore', invalid='ignore'):
            self.check_array_broadcasted_op(operator.imod)

    @testing.for_all_dtypes_combination(names=['x_type', 'y_type'])
    @testing.numpy_cupy_allclose(accept_error=TypeError)
    def check_array_doubly_broadcasted_op(self, op, xp, x_type, y_type):
        a = xp.array([[[0, 1, 2]], [[1, 0, 2]]], dtype=x_type)
        b = xp.array([[0], [0], [1]], dtype=y_type)
        return op(a, b)

    def test_doubly_broadcasted_lshift(self):
        self.check_array_doubly_broadcasted_op(operator.lshift)

    def test_doubly_broadcasted_rshift(self):
        self.check_array_doubly_broadcasted_op(operator.rshift)

    def test_doubly_broadcasted_and(self):
        self.check_array_doubly_broadcasted_op(operator.and_)

    def test_doubly_broadcasted_or(self):
        self.check_array_doubly_broadcasted_op(operator.or_)

    def test_doubly_broadcasted_xor(self):
        self.check_array_doubly_broadcasted_op(operator.xor)

    def test_doubly_broadcasted_mod(self):
        with numpy.errstate(divide='ignore', invalid='ignore'):
            self.check_array_doubly_broadcasted_op(operator.mod)


@pytest.mark.parametrize('value', [
    None,
    Ellipsis,
    object(),
    numpy._NoValue,
])
class TestArrayObjectComparison:

    @pytest.mark.parametrize('swap', [False, True])
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_eq_object(self, xp, dtype, value, swap):
        a = xp.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        if swap:
            return value == a
        else:
            return a == value

    @pytest.mark.parametrize('swap', [False, True])
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_ne_object(self, xp, dtype, value, swap):
        a = xp.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        if swap:
            return value != a
        else:
            return a != value


class HasEq:
    def __eq__(self, other):
        return (other == 2) | (other == 4)


class HasNe:
    def __ne__(self, other):
        return (other == 2) | (other == 4)


class HasEqSub(HasEq):
    pass


class CustomInt(int):
    pass


@pytest.mark.parametrize('dtype', ['int32', 'float64'])
@pytest.mark.parametrize('value', [
    HasEq(),
    HasNe(),  # eq test passes because `==` does not fall back to `__ne__`.
    HasEqSub(),
    CustomInt(3),
])
class TestArrayObjectComparisonDifficult:

    # OK to raise TypeError.
    # If CuPy returns a result, it should match with NumPy's result.

    def test_eq_object(self, dtype, value):
        expected = numpy.array([[1, 2, 3], [4, 5, 6]], dtype=dtype) == value

        a = cupy.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        try:
            res = a == value
        except TypeError:
            pytest.skip()

        cupy.testing.assert_array_equal(res, expected)

    def test_ne_object(self, dtype, value):
        expected = numpy.array([[1, 2, 3], [4, 5, 6]], dtype=dtype) != value

        a = cupy.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        try:
            res = a != value
        except TypeError:
            pytest.skip()

        cupy.testing.assert_array_equal(res, expected)
