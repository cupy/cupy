import ctypes
import pickle
import unittest

import pytest

cudss = pytest.importorskip('cupy_backends.cuda.libs.cudss')


@unittest.skipUnless(cudss.available, 'cuDSS is unavailable')
class TestCuDSS(unittest.TestCase):

    def test_version(self):
        version = tuple(cudss.getProperty(i) for i in range(3))
        assert version >= (0, 8, 0)

    def test_error(self):
        cudss.check_status(cudss.CUDSS_STATUS_SUCCESS)
        with pytest.raises(cudss.CuDSSError) as exc:
            cudss.check_status(cudss.CUDSS_STATUS_INVALID_VALUE)
        error = pickle.loads(pickle.dumps(exc.value))
        assert error.status == cudss.CUDSS_STATUS_INVALID_VALUE
        assert str(error) == str(exc.value)
        assert 'CUDSS_STATUS_INVALID_VALUE' in str(error)

    def test_config(self):
        config = cudss.configCreate()
        try:
            value = ctypes.c_int(3)
            result = ctypes.c_int()
            param = cudss.CUDSS_CONFIG_IR_N_STEPS
            cudss.configSet(config, param, ctypes.addressof(value),
                            ctypes.sizeof(value))
            size = cudss.configGet(config, param, ctypes.addressof(result),
                                   ctypes.sizeof(result))
            assert size == ctypes.sizeof(result)
            assert result.value == value.value
        finally:
            cudss.configDestroy(config)

    def test_matrix_dn(self):
        values = (ctypes.c_double * 4)(1, 2, 3, 4)
        ptr = ctypes.addressof(values)
        matrix = cudss.matrixCreateDn(
            2, 2, 2, ptr, cudss.CUDSS_R_64F, cudss.CUDSS_LAYOUT_COL_MAJOR)
        try:
            assert cudss.matrixGetDn(matrix) == (
                2, 2, 2, ptr, cudss.CUDSS_R_64F,
                cudss.CUDSS_LAYOUT_COL_MAJOR)
            assert cudss.matrixGetFormat(matrix) == cudss.CUDSS_MFORMAT_DENSE
        finally:
            cudss.matrixDestroy(matrix)

    def test_matrix_csr_mixed_indices(self):
        offsets = (ctypes.c_int64 * 3)(0, 1, 2)
        columns = (ctypes.c_int32 * 2)(0, 1)
        values = (ctypes.c_double * 2)(2, 3)
        start = ctypes.addressof(offsets)
        end = 0  # cuDSS expects contiguous CSR offsets with rowEnd=NULL.
        col = ctypes.addressof(columns)
        val = ctypes.addressof(values)
        metadata = (cudss.CUDSS_R_64I, cudss.CUDSS_R_32I,
                    cudss.CUDSS_R_64F, cudss.CUDSS_MTYPE_GENERAL,
                    cudss.CUDSS_MVIEW_FULL, cudss.CUDSS_BASE_ZERO)
        matrix = cudss.matrixCreateCsr(
            2, 2, 2, start, end, col, val, *metadata)
        try:
            assert cudss.matrixGetCsr(matrix) == (
                2, 2, 2, start, end, col, val, *metadata)
            assert cudss.matrixGetFormat(matrix) == cudss.CUDSS_MFORMAT_CSR
        finally:
            cudss.matrixDestroy(matrix)
