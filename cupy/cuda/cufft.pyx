cimport cython  # NOQA

import cupy
import numpy

from cupy.cuda cimport driver
from cupy.cuda cimport memory
from cupy.cuda cimport stream as stream_module


cdef extern from "cupy_cufft.h" nogil:
    ctypedef struct Complex 'cufftComplex':
        float x, y

    ctypedef struct DoubleComplex 'cufftDoubleComplex':
        double x, y

    # cuFFT Helper Function
    Result cufftCreate(Handle *plan)
    Result cufftDestroy(Handle plan)
    Result cufftSetAutoAllocation(Handle plan, int autoAllocate)
    Result cufftSetWorkArea(Handle plan, void *workArea)

    # cuFFT Stream Function
    Result cufftSetStream(Handle plan, driver.Stream streamId)

    # cuFFT Plan Function
    Result cufftMakePlan1d(Handle plan, int nx, Type type, int batch,
                           size_t *workSize)

    # cuFFT Exec Function
    Result cufftExecC2C(Handle plan, Complex *idata, Complex *odata,
                        int direction)
    Result cufftExecR2C(Handle plan, Float *idata, Complex *odata)
    Result cufftExecC2R(Handle plan, Complex *idata, Float *odata)
    Result cufftExecZ2Z(Handle plan, DoubleComplex *idata,
                        DoubleComplex *odata, int direction)
    Result cufftExecD2Z(Handle plan, Double *idata, DoubleComplex *odata)
    Result cufftExecZ2D(Handle plan, DoubleComplex *idata, Double *odata)


cdef dict RESULT = {
    0: 'CUFFT_SUCCESS',
    1: 'CUFFT_INVALID_PLAN',
    2: 'CUFFT_ALLOC_FAILED',
    3: 'CUFFT_INVALID_TYPE',
    4: 'CUFFT_INVALID_VALUE',
    5: 'CUFFT_INTERNAL_ERROR',
    6: 'CUFFT_EXEC_FAILED',
    7: 'CUFFT_SETUP_FAILED',
    8: 'CUFFT_INVALID_SIZE',
    9: 'CUFFT_UNALIGNED_DATA',
    10: 'CUFFT_INCOMPLETE_PARAMETER_LIST',
    11: 'CUFFT_INVALID_DEVICE',
    12: 'CUFFT_PARSE_ERROR',
    13: 'CUFFT_NO_WORKSPACE',
    14: 'CUFFT_NOT_IMPLEMENTED',
    15: 'CUFFT_LICENSE_ERROR',
    16: 'CUFFT_NOT_SUPPORTED',
}


class CuFFTError(RuntimeError):

    def __init__(self, int result):
        self.result = result
        super(CuFFTError, self).__init__('%s' % (RESULT[result]))


@cython.profile(False)
cpdef inline check_result(int result):
    if result != 0:
        raise CuFFTError(result)


class Plan1d(object):
    def __init__(self, int nx, int fft_type, int batch):
        cdef Handle plan
        cdef size_t work_size
        stream = stream_module.get_current_stream_ptr()
        with nogil:
            result = cufftCreate(&plan)
            if result == 0:
                result = cufftSetStream(<Handle>plan, <driver.Stream>stream)
            if result == 0:
                result = cufftSetAutoAllocation(plan, 0)
            if result == 0:
                result = cufftMakePlan1d(plan, nx, <Type>fft_type, batch,
                                         &work_size)

        # cufftMakePlan1d uses large memory when nx has large divisor.
        # See https://github.com/cupy/cupy/issues/1063
        if result == 2:
            cupy.get_default_memory_pool().free_all_blocks()
            with nogil:
                result = cufftMakePlan1d(plan, nx, <Type>fft_type, batch,
                                         &work_size)

        check_result(result)
        work_area = memory.alloc(work_size)
        with nogil:
            result = cufftSetWorkArea(plan, <void *>(work_area.ptr))
        check_result(result)
        self.nx = nx
        self.fft_type = fft_type
        self.plan = plan
        self.work_area = work_area

    def __del__(self):
        cdef Handle plan = self.plan
        with nogil:
            result = cufftDestroy(plan)
        check_result(result)

    def fft(self, a, out, direction):
        if self.fft_type == CUFFT_C2C:
            execC2C(self.plan, a.data, out.data, direction)
        elif self.fft_type == CUFFT_R2C:
            execR2C(self.plan, a.data, out.data)
        elif self.fft_type == CUFFT_C2R:
            execC2R(self.plan, a.data, out.data)
        elif self.fft_type == CUFFT_Z2Z:
            execZ2Z(self.plan, a.data, out.data, direction)
        elif self.fft_type == CUFFT_D2Z:
            execD2Z(self.plan, a.data, out.data)
        else:
            execZ2D(self.plan, a.data, out.data)

    def get_output_array(self, a):
        shape = list(a.shape)
        if self.fft_type == CUFFT_C2C:
            return cupy.empty(shape, numpy.complex64)
        elif self.fft_type == CUFFT_R2C:
            shape[-1] = shape[-1] // 2 + 1
            return cupy.empty(shape, numpy.complex64)
        elif self.fft_type == CUFFT_C2R:
            shape[-1] = self.nx
            return cupy.empty(shape, numpy.float32)
        elif self.fft_type == CUFFT_Z2Z:
            return cupy.empty(shape, numpy.complex128)
        elif self.fft_type == CUFFT_D2Z:
            shape[-1] = shape[-1] // 2 + 1
            return cupy.empty(shape, numpy.complex128)
        else:
            shape[-1] = self.nx
            return cupy.empty(shape, numpy.float64)


cpdef execC2C(size_t plan, size_t idata, size_t odata, int direction):
    with nogil:
        result = cufftExecC2C(plan, <Complex*>idata, <Complex*>odata,
                              direction)
    check_result(result)


cpdef execR2C(size_t plan, size_t idata, size_t odata):
    with nogil:
        result = cufftExecR2C(plan, <Float*>idata, <Complex*>odata)
    check_result(result)


cpdef execC2R(size_t plan, size_t idata, size_t odata):
    with nogil:
        result = cufftExecC2R(plan, <Complex*>idata, <Float*>odata)
    check_result(result)


cpdef execZ2Z(size_t plan, size_t idata, size_t odata, int direction):
    with nogil:
        result = cufftExecZ2Z(plan, <DoubleComplex*>idata,
                              <DoubleComplex*>odata, direction)
    check_result(result)


cpdef execD2Z(size_t plan, size_t idata, size_t odata):
    with nogil:
        result = cufftExecD2Z(plan, <Double*>idata, <DoubleComplex*>odata)
    check_result(result)


cpdef execZ2D(size_t plan, size_t idata, size_t odata):
    with nogil:
        result = cufftExecZ2D(plan, <DoubleComplex*>idata, <Double*>odata)
    check_result(result)
