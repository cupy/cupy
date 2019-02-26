cimport cython  # NOQA
import numpy

import cupy
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

    # cuFFT Plan Functions
    Result cufftMakePlan1d(Handle plan, int nx, Type type, int batch,
                           size_t *workSize)
    Result cufftMakePlanMany(Handle plan, int rank, int *n, int *inembed,
                             int istride, int idist, int *onembed, int ostride,
                             int odist, Type type, int batch,
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
        self.batch = batch

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

    def _output_dtype_and_shape(self, a):
        shape = list(a.shape)
        if self.fft_type == CUFFT_C2C:
            dtype = numpy.complex64
        elif self.fft_type == CUFFT_R2C:
            shape[-1] = shape[-1] // 2 + 1
            dtype = numpy.complex64
        elif self.fft_type == CUFFT_C2R:
            shape[-1] = self.nx
            dtype = numpy.float32
        elif self.fft_type == CUFFT_Z2Z:
            dtype = numpy.complex128
        elif self.fft_type == CUFFT_D2Z:
            shape[-1] = shape[-1] // 2 + 1
            dtype = numpy.complex128
        else:
            shape[-1] = self.nx
            dtype = numpy.float64
        return tuple(shape), dtype

    def get_output_array(self, a):
        shape, dtype = self._output_dtype_and_shape(a)
        return cupy.empty(shape, dtype)

    def check_output_array(self, a, out):
        """Verify shape and dtype of the output array.

        Parameters
        ----------
        a : cupy.array
            The input to the transform
        out : cupy.array
            The array where the output of the transform will be stored.
        """
        shape, dtype = self._output_dtype_and_shape(a)
        if out.shape != shape:
            raise ValueError(
                ("out must have shape {}.").format(shape))
        if out.dtype != dtype:
            raise ValueError(
                "out dtype mismatch: found {}, expected {}".format(
                    out.dtype, a.dtype))


class PlanNd(object):
    def __init__(self, object shape, object inembed, int istride,
                 int idist, object onembed, int ostride, int odist,
                 int fft_type, int batch):
        cdef Handle plan
        cdef size_t work_size
        cdef int ndim, i
        cdef int[:] shape_arr = numpy.asarray(shape, dtype=numpy.intc)
        cdef int[:] inembed_arr
        cdef int[:] onembed_arr
        cdef int* shape_ptr = &shape_arr[0]
        cdef int* inembed_ptr
        cdef int* onembed_ptr
        ndim = len(shape)

        if inembed is None:
            inembed_ptr = NULL  # ignore istride and use default strides
        else:
            inembed_arr = numpy.asarray(inembed, dtype=numpy.intc)
            inembed_ptr = &inembed_arr[0]

        if onembed is None:
            onembed_ptr = NULL  # ignore ostride and use default strides
        else:
            onembed_arr = numpy.asarray(onembed, dtype=numpy.intc)
            onembed_ptr = &onembed_arr[0]

        stream = stream_module.get_current_stream_ptr()
        with nogil:
            result = cufftCreate(&plan)
            if result == 0:
                result = cufftSetStream(<Handle>plan, <driver.Stream>stream)
            if result == 0:
                result = cufftSetAutoAllocation(plan, 0)
            if result == 0:
                result = cufftMakePlanMany(plan, ndim, shape_ptr,
                                           inembed_ptr, istride, idist,
                                           onembed_ptr, ostride, odist,
                                           <Type>fft_type, batch,
                                           &work_size)

        # cufftMakePlanMany could use a large amount of memory
        if result == 2:
            cupy.get_default_memory_pool().free_all_blocks()
            with nogil:
                result = cufftMakePlanMany(plan, ndim, shape_ptr,
                                           inembed_ptr, istride, idist,
                                           onembed_ptr, ostride, odist,
                                           <Type>fft_type, batch,
                                           &work_size)
        check_result(result)

        # TODO: for CUDA>=9.2 could also allow setting a work area policy
        # result = cufftXtSetWorkAreaPolicy(plan, policy, &work_size)

        work_area = memory.alloc(work_size)
        with nogil:
            result = cufftSetWorkArea(plan, <void *>(work_area.ptr))
        check_result(result)
        self.shape = tuple(shape)
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
        elif self.fft_type == CUFFT_Z2Z:
            execZ2Z(self.plan, a.data, out.data, direction)
        else:
            raise NotImplementedError("only C2C and Z2Z implemented")

    def get_output_array(self, a, order='C'):
        shape = list(a.shape)
        if self.fft_type == CUFFT_C2C:
            return cupy.empty(shape, numpy.complex64, order=order)
        elif self.fft_type == CUFFT_Z2Z:
            return cupy.empty(shape, numpy.complex128, order=order)
        else:
            raise NotImplementedError("only C2C and Z2Z implemented")

    def check_output_array(self, a, out):
        if out is a:
            return
        if out.dtype != a.dtype:
            raise ValueError("output dtype mismatch")
        if not ((out.flags.f_contiguous == a.flags.f_contiguous) and
                (out.flags.c_contiguous == a.flags.c_contiguous)):
            raise ValueError("output contiguity mismatch")
        if out.shape != a.shape:
            raise ValueError("output shape mismatch")


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
