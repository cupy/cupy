# distutils: language = c++

from libc.stdint cimport intptr_t

from cupy_backends.cuda.api cimport driver
from cupy.cuda.cufft cimport Handle, Type, Result
from cupy.cuda.cufft import check_result


cdef extern from 'cufftXt.h' nogil:
    # complex dtypes
    ctypedef struct Complex 'cufftComplex':
        float x, y

    ctypedef struct DoubleComplex 'cufftDoubleComplex':
        double x, y

    # cuFFT Helper Function
    Result cufftCreate(Handle *plan)
    Result cufftDestroy(Handle plan)

    # cuFFT Stream Function
    Result cufftSetStream(Handle plan, driver.Stream streamId)

    # cuFFT Plan Functions
    Result cufftPlan1d(Handle* plan, int nx, Type type, int batch)
    Result cufftPlanMany(Handle* plan, int rank, int *n, int *inembed,
                         int istride, int idist, int *onembed, int ostride,
                         int odist, Type type, int batch)

    # cuFFT Exec Function
    Result cufftExecC2C(Handle plan, Complex* idata, Complex* odata,
                        int direction)
    Result cufftExecR2C(Handle plan, float* idata, Complex* odata)
    Result cufftExecC2R(Handle plan, Complex* idata, float* odata)
    Result cufftExecZ2Z(Handle plan, DoubleComplex* idata,
                        DoubleComplex* odata, int direction)
    Result cufftExecD2Z(Handle plan, double* idata, DoubleComplex* odata)
    Result cufftExecZ2D(Handle plan, DoubleComplex* idata, double* odata)

    # Version
    Result cufftGetVersion(int* version)

    # cuFFT callback
    ctypedef enum callbackType 'cufftXtCallbackType':
        pass


cdef extern from 'cupy_cufftx.h' nogil:
    Result set_callback(Handle plan, int cb_type, bint cb_load)


cpdef createPlan(object plan):
    cdef Handle h
    cdef str plan_type
    cdef tuple plan_args
    cdef int result

    # for Plan1d
    cdef int nx
    cdef Type fft_type
    cdef int batch

    if not isinstance(plan, tuple):
        raise NotImplementedError
    plan_type, plan_args = plan
    if plan_type == 'Plan1d':
        nx = <int>(plan_args[0])
        fft_type = <Type>(plan_args[1])
        batch = <int>(plan_args[2])
        with nogil:
            result = cufftPlan1d(&h, nx, fft_type, batch)
        check_result(result)
    elif plan_type == 'PlanNd':
        pass
    #    result = cufftPlanMany(Handle plan, int rank, int *n, int *inembed,
    #                     int istride, int idist, int *onembed, int ostride,
    #                     int odist, Type type, int batch,
    #                     size_t *workSize)
    #    pN = PlanNd(plan.shape,
    #                plan.inembed, plan.istride, plan.idist,
    #                plan.onembed, plan.ostride, plan.odist,
    #                plan.fft_type, plan.batch, plan.order, plan.last_axis, plan.last_size)
    #    h = pN.handle
    else:
        raise NotImplementedError
    return <intptr_t>h, fft_type


cpdef destroyPlan(intptr_t plan):
    cdef Handle h = <Handle>plan
    cdef int result

    with nogil:
        result = cufftDestroy(h)
    check_result(result)


cpdef intptr_t setCallback(intptr_t plan, int cb_type, bint is_load):
    cdef Handle h = <Handle>plan
    cdef int result

    with nogil:
        result = set_callback(h, <callbackType>cb_type, is_load)
    check_result(result)


cpdef setStream(intptr_t plan, intptr_t stream):
    cdef Handle h = <Handle>plan
    cdef int result

    with nogil:
        result = cufftSetStream(<Handle>plan, <driver.Stream>stream)
    check_result(result)


cpdef execC2C(intptr_t plan, intptr_t idata, intptr_t odata, int direction):
    cdef Handle h = <Handle>plan
    cdef int result

    with nogil:
        result = cufftExecC2C(h, <Complex*>idata, <Complex*>odata,
                              direction)
    check_result(result)


cpdef execR2C(intptr_t plan, intptr_t idata, intptr_t odata):
    cdef Handle h = <Handle>plan
    cdef int result

    with nogil:
        result = cufftExecR2C(h, <float*>idata, <Complex*>odata)
    check_result(result)


cpdef execC2R(intptr_t plan, intptr_t idata, intptr_t odata):
    cdef Handle h = <Handle>plan
    cdef int result

    with nogil:
        result = cufftExecC2R(h, <Complex*>idata, <float*>odata)
    check_result(result)


cpdef execZ2Z(intptr_t plan, intptr_t idata, intptr_t odata, int direction):
    cdef Handle h = <Handle>plan
    cdef int result

    with nogil:
        result = cufftExecZ2Z(h, <DoubleComplex*>idata,
                              <DoubleComplex*>odata, direction)
    check_result(result)


cpdef execD2Z(intptr_t plan, intptr_t idata, intptr_t odata):
    cdef Handle h = <Handle>plan
    cdef int result

    with nogil:
        result = cufftExecD2Z(h, <double*>idata, <DoubleComplex*>odata)
    check_result(result)


cpdef execZ2D(intptr_t plan, intptr_t idata, intptr_t odata):
    cdef Handle h = <Handle>plan
    cdef int result

    with nogil:
        result = cufftExecZ2D(h, <DoubleComplex*>idata, <double*>odata)
    check_result(result)
