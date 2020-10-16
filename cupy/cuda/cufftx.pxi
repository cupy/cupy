# distutils: language = c++

from libc.stdint cimport intptr_t
from libcpp cimport vector

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
    Result set_callback(Handle plan, callbackType cb_type, bint cb_load)


cpdef createPlan(object plan):
    cdef Handle h
    cdef str plan_type
    cdef tuple plan_args
    cdef int result
    cdef Type fft_type
    cdef int batch

    # for Plan1d
    cdef int nx

    # for PlanNd
    cdef int ndim
    cdef vector.vector[int] shape_arr
    cdef tuple inembed
    cdef vector.vector[int] inembed_arr
    cdef int istride
    cdef int idist
    cdef tuple onembed
    cdef vector.vector[int] onembed_arr
    cdef int ostride
    cdef int odist
    cdef int* shape_ptr
    cdef int* inembed_ptr
    cdef int* onembed_ptr

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
        ndim = len(plan_args[0])
        shape_arr = plan_args[0]
        shape_ptr = shape_arr.data()

        inembed = plan_args[1]
        if inembed is None:
            inembed_ptr = NULL
        else:
            inembed_arr = inembed
            inembed_ptr = inembed_arr.data()
        istride = plan_args[2]
        idist = plan_args[3]

        onembed = plan_args[4]
        if onembed is None:
            onembed_ptr = NULL
        else:
            onembed_arr = onembed
            onembed_ptr = onembed_arr.data()
        ostride = plan_args[5]
        odist = plan_args[6]

        fft_type = <Type>(plan_args[7])
        batch = <int>(plan_args[8])

        with nogil:
            result = cufftPlanMany(&h, ndim, shape_ptr,
                                   inembed_ptr, istride, idist,
                                   onembed_ptr, ostride, odist,
                                   fft_type, batch)
        check_result(result)
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
