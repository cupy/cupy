cimport cython  # NOQA
from libc.stdint cimport intptr_t
from libcpp cimport vector
import numpy
import threading

import cupy
from cupy.cuda cimport driver
from cupy.cuda cimport memory
from cupy.cuda cimport stream as stream_module
from cupy.cuda.device import Device


cdef object _thread_local = threading.local()


cpdef get_current_plan():
    """Get current cuFFT plan.

    Returns:
        None or cupy.cuda.cufft.Plan1d or cupy.cuda.cufft.PlanNd
    """
    if not hasattr(_thread_local, '_current_plan'):
        _thread_local._current_plan = None
    return _thread_local._current_plan


cdef extern from 'cupy_cufft.h' nogil:
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

    # cufftXt data types
    ctypedef struct XtArrayDesc 'cudaXtDesc':
        # MAX_CUDA_DESCRIPTOR_GPUS is undocumented, so we put 16 here
        int version
        int nGPUs
        int GPUs[16]
        void* data[16]
        size_t size[16]
        void* cudaXtState

    ctypedef struct XtArray 'cudaLibXtDesc':
        int version
        XtArrayDesc* descriptor
        int library
        int subFormat
        void* libDescriptor

    # cufftXt functions
    Result cufftXtSetGPUs(Handle plan, int nGPUs, int* gpus)
    Result cufftXtSetWorkArea(Handle plan, void** workArea)
    Result cufftXtMalloc(Handle plan, XtArray** arr, int format)
    Result cufftXtFree(XtArray* arr)


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

    def __reduce__(self):
        return (type(self), (self.result,))


@cython.profile(False)
cpdef inline check_result(int result):
    if result != 0:
        raise CuFFTError(result)


class Plan1d(object):
    def __init__(self, int nx, int fft_type, int batch, *,
                 use_multi_gpus=False, devices=None, out=None):
        cdef Handle plan
        cdef size_t work_size
        cdef intptr_t ptr

        with nogil:
            result = cufftCreate(&plan)
            if result == 0:
                result = cufftSetAutoAllocation(plan, 0)
        check_result(result)

        if not use_multi_gpus:
            with nogil:
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
            ptr = work_area.ptr
            with nogil:
                result = cufftSetWorkArea(plan, <void*>(ptr))
            check_result(result)
        else:
            plan, work_area = self._get_multi_gpu_plan(
                plan, nx, fft_type, batch, devices, out)

        self.nx = nx
        self.fft_type = fft_type
        self.batch = batch
        self.plan = plan
        self.work_area = work_area

    def _get_multi_gpu_plan(self, Handle plan, int nx, int fft_type, int batch,
                            devices, out):
        cdef int nGPUs
        cdef vector.vector[int] gpus
        cdef vector.vector[size_t] work_size
        cdef list work_area = []
        cdef vector.vector[void*] work_area_ptr

        if isinstance(devices, list):
            nGPUs = len(devices)
            for i in range(nGPUs):
                gpus.push_back(devices[i])
        elif isinstance(devices, int):
            nGPUs = devices
            for i in range(nGPUs):
                gpus.push_back(i)
            #devices_ptr = devices.data()
        else:
            raise ValueError("\"devices\" should be an int or a list of int.")
        work_size.resize(nGPUs)

        with nogil:
            result = cufftXtSetGPUs(plan, nGPUs, gpus.data())
            if result == 0:
                result = cufftMakePlan1d(plan, nx, <Type>fft_type, batch,
                                         work_size.data())

        # cufftMakePlan1d uses large memory when nx has large divisor.
        # See https://github.com/cupy/cupy/issues/1063
        if result == 2:
            cupy.get_default_memory_pool().free_all_blocks()
            with nogil:
                result = cufftMakePlan1d(plan, nx, <Type>fft_type, batch,
                                         work_size.data())
        check_result(result)

        for i in range(nGPUs):
            with Device(gpus[i]):
                buf = memory.alloc(work_size[i])
                work_area.append(buf)
                work_area_ptr.push_back(<void*>buf.ptr)
        with nogil:
            result = cufftXtSetWorkArea(plan, work_area_ptr.data())
        check_result(result)

        return plan, work_area

    def __del__(self):
        cdef Handle plan = self.plan
        with nogil:
            result = cufftDestroy(plan)
        check_result(result)

    def __enter__(self):
        _thread_local._current_plan = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _thread_local._current_plan = None

    def fft(self, a, out, direction):
        cdef Handle plan = self.plan
        stream = stream_module.get_current_stream_ptr()
        with nogil:
            result = cufftSetStream(plan, <driver.Stream>stream)
        check_result(result)
        if self.fft_type == CUFFT_C2C:
            execC2C(plan, a.data.ptr, out.data.ptr, direction)
        elif self.fft_type == CUFFT_R2C:
            execR2C(plan, a.data.ptr, out.data.ptr)
        elif self.fft_type == CUFFT_C2R:
            execC2R(plan, a.data.ptr, out.data.ptr)
        elif self.fft_type == CUFFT_Z2Z:
            execZ2Z(plan, a.data.ptr, out.data.ptr, direction)
        elif self.fft_type == CUFFT_D2Z:
            execD2Z(plan, a.data.ptr, out.data.ptr)
        else:
            execZ2D(plan, a.data.ptr, out.data.ptr)

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
                ('out must have shape {}.').format(shape))
        if out.dtype != dtype:
            raise ValueError(
                'out dtype mismatch: found {}, expected {}'.format(
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

        with nogil:
            result = cufftCreate(&plan)
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

    def __enter__(self):
        _thread_local._current_plan = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _thread_local._current_plan = None

    def fft(self, a, out, direction):
        cdef Handle plan = self.plan
        stream = stream_module.get_current_stream_ptr()
        with nogil:
            result = cufftSetStream(plan, <driver.Stream>stream)
        check_result(result)
        if self.fft_type == CUFFT_C2C:
            execC2C(plan, a.data.ptr, out.data.ptr, direction)
        elif self.fft_type == CUFFT_Z2Z:
            execZ2Z(plan, a.data.ptr, out.data.ptr, direction)
        else:
            raise NotImplementedError('only C2C and Z2Z implemented')

    def get_output_array(self, a, order='C'):
        shape = list(a.shape)
        if self.fft_type == CUFFT_C2C:
            return cupy.empty(shape, numpy.complex64, order=order)
        elif self.fft_type == CUFFT_Z2Z:
            return cupy.empty(shape, numpy.complex128, order=order)
        else:
            raise NotImplementedError('only C2C and Z2Z implemented')

    def check_output_array(self, a, out):
        if out is a:
            return
        if out.dtype != a.dtype:
            raise ValueError('output dtype mismatch')
        if not ((out.flags.f_contiguous == a.flags.f_contiguous) and
                (out.flags.c_contiguous == a.flags.c_contiguous)):
            raise ValueError('output contiguity mismatch')
        if out.shape != a.shape:
            raise ValueError('output shape mismatch')


cpdef execC2C(Handle plan, intptr_t idata, intptr_t odata, int direction):
    with nogil:
        result = cufftExecC2C(plan, <Complex*>idata, <Complex*>odata,
                              direction)
    check_result(result)


cpdef execR2C(Handle plan, intptr_t idata, intptr_t odata):
    with nogil:
        result = cufftExecR2C(plan, <Float*>idata, <Complex*>odata)
    check_result(result)


cpdef execC2R(Handle plan, intptr_t idata, intptr_t odata):
    with nogil:
        result = cufftExecC2R(plan, <Complex*>idata, <Float*>odata)
    check_result(result)


cpdef execZ2Z(Handle plan, intptr_t idata, intptr_t odata, int direction):
    with nogil:
        result = cufftExecZ2Z(plan, <DoubleComplex*>idata,
                              <DoubleComplex*>odata, direction)
    check_result(result)


cpdef execD2Z(Handle plan, intptr_t idata, intptr_t odata):
    with nogil:
        result = cufftExecD2Z(plan, <Double*>idata, <DoubleComplex*>odata)
    check_result(result)


cpdef execZ2D(Handle plan, intptr_t idata, intptr_t odata):
    with nogil:
        result = cufftExecZ2D(plan, <DoubleComplex*>idata, <Double*>odata)
    check_result(result)
