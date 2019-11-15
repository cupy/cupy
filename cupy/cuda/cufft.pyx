cimport cython  # NOQA
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport intptr_t
from libc.string cimport memset as c_memset
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
        int version
        int nGPUs
        int GPUs[MAX_CUDA_DESCRIPTOR_GPUS]
        void* data[MAX_CUDA_DESCRIPTOR_GPUS]
        size_t size[MAX_CUDA_DESCRIPTOR_GPUS]
        void* cudaXtState

    ctypedef struct XtArray 'cudaLibXtDesc':
        int version
        XtArrayDesc* descriptor
        int library
        int subFormat
        void* libDescriptor

    ctypedef enum XtCopyType 'cufftXtCopyType':
        CUFFT_COPY_HOST_TO_DEVICE = 0x00
        CUFFT_COPY_DEVICE_TO_HOST = 0x01
        CUFFT_COPY_DEVICE_TO_DEVICE = 0x02

    ctypedef enum XtSubFormat 'cufftXtSubFormat':
        CUFFT_XT_FORMAT_INPUT
        CUFFT_XT_FORMAT_OUTPUT
        CUFFT_XT_FORMAT_INPLACE
        CUFFT_XT_FORMAT_INPLACE_SHUFFLED

    # for debugging
    ctypedef struct Xt1dFactors 'cufftXt1dFactors':
        long long int size
        long long int stringCount
        long long int stringLength
        long long int substringLength
        long long int factor1
        long long int factor2
        long long int stringMask
        long long int substringMask
        long long int factor1Mask
        long long int factor2Mask
        int stringShift
        int substringShift
        int factor1Shift
        int factor2Shift

    ctypedef enum XtQueryType 'cufftXtQueryType':
        CUFFT_QUERY_1D_FACTORS = 0x00

    # cufftXt functions
    Result cufftXtSetGPUs(Handle plan, int nGPUs, int* gpus)
    Result cufftXtSetWorkArea(Handle plan, void** workArea)
    Result cufftXtMalloc(Handle plan, XtArray** arr, XtSubFormat format)
    Result cufftXtFree(XtArray* arr)
    Result cufftXtMemcpy(Handle plan, void *dst, void *src, XtCopyType type)
    Result cufftXtExecDescriptorC2C(Handle plan, XtArray* idata,
                                    XtArray* odata, int direction)
    Result cufftXtExecDescriptorZ2Z(Handle plan, XtArray* idata,
                                    XtArray* odata, int direction)
    Result cufftXtQueryPlan(Handle plan, void* query, XtQueryType queryType)


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


cdef void _swap_mem_ptr(Handle plan, intptr_t old_xtArr, list old_xtArr_buffer):
    cdef int result, nGPUs
    cdef XtArray* arr
    cdef XtArray* old_arr
    cdef list garbage = []  # freed when out of scope

    # make a device copy to bring the data back to natural order
    nGPUs = len(old_xtArr_buffer)
    with nogil:
        result = cufftXtMalloc(plan, &arr, CUFFT_XT_FORMAT_INPLACE)
        if result == 0:
            result = cufftXtMemcpy(plan, <void*>arr, <void*>old_xtArr, CUFFT_COPY_DEVICE_TO_DEVICE)
    check_result(result)

    # Now the old buffer is not needed, but we still want its list container,
    # so we swap the pointers and free the "new" XtArray
    old_arr = <XtArray*>old_xtArr
    for i in range(nGPUs):
        # swap MemoryPointer in old_xtArr_buffer
        raw_mem = memory.BaseMemory()
        raw_mem.ptr = <intptr_t>(arr.descriptor.data[i])
        raw_mem.size = arr.descriptor.size[i]
        raw_mem.device_id = arr.descriptor.GPUs[i]
        mem = memory.MemoryPointer(raw_mem, 0)
        garbage.append(old_xtArr_buffer[i])
        old_xtArr_buffer[i] = mem

        # swap pointer in old_xtArr
        old_arr.descriptor.data[i] = arr.descriptor.data[i]
        arr.descriptor.data[i] = NULL
        arr.descriptor.size[i] = 0
    with nogil:
        result = cufftXtFree(arr)
    check_result(result)


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
            gpus = None
        else:
            plan, work_area, gpus = self._multi_gpu_get_plan(
                plan, nx, fft_type, batch, devices, out)

        self.nx = nx
        self.fft_type = fft_type
        self.batch = batch
        self.plan = plan
        self.work_area = work_area  # this is for cuFFT plan

        self.use_multi_gpus = use_multi_gpus
        self.gpus = gpus
        self.batch_share = None
        self.xtArr = <intptr_t>0  # pointer to metadata for multi-GPU buffer
        self.xtArr_buffer = None  # actual multi-GPU intermediate buffer

    def _multi_gpu_get_plan(self, Handle plan, int nx, int fft_type, int batch,
                            devices, out):
        cdef int nGPUs, min_len
        cdef vector.vector[int] gpus
        cdef vector.vector[size_t] work_size
        cdef list work_area = []
        cdef vector.vector[void*] work_area_ptr

        # some sanity checks
        if fft_type != CUFFT_C2C and fft_type != CUFFT_Z2Z:
            raise ValueError('Currently for multiple GPUs only C2C and Z2Z are'
                             ' supported.')
        if isinstance(devices, list):
            nGPUs = len(devices)
            for i in range(nGPUs):
                gpus.push_back(devices[i])
        elif isinstance(devices, int):
            nGPUs = devices
            for i in range(nGPUs):
                gpus.push_back(i)
        else:
            raise ValueError("\"devices\" should be an int or a list of int.")
        if batch == 1:
            if (nx & (nx - 1)) != 0:
                raise ValueError('For multi-GPU FFT with batch = 1, the array '
                                 'size must be a power of 2.')
            if nGPUs not in (2, 4, 8, 16):
                raise ValueError('For multi-GPU FFT with batch = 1, the number'
                                 ' of devices must be 2, 4, 8, or 16.')
            if nGPUs in (2, 4):
                min_len = 64
            elif nGPUs == 8:
                min_len = 128
            else:  # nGPU = 64
                min_len = 1024
            if nx < min_len:
                raise ValueError('For {} GPUs, the array length must be at '
                                 'least {} (you have {}).'.format(nGPUs,
                                 min_len, nx))
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

        return plan, work_area, list(gpus)

    def __del__(self):
        cdef Handle plan = self.plan
        cdef intptr_t ptr
        cdef XtArrayDesc* xtArr_desc
        cdef XtArray* xtArr

        with nogil:
            result = cufftDestroy(plan)
        check_result(result)

        if self.xtArr != 0:
            # WARNING: <XtArray*>self.xtAtr is the address of the object
            # self.xtAtr, not that of the actual XtArray struct...
            ptr = self.xtArr
            xtArr = <XtArray*>ptr
            xtArr_desc = xtArr.descriptor
            PyMem_Free(xtArr_desc)
            PyMem_Free(xtArr)

    def __enter__(self):
        _thread_local._current_plan = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _thread_local._current_plan = None

    def fft(self, a, out, direction):
        cdef Handle plan = self.plan

        if self.use_multi_gpus:
            # Note: mult-GPU plans cannot set stream
            self._multi_gpu_fft(a, out, direction)
            return

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

    def _multi_gpu_setup_buffer(self, a):
        cdef XtArrayDesc* xtArr_desc
        cdef XtArray* xtArr
        cdef list xtArr_buffer, share
        cdef int i, nGPUs, size, count

        # First, get the buffers:
        # We need to manage the buffers ourselves in order to avoid excessive,
        # uncessary memory usage. Note that these buffers are used for in-place
        # transforms, and are re-used (lifetime tied to the plan).
        if isinstance(a, cupy.ndarray):
            if self.xtArr == 0:
                xtArr_desc = <XtArrayDesc*>PyMem_Malloc(sizeof(XtArrayDesc))
                c_memset(xtArr_desc, 0, sizeof(XtArrayDesc))
                xtArr = <XtArray*>PyMem_Malloc(sizeof(XtArray))
                c_memset(xtArr, 0, sizeof(XtArray))
                nGPUs = len(self.gpus)

                # this is the rule for distributing the workload
                if self.batch > 1:
                    share = [self.batch // nGPUs] * nGPUs
                    for i in range(self.batch % nGPUs):
                        share[i] += 1
                else:
                    share = [1.0 / nGPUs] * nGPUs

                xtArr_buffer = []
                xtArr_desc.nGPUs = nGPUs
                for i in range(nGPUs):
                    size = int(share[i] * self.nx * a.dtype.itemsize)
                    with Device(self.gpus[i]):
                        buf = memory.alloc(size)
                    xtArr_buffer.append(buf)
                    xtArr_desc.GPUs[i] = self.gpus[i]
                    xtArr_desc.data[i] = <void*>buf.ptr
                    xtArr_desc.size[i] = size

                xtArr.descriptor = xtArr_desc
                xtArr.subFormat = CUFFT_XT_FORMAT_INPLACE
                self.batch_share = share
                self.xtArr = <intptr_t>xtArr
                self.xtArr_buffer = xtArr_buffer  # kept to ensure lifetime
        elif isinstance(a, list):
            # TODO(leofang): For users running Plan1d.fft() (bypassing all
            # checks in cupy.fft.fft), they are allowed to send in a list of
            # ndarrays, each of which is on a different GPU. Then, no data
            # copy is needed, just replace the pointers in the descriptor.
            raise NotImplementedError('User-managed buffer area is not yet '
                                      'supported.')
        else:
            raise ValueError('Impossible to reach.')

        # Next, copy data to buffer
        self._multi_gpu_memcpy(a, 'scatter')

    def _multi_gpu_memcpy(self, a, str action):
        cdef Handle plan = self.plan
        cdef list xtArr_buffer, share
        cdef int start, nGPUs, i, count, size
        cdef XtArray* arr
        cdef intptr_t ptr

        if isinstance(a, cupy.ndarray):
            start = 0
            b = a.ravel()
            assert b.flags['OWNDATA'] is False
            assert self.xtArr_buffer is not None
            ptr = self.xtArr
            xtArr_buffer = self.xtArr_buffer
            nGPUs = len(self.gpus)
            share = self.batch_share

            if action == 'scatter':
                for i in range(nGPUs):
                    count = int(share[i] * self.nx)
                    size = count * b.dtype.itemsize
                    xtArr_buffer[i].copy_from_device(
                        b[start:start+count].data, size)
                    start += count
                assert start == b.size
            else:  # = 'gather'
                if self.batch == 1:
                    _swap_mem_ptr(plan, self.xtArr, xtArr_buffer)

                for i in range(nGPUs):
                    count = int(share[i] * self.nx)
                    size = count * b.dtype.itemsize
                    b[start:start+count].data.copy_from_device(
                        xtArr_buffer[i], size)
                    start += count
                assert start == b.size

    def _multi_gpu_fft(self, a, out, direction):
        # When we arrive here, the normal CuPy call path ensures a and out
        # reside on the same GPU -> must distribute a to all of the GPUs
        self._multi_gpu_setup_buffer(a)

        if self.fft_type == CUFFT_C2C:
            multi_gpu_execC2C(self.plan, self.xtArr, self.xtArr, direction)
        elif self.fft_type == CUFFT_Z2Z:
            multi_gpu_execZ2Z(self.plan, self.xtArr, self.xtArr, direction)
        else:
            raise ValueError

        self._multi_gpu_memcpy(out, 'gather')

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

    # for debugging
    def _multi_gpu_check_data_integraty(self, intptr_t in_arr=0):
        cdef intptr_t ptr
        cdef XtArray* xtArr
        cdef Xt1dFactors factors

        if in_arr == 0:
            # WARNING: <XtArray*>self.ptr gets the address of the object
            # self.ptr, not that of the actual struct...
            ptr = <intptr_t>self.xtArr
            xtArr = <XtArray*>ptr
        else:
            xtArr = <XtArray*>in_arr
        print(xtArr.version)
        print(xtArr.descriptor.version)
        print(xtArr.descriptor.nGPUs)
        for i in range(MAX_CUDA_DESCRIPTOR_GPUS):
            print(xtArr.descriptor.GPUs[i], end=' ')
        print()
        for i in range(MAX_CUDA_DESCRIPTOR_GPUS):
            print(<intptr_t>xtArr.descriptor.data[i], end=' ')
        print()
        for i in range(MAX_CUDA_DESCRIPTOR_GPUS):
            print(xtArr.descriptor.size[i], end=' ')
        print()
        print(<intptr_t>xtArr.descriptor.cudaXtState)
        print(xtArr.library)
        print(xtArr.subFormat)
        print(<intptr_t>xtArr.libDescriptor)

        if self.batch == 1:
            # Seems the query only works in this corner case; for other cases, a Floating point exception
            # is raised here for some reason...
            result = cufftXtQueryPlan(<Handle>self.plan, <void*>&factors, CUFFT_QUERY_1D_FACTORS)
            check_result(result)
            print(factors.size)
            print(factors.stringCount)
            print(factors.stringLength)
            print(factors.substringLength)
            print(factors.factor1)
            print(factors.factor2)
            print(factors.stringMask)
            print(factors.substringMask)
            print(factors.factor1Mask)
            print(factors.factor2Mask)
            print(factors.stringShift)
            print(factors.substringShift)
            print(factors.factor1Shift)
            print(factors.factor2Shift)


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


cpdef multi_gpu_execC2C(Handle plan, intptr_t idata, intptr_t odata,
                        int direction):
    with nogil:
        result = cufftXtExecDescriptorC2C(plan, <XtArray*>idata,
                                          <XtArray*>odata, direction)
    check_result(result)


cpdef multi_gpu_execZ2Z(Handle plan, intptr_t idata, intptr_t odata,
                        int direction):
    with nogil:
        result = cufftXtExecDescriptorZ2Z(plan, <XtArray*>idata,
                                          <XtArray*>odata, direction)
    check_result(result)
