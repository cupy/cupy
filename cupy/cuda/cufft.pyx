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
from cupy.cuda cimport runtime
from cupy.cuda cimport stream as stream_module
from cupy.cuda.device import Device
from cupy.cuda.stream import Event, Stream


cdef object _thread_local = threading.local()


cpdef get_current_plan():
    """Get current cuFFT plan.

    Returns:
        None or cupy.cuda.cufft.Plan1d or cupy.cuda.cufft.PlanNd
    """
    if not hasattr(_thread_local, '_current_plan'):
        _thread_local._current_plan = None
    return _thread_local._current_plan


cdef enum:
    # Actually, this is 64, but it's undocumented. For the sake
    # of safety, let us use 16, which agrees with the cuFFT doc.
    MAX_CUDA_DESCRIPTOR_GPUS = 16


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

    # Version
    Result cufftGetVersion(int* version)

    # cufftXt data types
    ctypedef struct XtArrayDesc 'cudaXtDesc':
        int version
        int nGPUs
        int GPUs[MAX_CUDA_DESCRIPTOR_GPUS]
        void* data[MAX_CUDA_DESCRIPTOR_GPUS]
        size_t size[MAX_CUDA_DESCRIPTOR_GPUS]
        void* cudaXtState

    ctypedef enum XtSubFormat 'cufftXtSubFormat':
        CUFFT_XT_FORMAT_INPUT
        CUFFT_XT_FORMAT_OUTPUT
        CUFFT_XT_FORMAT_INPLACE
        CUFFT_XT_FORMAT_INPLACE_SHUFFLED
        CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED

    ctypedef struct XtArray 'cudaLibXtDesc':
        int version
        XtArrayDesc* descriptor
        int library
        XtSubFormat subFormat
        void* libDescriptor

    ctypedef enum XtCopyType 'cufftXtCopyType':
        CUFFT_COPY_HOST_TO_DEVICE = 0x00
        CUFFT_COPY_DEVICE_TO_HOST = 0x01
        CUFFT_COPY_DEVICE_TO_DEVICE = 0x02

    # cufftXt functions
    Result cufftXtSetGPUs(Handle plan, int nGPUs, int* gpus)
    Result cufftXtSetWorkArea(Handle plan, void** workArea)
    Result cufftXtMemcpy(Handle plan, void *dst, void *src, XtCopyType type)
    Result cufftXtExecDescriptorC2C(Handle plan, XtArray* idata,
                                    XtArray* odata, int direction)
    Result cufftXtExecDescriptorZ2Z(Handle plan, XtArray* idata,
                                    XtArray* odata, int direction)


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


cpdef size_t getVersion() except? -1:
    cdef int version
    result = cufftGetVersion(&version)
    check_result(result)
    return version


# This is necessary for single-batch transforms: "when batch is one, data is
# left in the GPU memory in a permutation of the natural output", see
# https://docs.nvidia.com/cuda/cufft/index.html#multiple-GPU-cufft-intermediate-helper  # NOQA
cdef _reorder_buffers(Handle plan, intptr_t xtArr, list xtArr_buffer):
    cdef int i, result, nGPUs
    cdef intptr_t temp_xtArr
    cdef XtArray* temp_arr
    cdef XtArray* arr
    cdef list gpus = []
    cdef list sizes = []
    cdef list temp_xtArr_buffer

    arr = <XtArray*>xtArr
    nGPUs = len(xtArr_buffer)
    assert nGPUs == arr.descriptor.nGPUs

    # allocate another buffer to prepare for order conversion
    for i in range(nGPUs):
        gpus.append(arr.descriptor.GPUs[i])
        sizes.append(arr.descriptor.size[i])
    temp_xtArr, temp_xtArr_buffer = _XtMalloc(gpus, sizes,
                                              CUFFT_XT_FORMAT_INPLACE)
    temp_arr = <XtArray*>temp_xtArr

    # Make a device copy to bring the data from the permuted order back to
    # the natural order. Note that this works because after FFT
    # arr.subFormat is silently changed to CUFFT_XT_FORMAT_INPLACE_SHUFFLED
    with nogil:
        result = cufftXtMemcpy(plan, <void*>temp_arr, <void*>arr,
                               CUFFT_COPY_DEVICE_TO_DEVICE)
    check_result(result)

    for i in range(nGPUs):
        # swap MemoryPointer in xtArr_buffer
        temp = temp_xtArr_buffer[i]
        temp_xtArr_buffer[i] = xtArr_buffer[i]
        xtArr_buffer[i] = temp

        # swap pointer in xtArr
        arr.descriptor.data[i] = temp_arr.descriptor.data[i]
        assert arr.descriptor.size[i] == temp_arr.descriptor.size[i]
        temp_arr.descriptor.data[i] = NULL
        temp_arr.descriptor.size[i] = 0

    # temp_xtArr now points to the old data, which is now in temp_xtArr_buffer
    # and will be deallocated after this line (out of scope)
    _XtFree(temp_xtArr)


# We need to manage the buffers ourselves in order to 1. avoid excessive,
# uncessary memory usage, and 2. use CuPy's memory pool.
# This is meant to replace cufftXtMalloc().
cdef _XtMalloc(list gpus, list sizes, XtSubFormat fmt):
    cdef XtArrayDesc* xtArr_desc
    cdef XtArray* xtArr
    cdef list xtArr_buffer = []
    cdef int i, nGPUs
    cdef size_t size

    assert len(gpus) == len(sizes)
    nGPUs = len(gpus)
    xtArr_desc = <XtArrayDesc*>PyMem_Malloc(sizeof(XtArrayDesc))
    xtArr = <XtArray*>PyMem_Malloc(sizeof(XtArray))
    c_memset(xtArr_desc, 0, sizeof(XtArrayDesc))
    c_memset(xtArr, 0, sizeof(XtArray))

    xtArr_desc.nGPUs = nGPUs
    for i, (gpu, size) in enumerate(zip(gpus, sizes)):
        with Device(gpu):
            buf = memory.alloc(size)
        assert gpu == buf.device_id
        xtArr_buffer.append(buf)
        xtArr_desc.GPUs[i] = gpu
        xtArr_desc.data[i] = <void*>buf.ptr
        xtArr_desc.size[i] = size

    xtArr.descriptor = xtArr_desc
    xtArr.subFormat = fmt

    return <intptr_t>xtArr, xtArr_buffer


# This is meant to replace cufftXtFree().
# We only free the C structs. The underlying GPU buffers are deallocated when
# going out of scope.
cdef _XtFree(intptr_t ptr):
    cdef XtArray* xtArr = <XtArray*>ptr
    cdef XtArrayDesc* xtArr_desc = xtArr.descriptor
    PyMem_Free(xtArr_desc)
    PyMem_Free(xtArr)


class Plan1d(object):
    def __init__(self, int nx, int fft_type, int batch, *,
                 devices=None, out=None):
        cdef Handle plan
        cdef bint use_multi_gpus = 0 if devices is None else 1

        with nogil:
            result = cufftCreate(&plan)
            if result == 0:
                result = cufftSetAutoAllocation(plan, 0)
        check_result(result)

        # set plan, work_area, gpus, streams, and events
        if not use_multi_gpus:
            self._single_gpu_get_plan(plan, nx, fft_type, batch)
        else:
            self._multi_gpu_get_plan(plan, nx, fft_type, batch, devices, out)

        self.nx = nx
        self.fft_type = fft_type
        self.batch = batch

        self._use_multi_gpus = use_multi_gpus
        self.batch_share = None
        self.xtArr = <intptr_t>0  # pointer to metadata for multi-GPU buffer
        self.xtArr_buffer = None  # actual multi-GPU intermediate buffer

    def _single_gpu_get_plan(self, Handle plan, int nx, int fft_type,
                             int batch):
        cdef int result
        cdef size_t work_size
        cdef intptr_t ptr

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

        self.plan = plan
        self.work_area = work_area  # this is for cuFFT plan
        self.gpus = None
        self.streams = None
        self.events = None

    def _multi_gpu_get_plan(self, Handle plan, int nx, int fft_type, int batch,
                            devices, out):
        cdef int nGPUs, min_len, result
        cdef vector.vector[int] gpus
        cdef vector.vector[size_t] work_size
        cdef list work_area = []
        cdef list streams = []
        cdef list events = []
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
            raise ValueError('\"devices\" should be an int or a list of int.')
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
            else:  # nGPU = 16
                min_len = 1024
            if nx < min_len:
                raise ValueError('For {} GPUs, the array length must be at '
                                 'least {} (you have {}).'
                                 .format(nGPUs, min_len, nx))
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
                stream = Stream()
                event = Event()
            work_area.append(buf)
            streams.append(stream)
            events.append(event)
            work_area_ptr.push_back(<void*>buf.ptr)
        with nogil:
            result = cufftXtSetWorkArea(plan, work_area_ptr.data())
        check_result(result)

        self.plan = plan
        self.work_area = work_area  # this is for cuFFT plan
        self.gpus = list(gpus)
        self.streams = streams      # for async, overlapped copies
        self.events = events        # for async, overlapped copies

    def __del__(self):
        cdef Handle plan = self.plan
        cdef int dev = runtime.getDevice()

        with nogil:
            result = cufftDestroy(plan)
        check_result(result)

        # cuFFT bug: after cufftDestroy(), the current device is mistakenly
        # set to the last device in self.gpus, so we must correct it. See
        # https://github.com/cupy/cupy/pull/2644#discussion_r347567899 and
        # NVIDIA internal ticket 2761341.
        runtime.setDevice(dev)

        if self.xtArr != 0:
            _XtFree(self.xtArr)
            self.xtArr = 0

    def __enter__(self):
        _thread_local._current_plan = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _thread_local._current_plan = None

    def fft(self, a, out, direction):
        if self._use_multi_gpus:
            # Note: mult-GPU plans cannot set stream
            self._multi_gpu_fft(a, out, direction)
        else:
            self._single_gpu_fft(a, out, direction)

    def _single_gpu_fft(self, a, out, direction):
        cdef Handle plan = self.plan
        cdef intptr_t stream

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
        cdef intptr_t ptr
        cdef list xtArr_buffer, share, sizes
        cdef int i, nGPUs, count
        cdef XtSubFormat fmt

        # First, get the buffers:
        # We need to manage the buffers ourselves in order to avoid excessive,
        # uncessary memory usage. Note that these buffers are used for in-place
        # transforms, and are re-used (lifetime tied to the plan).

        # TODO(leofang): revisit this when NumPy support is added
        # if isinstance(a, cupy.ndarray) or isinstance(a, numpy.ndarray):
        if isinstance(a, cupy.ndarray):
            if self.xtArr == 0 and self.xtArr_buffer is None:
                nGPUs = len(self.gpus)

                # this is the rule for distributing the workload
                if self.batch > 1:
                    share = [self.batch // nGPUs] * nGPUs
                    for i in range(self.batch % nGPUs):
                        share[i] += 1
                else:
                    share = [1.0 / nGPUs] * nGPUs
                sizes = [int(share[i] * self.nx * a.dtype.itemsize)
                         for i in range(nGPUs)]

                # get buffer
                if isinstance(a, cupy.ndarray):
                    fmt = CUFFT_XT_FORMAT_INPLACE
                # TODO(leofang): revisit this when NumPy support is added
                # else:  # from numpy
                #     fmt = CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED
                ptr, xtArr_buffer = _XtMalloc(self.gpus, sizes, fmt)

                xtArr = <XtArray*>ptr
                xtArr_desc = xtArr.descriptor
                assert xtArr_desc.nGPUs == nGPUs

                self.batch_share = share
                self.xtArr = ptr
                self.xtArr_buffer = xtArr_buffer  # kept to ensure lifetime
            else:
                # After FFT the subFormat flag is silently changed to
                # CUFFT_XT_FORMAT_INPLACE_SHUFFLED. For reuse we must correct
                # it, otherwise in the next run we would encounter
                # CUFFT_INVALID_TYPE!
                ptr = self.xtArr
                xtArr = <XtArray*>ptr
                if self.batch == 1:
                    if isinstance(a, cupy.ndarray):
                        fmt = CUFFT_XT_FORMAT_INPLACE
                    # TODO(leofang): revisit this when NumPy support is added
                    # else:  # from numpy
                    #     fmt = CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED
                    xtArr.subFormat = fmt
        elif isinstance(a, list):
            # TODO(leofang): For users running Plan1d.fft() (bypassing all
            # checks in cupy.fft.fft), they are allowed to send in a list of
            # ndarrays, each of which is on a different GPU. Then, no data
            # copy is needed, just replace the pointers in the descriptor.
            raise NotImplementedError('User-managed buffer area is not yet '
                                      'supported.')
        else:
            raise ValueError('Impossible to reach.')

    def _multi_gpu_memcpy(self, a, str action):
        cdef Handle plan = self.plan
        cdef list xtArr_buffer, share
        cdef int start, nGPUs, i, count, result
        cdef XtArray* arr
        cdef intptr_t ptr, ptr2
        cdef size_t size

        # TODO(leofang): revisit this when NumPy support is added
        # if isinstance(a, cupy.ndarray) or isinstance(a, numpy.ndarray):
        if isinstance(a, cupy.ndarray):
            start = 0
            b = a.ravel()
            assert b.flags['OWNDATA'] is False
            assert self.xtArr_buffer is not None
            ptr = self.xtArr
            arr = <XtArray*>ptr
            xtArr_buffer = self.xtArr_buffer
            nGPUs = len(self.gpus)
            share = self.batch_share

            if action == 'scatter':
                if isinstance(a, cupy.ndarray):
                    for i in range(nGPUs):
                        count = int(share[i] * self.nx)
                        size = count * b.dtype.itemsize
                        curr_stream = self.streams[i]
                        if i == 0:
                            # When we come here, another stream could still be
                            # copying data for us, so we wait patiently...
                            outer_stream = stream_module.get_current_stream()
                            outer_stream.synchronize()
                        xtArr_buffer[i].copy_from_device_async(
                            b[start:start+count].data, size, curr_stream)
                        if i != 0:
                            prev_event = self.events[i-1]
                            curr_stream.wait_event(prev_event)
                        curr_event = self.events[i]
                        curr_event.record(curr_stream)
                        if i == nGPUs - 1:
                            curr_event.synchronize()
                        start += count
                    assert start == b.size
                # TODO(leofang): revisit this when NumPy support is added
                # else:  # numpy
                #     ptr2 = b.ctypes.data
                #     with nogil:
                #         result = cufftXtMemcpy(
                #             plan, <void*>arr, <void*>ptr2,
                #             CUFFT_COPY_HOST_TO_DEVICE)
                #     check_result(result)
            elif action == 'gather':
                if isinstance(a, cupy.ndarray):
                    if self.batch == 1:
                        _reorder_buffers(plan, self.xtArr, xtArr_buffer)

                    for i in range(nGPUs):
                        count = int(share[i] * self.nx)
                        size = count * b.dtype.itemsize
                        curr_stream = self.streams[i]
                        if i == 0:
                            # When we come here, another stream could still be
                            # copying data for us, so we wait patiently...
                            outer_stream = stream_module.get_current_stream()
                            outer_stream.synchronize()
                        b[start:start+count].data.copy_from_device_async(
                            xtArr_buffer[i], size, curr_stream)
                        if i != 0:
                            prev_event = self.events[i-1]
                            curr_stream.wait_event(prev_event)
                        curr_event = self.events[i]
                        curr_event.record(curr_stream)
                        if i == nGPUs - 1:
                            curr_event.synchronize()
                        start += count
                    assert start == b.size
                # TODO(leofang): revisit this when NumPy support is added
                # else:  # numpy
                #     ptr2 = b.ctypes.data
                #     with nogil:
                #         result = cufftXtMemcpy(
                #             plan, <void*>ptr2, <void*>arr,
                #             CUFFT_COPY_DEVICE_TO_HOST)
                #     check_result(result)
            else:
                raise ValueError

    def _multi_gpu_fft(self, a, out, direction):
        # When we arrive here, the normal CuPy call path ensures a and out
        # reside on the same GPU -> must distribute a to all of the GPUs
        self._multi_gpu_setup_buffer(a)

        # Next, copy data to buffer
        self._multi_gpu_memcpy(a, 'scatter')

        # Actual workhorses
        if self.fft_type == CUFFT_C2C:
            multi_gpu_execC2C(self.plan, self.xtArr, self.xtArr, direction)
        elif self.fft_type == CUFFT_Z2Z:
            multi_gpu_execZ2Z(self.plan, self.xtArr, self.xtArr, direction)
        else:
            raise ValueError

        # Gather the distributed outputs
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
