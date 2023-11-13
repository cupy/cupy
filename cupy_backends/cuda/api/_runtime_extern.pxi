from cupy_backends.cuda.api cimport driver


cdef extern from '../../cupy_backend_runtime.h' nogil:
    # Error handling
    const char* cudaGetErrorName(Error error)
    const char* cudaGetErrorString(Error error)
    int cudaGetLastError()

    # Initialization
    int cudaDriverGetVersion(int* driverVersion)
    int cudaRuntimeGetVersion(int* runtimeVersion)

    # Device operations
    int cudaGetDevice(int* device)
    int cudaDeviceGetAttribute(int* value, DeviceAttr attr, int device)
    int cudaDeviceGetByPCIBusId(int* device, const char* pciBusId)
    int cudaDeviceGetPCIBusId(char* pciBusId, int len, int device)
    int cudaGetDeviceProperties(DeviceProp* prop, int device)
    int cudaGetDeviceCount(int* count)
    int cudaSetDevice(int device)
    int cudaDeviceSynchronize()

    int cudaDeviceCanAccessPeer(int* canAccessPeer, int device,
                                int peerDevice)
    int cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
    int cudaDeviceDisablePeerAccess(int peerDevice)

    int cudaDeviceGetLimit(size_t* value, Limit limit)
    int cudaDeviceSetLimit(Limit limit, size_t value)

    # IPC
    int cudaIpcCloseMemHandle(void* devPtr)
    int cudaIpcGetEventHandle(IpcEventHandle* handle, driver.Event event)
    int cudaIpcGetMemHandle(IpcMemHandle*, void* devPtr)
    int cudaIpcOpenEventHandle(driver.Event* event, IpcEventHandle handle)
    int cudaIpcOpenMemHandle(void** devPtr, IpcMemHandle handle,
                             unsigned int  flags)

    # Memory management
    int cudaMalloc(void** devPtr, size_t size)
    int cudaMallocManaged(void** devPtr, size_t size, unsigned int flags)
    int cudaMalloc3DArray(Array* array, const ChannelFormatDesc* desc,
                          Extent extent, unsigned int flags)
    int cudaMallocArray(Array* array, const ChannelFormatDesc* desc,
                        size_t width, size_t height, unsigned int flags)
    int cudaMallocAsync(void**, size_t, driver.Stream)
    int cudaMallocFromPoolAsync(void**, size_t, MemPool, driver.Stream)
    int cudaHostAlloc(void** ptr, size_t size, unsigned int flags)
    int cudaHostRegister(void *ptr, size_t size, unsigned int flags)
    int cudaHostUnregister(void *ptr)
    int cudaFree(void* devPtr)
    int cudaFreeHost(void* ptr)
    int cudaFreeArray(Array array)
    int cudaFreeAsync(void*, driver.Stream)
    int cudaMemGetInfo(size_t* free, size_t* total)
    int cudaMemcpy(void* dst, const void* src, size_t count,
                   MemoryKind kind)
    int cudaMemcpyAsync(void* dst, const void* src, size_t count,
                        MemoryKind kind, driver.Stream stream)
    int cudaMemcpyPeer(void* dst, int dstDevice, const void* src,
                       int srcDevice, size_t count)
    int cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src,
                            int srcDevice, size_t count,
                            driver.Stream stream)
    int cudaMemcpy2DFromArray(void* dst, size_t dpitch, Array src,
                              size_t wOffset, size_t hOffset, size_t width,
                              size_t height, MemoryKind kind)
    int cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, Array src,
                                   size_t wOffset, size_t hOffset,
                                   size_t width, size_t height,
                                   MemoryKind kind, driver.Stream stream)
    int cudaMemcpy2DToArray(Array dst, size_t wOffset, size_t hOffset,
                            const void* src, size_t spitch, size_t width,
                            size_t height, MemoryKind kind)
    int cudaMemcpy2DToArrayAsync(Array dst, size_t wOffset, size_t hOffset,
                                 const void* src, size_t spitch, size_t width,
                                 size_t height, MemoryKind kind,
                                 driver.Stream stream)
    int cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                     size_t width, size_t height, MemoryKind kind)
    int cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src,
                          size_t spitch, size_t width, size_t height,
                          MemoryKind kind, driver.Stream stream)
    int cudaMemcpy3D(Memcpy3DParms* Memcpy3DParmsPtr)
    int cudaMemcpy3DAsync(Memcpy3DParms* Memcpy3DParmsPtr,
                          driver.Stream stream)
    int cudaMemset(void* devPtr, int value, size_t count)
    int cudaMemsetAsync(void* devPtr, int value, size_t count,
                        driver.Stream stream)
    int cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice,
                             driver.Stream stream)
    int cudaMemAdvise(const void *devPtr, size_t count,
                      MemoryAdvise advice, int device)
    int cudaDeviceGetDefaultMemPool(MemPool*, int)
    int cudaDeviceGetMemPool(MemPool*, int)
    int cudaDeviceSetMemPool(int, MemPool)
    int cudaMemPoolCreate(MemPool*, _MemPoolProps*)
    int cudaMemPoolDestroy(MemPool)
    int cudaMemPoolTrimTo(MemPool, size_t)
    int cudaMemPoolGetAttribute(MemPool, MemPoolAttr, void*)
    int cudaMemPoolSetAttribute(MemPool, MemPoolAttr, void*)
    int cudaPointerGetAttributes(_PointerAttributes* attributes,
                                 const void* ptr)
    Extent make_cudaExtent(size_t w, size_t h, size_t d)
    Pos make_cudaPos(size_t x, size_t y, size_t z)
    PitchedPtr make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz)

    # Stream and Event
    int cudaStreamCreate(driver.Stream* pStream)
    int cudaStreamCreateWithFlags(driver.Stream* pStream,
                                  unsigned int flags)
    int cudaStreamDestroy(driver.Stream stream)
    int cudaStreamSynchronize(driver.Stream stream)
    int cudaStreamAddCallback(driver.Stream stream, StreamCallback callback,
                              void* userData, unsigned int flags)
    int cudaLaunchHostFunc(driver.Stream stream, HostFn fn, void* userData)
    int cudaStreamQuery(driver.Stream stream)
    int cudaStreamWaitEvent(driver.Stream stream, driver.Event event,
                            unsigned int flags)
    int cudaStreamBeginCapture(driver.Stream stream, StreamCaptureMode mode)
    int cudaStreamEndCapture(driver.Stream stream, Graph*)
    int cudaStreamIsCapturing(driver.Stream stream, StreamCaptureStatus*)
    int cudaEventCreate(driver.Event* event)
    int cudaEventCreateWithFlags(driver.Event* event, unsigned int flags)
    int cudaEventDestroy(driver.Event event)
    int cudaEventElapsedTime(float* ms, driver.Event start,
                             driver.Event end)
    int cudaEventQuery(driver.Event event)
    int cudaEventRecord(driver.Event event, driver.Stream stream)
    int cudaEventSynchronize(driver.Event event)

    # Texture
    int cudaCreateTextureObject(TextureObject* pTexObject,
                                const ResourceDesc* pResDesc,
                                const TextureDesc* pTexDesc,
                                const ResourceViewDesc* pResViewDesc)
    int cudaDestroyTextureObject(TextureObject texObject)
    int cudaGetChannelDesc(ChannelFormatDesc* desc, Array array)
    int cudaGetTextureObjectResourceDesc(ResourceDesc* desc, TextureObject obj)
    int cudaGetTextureObjectTextureDesc(TextureDesc* desc, TextureObject obj)

    # Surface
    int cudaCreateSurfaceObject(SurfaceObject* pSurObject,
                                const ResourceDesc* pResDesc)
    int cudaDestroySurfaceObject(SurfaceObject surObject)

    # Graph
    int cudaGraphDestroy(Graph graph)
    int cudaGraphExecDestroy(GraphExec graph)
    int cudaGraphInstantiate(GraphExec*, Graph, GraphNode*, char*, size_t)
    int cudaGraphLaunch(GraphExec, driver.Stream)
    int cudaGraphUpload(GraphExec, driver.Stream)

    # Constants
    int cudaDevAttrComputeCapabilityMajor
    int cudaDevAttrComputeCapabilityMinor

    # Error code
    int cudaErrorMemoryAllocation
    int cudaErrorInvalidValue
    int cudaErrorPeerAccessAlreadyEnabled
    int cudaErrorContextIsDestroyed
    int cudaErrorInvalidResourceHandle


cdef extern from '../../cupy_profiler.h' nogil:
    # Profiler
    int cudaProfilerStart()
    int cudaProfilerStop()
