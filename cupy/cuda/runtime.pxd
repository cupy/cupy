from libc.stdint cimport intptr_t, uintmax_t


###############################################################################
# Types
###############################################################################

cdef class PointerAttributes:
    cdef:
        public int device
        public intptr_t devicePointer
        public intptr_t hostPointer


cdef extern from *:
    ctypedef int Error 'cudaError_t'
    ctypedef int DataType 'cudaDataType'

    ctypedef int ChannelFormatKind 'cudaChannelFormatKind'
    ctypedef struct ChannelFormatDesc 'cudaChannelFormatDesc':
        int x, y, z, w
        ChannelFormatKind f
    ctypedef uintmax_t TextureObject 'cudaTextureObject_t'
    ctypedef int ResourceType 'cudaResourceType'
    ctypedef int TextureAddressMode 'cudaTextureAddressMode'
    ctypedef int TextureFilterMode 'cudaTextureFilterMode'
    ctypedef int TextureReadMode 'cudaTextureReadMode'
    ctypedef struct ResourceViewDesc 'cudaResourceViewDesc'
    ctypedef void* Array 'cudaArray_t'
    ctypedef struct Extent 'cudaExtent':
        size_t width, height, depth
    ctypedef struct Pos 'cudaPos':
        size_t x, y, z
    ctypedef struct PitchedPtr 'cudaPitchedPtr':
        size_t pitch
        void* ptr
        size_t xsize, ysize
    ctypedef int MemoryKind 'cudaMemcpyKind'
    ctypedef void* MipmappedArray 'cudaMipmappedArray_t'

    # This is for the annoying nested struct cudaResourceDesc, which is not
    # perfectly supprted in Cython
    ctypedef struct _array:
        Array array

    ctypedef struct _mipmap:
        MipmappedArray mipmap

    ctypedef struct _linear:
        void* devPtr
        ChannelFormatDesc desc
        size_t sizeInBytes

    ctypedef struct _pitch2D:
        void* devPtr
        ChannelFormatDesc desc
        size_t width
        size_t height
        size_t pitchInBytes

    ctypedef union _res:
        _array array
        _mipmap mipmap
        _linear linear
        _pitch2D pitch2D

    ctypedef struct ResourceDesc 'cudaResourceDesc':
        int resType
        _res res
    # typedef cudaResourceDesc done

    ctypedef struct Memcpy3DParms 'cudaMemcpy3DParms':
        Array srcArray
        Pos srcPos
        PitchedPtr srcPtr

        Array dstArray
        Pos dstPos
        PitchedPtr dstPtr

        Extent extent
        MemoryKind kind

    ctypedef struct TextureDesc 'cudaTextureDesc':
        int addressMode[3]
        int filterMode
        int readMode
        int sRGB
        float borderColor[4]
        int normalizedCoords
        unsigned int maxAnisotropy
        # TODO(leofang): support mipmap?


###############################################################################
# Enum
###############################################################################

cpdef enum:
    memcpyHostToHost = 0
    memcpyHostToDevice = 1
    memcpyDeviceToHost = 2
    memcpyDeviceToDevice = 3
    memcpyDefault = 4

    cudaMemoryTypeHost = 1
    cudaMemoryTypeDevice = 2

    cudaMemAttachGlobal = 1
    cudaMemAttachHost = 2
    cudaMemAttachSingle = 4

    hostAllocDefault = 0
    hostAllocPortable = 1
    hostAllocMapped = 2
    hostAllocWriteCombined = 4

    cudaMemAdviseSetReadMostly = 1
    cudaMemAdviseUnsetReadMostly = 2
    cudaMemAdviseSetPreferredLocation = 3
    cudaMemAdviseUnsetPreferredLocation = 4
    cudaMemAdviseSetAccessedBy = 5
    cudaMemAdviseUnsetAccessedBy = 6

    streamDefault = 0
    streamNonBlocking = 1

    eventDefault = 0
    eventBlockingSync = 1
    eventDisableTiming = 2
    eventInterprocess = 4

    CUDA_R_32F = 0  # 32 bit real
    CUDA_R_64F = 1  # 64 bit real
    CUDA_R_16F = 2  # 16 bit real
    CUDA_R_8I = 3  # 8 bit real as a signed integer
    CUDA_C_32F = 4  # 32 bit complex
    CUDA_C_64F = 5  # 64 bit complex
    CUDA_C_16F = 6  # 16 bit complex
    CUDA_C_8I = 7  # 8 bit complex as a pair of signed integers
    CUDA_R_8U = 8  # 8 bit real as a signed integer
    CUDA_C_8U = 9  # 8 bit complex as a pair of signed integers

    cudaDevAttrMaxThreadsPerBlock = 1
    cudaDevAttrMaxBlockDimX = 2
    cudaDevAttrMaxBlockDimY = 3
    cudaDevAttrMaxBlockDimZ = 4
    cudaDevAttrMaxGridDimX = 5
    cudaDevAttrMaxGridDimY = 6
    cudaDevAttrMaxGridDimZ = 7
    cudaDevAttrMaxSharedMemoryPerBlock = 8
    cudaDevAttrTotalConstantMemory = 9
    cudaDevAttrWarpSize = 10
    cudaDevAttrMaxPitch = 11
    cudaDevAttrMaxRegistersPerBlock = 12
    cudaDevAttrClockRate = 13
    cudaDevAttrTextureAlignment = 14
    cudaDevAttrGpuOverlap = 15
    cudaDevAttrMultiProcessorCount = 16
    cudaDevAttrKernelExecTimeout = 17
    cudaDevAttrIntegrated = 18
    cudaDevAttrCanMapHostMemory = 19
    cudaDevAttrComputeMode = 20
    cudaDevAttrMaxTexture1DWidth = 21
    cudaDevAttrMaxTexture2DWidth = 22
    cudaDevAttrMaxTexture2DHeight = 23
    cudaDevAttrMaxTexture3DWidth = 24
    cudaDevAttrMaxTexture3DHeight = 25
    cudaDevAttrMaxTexture3DDepth = 26
    cudaDevAttrMaxTexture2DLayeredWidth = 27
    cudaDevAttrMaxTexture2DLayeredHeight = 28
    cudaDevAttrMaxTexture2DLayeredLayers = 29
    cudaDevAttrSurfaceAlignment = 30
    cudaDevAttrConcurrentKernels = 31
    cudaDevAttrEccEnabled = 32
    cudaDevAttrPciBusId = 33
    cudaDevAttrPciDeviceId = 34
    cudaDevAttrTccDriver = 35
    cudaDevAttrMemoryClockRate = 36
    cudaDevAttrGlobalMemoryBusWidth = 37
    cudaDevAttrL2CacheSize = 38
    cudaDevAttrMaxThreadsPerMultiProcessor = 39
    cudaDevAttrAsyncEngineCount = 40
    cudaDevAttrUnifiedAddressing = 41
    cudaDevAttrMaxTexture1DLayeredWidth = 42
    cudaDevAttrMaxTexture1DLayeredLayers = 43
    cudaDevAttrMaxTexture2DGatherWidth = 45
    cudaDevAttrMaxTexture2DGatherHeight = 46
    cudaDevAttrMaxTexture3DWidthAlt = 47
    cudaDevAttrMaxTexture3DHeightAlt = 48
    cudaDevAttrMaxTexture3DDepthAlt = 49
    cudaDevAttrPciDomainId = 50
    cudaDevAttrTexturePitchAlignment = 51
    cudaDevAttrMaxTextureCubemapWidth = 52
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54
    cudaDevAttrMaxSurface1DWidth = 55
    cudaDevAttrMaxSurface2DWidth = 56
    cudaDevAttrMaxSurface2DHeight = 57
    cudaDevAttrMaxSurface3DWidth = 58
    cudaDevAttrMaxSurface3DHeight = 59
    cudaDevAttrMaxSurface3DDepth = 60
    cudaDevAttrMaxSurface1DLayeredWidth = 61
    cudaDevAttrMaxSurface1DLayeredLayers = 62
    cudaDevAttrMaxSurface2DLayeredWidth = 63
    cudaDevAttrMaxSurface2DLayeredHeight = 64
    cudaDevAttrMaxSurface2DLayeredLayers = 65
    cudaDevAttrMaxSurfaceCubemapWidth = 66
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68
    cudaDevAttrMaxTexture1DLinearWidth = 69
    cudaDevAttrMaxTexture2DLinearWidth = 70
    cudaDevAttrMaxTexture2DLinearHeight = 71
    cudaDevAttrMaxTexture2DLinearPitch = 72
    cudaDevAttrMaxTexture2DMipmappedWidth = 73
    cudaDevAttrMaxTexture2DMipmappedHeight = 74
    # Use header version
    # cudaDevAttrComputeCapabilityMajor = 75
    # cudaDevAttrComputeCapabilityMinor = 76
    cudaDevAttrMaxTexture1DMipmappedWidth = 77
    cudaDevAttrStreamPrioritiesSupported = 78
    cudaDevAttrGlobalL1CacheSupported = 79
    cudaDevAttrLocalL1CacheSupported = 80
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81
    cudaDevAttrMaxRegistersPerMultiprocessor = 82
    cudaDevAttrManagedMemory = 83
    cudaDevAttrIsMultiGpuBoard = 84
    cudaDevAttrMultiGpuBoardGroupID = 85
    cudaDevAttrHostNativeAtomicSupported = 86
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87
    cudaDevAttrPageableMemoryAccess = 88
    cudaDevAttrConcurrentManagedAccess = 89
    cudaDevAttrComputePreemptionSupported = 90
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91
    cudaDevAttrReserved92 = 92
    cudaDevAttrReserved93 = 93
    cudaDevAttrReserved94 = 94
    cudaDevAttrCooperativeLaunch = 95
    cudaDevAttrCooperativeMultiDeviceLaunch = 96
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97
    cudaDevAttrCanFlushRemoteWrites = 98
    cudaDevAttrHostRegisterSupported = 99
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100
    cudaDevAttrDirectManagedMemAccessFromHost = 101

    # cudaChannelFormatKind
    cudaChannelFormatKindSigned = 0
    cudaChannelFormatKindUnsigned = 1
    cudaChannelFormatKindFloat = 2
    cudaChannelFormatKindNone = 3

    # cudaResourceType
    cudaResourceTypeArray = 0
    cudaResourceTypeMipmappedArray = 1
    cudaResourceTypeLinear = 2
    cudaResourceTypePitch2D = 3

    # cudaTextureAddressMode
    cudaAddressModeWrap = 0
    cudaAddressModeClamp = 1
    cudaAddressModeMirror = 2
    cudaAddressModeBorder = 3

    # cudaTextureFilterMode
    cudaFilterModePoint = 0
    cudaFilterModeLinear = 1

    # cudaTextureReadMode
    cudaReadModeElementType = 0
    cudaReadModeNormalizedFloat = 1


###############################################################################
# Error codes
###############################################################################

cdef extern from '../cuda/cupy_cuda.h':  # thru parent to import in core
    int cudaErrorMemoryAllocation
    int cudaErrorInvalidValue

###############################################################################
# Const value
###############################################################################
cpdef bint _is_hip_environment
cpdef int deviceAttributeComputeCapabilityMajor
cpdef int deviceAttributeComputeCapabilityMinor


###############################################################################
# Error handling
###############################################################################

cpdef check_status(int status)


###############################################################################
# Initialization
###############################################################################

cpdef int driverGetVersion() except? -1
cpdef int runtimeGetVersion() except? -1


###############################################################################
# Device and context operations
###############################################################################

cpdef int getDevice() except? -1
cpdef int deviceGetAttribute(int attrib, int device) except? -1
cpdef int deviceGetByPCIBusId(str pci_bus_id) except? -1
cpdef str deviceGetPCIBusId(int device)
cpdef int getDeviceCount() except? -1
cpdef setDevice(int device)
cpdef deviceSynchronize()

cpdef int deviceCanAccessPeer(int device, int peerDevice) except? -1
cpdef deviceEnablePeerAccess(int peerDevice)


###############################################################################
# Memory management
###############################################################################

cpdef intptr_t malloc(size_t size) except? 0
cpdef intptr_t mallocManaged(size_t size, unsigned int flags=*) except? 0
cpdef intptr_t malloc3DArray(intptr_t desc, size_t width, size_t height,
                             size_t depth, unsigned int flags=*) except? 0
cpdef intptr_t mallocArray(intptr_t desc, size_t width, size_t height,
                           unsigned int flags=*) except? 0
cpdef intptr_t hostAlloc(size_t size, unsigned int flags) except? 0
cpdef hostRegister(intptr_t ptr, size_t size, unsigned int flags)
cpdef hostUnregister(intptr_t ptr)
cpdef free(intptr_t ptr)
cpdef freeHost(intptr_t ptr)
cpdef freeArray(intptr_t ptr)
cpdef memGetInfo()
cpdef memcpy(intptr_t dst, intptr_t src, size_t size, int kind)
cpdef memcpyAsync(intptr_t dst, intptr_t src, size_t size, int kind,
                  intptr_t stream)
cpdef memcpyPeer(intptr_t dst, int dstDevice, intptr_t src, int srcDevice,
                 size_t size)
cpdef memcpyPeerAsync(intptr_t dst, int dstDevice,
                      intptr_t src, int srcDevice,
                      size_t size, intptr_t stream)
cpdef memcpy2D(intptr_t dst, size_t dpitch, intptr_t src, size_t spitch,
               size_t width, size_t height, MemoryKind kind)
cpdef memcpy2DAsync(intptr_t dst, size_t dpitch, intptr_t src, size_t spitch,
                    size_t width, size_t height, MemoryKind kind,
                    intptr_t stream)
cpdef memcpy2DFromArray(intptr_t dst, size_t dpitch, intptr_t src,
                        size_t wOffset, size_t hOffset, size_t width,
                        size_t height, int kind)
cpdef memcpy2DFromArrayAsync(intptr_t dst, size_t dpitch, intptr_t src,
                             size_t wOffset, size_t hOffset, size_t width,
                             size_t height, int kind, intptr_t stream)
cpdef memcpy2DToArray(intptr_t dst, size_t wOffset, size_t hOffset,
                      intptr_t src, size_t spitch, size_t width, size_t height,
                      int kind)
cpdef memcpy2DToArrayAsync(intptr_t dst, size_t wOffset, size_t hOffset,
                           intptr_t src, size_t spitch, size_t width,
                           size_t height, int kind, intptr_t stream)
cpdef memcpy3D(intptr_t Memcpy3DParmsPtr)
cpdef memcpy3DAsync(intptr_t Memcpy3DParmsPtr, intptr_t stream)
cpdef memset(intptr_t ptr, int value, size_t size)
cpdef memsetAsync(intptr_t ptr, int value, size_t size, intptr_t stream)
cpdef memPrefetchAsync(intptr_t devPtr, size_t count, int dstDevice,
                       intptr_t stream)
cpdef memAdvise(intptr_t devPtr, size_t count, int advice, int device)
cpdef PointerAttributes pointerGetAttributes(intptr_t ptr)


###############################################################################
# Stream and Event
###############################################################################

cpdef intptr_t streamCreate() except? 0
cpdef intptr_t streamCreateWithFlags(unsigned int flags) except? 0
cpdef streamDestroy(intptr_t stream)
cpdef streamSynchronize(intptr_t stream)
cpdef streamAddCallback(intptr_t stream, callback, intptr_t arg,
                        unsigned int flags=*)
cpdef streamQuery(intptr_t stream)
cpdef streamWaitEvent(intptr_t stream, intptr_t event, unsigned int flags=*)
cpdef intptr_t eventCreate() except? 0
cpdef intptr_t eventCreateWithFlags(unsigned int flags) except? 0
cpdef eventDestroy(intptr_t event)
cpdef float eventElapsedTime(intptr_t start, intptr_t end) except? 0
cpdef eventQuery(intptr_t event)
cpdef eventRecord(intptr_t event, intptr_t stream)
cpdef eventSynchronize(intptr_t event)


##############################################################################
# util
##############################################################################

cdef _ensure_context()


##############################################################################
# Texture
##############################################################################

cpdef uintmax_t createTextureObject(intptr_t ResDesc, intptr_t TexDesc)
cpdef destroyTextureObject(uintmax_t texObject)
cdef ChannelFormatDesc getChannelDesc(intptr_t array)
cdef ResourceDesc getTextureObjectResourceDesc(uintmax_t texobj)
cdef TextureDesc getTextureObjectTextureDesc(uintmax_t texobj)
cdef Extent make_Extent(size_t w, size_t h, size_t d)
cdef Pos make_Pos(size_t x, size_t y, size_t z)
cdef PitchedPtr make_PitchedPtr(intptr_t d, size_t p, size_t xsz, size_t ysz)
