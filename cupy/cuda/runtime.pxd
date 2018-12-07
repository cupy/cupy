###############################################################################
# Types
###############################################################################

cdef class PointerAttributes:
    cdef:
        public int device
        public size_t devicePointer
        public size_t hostPointer
        public int isManaged
        public int memoryType


cdef extern from *:
    ctypedef int Error 'cudaError_t'
    ctypedef int DataType 'cudaDataType'


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

    errorMemoryAllocation = 2

    cudaDevAttrMaxThreadsPerBlock             = 1
    cudaDevAttrMaxBlockDimX                   = 2
    cudaDevAttrMaxBlockDimY                   = 3
    cudaDevAttrMaxBlockDimZ                   = 4
    cudaDevAttrMaxGridDimX                    = 5
    cudaDevAttrMaxGridDimY                    = 6
    cudaDevAttrMaxGridDimZ                    = 7
    cudaDevAttrMaxSharedMemoryPerBlock        = 8
    cudaDevAttrTotalConstantMemory            = 9
    cudaDevAttrWarpSize                       = 10
    cudaDevAttrMaxPitch                       = 11
    cudaDevAttrMaxRegistersPerBlock           = 12
    cudaDevAttrClockRate                      = 13
    cudaDevAttrTextureAlignment               = 14
    cudaDevAttrGpuOverlap                     = 15
    cudaDevAttrMultiProcessorCount            = 16
    cudaDevAttrKernelExecTimeout              = 17
    cudaDevAttrIntegrated                     = 18
    cudaDevAttrCanMapHostMemory               = 19
    cudaDevAttrComputeMode                    = 20
    cudaDevAttrMaxTexture1DWidth              = 21
    cudaDevAttrMaxTexture2DWidth              = 22
    cudaDevAttrMaxTexture2DHeight             = 23
    cudaDevAttrMaxTexture3DWidth              = 24
    cudaDevAttrMaxTexture3DHeight             = 25
    cudaDevAttrMaxTexture3DDepth              = 26
    cudaDevAttrMaxTexture2DLayeredWidth       = 27
    cudaDevAttrMaxTexture2DLayeredHeight      = 28
    cudaDevAttrMaxTexture2DLayeredLayers      = 29
    cudaDevAttrSurfaceAlignment               = 30
    cudaDevAttrConcurrentKernels              = 31
    cudaDevAttrEccEnabled                     = 32
    cudaDevAttrPciBusId                       = 33
    cudaDevAttrPciDeviceId                    = 34
    cudaDevAttrTccDriver                      = 35
    cudaDevAttrMemoryClockRate                = 36
    cudaDevAttrGlobalMemoryBusWidth           = 37
    cudaDevAttrL2CacheSize                    = 38
    cudaDevAttrMaxThreadsPerMultiProcessor    = 39
    cudaDevAttrAsyncEngineCount               = 40
    cudaDevAttrUnifiedAddressing              = 41
    cudaDevAttrMaxTexture1DLayeredWidth       = 42
    cudaDevAttrMaxTexture1DLayeredLayers      = 43
    cudaDevAttrMaxTexture2DGatherWidth        = 45
    cudaDevAttrMaxTexture2DGatherHeight       = 46
    cudaDevAttrMaxTexture3DWidthAlt           = 47
    cudaDevAttrMaxTexture3DHeightAlt          = 48
    cudaDevAttrMaxTexture3DDepthAlt           = 49
    cudaDevAttrPciDomainId                    = 50
    cudaDevAttrTexturePitchAlignment          = 51
    cudaDevAttrMaxTextureCubemapWidth         = 52
    cudaDevAttrMaxTextureCubemapLayeredWidth  = 53
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54
    cudaDevAttrMaxSurface1DWidth              = 55
    cudaDevAttrMaxSurface2DWidth              = 56
    cudaDevAttrMaxSurface2DHeight             = 57
    cudaDevAttrMaxSurface3DWidth              = 58
    cudaDevAttrMaxSurface3DHeight             = 59
    cudaDevAttrMaxSurface3DDepth              = 60
    cudaDevAttrMaxSurface1DLayeredWidth       = 61
    cudaDevAttrMaxSurface1DLayeredLayers      = 62
    cudaDevAttrMaxSurface2DLayeredWidth       = 63
    cudaDevAttrMaxSurface2DLayeredHeight      = 64
    cudaDevAttrMaxSurface2DLayeredLayers      = 65
    cudaDevAttrMaxSurfaceCubemapWidth         = 66
    cudaDevAttrMaxSurfaceCubemapLayeredWidth  = 67
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68
    cudaDevAttrMaxTexture1DLinearWidth        = 69
    cudaDevAttrMaxTexture2DLinearWidth        = 70
    cudaDevAttrMaxTexture2DLinearHeight       = 71
    cudaDevAttrMaxTexture2DLinearPitch        = 72
    cudaDevAttrMaxTexture2DMipmappedWidth     = 73
    cudaDevAttrMaxTexture2DMipmappedHeight    = 74
    cudaDevAttrComputeCapabilityMajor         = 75
    cudaDevAttrComputeCapabilityMinor         = 76
    cudaDevAttrMaxTexture1DMipmappedWidth     = 77
    cudaDevAttrStreamPrioritiesSupported      = 78
    cudaDevAttrGlobalL1CacheSupported         = 79
    cudaDevAttrLocalL1CacheSupported          = 80
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81
    cudaDevAttrMaxRegistersPerMultiprocessor  = 82
    cudaDevAttrManagedMemory                  = 83
    cudaDevAttrIsMultiGpuBoard                = 84
    cudaDevAttrMultiGpuBoardGroupID           = 85
    cudaDevAttrHostNativeAtomicSupported      = 86
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87
    cudaDevAttrPageableMemoryAccess           = 88
    cudaDevAttrConcurrentManagedAccess        = 89
    cudaDevAttrComputePreemptionSupported     = 90
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91
    cudaDevAttrReserved92                     = 92
    cudaDevAttrReserved93                     = 93
    cudaDevAttrReserved94                     = 94
    cudaDevAttrCooperativeLaunch              = 95
    cudaDevAttrCooperativeMultiDeviceLaunch   = 96
    cudaDevAttrMaxSharedMemoryPerBlockOptin   = 97
    cudaDevAttrCanFlushRemoteWrites           = 98
    cudaDevAttrHostRegisterSupported          = 99
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100
    cudaDevAttrDirectManagedMemAccessFromHost = 101

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
cpdef int getDeviceCount() except? -1
cpdef setDevice(int device)
cpdef deviceSynchronize()

cpdef int deviceCanAccessPeer(int device, int peerDevice) except? -1
cpdef deviceEnablePeerAccess(int peerDevice)


###############################################################################
# Memory management
###############################################################################

cpdef size_t malloc(size_t size) except? 0
cpdef size_t mallocManaged(size_t size, unsigned int flags=*) except? 0
cpdef size_t hostAlloc(size_t size, unsigned int flags) except? 0
cpdef free(size_t ptr)
cpdef freeHost(size_t ptr)
cpdef memGetInfo()
cpdef memcpy(size_t dst, size_t src, size_t size, int kind)
cpdef memcpyAsync(size_t dst, size_t src, size_t size, int kind,
                  size_t stream)
cpdef memcpyPeer(size_t dst, int dstDevice, size_t src, int srcDevice,
                 size_t size)
cpdef memcpyPeerAsync(size_t dst, int dstDevice,
                      size_t src, int srcDevice,
                      size_t size, size_t stream)
cpdef memset(size_t ptr, int value, size_t size)
cpdef memsetAsync(size_t ptr, int value, size_t size, size_t stream)
cpdef memPrefetchAsync(size_t devPtr, size_t count, int dstDevice,
                       size_t stream)
cpdef memAdvise(size_t devPtr, int count, int advice, int device)
cpdef PointerAttributes pointerGetAttributes(size_t ptr)


###############################################################################
# Stream and Event
###############################################################################

cpdef size_t streamCreate() except? 0
cpdef size_t streamCreateWithFlags(unsigned int flags) except? 0
cpdef streamDestroy(size_t stream)
cpdef streamSynchronize(size_t stream)
cpdef streamAddCallback(size_t stream, callback, size_t arg,
                        unsigned int flags=*)
cpdef streamQuery(size_t stream)
cpdef streamWaitEvent(size_t stream, size_t event, unsigned int flags=*)
cpdef size_t eventCreate() except? 0
cpdef size_t eventCreateWithFlags(unsigned int flags) except? 0
cpdef eventDestroy(size_t event)
cpdef float eventElapsedTime(size_t start, size_t end) except? 0
cpdef eventQuery(size_t event)
cpdef eventRecord(size_t event, size_t stream)
cpdef eventSynchronize(size_t event)


##############################################################################
# util
##############################################################################

cdef _ensure_context()
