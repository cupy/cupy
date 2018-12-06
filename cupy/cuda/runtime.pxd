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

    cudaDevAttrMaxThreadsPerBlock             = 1  # Maximum number of threads per block
    cudaDevAttrMaxBlockDimX                   = 2  # Maximum block dimension X
    cudaDevAttrMaxBlockDimY                   = 3  # Maximum block dimension Y
    cudaDevAttrMaxBlockDimZ                   = 4  # Maximum block dimension Z
    cudaDevAttrMaxGridDimX                    = 5  # Maximum grid dimension X
    cudaDevAttrMaxGridDimY                    = 6  # Maximum grid dimension Y
    cudaDevAttrMaxGridDimZ                    = 7  # Maximum grid dimension Z
    cudaDevAttrMaxSharedMemoryPerBlock        = 8  # Maximum shared memory available per block in bytes
    cudaDevAttrTotalConstantMemory            = 9  # Memory available on device for __constant__ variables in a CUDA C kernel in bytes
    cudaDevAttrWarpSize                       = 10 # Warp size in threads
    cudaDevAttrMaxPitch                       = 11 # Maximum pitch in bytes allowed by memory copies
    cudaDevAttrMaxRegistersPerBlock           = 12 # Maximum number of 32-bit registers available per block
    cudaDevAttrClockRate                      = 13 # Peak clock frequency in kilohertz
    cudaDevAttrTextureAlignment               = 14 # Alignment requirement for textures
    cudaDevAttrGpuOverlap                     = 15 # Device can possibly copy memory and execute a kernel concurrently
    cudaDevAttrMultiProcessorCount            = 16 # Number of multiprocessors on device
    cudaDevAttrKernelExecTimeout              = 17 # Specifies whether there is a run time limit on kernels
    cudaDevAttrIntegrated                     = 18 # Device is integrated with host memory
    cudaDevAttrCanMapHostMemory               = 19 # Device can map host memory into CUDA address space
    cudaDevAttrComputeMode                    = 20 # Compute mode (See ::cudaComputeMode for details)
    cudaDevAttrMaxTexture1DWidth              = 21 # Maximum 1D texture width
    cudaDevAttrMaxTexture2DWidth              = 22 # Maximum 2D texture width
    cudaDevAttrMaxTexture2DHeight             = 23 # Maximum 2D texture height
    cudaDevAttrMaxTexture3DWidth              = 24 # Maximum 3D texture width
    cudaDevAttrMaxTexture3DHeight             = 25 # Maximum 3D texture height
    cudaDevAttrMaxTexture3DDepth              = 26 # Maximum 3D texture depth
    cudaDevAttrMaxTexture2DLayeredWidth       = 27 # Maximum 2D layered texture width
    cudaDevAttrMaxTexture2DLayeredHeight      = 28 # Maximum 2D layered texture height
    cudaDevAttrMaxTexture2DLayeredLayers      = 29 # Maximum layers in a 2D layered texture
    cudaDevAttrSurfaceAlignment               = 30 # Alignment requirement for surfaces
    cudaDevAttrConcurrentKernels              = 31 # Device can possibly execute multiple kernels concurrently
    cudaDevAttrEccEnabled                     = 32 # Device has ECC support enabled
    cudaDevAttrPciBusId                       = 33 # PCI bus ID of the device
    cudaDevAttrPciDeviceId                    = 34 # PCI device ID of the device
    cudaDevAttrTccDriver                      = 35 # Device is using TCC driver model
    cudaDevAttrMemoryClockRate                = 36 # Peak memory clock frequency in kilohertz
    cudaDevAttrGlobalMemoryBusWidth           = 37 # Global memory bus width in bits
    cudaDevAttrL2CacheSize                    = 38 # Size of L2 cache in bytes
    cudaDevAttrMaxThreadsPerMultiProcessor    = 39 # Maximum resident threads per multiprocessor
    cudaDevAttrAsyncEngineCount               = 40 # Number of asynchronous engines
    cudaDevAttrUnifiedAddressing              = 41 # Device shares a unified address space with the host
    cudaDevAttrMaxTexture1DLayeredWidth       = 42 # Maximum 1D layered texture width
    cudaDevAttrMaxTexture1DLayeredLayers      = 43 # Maximum layers in a 1D layered texture
    cudaDevAttrMaxTexture2DGatherWidth        = 45 # Maximum 2D texture width if cudaArrayTextureGather is set
    cudaDevAttrMaxTexture2DGatherHeight       = 46 # Maximum 2D texture height if cudaArrayTextureGather is set
    cudaDevAttrMaxTexture3DWidthAlt           = 47 # Alternate maximum 3D texture width
    cudaDevAttrMaxTexture3DHeightAlt          = 48 # Alternate maximum 3D texture height
    cudaDevAttrMaxTexture3DDepthAlt           = 49 # Alternate maximum 3D texture depth
    cudaDevAttrPciDomainId                    = 50 # PCI domain ID of the device
    cudaDevAttrTexturePitchAlignment          = 51 # Pitch alignment requirement for textures
    cudaDevAttrMaxTextureCubemapWidth         = 52 # Maximum cubemap texture width/height
    cudaDevAttrMaxTextureCubemapLayeredWidth  = 53 # Maximum cubemap layered texture width/height
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54 # Maximum layers in a cubemap layered texture
    cudaDevAttrMaxSurface1DWidth              = 55 # Maximum 1D surface width
    cudaDevAttrMaxSurface2DWidth              = 56 # Maximum 2D surface width
    cudaDevAttrMaxSurface2DHeight             = 57 # Maximum 2D surface height
    cudaDevAttrMaxSurface3DWidth              = 58 # Maximum 3D surface width
    cudaDevAttrMaxSurface3DHeight             = 59 # Maximum 3D surface height
    cudaDevAttrMaxSurface3DDepth              = 60 # Maximum 3D surface depth
    cudaDevAttrMaxSurface1DLayeredWidth       = 61 # Maximum 1D layered surface width
    cudaDevAttrMaxSurface1DLayeredLayers      = 62 # Maximum layers in a 1D layered surface
    cudaDevAttrMaxSurface2DLayeredWidth       = 63 # Maximum 2D layered surface width
    cudaDevAttrMaxSurface2DLayeredHeight      = 64 # Maximum 2D layered surface height
    cudaDevAttrMaxSurface2DLayeredLayers      = 65 # Maximum layers in a 2D layered surface
    cudaDevAttrMaxSurfaceCubemapWidth         = 66 # Maximum cubemap surface width
    cudaDevAttrMaxSurfaceCubemapLayeredWidth  = 67 # Maximum cubemap layered surface width
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68 # Maximum layers in a cubemap layered surface
    cudaDevAttrMaxTexture1DLinearWidth        = 69 # Maximum 1D linear texture width
    cudaDevAttrMaxTexture2DLinearWidth        = 70 # Maximum 2D linear texture width
    cudaDevAttrMaxTexture2DLinearHeight       = 71 # Maximum 2D linear texture height
    cudaDevAttrMaxTexture2DLinearPitch        = 72 # Maximum 2D linear texture pitch in bytes
    cudaDevAttrMaxTexture2DMipmappedWidth     = 73 # Maximum mipmapped 2D texture width
    cudaDevAttrMaxTexture2DMipmappedHeight    = 74 # Maximum mipmapped 2D texture height
    cudaDevAttrComputeCapabilityMajor         = 75 # Major compute capability version number
    cudaDevAttrComputeCapabilityMinor         = 76 # Minor compute capability version number
    cudaDevAttrMaxTexture1DMipmappedWidth     = 77 # Maximum mipmapped 1D texture width
    cudaDevAttrStreamPrioritiesSupported      = 78 # Device supports stream priorities
    cudaDevAttrGlobalL1CacheSupported         = 79 # Device supports caching globals in L1
    cudaDevAttrLocalL1CacheSupported          = 80 # Device supports caching locals in L1
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81 # Maximum shared memory available per multiprocessor in bytes
    cudaDevAttrMaxRegistersPerMultiprocessor  = 82 # Maximum number of 32-bit registers available per multiprocessor
    cudaDevAttrManagedMemory                  = 83 # Device can allocate managed memory on this system
    cudaDevAttrIsMultiGpuBoard                = 84 # Device is on a multi-GPU board
    cudaDevAttrMultiGpuBoardGroupID           = 85 # Unique identifier for a group of devices on the same multi-GPU board
    cudaDevAttrHostNativeAtomicSupported      = 86 # Link between the device and the host supports native atomic operations
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87 # Ratio of single precision performance (in floating-point operations per second) to double precision performance
    cudaDevAttrPageableMemoryAccess           = 88 # Device supports coherently accessing pageable memory without calling cudaHostRegister on it
    cudaDevAttrConcurrentManagedAccess        = 89 # Device can coherently access managed memory concurrently with the CPU
    cudaDevAttrComputePreemptionSupported     = 90 # Device supports Compute Preemption
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91 # Device can access host registered memory at the same virtual address as the CPU
    cudaDevAttrReserved92                     = 92,
    cudaDevAttrReserved93                     = 93,
    cudaDevAttrReserved94                     = 94,
    cudaDevAttrCooperativeLaunch              = 95 # Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel
    cudaDevAttrCooperativeMultiDeviceLaunch   = 96 # Device can participate in cooperative kernels launched via ::cudaLaunchCooperativeKernelMultiDevice
    cudaDevAttrMaxSharedMemoryPerBlockOptin   = 97 # The maximum optin shared memory per block. This value may vary by chip. See ::cudaFuncSetAttribute
    cudaDevAttrCanFlushRemoteWrites           = 98 # Device supports flushing of outstanding remote writes.
    cudaDevAttrHostRegisterSupported          = 99 # Device supports host memory registration via ::cudaHostRegister.
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100 # Device accesses pageable memory via the host's page tables.
    cudaDevAttrDirectManagedMemAccessFromHost = 101 # Host can directly access managed memory on the device without migration.

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
