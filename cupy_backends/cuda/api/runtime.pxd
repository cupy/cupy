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
    ctypedef uintmax_t SurfaceObject 'cudaSurfaceObject_t'
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

    ctypedef int Limit 'cudaLimit'

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

    ctypedef struct IpcMemHandle 'cudaIpcMemHandle_t':
        unsigned char[64] reserved

    ctypedef struct IpcEventHandle 'cudaIpcEventHandle_t':
        unsigned char[64] reserved

    ctypedef struct cudaUUID 'cudaUUID_t':
        char bytes[16]

    ctypedef void* MemPool 'cudaMemPool_t'

    IF CUDA_VERSION >= 11000:
        # We can't use IF in the middle of structs declaration
        # to add or ignore fields in compile time so we have to
        # replicate the struct definition
        ctypedef struct DeviceProp 'cudaDeviceProp':
            char         name[256]
            cudaUUID     uuid
            char         luid[8]
            unsigned int luidDeviceNodeMask
            size_t       totalGlobalMem
            size_t       sharedMemPerBlock
            int          regsPerBlock
            int          warpSize
            size_t       memPitch
            int          maxThreadsPerBlock
            int          maxThreadsDim[3]
            int          maxGridSize[3]
            int          clockRate
            size_t       totalConstMem
            int          major
            int          minor
            size_t       textureAlignment
            size_t       texturePitchAlignment
            int          deviceOverlap
            int          multiProcessorCount
            int          kernelExecTimeoutEnabled
            int          integrated
            int          canMapHostMemory
            int          computeMode
            int          maxTexture1D
            int          maxTexture1DMipmap
            int          maxTexture1DLinear
            int          maxTexture2D[2]
            int          maxTexture2DMipmap[2]
            int          maxTexture2DLinear[3]
            int          maxTexture2DGather[2]
            int          maxTexture3D[3]
            int          maxTexture3DAlt[3]
            int          maxTextureCubemap
            int          maxTexture1DLayered[2]
            int          maxTexture2DLayered[3]
            int          maxTextureCubemapLayered[2]
            int          maxSurface1D
            int          maxSurface2D[2]
            int          maxSurface3D[3]
            int          maxSurface1DLayered[2]
            int          maxSurface2DLayered[3]
            int          maxSurfaceCubemap
            int          maxSurfaceCubemapLayered[2]
            size_t       surfaceAlignment
            int          concurrentKernels
            int          ECCEnabled
            int          pciBusID
            int          pciDeviceID
            int          pciDomainID
            int          tccDriver
            int          asyncEngineCount
            int          unifiedAddressing
            int          memoryClockRate
            int          memoryBusWidth
            int          l2CacheSize
            int          persistingL2CacheMaxSize  # CUDA 11.0 field
            int          maxThreadsPerMultiProcessor
            int          streamPrioritiesSupported
            int          globalL1CacheSupported
            int          localL1CacheSupported
            size_t       sharedMemPerMultiprocessor
            int          regsPerMultiprocessor
            int          managedMemory
            int          isMultiGpuBoard
            int          multiGpuBoardGroupID
            int          hostNativeAtomicSupported
            int          singleToDoublePrecisionPerfRatio
            int          pageableMemoryAccess
            int          concurrentManagedAccess
            int          computePreemptionSupported
            int          canUseHostPointerForRegisteredMem
            int          cooperativeLaunch
            int          cooperativeMultiDeviceLaunch
            size_t       sharedMemPerBlockOptin
            int          pageableMemoryAccessUsesHostPageTables
            int          directManagedMemAccessFromHost
            int          maxBlocksPerMultiProcessor  # CUDA 11.0 field
            int          accessPolicyMaxWindowSize  # CUDA 11.0 field
            size_t       reservedSharedMemPerBlock  # CUDA 11.0 field
    ELIF CUDA_VERSION >= 10000:
        ctypedef struct DeviceProp 'cudaDeviceProp':
            char         name[256]
            cudaUUID     uuid
            char         luid[8]
            unsigned int luidDeviceNodeMask
            size_t       totalGlobalMem
            size_t       sharedMemPerBlock
            int          regsPerBlock
            int          warpSize
            size_t       memPitch
            int          maxThreadsPerBlock
            int          maxThreadsDim[3]
            int          maxGridSize[3]
            int          clockRate
            size_t       totalConstMem
            int          major
            int          minor
            size_t       textureAlignment
            size_t       texturePitchAlignment
            int          deviceOverlap
            int          multiProcessorCount
            int          kernelExecTimeoutEnabled
            int          integrated
            int          canMapHostMemory
            int          computeMode
            int          maxTexture1D
            int          maxTexture1DMipmap
            int          maxTexture1DLinear
            int          maxTexture2D[2]
            int          maxTexture2DMipmap[2]
            int          maxTexture2DLinear[3]
            int          maxTexture2DGather[2]
            int          maxTexture3D[3]
            int          maxTexture3DAlt[3]
            int          maxTextureCubemap
            int          maxTexture1DLayered[2]
            int          maxTexture2DLayered[3]
            int          maxTextureCubemapLayered[2]
            int          maxSurface1D
            int          maxSurface2D[2]
            int          maxSurface3D[3]
            int          maxSurface1DLayered[2]
            int          maxSurface2DLayered[3]
            int          maxSurfaceCubemap
            int          maxSurfaceCubemapLayered[2]
            size_t       surfaceAlignment
            int          concurrentKernels
            int          ECCEnabled
            int          pciBusID
            int          pciDeviceID
            int          pciDomainID
            int          tccDriver
            int          asyncEngineCount
            int          unifiedAddressing
            int          memoryClockRate
            int          memoryBusWidth
            int          l2CacheSize
            int          maxThreadsPerMultiProcessor
            int          streamPrioritiesSupported
            int          globalL1CacheSupported
            int          localL1CacheSupported
            size_t       sharedMemPerMultiprocessor
            int          regsPerMultiprocessor
            int          managedMemory
            int          isMultiGpuBoard
            int          multiGpuBoardGroupID
            int          hostNativeAtomicSupported
            int          singleToDoublePrecisionPerfRatio
            int          pageableMemoryAccess
            int          concurrentManagedAccess
            int          computePreemptionSupported
            int          canUseHostPointerForRegisteredMem
            int          cooperativeLaunch
            int          cooperativeMultiDeviceLaunch
            size_t       sharedMemPerBlockOptin
            int          pageableMemoryAccessUsesHostPageTables
            int          directManagedMemAccessFromHost
    ELIF CUDA_VERSION == 9020:
        ctypedef struct DeviceProp 'cudaDeviceProp':
            char         name[256]
            size_t       totalGlobalMem
            size_t       sharedMemPerBlock
            int          regsPerBlock
            int          warpSize
            size_t       memPitch
            int          maxThreadsPerBlock
            int          maxThreadsDim[3]
            int          maxGridSize[3]
            int          clockRate
            size_t       totalConstMem
            int          major
            int          minor
            size_t       textureAlignment
            size_t       texturePitchAlignment
            int          deviceOverlap
            int          multiProcessorCount
            int          kernelExecTimeoutEnabled
            int          integrated
            int          canMapHostMemory
            int          computeMode
            int          maxTexture1D
            int          maxTexture1DMipmap
            int          maxTexture1DLinear
            int          maxTexture2D[2]
            int          maxTexture2DMipmap[2]
            int          maxTexture2DLinear[3]
            int          maxTexture2DGather[2]
            int          maxTexture3D[3]
            int          maxTexture3DAlt[3]
            int          maxTextureCubemap
            int          maxTexture1DLayered[2]
            int          maxTexture2DLayered[3]
            int          maxTextureCubemapLayered[2]
            int          maxSurface1D
            int          maxSurface2D[2]
            int          maxSurface3D[3]
            int          maxSurface1DLayered[2]
            int          maxSurface2DLayered[3]
            int          maxSurfaceCubemap
            int          maxSurfaceCubemapLayered[2]
            size_t       surfaceAlignment
            int          concurrentKernels
            int          ECCEnabled
            int          pciBusID
            int          pciDeviceID
            int          pciDomainID
            int          tccDriver
            int          asyncEngineCount
            int          unifiedAddressing
            int          memoryClockRate
            int          memoryBusWidth
            int          l2CacheSize
            int          maxThreadsPerMultiProcessor
            int          streamPrioritiesSupported
            int          globalL1CacheSupported
            int          localL1CacheSupported
            size_t       sharedMemPerMultiprocessor
            int          regsPerMultiprocessor
            int          managedMemory
            int          isMultiGpuBoard
            int          multiGpuBoardGroupID
            int          hostNativeAtomicSupported
            int          singleToDoublePrecisionPerfRatio
            int          pageableMemoryAccess
            int          concurrentManagedAccess
            int          computePreemptionSupported
            int          canUseHostPointerForRegisteredMem
            int          cooperativeLaunch
            int          cooperativeMultiDeviceLaunch
            size_t       sharedMemPerBlockOptin
            int          pageableMemoryAccessUsesHostPageTables
            int          directManagedMemAccessFromHost
    ELIF HIP_VERSION > 0:
        ctypedef struct deviceArch 'hipDeviceArch_t':
            unsigned hasGlobalInt32Atomics
            unsigned hasGlobalFloatAtomicExch
            unsigned hasSharedInt32Atomics
            unsigned hasSharedFloatAtomicExch
            unsigned hasFloatAtomicAdd

            unsigned hasGlobalInt64Atomics
            unsigned hasSharedInt64Atomics

            unsigned hasDoubles

            unsigned hasWarpVote
            unsigned hasWarpBallot
            unsigned hasWarpShuffle
            unsigned hasFunnelShift

            unsigned hasThreadFenceSystem
            unsigned hasSyncThreadsExt

            unsigned hasSurfaceFuncs
            unsigned has3dGrid
            unsigned hasDynamicParallelism

        IF HIP_VERSION >= 310:
            ctypedef struct DeviceProp 'cudaDeviceProp':
                char name[256]
                size_t totalGlobalMem
                size_t sharedMemPerBlock
                int regsPerBlock
                int warpSize
                int maxThreadsPerBlock
                int maxThreadsDim[3]
                int maxGridSize[3]
                int clockRate
                int memoryClockRate
                int memoryBusWidth
                size_t totalConstMem
                int major
                int minor
                int multiProcessorCount
                int l2CacheSize
                int maxThreadsPerMultiProcessor
                int computeMode
                int clockInstructionRate
                deviceArch arch
                int concurrentKernels
                int pciDomainID
                int pciBusID
                int pciDeviceID
                size_t maxSharedMemoryPerMultiProcessor
                int isMultiGpuBoard
                int canMapHostMemory
                int gcnArch
                # gcnArchName is added since ROCm 3.6, but given it's just
                # 'gfx'+str(gcnArch), in order not to duplicate another struct
                # we add it here
                char gcnArchName[256]
                int integrated
                int cooperativeLaunch
                int cooperativeMultiDeviceLaunch
                int maxTexture1D
                int maxTexture2D[2]
                int maxTexture3D[3]
                unsigned int* hdpMemFlushCntl
                unsigned int* hdpRegFlushCntl
                size_t memPitch
                size_t textureAlignment
                size_t texturePitchAlignment
                int kernelExecTimeoutEnabled
                int ECCEnabled
                int tccDriver
                int cooperativeMultiDeviceUnmatchedFunc
                int cooperativeMultiDeviceUnmatchedGridDim
                int cooperativeMultiDeviceUnmatchedBlockDim
                int cooperativeMultiDeviceUnmatchedSharedMem
                int isLargeBar
                # New since ROCm 3.10.0
                int asicRevision
                int managedMemory
                int directManagedMemAccessFromHost
                int concurrentManagedAccess
                int pageableMemoryAccess
                int pageableMemoryAccessUsesHostPageTables
        ELSE:
            ctypedef struct DeviceProp 'cudaDeviceProp':
                char name[256]
                size_t totalGlobalMem
                size_t sharedMemPerBlock
                int regsPerBlock
                int warpSize
                int maxThreadsPerBlock
                int maxThreadsDim[3]
                int maxGridSize[3]
                int clockRate
                int memoryClockRate
                int memoryBusWidth
                size_t totalConstMem
                int major
                int minor
                int multiProcessorCount
                int l2CacheSize
                int maxThreadsPerMultiProcessor
                int computeMode
                int clockInstructionRate
                deviceArch arch
                int concurrentKernels
                int pciDomainID
                int pciBusID
                int pciDeviceID
                size_t maxSharedMemoryPerMultiProcessor
                int isMultiGpuBoard
                int canMapHostMemory
                int gcnArch
                int integrated
                int cooperativeLaunch
                int cooperativeMultiDeviceLaunch
                int maxTexture1D
                int maxTexture2D[2]
                int maxTexture3D[3]
                unsigned int* hdpMemFlushCntl
                unsigned int* hdpRegFlushCntl
                size_t memPitch
                size_t textureAlignment
                size_t texturePitchAlignment
                int kernelExecTimeoutEnabled
                int ECCEnabled
                int tccDriver
                int cooperativeMultiDeviceUnmatchedFunc
                int cooperativeMultiDeviceUnmatchedGridDim
                int cooperativeMultiDeviceUnmatchedBlockDim
                int cooperativeMultiDeviceUnmatchedSharedMem
                int isLargeBar
    ELSE:  # for RTD
        ctypedef struct DeviceProp 'cudaDeviceProp':
            char         name[256]


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

    cudaIpcMemLazyEnablePeerAccess = 1

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

    # cudaStream flags
    streamDefault = 0
    streamNonBlocking = 1

    # cudaStream handles
    streamLegacy = 1
    streamPerThread = 2

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

    # CUDA Limits
    cudaLimitStackSize = 0x00
    cudaLimitPrintfFifoSize = 0x01
    cudaLimitMallocHeapSize = 0x02
    cudaLimitDevRuntimeSyncDepth = 0x03
    cudaLimitDevRuntimePendingLaunchCount = 0x04
    cudaLimitMaxL2FetchGranularity = 0x05

    # cudaChannelFormatKind
    cudaChannelFormatKindSigned = 0
    cudaChannelFormatKindUnsigned = 1
    cudaChannelFormatKindFloat = 2
    cudaChannelFormatKindNone = 3

    # CUDA array flags
    cudaArrayDefault = 0
    # cudaArrayLayered = 1
    cudaArraySurfaceLoadStore = 2
    # cudaArrayCubemap = 4
    # cudaArrayTextureGather = 8

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


# This was a legacy mistake: the prefix "cuda" should have been removed
# so that we can directly assign their C counterparts here. Now because
# of backward compatibility and no flexible Cython macro (IF/ELSE), we
# have to duplicate the enum. (CUDA and HIP use different values!)
IF HIP_VERSION > 0:
    # separate in groups of 10 for easier counting...
    cpdef enum:
        cudaDevAttrMaxThreadsPerBlock = 0
        cudaDevAttrMaxBlockDimX
        cudaDevAttrMaxBlockDimY
        cudaDevAttrMaxBlockDimZ
        cudaDevAttrMaxGridDimX
        cudaDevAttrMaxGridDimY
        cudaDevAttrMaxGridDimZ
        cudaDevAttrMaxSharedMemoryPerBlock
        cudaDevAttrTotalConstantMemory
        cudaDevAttrWarpSize

        cudaDevAttrMaxRegistersPerBlock
        cudaDevAttrClockRate
        cudaDevAttrMemoryClockRate
        cudaDevAttrGlobalMemoryBusWidth
        cudaDevAttrMultiProcessorCount
        cudaDevAttrComputeMode
        cudaDevAttrL2CacheSize
        cudaDevAttrMaxThreadsPerMultiProcessor
        # The following are exposed as "deviceAttributeCo..."
        # cudaDevAttrComputeCapabilityMajor
        # cudaDevAttrComputeCapabilityMinor

        cudaDevAttrConcurrentKernels = 20
        cudaDevAttrPciBusId
        cudaDevAttrPciDeviceId
        cudaDevAttrMaxSharedMemoryPerMultiprocessor
        cudaDevAttrIsMultiGpuBoard
        cudaDevAttrIntegrated
        cudaDevAttrCooperativeLaunch
        cudaDevAttrCooperativeMultiDeviceLaunch
        cudaDevAttrMaxTexture1DWidth
        cudaDevAttrMaxTexture2DWidth

        cudaDevAttrMaxTexture2DHeight
        cudaDevAttrMaxTexture3DWidth
        cudaDevAttrMaxTexture3DHeight
        cudaDevAttrMaxTexture3DDepth
        # The following attributes do not exist in CUDA and cause segfualts
        # if we try to access them
        # hipDeviceAttributeHdpMemFlushCntl
        # hipDeviceAttributeHdpRegFlushCntl
        cudaDevAttrMaxPitch = 36
        cudaDevAttrTextureAlignment
        cudaDevAttrTexturePitchAlignment
        cudaDevAttrKernelExecTimeout

        cudaDevAttrCanMapHostMemory
        cudaDevAttrEccEnabled
        # The following attributes do not exist in CUDA
        # hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc
        # hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim
        # hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim
        # hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem

        # The rest do not have HIP correspondence...
        # TODO(leofang): should we expose them anyway, with a value -1 to
        # indicate they cannot be used in HIP?
        # cudaDevAttrGpuOverlap
        # cudaDevAttrMaxTexture2DLayeredWidth
        # cudaDevAttrMaxTexture2DLayeredHeight
        # cudaDevAttrMaxTexture2DLayeredLayers
        # cudaDevAttrSurfaceAlignment
        # cudaDevAttrTccDriver
        # cudaDevAttrAsyncEngineCount
        # cudaDevAttrUnifiedAddressing
        # cudaDevAttrMaxTexture1DLayeredWidth
        # cudaDevAttrMaxTexture1DLayeredLayers
        # cudaDevAttrMaxTexture2DGatherWidth
        # cudaDevAttrMaxTexture2DGatherHeight
        # cudaDevAttrMaxTexture3DWidthAlt
        # cudaDevAttrMaxTexture3DHeightAlt
        # cudaDevAttrMaxTexture3DDepthAlt
        # cudaDevAttrPciDomainId
        # cudaDevAttrMaxTextureCubemapWidth
        # cudaDevAttrMaxTextureCubemapLayeredWidth
        # cudaDevAttrMaxTextureCubemapLayeredLayers
        # cudaDevAttrMaxSurface1DWidth
        # cudaDevAttrMaxSurface2DWidth
        # cudaDevAttrMaxSurface2DHeight
        # cudaDevAttrMaxSurface3DWidth
        # cudaDevAttrMaxSurface3DHeight
        # cudaDevAttrMaxSurface3DDepth
        # cudaDevAttrMaxSurface1DLayeredWidth
        # cudaDevAttrMaxSurface1DLayeredLayers
        # cudaDevAttrMaxSurface2DLayeredWidth
        # cudaDevAttrMaxSurface2DLayeredHeight
        # cudaDevAttrMaxSurface2DLayeredLayers
        # cudaDevAttrMaxSurfaceCubemapWidth
        # cudaDevAttrMaxSurfaceCubemapLayeredWidth
        # cudaDevAttrMaxSurfaceCubemapLayeredLayers
        # cudaDevAttrMaxTexture1DLinearWidth
        # cudaDevAttrMaxTexture2DLinearWidth
        # cudaDevAttrMaxTexture2DLinearHeight
        # cudaDevAttrMaxTexture2DLinearPitch
        # cudaDevAttrMaxTexture2DMipmappedWidth
        # cudaDevAttrMaxTexture2DMipmappedHeight
        # cudaDevAttrMaxTexture1DMipmappedWidth
        # cudaDevAttrStreamPrioritiesSupported
        # cudaDevAttrGlobalL1CacheSupported
        # cudaDevAttrLocalL1CacheSupported
        # cudaDevAttrMaxRegistersPerMultiprocessor
        # cudaDevAttrMultiGpuBoardGroupID
        # cudaDevAttrHostNativeAtomicSupported
        # cudaDevAttrSingleToDoublePrecisionPerfRatio
        # cudaDevAttrComputePreemptionSupported
        # cudaDevAttrCanUseHostPointerForRegisteredMem
        # cudaDevAttrReserved92
        # cudaDevAttrReserved93
        # cudaDevAttrReserved94
        # cudaDevAttrMaxSharedMemoryPerBlockOptin
        # cudaDevAttrCanFlushRemoteWrites
        # cudaDevAttrHostRegisterSupported
    IF HIP_VERSION >= 310:
        cpdef enum:
            # hipDeviceAttributeAsicRevision  # does not exist in CUDA
            cudaDevAttrManagedMemory = 47
            cudaDevAttrDirectManagedMemAccessFromHost
            cudaDevAttrConcurrentManagedAccess

            cudaDevAttrPageableMemoryAccess
            cudaDevAttrPageableMemoryAccessUsesHostPageTables
ELSE:
    # For CUDA/RTD
    cpdef enum:
        cudaDevAttrMaxThreadsPerBlock = 1
        cudaDevAttrMaxBlockDimX
        cudaDevAttrMaxBlockDimY
        cudaDevAttrMaxBlockDimZ
        cudaDevAttrMaxGridDimX
        cudaDevAttrMaxGridDimY
        cudaDevAttrMaxGridDimZ
        cudaDevAttrMaxSharedMemoryPerBlock
        cudaDevAttrTotalConstantMemory
        cudaDevAttrWarpSize
        cudaDevAttrMaxPitch
        cudaDevAttrMaxRegistersPerBlock
        cudaDevAttrClockRate
        cudaDevAttrTextureAlignment
        cudaDevAttrGpuOverlap
        cudaDevAttrMultiProcessorCount
        cudaDevAttrKernelExecTimeout
        cudaDevAttrIntegrated
        cudaDevAttrCanMapHostMemory
        cudaDevAttrComputeMode
        cudaDevAttrMaxTexture1DWidth
        cudaDevAttrMaxTexture2DWidth
        cudaDevAttrMaxTexture2DHeight
        cudaDevAttrMaxTexture3DWidth
        cudaDevAttrMaxTexture3DHeight
        cudaDevAttrMaxTexture3DDepth
        cudaDevAttrMaxTexture2DLayeredWidth
        cudaDevAttrMaxTexture2DLayeredHeight
        cudaDevAttrMaxTexture2DLayeredLayers
        cudaDevAttrSurfaceAlignment
        cudaDevAttrConcurrentKernels
        cudaDevAttrEccEnabled
        cudaDevAttrPciBusId
        cudaDevAttrPciDeviceId
        cudaDevAttrTccDriver
        cudaDevAttrMemoryClockRate
        cudaDevAttrGlobalMemoryBusWidth
        cudaDevAttrL2CacheSize
        cudaDevAttrMaxThreadsPerMultiProcessor
        cudaDevAttrAsyncEngineCount
        cudaDevAttrUnifiedAddressing
        cudaDevAttrMaxTexture1DLayeredWidth
        cudaDevAttrMaxTexture1DLayeredLayers  # = 43; 44 is missing
        cudaDevAttrMaxTexture2DGatherWidth = 45
        cudaDevAttrMaxTexture2DGatherHeight
        cudaDevAttrMaxTexture3DWidthAlt
        cudaDevAttrMaxTexture3DHeightAlt
        cudaDevAttrMaxTexture3DDepthAlt
        cudaDevAttrPciDomainId
        cudaDevAttrTexturePitchAlignment
        cudaDevAttrMaxTextureCubemapWidth
        cudaDevAttrMaxTextureCubemapLayeredWidth
        cudaDevAttrMaxTextureCubemapLayeredLayers
        cudaDevAttrMaxSurface1DWidth
        cudaDevAttrMaxSurface2DWidth
        cudaDevAttrMaxSurface2DHeight
        cudaDevAttrMaxSurface3DWidth
        cudaDevAttrMaxSurface3DHeight
        cudaDevAttrMaxSurface3DDepth
        cudaDevAttrMaxSurface1DLayeredWidth
        cudaDevAttrMaxSurface1DLayeredLayers
        cudaDevAttrMaxSurface2DLayeredWidth
        cudaDevAttrMaxSurface2DLayeredHeight
        cudaDevAttrMaxSurface2DLayeredLayers
        cudaDevAttrMaxSurfaceCubemapWidth
        cudaDevAttrMaxSurfaceCubemapLayeredWidth
        cudaDevAttrMaxSurfaceCubemapLayeredLayers
        cudaDevAttrMaxTexture1DLinearWidth
        cudaDevAttrMaxTexture2DLinearWidth
        cudaDevAttrMaxTexture2DLinearHeight
        cudaDevAttrMaxTexture2DLinearPitch
        cudaDevAttrMaxTexture2DMipmappedWidth
        cudaDevAttrMaxTexture2DMipmappedHeight
        # The following are exposed as "deviceAttributeCo..."
        # cudaDevAttrComputeCapabilityMajor  # = 75
        # cudaDevAttrComputeCapabilityMinor  # = 76
        cudaDevAttrMaxTexture1DMipmappedWidth = 77
        cudaDevAttrStreamPrioritiesSupported
        cudaDevAttrGlobalL1CacheSupported
        cudaDevAttrLocalL1CacheSupported
        cudaDevAttrMaxSharedMemoryPerMultiprocessor
        cudaDevAttrMaxRegistersPerMultiprocessor
        cudaDevAttrManagedMemory
        cudaDevAttrIsMultiGpuBoard
        cudaDevAttrMultiGpuBoardGroupID
        cudaDevAttrHostNativeAtomicSupported
        cudaDevAttrSingleToDoublePrecisionPerfRatio
        cudaDevAttrPageableMemoryAccess
        cudaDevAttrConcurrentManagedAccess
        cudaDevAttrComputePreemptionSupported
        cudaDevAttrCanUseHostPointerForRegisteredMem
        cudaDevAttrReserved92
        cudaDevAttrReserved93
        cudaDevAttrReserved94
        cudaDevAttrCooperativeLaunch
        cudaDevAttrCooperativeMultiDeviceLaunch
        cudaDevAttrMaxSharedMemoryPerBlockOptin
        cudaDevAttrCanFlushRemoteWrites
        cudaDevAttrHostRegisterSupported
        cudaDevAttrPageableMemoryAccessUsesHostPageTables
        cudaDevAttrDirectManagedMemAccessFromHost  # = 101
        # added since CUDA 11.0
        cudaDevAttrMaxBlocksPerMultiprocessor = 106
        cudaDevAttrReservedSharedMemoryPerBlock = 111
        # added since CUDA 11.1
        cudaDevAttrSparseCudaArraySupported = 112
        cudaDevAttrHostRegisterReadOnlySupported = 113
        # added since CUDA 11.2
        cudaDevAttrMaxTimelineSemaphoreInteropSupported = 114
        cudaDevAttrMemoryPoolsSupported = 115


###############################################################################
# Error codes
###############################################################################

cdef int errorMemoryAllocation
cdef int errorInvalidValue
cdef int errorPeerAccessAlreadyEnabled
cdef int errorContextIsDestroyed
cdef int errorInvalidResourceHandle


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
cpdef getDeviceProperties(int device)

cpdef int deviceCanAccessPeer(int device, int peerDevice) except? -1
cpdef deviceEnablePeerAccess(int peerDevice)

cpdef size_t deviceGetLimit(int limit) except? -1
cpdef deviceSetLimit(int limit, size_t value)


###############################################################################
# Memory management
###############################################################################

cpdef intptr_t malloc(size_t size) except? 0
cpdef intptr_t mallocManaged(size_t size, unsigned int flags=*) except? 0
cpdef intptr_t malloc3DArray(intptr_t desc, size_t width, size_t height,
                             size_t depth, unsigned int flags=*) except? 0
cpdef intptr_t mallocArray(intptr_t desc, size_t width, size_t height,
                           unsigned int flags=*) except? 0
cpdef intptr_t mallocAsync(size_t size, intptr_t stream) except? 0
cpdef intptr_t hostAlloc(size_t size, unsigned int flags) except? 0
cpdef hostRegister(intptr_t ptr, size_t size, unsigned int flags)
cpdef hostUnregister(intptr_t ptr)
cpdef free(intptr_t ptr)
cpdef freeHost(intptr_t ptr)
cpdef freeArray(intptr_t ptr)
cpdef freeAsync(intptr_t ptr, intptr_t stream)
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
cpdef intptr_t deviceGetDefaultMemPool(int) except? 0
cpdef intptr_t deviceGetMemPool(int) except? 0
cpdef deviceSetMemPool(int, intptr_t)
cpdef memPoolTrimTo(intptr_t, size_t)


###############################################################################
# Stream and Event
###############################################################################

cpdef intptr_t streamCreate() except? 0
cpdef intptr_t streamCreateWithFlags(unsigned int flags) except? 0
cpdef streamDestroy(intptr_t stream)
cpdef streamSynchronize(intptr_t stream)
cpdef streamAddCallback(intptr_t stream, callback, intptr_t arg,
                        unsigned int flags=*)
cpdef launchHostFunc(intptr_t stream, callback, intptr_t arg)
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

cpdef uintmax_t createSurfaceObject(intptr_t ResDesc)
cpdef destroySurfaceObject(uintmax_t surfObject)
# TODO(leofang): add cudaGetSurfaceObjectResourceDesc
