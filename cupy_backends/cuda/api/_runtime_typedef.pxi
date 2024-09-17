# Keep in sync with typenames exported in `runtime.pxd`.

from cupy_backends.cuda.api cimport driver


cdef extern from *:
    ctypedef int Error 'cudaError_t'
    ctypedef int DataType 'cudaDataType'

    ctypedef int DeviceAttr 'cudaDeviceAttr'
    ctypedef int MemoryAdvise 'cudaMemoryAdvise'

    ctypedef void* Stream 'cudaStream_t'
    ctypedef void _StreamCallbackDef(
        driver.Stream stream, Error status, void* userData)
    ctypedef _StreamCallbackDef* StreamCallback 'cudaStreamCallback_t'
    ctypedef void* StreamCaptureStatus 'cudaStreamCaptureStatus'
    ctypedef void* GraphNode 'cudaGraphNode_t'

    ctypedef void _HostFnDef(void* userData)
    ctypedef _HostFnDef* HostFn 'cudaHostFn_t'

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

    ctypedef int StreamCaptureMode 'cudaStreamCaptureMode'
    ctypedef void* Graph 'cudaGraph_t'
    ctypedef void* GraphExec 'cudaGraphExec_t'

    # This is for the annoying nested struct cudaResourceDesc, which is not
    # perfectly supported in Cython
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
    ctypedef int MemPoolAttr 'cudaMemPoolAttr'

    ctypedef int MemAllocationType 'cudaMemAllocationType'
    ctypedef int MemAllocationHandleType 'cudaMemAllocationHandleType'
    ctypedef int MemLocationType 'cudaMemLocationType'
    IF CUPY_CUDA_VERSION > 0:
        # This is for the annoying nested struct, which is not
        # perfectly supported in Cython
        ctypedef struct _MemLocation 'cudaMemLocation':
            MemLocationType type
            int id

        ctypedef struct _MemPoolProps 'cudaMemPoolProps':
            MemAllocationType allocType
            MemAllocationHandleType handleTypes
            _MemLocation location
    ELSE:
        ctypedef struct _MemPoolProps 'cudaMemPoolProps':
            pass  # for HIP & RTD

    IF CUPY_CUDA_VERSION > 0:
        ctypedef struct _PointerAttributes 'cudaPointerAttributes':
            int type
            int device
            void* devicePointer
            void* hostPointer
    ELIF CUPY_HIP_VERSION > 0:
        ctypedef struct _PointerAttributes 'cudaPointerAttributes':
            int memoryType
            int device
            void* devicePointer
            void* hostPointer
    ELSE:
        ctypedef struct _PointerAttributes 'cudaPointerAttributes':
            pass  # for RTD

    IF CUPY_CUDA_VERSION >= 11000:
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
    ELIF CUPY_CUDA_VERSION >= 10000:
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
    ELIF CUPY_HIP_VERSION > 0:
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

        IF CUPY_HIP_VERSION >= 310:
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
