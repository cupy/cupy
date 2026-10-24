# Keep in sync with typenames exported in `runtime.pxd`.

from cupy.backends.backend.api cimport driver

cdef extern from *:
    IF CUPY_CUDA_VERSION >= 13000:
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
            size_t       totalConstMem
            int          major
            int          minor
            size_t       textureAlignment
            size_t       texturePitchAlignment
            int          multiProcessorCount
            int          integrated
            int          canMapHostMemory
            int          maxTexture1D
            int          maxTexture1DMipmap
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
            int          memoryBusWidth
            int          l2CacheSize
            int          persistingL2CacheMaxSize
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
            int          pageableMemoryAccess
            int          concurrentManagedAccess
            int          computePreemptionSupported
            int          canUseHostPointerForRegisteredMem
            int          cooperativeLaunch
            size_t       sharedMemPerBlockOptin
            int          pageableMemoryAccessUsesHostPageTables
            int          directManagedMemAccessFromHost
            int          maxBlocksPerMultiProcessor
            int          accessPolicyMaxWindowSize
            size_t       reservedSharedMemPerBlock
            int          hostRegisterSupported              # CUDA 12.0 field
            int          sparseCudaArraySupported           # CUDA 12.0 field
            int          hostRegisterReadOnlySupported      # CUDA 12.0 field
            int          timelineSemaphoreInteropSupported  # CUDA 12.0 field
            int          memoryPoolsSupported               # CUDA 12.0 field
            int          gpuDirectRDMASupported             # CUDA 12.0 field
            unsigned int gpuDirectRDMAFlushWritesOptions    # CUDA 12.0 field
            int          gpuDirectRDMAWritesOrdering        # CUDA 12.0 field
            unsigned int memoryPoolSupportedHandleTypes     # CUDA 12.0 field
            int          deferredMappingCudaArraySupported  # CUDA 12.0 field
            int          ipcEventSupported                  # CUDA 12.0 field
            int          clusterLaunch                      # CUDA 12.0 field
            int          unifiedFunctionPointers            # CUDA 12.0 field
            int          deviceNumaConfig                   # CUDA 13.0 field
            int          deviceNumaId                       # CUDA 13.0 field
            int          mpsEnabled                         # CUDA 13.0 field
            int          hostNumaId                         # CUDA 13.0 field
            unsigned int gpuPciDeviceID                     # CUDA 13.0 field
            unsigned int gpuPciSubsystemID                  # CUDA 13.0 field
            int          hostNumaMultinodeIpcSupported      # CUDA 13.0 field
    ELIF CUPY_CUDA_VERSION >= 12000:
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
            int          persistingL2CacheMaxSize
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
            int          maxBlocksPerMultiProcessor
            int          accessPolicyMaxWindowSize
            size_t       reservedSharedMemPerBlock
            int          hostRegisterSupported              # CUDA 12.0 field
            int          sparseCudaArraySupported           # CUDA 12.0 field
            int          hostRegisterReadOnlySupported      # CUDA 12.0 field
            int          timelineSemaphoreInteropSupported  # CUDA 12.0 field
            int          memoryPoolsSupported               # CUDA 12.0 field
            int          gpuDirectRDMASupported             # CUDA 12.0 field
            unsigned int gpuDirectRDMAFlushWritesOptions    # CUDA 12.0 field
            int          gpuDirectRDMAWritesOrdering        # CUDA 12.0 field
            unsigned int memoryPoolSupportedHandleTypes     # CUDA 12.0 field
            int          deferredMappingCudaArraySupported  # CUDA 12.0 field
            int          ipcEventSupported                  # CUDA 12.0 field
            int          clusterLaunch                      # CUDA 12.0 field
            int          unifiedFunctionPointers            # CUDA 12.0 field
    ELIF CUPY_CUDA_VERSION >= 11000:
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
