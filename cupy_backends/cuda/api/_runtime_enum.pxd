cpdef enum:
    # need to revisit this when cython supports C++ enums (in 3.0)
    # https://stackoverflow.com/a/67138945

    cudaMemoryTypeHost = 1
    cudaMemoryTypeDevice = 2

    cudaIpcMemLazyEnablePeerAccess = 1

    cudaMemAttachGlobal = 1
    cudaMemAttachHost = 2
    cudaMemAttachSingle = 4

    cudaCpuDeviceId = -1
    cudaInvalidDeviceId = -2

    cudaMemAdviseSetReadMostly = 1
    cudaMemAdviseUnsetReadMostly = 2
    cudaMemAdviseSetPreferredLocation = 3
    cudaMemAdviseUnsetPreferredLocation = 4
    cudaMemAdviseSetAccessedBy = 5
    cudaMemAdviseUnsetAccessedBy = 6

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

    # cudaMemPoolAttr
    # ----- added since 11.2 -----
    cudaMemPoolReuseFollowEventDependencies = 0x1
    cudaMemPoolReuseAllowOpportunistic = 0x2
    cudaMemPoolReuseAllowInternalDependencies = 0x3
    cudaMemPoolAttrReleaseThreshold = 0x4
    # ----- added since 11.3 -----
    cudaMemPoolAttrReservedMemCurrent = 0x5
    cudaMemPoolAttrReservedMemHigh = 0x6
    cudaMemPoolAttrUsedMemCurrent = 0x7
    cudaMemPoolAttrUsedMemHigh = 0x8

    # cudaMemAllocationType
    cudaMemAllocationTypePinned = 0x1

    # cudaMemAllocationHandleType
    cudaMemHandleTypeNone = 0x0
    cudaMemHandleTypePosixFileDescriptor = 0x1
    # cudaMemHandleTypeWin32 = 0x2
    # cudaMemHandleTypeWin32Kmt = 0x4

    # cudaMemLocationType
    cudaMemLocationTypeInvalid = 0          # CUDA 12.0
    cudaMemLocationTypeDevice = 1
    cudaMemLocationTypeHost = 2             # CUDA 12.0
    cudaMemLocationTypeHostNuma = 3         # CUDA 12.0
    cudaMemLocationTypeHostNumaCurrent = 4  # CUDA 12.0

    # cudaGraphDebugDotFlags
    cudaGraphDebugDotFlagsVerbose = 1<<0
    cudaGraphDebugDotFlagsKernelNodeParams = 1<<2
    cudaGraphDebugDotFlagsMemcpyNodeParams = 1<<3
    cudaGraphDebugDotFlagsMemsetNodeParams = 1<<4
    cudaGraphDebugDotFlagsHostNodeParams = 1<<5
    cudaGraphDebugDotFlagsEventNodeParams = 1<<6
    cudaGraphDebugDotFlagsExtSemasSignalNodeParams = 1<<7
    cudaGraphDebugDotFlagsExtSemasWaitNodeParams = 1<<8
    cudaGraphDebugDotFlagsKernelNodeAttributes = 1<<9
    cudaGraphDebugDotFlagsHandles = 1<<10
    cudaGraphDebugDotFlagsConditionalNodeParams = 1<<15

# This was a legacy mistake: the prefix "cuda" should have been removed
# so that we can directly assign their C counterparts here. Now because
# of backward compatibility and no flexible Cython macro (IF/ELSE), we
# have to duplicate the enum. (CUDA and HIP use different values!)
IF CUPY_HIP_VERSION >= 40300000:
    # HIP >= 4.3: hipDeviceAttribute_t was rearranged (ca50ac83...),
    # define the full HIP enum and provide CUDA-style aliases where applicable.
    cpdef enum:
        # hipDeviceAttribute_t (Cuda-compatible region)
        hipDeviceAttributeCudaCompatibleBegin = 0

        hipDeviceAttributeEccEnabled = hipDeviceAttributeCudaCompatibleBegin
        hipDeviceAttributeAccessPolicyMaxWindowSize
        hipDeviceAttributeAsyncEngineCount
        hipDeviceAttributeCanMapHostMemory
        hipDeviceAttributeCanUseHostPointerForRegisteredMem
        hipDeviceAttributeClockRate
        hipDeviceAttributeComputeMode
        hipDeviceAttributeComputePreemptionSupported
        hipDeviceAttributeConcurrentKernels
        hipDeviceAttributeConcurrentManagedAccess
        hipDeviceAttributeCooperativeLaunch
        hipDeviceAttributeCooperativeMultiDeviceLaunch
        hipDeviceAttributeDeviceOverlap
        hipDeviceAttributeDirectManagedMemAccessFromHost
        hipDeviceAttributeGlobalL1CacheSupported
        hipDeviceAttributeHostNativeAtomicSupported
        hipDeviceAttributeIntegrated
        hipDeviceAttributeIsMultiGpuBoard
        hipDeviceAttributeKernelExecTimeout
        hipDeviceAttributeL2CacheSize
        hipDeviceAttributeLocalL1CacheSupported
        hipDeviceAttributeLuid
        hipDeviceAttributeLuidDeviceNodeMask
        hipDeviceAttributeComputeCapabilityMajor
        hipDeviceAttributeManagedMemory
        hipDeviceAttributeMaxBlocksPerMultiProcessor
        hipDeviceAttributeMaxBlockDimX
        hipDeviceAttributeMaxBlockDimY
        hipDeviceAttributeMaxBlockDimZ
        hipDeviceAttributeMaxGridDimX
        hipDeviceAttributeMaxGridDimY
        hipDeviceAttributeMaxGridDimZ
        hipDeviceAttributeMaxSurface1D
        hipDeviceAttributeMaxSurface1DLayered
        hipDeviceAttributeMaxSurface2D
        hipDeviceAttributeMaxSurface2DLayered
        hipDeviceAttributeMaxSurface3D
        hipDeviceAttributeMaxSurfaceCubemap
        hipDeviceAttributeMaxSurfaceCubemapLayered
        hipDeviceAttributeMaxTexture1DWidth
        hipDeviceAttributeMaxTexture1DLayered
        hipDeviceAttributeMaxTexture1DLinear
        hipDeviceAttributeMaxTexture1DMipmap
        hipDeviceAttributeMaxTexture2DWidth
        hipDeviceAttributeMaxTexture2DHeight
        hipDeviceAttributeMaxTexture2DGather
        hipDeviceAttributeMaxTexture2DLayered
        hipDeviceAttributeMaxTexture2DLinear
        hipDeviceAttributeMaxTexture2DMipmap
        hipDeviceAttributeMaxTexture3DWidth
        hipDeviceAttributeMaxTexture3DHeight
        hipDeviceAttributeMaxTexture3DDepth
        hipDeviceAttributeMaxTexture3DAlt
        hipDeviceAttributeMaxTextureCubemap
        hipDeviceAttributeMaxTextureCubemapLayered
        hipDeviceAttributeMaxThreadsDim
        hipDeviceAttributeMaxThreadsPerBlock
        hipDeviceAttributeMaxThreadsPerMultiProcessor
        hipDeviceAttributeMaxPitch
        hipDeviceAttributeMemoryBusWidth
        hipDeviceAttributeMemoryClockRate
        hipDeviceAttributeComputeCapabilityMinor
        hipDeviceAttributeMultiGpuBoardGroupID
        hipDeviceAttributeMultiprocessorCount
        hipDeviceAttributeName
        hipDeviceAttributePageableMemoryAccess
        hipDeviceAttributePageableMemoryAccessUsesHostPageTables
        hipDeviceAttributePciBusId
        hipDeviceAttributePciDeviceId
        hipDeviceAttributePciDomainID
        hipDeviceAttributePersistingL2CacheMaxSize
        hipDeviceAttributeMaxRegistersPerBlock
        hipDeviceAttributeMaxRegistersPerMultiprocessor
        hipDeviceAttributeReservedSharedMemPerBlock
        hipDeviceAttributeMaxSharedMemoryPerBlock
        hipDeviceAttributeSharedMemPerBlockOptin
        hipDeviceAttributeSharedMemPerMultiprocessor
        hipDeviceAttributeSingleToDoublePrecisionPerfRatio
        hipDeviceAttributeStreamPrioritiesSupported
        hipDeviceAttributeSurfaceAlignment
        hipDeviceAttributeTccDriver
        hipDeviceAttributeTextureAlignment
        hipDeviceAttributeTexturePitchAlignment
        hipDeviceAttributeTotalConstantMemory
        hipDeviceAttributeTotalGlobalMem
        hipDeviceAttributeUnifiedAddressing
        hipDeviceAttributeUuid
        hipDeviceAttributeWarpSize

        hipDeviceAttributeCudaCompatibleEnd = 9999

        # AMD-specific region
        hipDeviceAttributeAmdSpecificBegin = 10000
        hipDeviceAttributeClockInstructionRate = \
                hipDeviceAttributeAmdSpecificBegin
        hipDeviceAttributeArch
        hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
        hipDeviceAttributeGcnArch
        hipDeviceAttributeGcnArchName
        hipDeviceAttributeHdpMemFlushCntl
        hipDeviceAttributeHdpRegFlushCntl
        hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc
        hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim
        hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim
        hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem
        hipDeviceAttributeIsLargeBar
        hipDeviceAttributeAsicRevision
        hipDeviceAttributeCanUseStreamWaitValue

        hipDeviceAttributeAmdSpecificEnd = 19999
        hipDeviceAttributeVendorSpecificBegin = 20000

        # CUDA-style aliases for attributes that have CUDA counterparts
        cudaDevAttrEccEnabled = hipDeviceAttributeEccEnabled
        cudaDevAttrAccessPolicyMaxWindowSize = \
            hipDeviceAttributeAccessPolicyMaxWindowSize
        cudaDevAttrAsyncEngineCount = hipDeviceAttributeAsyncEngineCount
        cudaDevAttrCanMapHostMemory = hipDeviceAttributeCanMapHostMemory
        cudaDevAttrCanUseHostPointerForRegisteredMem = \
            hipDeviceAttributeCanUseHostPointerForRegisteredMem
        cudaDevAttrClockRate = hipDeviceAttributeClockRate
        cudaDevAttrComputeMode = hipDeviceAttributeComputeMode
        cudaDevAttrComputePreemptionSupported = \
            hipDeviceAttributeComputePreemptionSupported
        cudaDevAttrConcurrentKernels = hipDeviceAttributeConcurrentKernels
        cudaDevAttrConcurrentManagedAccess = \
            hipDeviceAttributeConcurrentManagedAccess
        cudaDevAttrCooperativeLaunch = hipDeviceAttributeCooperativeLaunch
        cudaDevAttrCooperativeMultiDeviceLaunch = \
            hipDeviceAttributeCooperativeMultiDeviceLaunch
        cudaDevAttrGpuOverlap = hipDeviceAttributeDeviceOverlap
        cudaDevAttrDirectManagedMemAccessFromHost = \
            hipDeviceAttributeDirectManagedMemAccessFromHost
        cudaDevAttrGlobalL1CacheSupported = \
            hipDeviceAttributeGlobalL1CacheSupported
        cudaDevAttrHostNativeAtomicSupported = \
            hipDeviceAttributeHostNativeAtomicSupported
        cudaDevAttrIntegrated = hipDeviceAttributeIntegrated
        cudaDevAttrIsMultiGpuBoard = hipDeviceAttributeIsMultiGpuBoard
        cudaDevAttrKernelExecTimeout = hipDeviceAttributeKernelExecTimeout
        cudaDevAttrL2CacheSize = hipDeviceAttributeL2CacheSize
        cudaDevAttrLocalL1CacheSupported = hipDeviceAttributeLocalL1CacheSupported
        cudaDevAttrLuid = hipDeviceAttributeLuid
        cudaDevAttrLuidDeviceNodeMask = hipDeviceAttributeLuidDeviceNodeMask
        cudaDevAttrManagedMemory = hipDeviceAttributeManagedMemory
        cudaDevAttrMaxBlocksPerMultiprocessor = \
            hipDeviceAttributeMaxBlocksPerMultiProcessor
        cudaDevAttrMaxBlockDimX = hipDeviceAttributeMaxBlockDimX
        cudaDevAttrMaxBlockDimY = hipDeviceAttributeMaxBlockDimY
        cudaDevAttrMaxBlockDimZ = hipDeviceAttributeMaxBlockDimZ
        cudaDevAttrMaxGridDimX = hipDeviceAttributeMaxGridDimX
        cudaDevAttrMaxGridDimY = hipDeviceAttributeMaxGridDimY
        cudaDevAttrMaxGridDimZ = hipDeviceAttributeMaxGridDimZ
        cudaDevAttrMaxSurface1DWidth = hipDeviceAttributeMaxSurface1D
        cudaDevAttrMaxSurface1DLayeredWidth = \
            hipDeviceAttributeMaxSurface1DLayered
        cudaDevAttrMaxSurface2DWidth = hipDeviceAttributeMaxSurface2D
        cudaDevAttrMaxSurface2DHeight = hipDeviceAttributeMaxSurface2D
        cudaDevAttrMaxSurface2DLayeredWidth = \
            hipDeviceAttributeMaxSurface2DLayered
        cudaDevAttrMaxSurface2DLayeredHeight = \
            hipDeviceAttributeMaxSurface2DLayered
        cudaDevAttrMaxSurface2DLayeredLayers = \
            hipDeviceAttributeMaxSurface2DLayered
        cudaDevAttrMaxSurface3DWidth = hipDeviceAttributeMaxSurface3D
        cudaDevAttrMaxSurface3DHeight = hipDeviceAttributeMaxSurface3D
        cudaDevAttrMaxSurface3DDepth = hipDeviceAttributeMaxSurface3D
        cudaDevAttrMaxSurfaceCubemapWidth = \
            hipDeviceAttributeMaxSurfaceCubemap
        cudaDevAttrMaxSurfaceCubemapLayeredWidth = \
            hipDeviceAttributeMaxSurfaceCubemapLayered
        cudaDevAttrMaxSurfaceCubemapLayeredLayers = \
            hipDeviceAttributeMaxSurfaceCubemapLayered
        cudaDevAttrMaxTexture1DWidth = hipDeviceAttributeMaxTexture1DWidth
        cudaDevAttrMaxTexture1DLayeredWidth = \
            hipDeviceAttributeMaxTexture1DLayered
        cudaDevAttrMaxTexture1DLayeredLayers = \
            hipDeviceAttributeMaxTexture1DLayered
        cudaDevAttrMaxTexture1DLinearWidth = \
            hipDeviceAttributeMaxTexture1DLinear
        cudaDevAttrMaxTexture1DMipmappedWidth = \
            hipDeviceAttributeMaxTexture1DMipmap
        cudaDevAttrMaxTexture2DWidth = hipDeviceAttributeMaxTexture2DWidth
        cudaDevAttrMaxTexture2DHeight = hipDeviceAttributeMaxTexture2DHeight
        cudaDevAttrMaxTexture2DGatherWidth = \
            hipDeviceAttributeMaxTexture2DGather
        cudaDevAttrMaxTexture2DGatherHeight = \
            hipDeviceAttributeMaxTexture2DGather
        cudaDevAttrMaxTexture2DLayeredWidth = \
            hipDeviceAttributeMaxTexture2DLayered
        cudaDevAttrMaxTexture2DLayeredHeight = \
            hipDeviceAttributeMaxTexture2DLayered
        cudaDevAttrMaxTexture2DLayeredLayers = \
            hipDeviceAttributeMaxTexture2DLayered
        cudaDevAttrMaxTexture2DLinearWidth = \
            hipDeviceAttributeMaxTexture2DLinear
        cudaDevAttrMaxTexture2DLinearHeight = \
            hipDeviceAttributeMaxTexture2DLinear
        cudaDevAttrMaxTexture2DLinearPitch = \
            hipDeviceAttributeMaxTexture2DLinear
        cudaDevAttrMaxTexture2DMipmappedWidth = \
            hipDeviceAttributeMaxTexture2DMipmap
        cudaDevAttrMaxTexture2DMipmappedHeight = \
            hipDeviceAttributeMaxTexture2DMipmap
        cudaDevAttrMaxTexture3DWidth = hipDeviceAttributeMaxTexture3DWidth
        cudaDevAttrMaxTexture3DHeight = hipDeviceAttributeMaxTexture3DHeight
        cudaDevAttrMaxTexture3DDepth = hipDeviceAttributeMaxTexture3DDepth
        cudaDevAttrMaxTexture3DWidthAlt = hipDeviceAttributeMaxTexture3DAlt
        cudaDevAttrMaxTexture3DHeightAlt = hipDeviceAttributeMaxTexture3DAlt
        cudaDevAttrMaxTexture3DDepthAlt = hipDeviceAttributeMaxTexture3DAlt
        cudaDevAttrMaxTextureCubemapWidth = \
            hipDeviceAttributeMaxTextureCubemap
        cudaDevAttrMaxTextureCubemapLayeredWidth = \
            hipDeviceAttributeMaxTextureCubemapLayered
        cudaDevAttrMaxTextureCubemapLayeredLayers = \
            hipDeviceAttributeMaxTextureCubemapLayered
        cudaDevAttrMaxThreadsPerBlock = hipDeviceAttributeMaxThreadsPerBlock
        cudaDevAttrMaxThreadsPerMultiProcessor = \
            hipDeviceAttributeMaxThreadsPerMultiProcessor
        cudaDevAttrMaxPitch = hipDeviceAttributeMaxPitch
        cudaDevAttrGlobalMemoryBusWidth = hipDeviceAttributeMemoryBusWidth
        cudaDevAttrMemoryClockRate = hipDeviceAttributeMemoryClockRate
        cudaDevAttrMultiGpuBoardGroupID = hipDeviceAttributeMultiGpuBoardGroupID
        cudaDevAttrMultiProcessorCount = hipDeviceAttributeMultiprocessorCount
        cudaDevAttrPageableMemoryAccess = \
            hipDeviceAttributePageableMemoryAccess
        cudaDevAttrPageableMemoryAccessUsesHostPageTables = \
            hipDeviceAttributePageableMemoryAccessUsesHostPageTables
        cudaDevAttrPciBusId = hipDeviceAttributePciBusId
        cudaDevAttrPciDeviceId = hipDeviceAttributePciDeviceId
        cudaDevAttrPciDomainId = hipDeviceAttributePciDomainID
        cudaDevAttrPersistingL2CacheMaxSize = \
            hipDeviceAttributePersistingL2CacheMaxSize
        cudaDevAttrMaxRegistersPerBlock = \
            hipDeviceAttributeMaxRegistersPerBlock
        cudaDevAttrMaxRegistersPerMultiprocessor = \
            hipDeviceAttributeMaxRegistersPerMultiprocessor
        cudaDevAttrReservedSharedMemoryPerBlock = \
            hipDeviceAttributeReservedSharedMemPerBlock
        cudaDevAttrMaxSharedMemoryPerBlock = \
            hipDeviceAttributeMaxSharedMemoryPerBlock
        cudaDevAttrMaxSharedMemoryPerBlockOptin = \
            hipDeviceAttributeSharedMemPerBlockOptin
        cudaDevAttrMaxSharedMemoryPerMultiprocessor = \
            hipDeviceAttributeSharedMemPerMultiprocessor
        cudaDevAttrSingleToDoublePrecisionPerfRatio = \
            hipDeviceAttributeSingleToDoublePrecisionPerfRatio
        cudaDevAttrStreamPrioritiesSupported = \
            hipDeviceAttributeStreamPrioritiesSupported
        cudaDevAttrSurfaceAlignment = hipDeviceAttributeSurfaceAlignment
        cudaDevAttrTccDriver = hipDeviceAttributeTccDriver
        cudaDevAttrTextureAlignment = hipDeviceAttributeTextureAlignment
        cudaDevAttrTexturePitchAlignment = \
            hipDeviceAttributeTexturePitchAlignment
        cudaDevAttrTotalConstantMemory = hipDeviceAttributeTotalConstantMemory
        cudaDevAttrTotalGlobalMem = hipDeviceAttributeTotalGlobalMem
        cudaDevAttrUnifiedAddressing = hipDeviceAttributeUnifiedAddressing
        cudaDevAttrUuid = hipDeviceAttributeUuid
        cudaDevAttrWarpSize = hipDeviceAttributeWarpSize
ELIF CUPY_HIP_VERSION > 0:
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
        # The following attributes do not exist in CUDA and cause segfaults
        # if we try to access them
        # hipDeviceAttributeHdpMemFlushCntl
        # hipDeviceAttributeHdpRegFlushCntl
        cudaDevAttrMaxPitch = 36
        cudaDevAttrTextureAlignment
        cudaDevAttrTexturePitchAlignment
        cudaDevAttrKernelExecTimeout

        cudaDevAttrCanMapHostMemory
        cudaDevAttrEccEnabled
        cudaDevAttrMemoryPoolsSupported = 0
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
    IF CUPY_HIP_VERSION >= 310:
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
        # added since CUDA 11.3
        cudaDevAttrGPUDirectRDMASupported
        cudaDevAttrGPUDirectRDMAFlushWritesOptions
        cudaDevAttrGPUDirectRDMAWritesOrdering
        cudaDevAttrMemoryPoolSupportedHandleTypes
