cpdef enum:
    # need to revisit this when cython supports C++ enums (in 3.0)
    # https://stackoverflow.com/a/67138945

    cudaMemoryTypeHost = 1
    cudaMemoryTypeDevice = 2

    cudaIpcMemLazyEnablePeerAccess = 1

    cudaMemAttachGlobal = 1
    cudaMemAttachHost = 2
    cudaMemAttachSingle = 4

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
    cudaMemLocationTypeDevice = 1


# This was a legacy mistake: the prefix "cuda" should have been removed
# so that we can directly assign their C counterparts here. Now because
# of backward compatibility and no flexible Cython macro (IF/ELSE), we
# have to duplicate the enum. (CUDA and HIP use different values!)
IF CUPY_HIP_VERSION > 0:
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
