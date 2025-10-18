# Keep in sync with typenames exported in `runtime.pxd`.

from backends.backend.api cimport driver


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

    IF 0 < CUPY_CUDA_VERSION:
        ctypedef struct _PointerAttributes 'cudaPointerAttributes':
            int type
            int device
            void* devicePointer
            void* hostPointer
    ELIF 0 < CUPY_HIP_VERSION < 60000000:
        ctypedef struct _PointerAttributes 'cudaPointerAttributes':
            int memoryType
            int device
            void* devicePointer
            void* hostPointer
    ELIF 60000000 <= CUPY_HIP_VERSION:
        ctypedef struct _PointerAttributes 'cudaPointerAttributes':
            int type
            int device
            void* devicePointer
            void* hostPointer
    ELSE:
        ctypedef struct _PointerAttributes 'cudaPointerAttributes':
            pass  # for RTD

