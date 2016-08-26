"""
Wrapper for NVIDIA Tools Extension Library (NVTX)

"""
cdef extern from "cupy_cuda.h":
    cdef int NVTX_VERSION
    cdef int NVTX_EVENT_ATTRIB_STRUCT_SIZE
    cdef int NVTX_NO_PUSH_POP_TRACKING
    cdef enum nvtxColorType_t:
        NVTX_COLOR_UNKNOWN
        NVTX_COLOR_ARGB
    cdef enum nvtxPayloadType_t:
        NVTX_PAYLOAD_UNKNOWN
        NVTX_PAYLOAD_TYPE_UNSIGNED_INT64
        NVTX_PAYLOAD_TYPE_INT64
        NVTX_PAYLOAD_TYPE_DOUBLE
    cdef enum nvtxMessageType_t:
        NVTX_MESSAGE_UNKNOWN
        NVTX_MESSAGE_TYPE_ASCII
        NVTX_MESSAGE_TYPE_UNICODE
    ctypedef   signed char         int8_t
    ctypedef   signed short       int16_t
    ctypedef   signed int         int32_t
    ctypedef   signed long long   int64_t
    ctypedef unsigned char        uint8_t
    ctypedef unsigned short      uint16_t
    ctypedef unsigned int        uint32_t
    ctypedef unsigned long long  uint64_t
    ctypedef          void        wchar_t
    cdef union payload_t:
        uint64_t  ullValue
        int64_t    llValue
        double      dValue
    cdef union message_t:
        char* ascii
        wchar_t* unicode
    cdef struct nvtxEventAttributes_v1:
        uint16_t   version
        uint16_t   size
        uint32_t   category
        int32_t    colorType
        uint32_t   color
        int32_t    payloadType
        int32_t    reserved0
        payload_t  payload
        int32_t    messageType
        message_t  message
    ctypedef nvtxEventAttributes_v1 nvtxEventAttributes_t
    void nvtxMarkA(const char *message)
    void nvtxMarkEx(const nvtxEventAttributes_t *eventAttrib)
    int nvtxRangePushA(const char *message)
    int nvtxRangePushEx(const nvtxEventAttributes_t *eventAttrib)
    int nvtxRangePop()
