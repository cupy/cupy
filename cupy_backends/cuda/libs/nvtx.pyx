# distutils: language = c++

"""
Wrapper for NVIDIA Tools Extension Library (NVTX)

"""
from libc cimport string

cdef extern from '../../cupy_tx.h' nogil:
    cdef int NVTX_VERSION
    cdef enum nvtxColorType_t:
        NVTX_COLOR_UNKNOWN
        NVTX_COLOR_ARGB
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
    ctypedef unsigned long long range_id_t
    void nvtxMarkA(const char *message)
    void nvtxMarkEx(const nvtxEventAttributes_t *eventAttrib)
    int nvtxRangePushA(const char *message)
    int nvtxRangePushEx(const nvtxEventAttributes_t *eventAttrib)
    int nvtxRangePop()
    range_id_t nvtxRangeStartEx(const nvtxEventAttributes_t *eventAttrib)
    void nvtxRangeEnd(range_id_t)

cdef int num_colors = 10
cdef uint32_t colors[10]
colors[0] = 0xFF00FF00
colors[1] = 0xFF007FFF
colors[2] = 0xFF7F00FF
colors[3] = 0xFFFF0000
colors[4] = 0xFF7FFF00
colors[5] = 0xFF00FF7F
colors[6] = 0xFF0000FF
colors[7] = 0xFFFF007F
colors[8] = 0xFFFF7F00
colors[9] = 0xFF7F7F7F

available = True


cdef nvtxEventAttributes_t make_event_attributes(message, color):
    cdef bytes b_message
    cdef nvtxEventAttributes_t attrib

    string.memset(&attrib, 0, sizeof(attrib))
    attrib.version = NVTX_VERSION
    attrib.size = sizeof(attrib)

    if color is None:
        attrib.colorType = NVTX_COLOR_UNKNOWN
    else:
        attrib.color = color
        attrib.colorType = NVTX_COLOR_ARGB

    if message is None:
        attrib.messageType = NVTX_MESSAGE_UNKNOWN
    else:
        attrib.messageType = NVTX_MESSAGE_TYPE_ASCII
        b_message = message.encode()
        attrib.message.ascii = b_message

    return attrib


cpdef MarkC(message, uint32_t color=0):
    """
    Marks an instantaneous event (marker) in the application.

    Markers are used to describe events at a specific time during execution of
    the application.

    Args:
        message (str): Name of a marker.
        color (uint32): Color code for a marker.
    """
    cdef bytes b_message = message.encode()
    if NVTX_VERSION != 1 and NVTX_VERSION != 2:
        nvtxMarkA(<const char*>b_message)
        return

    cdef nvtxEventAttributes_t attrib
    string.memset(&attrib, 0, sizeof(attrib))
    attrib.version = NVTX_VERSION
    attrib.size = sizeof(attrib)
    attrib.messageType = NVTX_MESSAGE_TYPE_ASCII
    attrib.message.ascii = b_message
    attrib.colorType = NVTX_COLOR_ARGB
    attrib.color = color

    nvtxMarkEx(&attrib)


cpdef Mark(message, int id_color=-1):
    """
    Marks an instantaneous event (marker) in the application.

    Markers are used to describe events at a specific time during execution of
    the application.

    Args:
        message (str): Name of a marker.
        id_color (int): ID of color for a marker.
    """
    cdef bytes b_message = message.encode()
    if id_color < 0 or (NVTX_VERSION != 1 and NVTX_VERSION != 2):
        nvtxMarkA(<const char*>b_message)
        return

    cdef uint32_t color = colors[id_color % num_colors]
    MarkC(message, color)


cpdef RangePushC(message, uint32_t color=0):
    """
    Starts a nested range.

    Ranges are used to describe events over a time span during execution of
    the application. The duration of a range is defined by the corresponding
    pair of ``RangePush*()`` to ``RangePop()`` calls.

    Args:
        message (str): Name of a range.
        color (uint32): ARGB color for a range.
    """
    cdef bytes b_message = message.encode()
    if NVTX_VERSION != 1 and NVTX_VERSION != 2:
        nvtxRangePushA(<const char*>b_message)
        return

    cdef nvtxEventAttributes_t attrib
    string.memset(&attrib, 0, sizeof(attrib))
    attrib.version = NVTX_VERSION
    attrib.size = sizeof(attrib)
    attrib.messageType = NVTX_MESSAGE_TYPE_ASCII
    attrib.message.ascii = b_message
    attrib.colorType = NVTX_COLOR_ARGB
    attrib.color = color

    nvtxRangePushEx(&attrib)


cpdef RangePush(message, int id_color=-1):
    """
    Starts a nested range.

    Ranges are used to describe events over a time span during execution of
    the application. The duration of a range is defined by the corresponding
    pair of ``RangePush*()`` to ``RangePop()`` calls.

    Args:
        message (str): Name of a range.
        id_color (int): ID of color for a range.
    """
    cdef bytes b_message = message.encode()
    if id_color < 0 or (NVTX_VERSION != 1 and NVTX_VERSION != 2):
        nvtxRangePushA(<const char*>b_message)
        return

    cdef uint32_t color = colors[id_color % num_colors]
    RangePushC(message, color)


cpdef RangePop():
    """
    Ends a nested range.

    Ranges are used to describe events over a time span during execution of
    the application. The duration of a range is defined by the corresponding
    pair of ``RangePush*()`` to ``RangePop()`` calls.
    """
    nvtxRangePop()


cpdef unsigned long long RangeStart(message, color) except? 0:
    cdef nvtxEventAttributes_t attrib = make_event_attributes(message, color)
    return nvtxRangeStartEx(&attrib)


cpdef RangeEnd(unsigned long long range_id):
    nvtxRangeEnd(range_id)
