"""
Wrapper for NVIDIA Tools Extension Library (NVTX)

"""
from libc cimport string

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


cpdef void MarkC(str message, uint32_t color=0) except *:
    """
    Marks an instantaneous event (marker) in the application.

    Markes are used to describe events at a specific time during execution of
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
    attrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE
    attrib.messageType = NVTX_MESSAGE_TYPE_ASCII
    attrib.message.ascii = b_message
    attrib.colorType = NVTX_COLOR_ARGB
    attrib.color = color

    nvtxMarkEx(&attrib)


cpdef void Mark(str message, int id_color=-1) except *:
    """
    Marks an instantaneous event (marker) in the application.

    Markes are used to describe events at a specific time during execution of
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


cpdef void RangePushC(str message, uint32_t color=0) except *:
    """
    Starts a nestead range.

    Ranges are used to describe events over a time span during execution of
    the application. The duration of a range is defined by the corresponding
    pair of RangePush*() to RangePop() calls.

    Args:
        message (str): Name of a range.
        color (uint32): Color for a range.
    """
    cdef bytes b_message = message.encode()
    if NVTX_VERSION != 1 and NVTX_VERSION != 2:
        nvtxRangePushA(<const char*>b_message)
        return

    cdef nvtxEventAttributes_t attrib
    string.memset(&attrib, 0, sizeof(attrib))
    attrib.version = NVTX_VERSION
    attrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE
    attrib.messageType = NVTX_MESSAGE_TYPE_ASCII
    attrib.message.ascii = b_message
    attrib.colorType = NVTX_COLOR_ARGB
    attrib.color = color

    nvtxRangePushEx(&attrib)


cpdef void RangePush(str message, int id_color=-1) except *:
    """
    Starts a nestead range.

    Ranges are used to describe events over a time span during execution of
    the application. The duration of a range is defined by the corresponding
    pair of RangePush*() to RangePop() calls.

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


cpdef void RangePop() except *:
    """
    Ends a nestead range.

    Ranges are used to describe events over a time span during execution of
    the application. The duration of a range is defined by the corresponding
    pair of RangePush*() to RangePop() calls.
    """
    nvtxRangePop()
