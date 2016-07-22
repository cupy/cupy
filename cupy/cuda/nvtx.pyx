"""
Wrapper for NVIDIA Tools Extension Library (NVTX)

"""
cimport cython


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


cpdef void Mark(str message, int id_color=-1) except *:
    cdef bytes b_message = message.encode()
    if id_color < 0 or NVTX_VERSION != 1:
        nvtxMarkA(<const char*>b_message)
        return

    cdef nvtxEventAttributes_t attrib[1]
    if NVTX_VERSION == 1:
        attrib[0].version = 0
        attrib[0].size = 0
        attrib[0].category = 0
        attrib[0].colorType = 0
        attrib[0].color = 0
        attrib[0].payloadType = 0
        attrib[0].reserved0 = 0
        attrib[0].payload.ullValue = 0
        attrib[0].messageType = 0
        attrib[0].message.ascii = NULL

    attrib[0].version = NVTX_VERSION
    attrib[0].size = NVTX_EVENT_ATTRIB_STRUCT_SIZE
    attrib[0].messageType = NVTX_MESSAGE_TYPE_ASCII
    attrib[0].message.ascii = b_message
    attrib[0].colorType = NVTX_COLOR_ARGB
    attrib[0].color = colors[id_color % num_colors]

    nvtxMarkEx(attrib)


cpdef void RangePush(str message, int id_color=-1) except *:
    cdef bytes b_message = message.encode()
    if id_color < 0 or NVTX_VERSION != 1:
        ret = nvtxRangePushA(<const char*>b_message)
        # assert ret >= 0, 'nvtxRangePushA(): {}'.format(ret)
        return

    cdef nvtxEventAttributes_t attrib[1]
    if NVTX_VERSION == 1:
        attrib[0].version = 0
        attrib[0].size = 0
        attrib[0].category = 0
        attrib[0].colorType = 0
        attrib[0].color = 0
        attrib[0].payloadType = 0
        attrib[0].reserved0 = 0
        attrib[0].payload.ullValue = 0
        attrib[0].messageType = 0
        attrib[0].message.ascii = NULL

    attrib[0].version = NVTX_VERSION
    attrib[0].size = NVTX_EVENT_ATTRIB_STRUCT_SIZE
    attrib[0].messageType = NVTX_MESSAGE_TYPE_ASCII
    attrib[0].message.ascii = b_message
    attrib[0].colorType = NVTX_COLOR_ARGB
    attrib[0].color = colors[id_color % num_colors]

    ret = nvtxRangePushEx(attrib)
    # assert ret >= 0, 'nvtxRangePushEx(): {}'.format(ret)


cpdef void RangePop() except *:
    ret = nvtxRangePop()
    # assert ret >= 0, 'nvtxRangePop(): {}'.format(ret)
