// NVTX event-attribute compatibility backed by real rocTX operations.

#ifndef INCLUDE_GUARD_CUPY_HIP_NVTX_H
#define INCLUDE_GUARD_CUPY_HIP_NVTX_H

#include <stdint.h>
#include "cupy_roctx.h"

extern "C" {

typedef enum nvtxColorType_t
{
    NVTX_COLOR_UNKNOWN  = 0,
    NVTX_COLOR_ARGB     = 1
} nvtxColorType_t;

typedef enum nvtxMessageType_t
{
    NVTX_MESSAGE_UNKNOWN          = 0,
    NVTX_MESSAGE_TYPE_ASCII       = 1,
    NVTX_MESSAGE_TYPE_UNICODE     = 2,
} nvtxMessageType_t;

typedef union nvtxMessageValue_t
{
    const char* ascii;
    const wchar_t* unicode;
} nvtxMessageValue_t;

typedef struct nvtxEventAttributes_v1
{
    uint16_t version;
    uint16_t size;
    uint32_t category;
    int32_t colorType;
    uint32_t color;
    int32_t payloadType;
    int32_t reserved0;
    union payload_t
    {
        uint64_t ullValue;
        int64_t llValue;
        double dValue;
    } payload;
    int32_t messageType;
    nvtxMessageValue_t message;
} nvtxEventAttributes_v1;

typedef nvtxEventAttributes_v1 nvtxEventAttributes_t;

static const char* cupy_roctx_message(
        const nvtxEventAttributes_t* eventAttrib) {
    return (eventAttrib->messageType == NVTX_MESSAGE_TYPE_ASCII &&
            eventAttrib->message.ascii != nullptr)
        ? eventAttrib->message.ascii : "";
}

void nvtxMarkEx(const nvtxEventAttributes_t* eventAttrib) {
    roctxMarkA(cupy_roctx_message(eventAttrib));
}

int nvtxRangePushEx(const nvtxEventAttributes_t* eventAttrib) {
    return roctxRangePushA(cupy_roctx_message(eventAttrib));
}

uint64_t nvtxRangeStartEx(const nvtxEventAttributes_t* eventAttrib) {
    return roctxRangeStartA(cupy_roctx_message(eventAttrib));
}

void nvtxRangeEnd(uint64_t range_id) {
    roctxRangeStop(static_cast<roctx_range_id_t>(range_id));
}

} // extern "C"


#endif
