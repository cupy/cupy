// This file is a stub header file of cuda for Read the Docs.

#ifndef INCLUDE_GUARD_STUB_CUPY_NVTX_H
#define INCLUDE_GUARD_STUB_CUPY_NVTX_H

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

void nvtxMarkEx(...) {
}

int nvtxRangePushEx(...) {
    return 0;
}

uint64_t nvtxRangeStartEx(...) {
    return 0;
}

void nvtxRangeEnd(...) {
}

} // extern "C"


#endif
