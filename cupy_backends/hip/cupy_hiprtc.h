#ifndef INCLUDE_GUARD_HIP_CUPY_HIPRTC_H
#define INCLUDE_GUARD_HIP_CUPY_HIPRTC_H

#include <hip/hiprtc.h>

extern "C" {

typedef hiprtcResult nvrtcResult;
const nvrtcResult NVRTC_SUCCESS = HIPRTC_SUCCESS;

typedef hiprtcProgram nvrtcProgram;

const char *nvrtcGetErrorString(nvrtcResult result) {
    return hiprtcGetErrorString(result);
}

nvrtcResult nvrtcVersion(int* major, int* minor) {
    return hiprtcVersion(major, minor);
}

nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog, const char* src,
                               const char* name, int numHeaders,
                               const char** headers,
                               const char** includeNames) {
    return hiprtcCreateProgram(prog, src, name, numHeaders, headers, includeNames);
}

nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog) {
    return hiprtcDestroyProgram(prog);
}

nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions,
                                const char** options) {
    return hiprtcCompileProgram(prog, numOptions, options);
}

nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, std::size_t* codeSizeRet) {
    return hiprtcGetCodeSize(prog, codeSizeRet);
}

nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* code) {
    return hiprtcGetCode(prog, code);
}

nvrtcResult nvrtcGetCUBINSize(...) {
    return HIPRTC_ERROR_COMPILATION;
}

nvrtcResult nvrtcGetCUBIN(...) {
    return HIPRTC_ERROR_COMPILATION;
}

nvrtcResult nvrtcGetNumSupportedArchs(...) {
    return HIPRTC_ERROR_INTERNAL_ERROR;
}

nvrtcResult nvrtcGetSupportedArchs(...) {
    return HIPRTC_ERROR_INTERNAL_ERROR;
}

nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, std::size_t* logSizeRet) {
    return hiprtcGetProgramLogSize(prog, logSizeRet);
}

nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log) {
    return hiprtcGetProgramLog(prog, log);
}

nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog,
                                   const char* name_expression) {
    return hiprtcAddNameExpression(prog, name_expression);
}

nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog,
                                const char* name_expression,
                                const char** lowered_name ) {
    return hiprtcGetLoweredName(prog, name_expression, lowered_name);
}

}

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_HIPRTC_H
