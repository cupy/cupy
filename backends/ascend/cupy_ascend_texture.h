// ================ GPU render API is not supported on ASCEND NPU============================
#ifndef CUPY_INSTALL_USE_ASCEND
// Texture is not supported on NPU
cudaError_t cudaCreateTextureObject(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaDestroyTextureObject(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaGetChannelDesc(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaGetTextureObjectResourceDesc(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaGetTextureObjectTextureDesc(...) {
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaExtent make_cudaExtent(...) {
    cudaExtent ex = {};
    return ex;
}

cudaPitchedPtr make_cudaPitchedPtr(...) {
    cudaPitchedPtr ptr = {};
    return ptr;
}

cudaPos make_cudaPos(...) {
    cudaPos pos = {};
    return pos;
}

// Surface
cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject,
                                    const cudaResourceDesc* pResDesc) {
    // return hipCreateSurfaceObject(pSurfObject, pResDesc);
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) {
    // return hipDestroySurfaceObject(surfObject);
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}
#endif