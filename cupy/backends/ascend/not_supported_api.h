

#ifndef CUPY_INSTALL_USE_ASCEND
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    // // WARNING: AscendCL device properties structure (aclrtDeviceProp) is different from cudaDeviceProp.
    // // Requires manual mapping of fields.
    // aclrtDeviceProp aclProp;
    // aclError ret = aclrtGetDeviceProperties(&aclProp, device);
    // if (ret != ACL_SUCCESS) {
    //     return ret;
    // }
    // // Map relevant fields from aclProp to prop (e.g., name, compute capability)
    // // prop->name = aclProp.name; // Example
    // // ... other fields
    // return ACL_SUCCESS;
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaDeviceGetAttribute(int* pi, cudaDeviceAttr attr,
                                   int deviceId) {
    // // WARNING: AscendCL device attributes are different from CUDA/HIP.
    // // Use aclrtGetDeviceProperties or specific aclGet* functions. 
    // // This is a placeholder and requires mapping CUDA attributes to AscendCL properties.
    // aclrtDeviceProp prop;
    // aclError ret = aclrtGetDeviceProperties(&prop, deviceId);
    // if (ret != ACL_SUCCESS) {
    //     return ret;
    // }
    // // Map specific attr to prop fields (example for compute capability)
    // switch(attr) {
    //     // case cudaDevAttrComputeCapabilityMajor: // WARNING: No direct equivalent
    //     //     *pi = 0; // Placeholder
    //     //     break;
    //     default:
    //         return ACL_ERROR_INVALID_PARAM;
    // }
    // return ACL_SUCCESS;
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId) {
    // WARNING: Missing direct equivalent in AscendCL for PCI Bus ID lookup.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    // WARNING: Missing direct equivalent in AscendCL for getting PCI Bus ID.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}
#endif

#ifndef CUPY_INSTALL_USE_ASCEND
cudaError_t cudaMalloc3DArray(...) {
    // WARNING: Missing direct equivalent in AscendCL for 3D array allocation.
    // AscendCL uses aclDataBuffer and aclTensorDesc for data management.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMallocArray(...) {
    // WARNING: Missing direct equivalent in AscendCL for array allocation.
    // AscendCL uses aclDataBuffer and aclTensorDesc for data management.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaFreeArray(...) {
    // WARNING: Missing direct equivalent in AscendCL for freeing arrays.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemcpy2DFromArray(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemcpy2DFromArrayAsync(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemcpy2DToArray(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemcpy2DToArrayAsync(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemcpy3D(...) {
    // WARNING: Missing direct equivalent in AscendCL for 3D memory copy.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemcpy3DAsync(...) {
    // WARNING: Missing direct equivalent in AscendCL for async 3D memory copy.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}
#endif


#ifndef CUPY_INSTALL_USE_ASCEND
// ====================== MemPool is supported by ascend ===================
// ====================== MemPool is also not supported on ROCm ===================
cudaError_t cudaMallocFromPoolAsync(...) {
    // WARNING: Missing direct equivalent in AscendCL for memory pools.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemPoolCreate(...) {
    // WARNING: Missing direct equivalent in AscendCL for memory pool creation.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemPoolDestroy(...) {
    // WARNING: Missing direct equivalent in AscendCL for memory pool destruction.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaDeviceGetDefaultMemPool(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaDeviceGetMemPool(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaDeviceSetMemPool(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemPoolTrimTo(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemPoolGetAttribute(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}

cudaError_t cudaMemPoolSetAttribute(...) {
    // WARNING: Missing direct equivalent in AscendCL.
    return ACL_ERROR_FEATURE_UNSUPPORTED;
}
#endif