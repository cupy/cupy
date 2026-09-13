// Host-side launcher for custom AscendC kernels (numpy-ascend).
//
// Kernels are compiled out-of-tree by bisheng into an aicore fatbin (.o) and
// loaded through the aclrt binary API -- no dlopen / no host .so involved:
//   aclrtBinaryLoadFromFile -> aclrtBinaryGetFunction ->
//   aclrtKernelArgsInit/Append/Finalize -> aclrtLaunchKernelWithConfig
//
// Kernel calling convention (see kernels/ascendc_elementwise.cpp):
//   kernel(out0, out1, in0, in1, n, perBlock)
// with unused pointer slots = nullptr. Block instance b handles
// [b*perBlock, min(n, (b+1)*perBlock)).
//
// NOTE: binary/function handles are cached per fatbin path. Cache access is
// guarded by the GIL (all callers run under Python), so no extra locking.
#ifndef CUPY_ACL_CUSTOM_KERNELS_H
#define CUPY_ACL_CUSTOM_KERNELS_H

#include <map>
#include <string>
#include <utility>
#include <cstdint>
#include "acl/acl.h"

// Elements processed per AI-core block instance. Must be a multiple of 128
// (the vector width / 32B alignment expected by the kernels).
#define CUPY_CUSTOM_KERNEL_PER_BLOCK 8192ULL

inline aclError aclop_LaunchCustomKernel(
        const char* bin_path, const char* func_name,
        void* out0, void* out1, void* in0, void* in1,
        uint64_t n, aclrtStream stream) {
    typedef std::map<std::string, std::pair<aclrtBinHandle, aclrtFuncHandle> > HandleCache;
    static HandleCache cache;

    aclrtFuncHandle func = nullptr;
    std::string key(bin_path);
    HandleCache::iterator it = cache.find(key);
    if (it == cache.end()) {
        aclrtBinHandle bin = nullptr;
        aclError ret = aclrtBinaryLoadFromFile(bin_path, nullptr, &bin);
        if (ret != 0) { return ret; }
        ret = aclrtBinaryGetFunction(bin, func_name, &func);
        if (ret != 0) { return ret; }
        it = cache.insert(std::make_pair(key, std::make_pair(bin, func))).first;
    }
    func = it->second.second;

    aclrtArgsHandle args = nullptr;
    aclrtParamHandle param = nullptr;
    aclError ret = aclrtKernelArgsInit(func, &args);
    if (ret != 0) { return ret; }

    // pointer params (value-copied, c.f. the aclrt demo)
    ret = aclrtKernelArgsAppend(args, &out0, sizeof(void*), &param);
    if (ret != 0) { return ret; }
    ret = aclrtKernelArgsAppend(args, &out1, sizeof(void*), &param);
    if (ret != 0) { return ret; }
    ret = aclrtKernelArgsAppend(args, &in0, sizeof(void*), &param);
    if (ret != 0) { return ret; }
    ret = aclrtKernelArgsAppend(args, &in1, sizeof(void*), &param);
    if (ret != 0) { return ret; }

    uint64_t per_block = CUPY_CUSTOM_KERNEL_PER_BLOCK;
    ret = aclrtKernelArgsAppend(args, &n, sizeof(uint64_t), &param);
    if (ret != 0) { return ret; }
    ret = aclrtKernelArgsAppend(args, &per_block, sizeof(uint64_t), &param);
    if (ret != 0) { return ret; }

    ret = aclrtKernelArgsFinalize(args);
    if (ret != 0) { return ret; }

    uint32_t num_blocks = static_cast<uint32_t>((n + per_block - 1) / per_block);
    if (num_blocks == 0) { return 0; }  // nothing to do
    return aclrtLaunchKernelWithConfig(func, num_blocks, stream, nullptr, args, nullptr);
}

#endif  // CUPY_ACL_CUSTOM_KERNELS_H
