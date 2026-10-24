#ifndef CUPY_ACL_RANDOM_OPS_HEADER
#define CUPY_ACL_RANDOM_OPS_HEADER

// Random number generation for cupy.random on Ascend.
// See docs/ascend/DeveloperNotes.md §"ops-rand 随机数库与 cupy.random 移植".
//
// Two tiers:
// * BASE OPS (unconditional): aclnnMultinomial belongs to the core CANN op
//   library (libopapi) present in every supported CANN release — it is NOT
//   part of the ops-rand family, so it is compiled without any gate.
// * OPS-RAND FAMILY (feature-detected): aclnnInplaceUniform/Normal/Random
//   do not exist in every CANN release / SoC package. Two independent
//   gates, both must pass:
//     1. CUPY_CANN_HAS_RAND — computed at build time by
//        AscendBackend.has_aclnn_rand() (header-presence probe, overridable
//        with CUPY_ENABLE_ACLRAND=0/1) and injected as a C macro AND a
//        Cython compile-time constant (DeveloperNotes.md §3.5 dual channel).
//     2. __has_include — a second, per-build guard so a mis-detected SDK
//        degrades to "no random ops" instead of a hard compile error.
//   Cython cannot express __has_include, so acl_utils.pyx gates only on
//   `IF CUPY_CANN_HAS_RAND:`; when the two gates disagree the link step
//   fails loudly rather than silently dropping the ops.

#include "./acl_op_template.h"
#include "./acl_scalar_arg.h"

// ---------------------------------------------------------------------------
// BASE OP: aclnnMultinomial (cupy.random.multinomial)
// NOTE the semantics: aclnnMultinomial has torch.multinomial semantics
// (draws numsamples category *indices* per row from an (N, C) probability
// tensor); numpy-style *counts* are derived by the caller with a flat
// bincount over row offsets (cupy/random/_sample.py).
// ---------------------------------------------------------------------------
#include "aclnnop/aclnn_multinomial.h"

// ins = [pvals (1-D or (N, C), float)], outs = [indices (numsamples,) or
// (N, numsamples), INT64], args = [numsamples, replacement, seed, offset]
aclError aclop_Multinomial(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
    const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
    if (ins.empty() || outs.empty()) {
        PrintArgs(__func__, args, kwargs, std::cout);
        return ACL_ERROR_INVALID_PARAM;
    }
    int64_t numsamples = GetScalarArg<int64_t>(args, 0, kwargs, "numsamples", 0);
    bool replacement = GetScalarArg<bool>(args, 1, kwargs, "replacement", true);
    int64_t seed = GetScalarArg<int64_t>(args, 2, kwargs, "seed", 0);
    int64_t offset = GetScalarArg<int64_t>(args, 3, kwargs, "offset", 0);
    return aclIrregularOpRun(aclnnMultinomialGetWorkspaceSize, aclnnMultinomial, stream,
        ins[0], numsamples, replacement, seed, offset, outs[0]);
}

// ---------------------------------------------------------------------------
// OPS-RAND FAMILY (feature-detected, see the two gates above)
// ---------------------------------------------------------------------------

#if defined(CUPY_CANN_HAS_RAND) && CUPY_CANN_HAS_RAND
#if __has_include(<aclnnop/aclnn_uniform.h>) \
    && __has_include(<aclnnop/aclnn_normal.h>) \
    && __has_include(<aclnnop/aclnn_random.h>)
#define CUPY_ACL_RAND_OPS_AVAILABLE 1
#endif
#endif

#ifdef CUPY_ACL_RAND_OPS_AVAILABLE

// All aclnn "rand" ops are STATELESS: (seed, offset) are passed explicitly on
// every call, which maps 1:1 onto cupy.random.RandomState's host-side
// monotonic counter (_rk_seed / _ascend_offset):
//   aclnnInplaceUniform(selfRef, from, to, seed, offset)   float  [from, to)
//   aclnnInplaceNormal (selfRef, mean, std, seed, offset)  normal(mean, std^2)
//   aclnnInplaceRandom (selfRef, from, to, seed, offset)   int    [from, to-1]
//
// These are pure "fill" ops: the destination tensor is outs[0] and `ins` is
// empty. Called from pure Python via
//   acl_utils.py_launch_general("ascend_random_uniform", [], [out],
//                               [from, to, seed, offset], {})

#include "aclnnop/aclnn_uniform.h"
#include "aclnnop/aclnn_normal.h"
#include "aclnnop/aclnn_random.h"

// cupy.random.RandomState._random_sample_raw / .uniform (scalar low/high):
// fill `out` with uniform floats over [from, to).
aclError aclop_RandomUniform(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
    const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
    if (outs.empty()) {
        PrintArgs(__func__, args, kwargs, std::cout);
        return ACL_ERROR_INVALID_PARAM;
    }
    aclTensor* self = outs[0];
    double from = GetScalarArg<double>(args, 0, kwargs, "from", 0.0);
    double to = GetScalarArg<double>(args, 1, kwargs, "to", 1.0);
    int64_t seed = GetScalarArg<int64_t>(args, 2, kwargs, "seed", 0);
    int64_t offset = GetScalarArg<int64_t>(args, 3, kwargs, "offset", 0);
    return aclIrregularOpRun(aclnnInplaceUniformGetWorkspaceSize, aclnnInplaceUniform, stream,
        self, from, to, static_cast<uint64_t>(seed), static_cast<uint64_t>(offset));
}

// cupy.random.RandomState.normal (scalar loc/scale):
// fill `out` with N(mean, std^2) samples.
aclError aclop_RandomNormal(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
    const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
    if (outs.empty()) {
        PrintArgs(__func__, args, kwargs, std::cout);
        return ACL_ERROR_INVALID_PARAM;
    }
    aclTensor* self = outs[0];
    float mean = GetScalarArg<float>(args, 0, kwargs, "mean", 0.0f);
    float std = GetScalarArg<float>(args, 1, kwargs, "std", 1.0f);
    int64_t seed = GetScalarArg<int64_t>(args, 2, kwargs, "seed", 0);
    int64_t offset = GetScalarArg<int64_t>(args, 3, kwargs, "offset", 0);
    return aclIrregularOpRun(aclnnInplaceNormalGetWorkspaceSize, aclnnInplaceNormal, stream,
        self, mean, std, seed, offset);
}

// cupy.random.RandomState.randint (scalar bounds) / _permutation / tomaxint:
// fill `out` with discrete uniform integers over [from, to-1].
// Supported out dtypes (aclnnInplaceRandom): INT8/16/32/64, UINT8, BOOL,
// FLOAT/FLOAT16/DOUBLE (not useful for randint).
aclError aclop_RandomInt(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
    const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
    if (outs.empty()) {
        PrintArgs(__func__, args, kwargs, std::cout);
        return ACL_ERROR_INVALID_PARAM;
    }
    aclTensor* self = outs[0];
    int64_t from = GetScalarArg<int64_t>(args, 0, kwargs, "from", 0);
    int64_t to = GetScalarArg<int64_t>(args, 1, kwargs, "to", 1);
    int64_t seed = GetScalarArg<int64_t>(args, 2, kwargs, "seed", 0);
    int64_t offset = GetScalarArg<int64_t>(args, 3, kwargs, "offset", 0);
    return aclIrregularOpRun(aclnnInplaceRandomGetWorkspaceSize, aclnnInplaceRandom, stream,
        self, from, to, seed, offset);
}

#else  // !CUPY_ACL_RAND_OPS_AVAILABLE — no aclnn rand ops in this SDK;
       // the wrappers compile out entirely and cupy.random reports
       // NotImplementedError at runtime (see cupy/random/_generator.py).
#endif  // CUPY_ACL_RAND_OPS_AVAILABLE

#endif  // CUPY_ACL_RANDOM_OPS_HEADER
