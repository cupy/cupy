#ifndef CUPY_ACL_REDUCTION_OPS_HEADER
#define CUPY_ACL_REDUCTION_OPS_HEADER

#include <cmath>

// bool reduction op
#include "aclnnop/aclnn_all.h"
#include "aclnnop/aclnn_any.h"

#include "aclnnop/aclnn_mean.h"
#include "aclnnop/aclnn_std.h"
#include "aclnnop/aclnn_var.h"
#include "aclnnop/aclnn_bincount.h"
#include "aclnnop/aclnn_median.h"
#include "aclnnop/aclnn_median.h" // nan version
#include "aclnnop/aclnn_aminmax.h" // ptp :  aminmax
// AMax/Amin: reduction over the given axes (aclIntArray), keepDim controlled.
// Passing ALL axes in dim == whole-tensor reduction with keepdim support,
// which aclnnMax/aclnnMin (no dim, no keepdim) cannot express.
#include "aclnnop/aclnn_amax.h"
#include "aclnnop/aclnn_amin.h"
// missing quantile, percentile, impl in Python
#include "aclnnop/aclnn_histc.h"
#include "aclnnop/aclnn_reduce_nansum.h"
#include "aclnnop/aclnn_reduce_sum.h"
#include "aclnnop/aclnn_cast.h"  // any/all dtype fallback: cast input -> BOOL

// count_non_nan (= sum(!isnan(x))) has no aclnn counterpart and is composed in
// aclop_CountNonNaN below from ne_tensor + s_where + inplace fill + reduce_sum.
#include "aclnnop/aclnn_ne_tensor.h"
#include "aclnnop/aclnn_s_where.h"
#include "aclnnop/aclnn_fill_scalar.h"

// argmax/argmin are referenced by aclop_ArgMax/aclop_ArgMin/aclop_NanArgMax/
// aclop_NanArgMin below. acl_math_ops.h also includes them, but the generated
// acl_utils.cpp includes all backend headers and relies on include order, so
// declare the dependency explicitly here.
#include "aclnnop/aclnn_argmax.h"
#include "aclnnop/aclnn_argmin.h"

#include "./acl_op_template.h"
#include "acl/acl.h"
#include "acl_scalar_arg.h"


#ifdef __cplusplus
extern "C" {
#endif

// ================================================================================================================
// aclop_Any / aclop_All: aclnnAny/aclnnAll 直接接受的 dtype 白名单很窄
// (BOOL / INT32 / INT64 / FLOAT16 / FLOAT32，实测)；DOUBLE、COMPLEX64/128、
// 窄整型 (int8/int16) 被拒。分层处理（AscendSpecialization.md §A.1.1/A.2）：
//
//   * uint 全系：上层 dispatcher 已处理 —— cupy/_core/_ascend/_reduction.pyx
//     ::_call 的 _UINT_PROMOTE 在 launch 前把 uint astype 成 int32/int64，
//     这里正常收不到 uint；cast 兜底仅作防御。
//   * int8/int16：dispatcher 刻意不提升（aclnnAmax/Amin/mean 原生收窄整型，
//     见 _reduction.pyx 头注释），这里是它们的**真实处理器**。
//   * DOUBLE：白名单无 DOUBLE，cast 到 FLOAT32 后归约。注意 |x| < 2^-126
//     的下溢会把非零元素错判成 0（any() 假阴性）；enable_float64_to_float32
//     打开时 dispatcher 已把 float64 降为 float32（直接命中白名单），本路由
//     只服务开关关闭的场景。
//   * COMPLEX64/128：已知差距 —— aclnnCast 的输入 doc 未列 complex（8.5.1
//     也没有 aclnnImag/复数版 abs，无法在 C++ 内做精确的非零判断），设备上
//     可能为取实部或直接报错：取实部会把纯虚数 (如 1j) 错判成 0，与
//     numpy any(1j) == True 不符。精确语义需上层 real/imag 组合（TODO）。
//   * out：aclnnAll doc 只列 BOOL（8.5.1 的 aclnn_any.h 无 dtype doc）。
//     numpy 的 any/all 返回恒为 bool（全量归约 -> numpy.bool_ 标量、
//     带 axis -> bool ndarray），out=BOOL 恒成立。
//
// 该 cast 兜底取代了曾在 launch_reduction_op_raw 的 astype('?') 补丁
// (_BOOL_CAST_INPUT_OPS) —— dtype 处理现在跟算子放在一起。
#ifdef __cplusplus
}  // leave extern "C": the helper below is a C++ template
#endif

static bool _any_all_dtype_ok(aclDataType dtype) {
    switch (dtype) {
        case ACL_BOOL:
        case ACL_INT32:
        case ACL_INT64:
        case ACL_FLOAT16:
        case ACL_FLOAT:
            return true;
        default:
            return false;
    }
}

template <typename WsFunc, typename KernFunc>
static aclError _run_any_all(const aclTensor* self, const aclIntArray* dim, bool keepdim,
    aclTensor* out, aclrtStream stream, WsFunc wsfunc, KernFunc kfunc) {
    aclDataType dtype = ACL_DT_UNDEFINED;
    aclGetDataType(self, &dtype);
    if (_any_all_dtype_ok(dtype)) {
        return aclReductionOpRun(self, out, wsfunc, kfunc, stream, dim, keepdim);
    }
    // cast self -> FLOAT32（aclnnAny/aclnnAll 白名单内的浮点家族，用户真机
    // 结论：BOOL 输入不可靠），归约跑在 cast 副本上；输出恒为 BOOL。
    aclTensor* temp = aclTensorLike(self, ACL_FLOAT);
    if (temp == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    aclError ret = aclIrregularOpRun(aclnnCastGetWorkspaceSize, aclnnCast, stream,
        self, ACL_FLOAT, temp);
    if (ret == ACL_SUCCESS) {
        ret = aclReductionOpRun(temp, out, wsfunc, kfunc, stream, dim, keepdim);
    }
    // aclTensorLike aclrtMallocs device memory; DestroyTensorLike frees it
    aclDestroyTensorLike(temp);
    return ret;
}

#ifdef __cplusplus
extern "C" {  // re-enter extern "C" for the aclop_* wrappers
#endif

aclError aclop_Any(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    return _run_any_all(self, dim, keepdim, out, stream,
        aclnnAnyGetWorkspaceSize, aclnnAny);
}

aclError aclop_All(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    return _run_any_all(self, dim, keepdim, out, stream,
        aclnnAllGetWorkspaceSize, aclnnAll);
}

// aclnnMax/aclnnMin are whole-tensor reductions: no dim, no keepdim (they
// ignore the dispatcher's dim/keepdim, so an axis-wise numpy.max was wrong).
// aclnnAmax/aclnnAmin take the aclIntArray of dims + keepDim. NumPy semantics:
// max over the given axes; the caller passes ALL axes for a global reduction.
aclError aclop_Max(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    return aclReductionOpRun(self, out,
        aclnnAmaxGetWorkspaceSize, aclnnAmax, stream, dim, keepdim);
}
aclError aclop_Min(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    return aclReductionOpRun(self, out,
        aclnnAminGetWorkspaceSize, aclnnAmin, stream, dim, keepdim);
}

// return index type is decided by input, if not specified, it should be int64
aclError aclop_ArgMax(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    int64_t dim_index = dim->GetData()[0];  // TODO, not sure how to convert
    return aclReductionOpRun(self, out,
        aclnnArgMaxGetWorkspaceSize, aclnnArgMax, stream, dim_index, keepdim); 
}
aclError aclop_ArgMin(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    int64_t dim_index = dim->GetData()[0];  // TODO caller will put int64_t dim into aclIntArray
    return aclReductionOpRun(self, out,
        aclnnArgMinGetWorkspaceSize, aclnnArgMin, stream, dim_index, keepdim); 
}

aclError aclop_Mean(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    aclDataType dtype = GetDataType(out, self);
    return aclReductionOpRun(self, out,
        aclnnMeanGetWorkspaceSize, aclnnMean, stream, dim, keepdim, dtype); 
}
// aclnn also provide nanMedian version
aclError aclop_Median(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    return aclReductionOpRun(self, out,
        aclnnMedianGetWorkspaceSize, aclnnMedian, stream); 
}


aclError aclop_Std(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    int64_t correction = 0; // numpy ddof default to 0, correction is added in numpy 2.0
    return aclReductionOpRun(self, out,
        aclnnStdGetWorkspaceSize, aclnnStd, stream, dim, correction, keepdim); 
}
aclError aclop_Var(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    bool unbiased = true;
    // TODO:  ddof/correction, default to zero, if keyword arg ddof is given, add impl here
    return aclReductionOpRun(self, out,
        aclnnVarGetWorkspaceSize, aclnnVar, stream, dim, unbiased, keepdim); 
}

aclError aclop_Prod(const aclTensor* self, const aclIntArray* axis, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    int64_t dim_index = axis->GetData()[0];  // TODO, not sure how to convert
    aclDataType dtype = GetDataType(out, self);
    return aclReductionOpRun(self, out,
        aclnnProdDimGetWorkspaceSize, aclnnProdDim, stream, dim_index, keepdim, dtype); 
}

aclError aclop_Sum(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    aclDataType dtype = GetDataType(out, self);
    return aclReductionOpRun(self, out,
        aclnnReduceSumGetWorkspaceSize, aclnnReduceSum, stream, dim, keepdim, dtype); 
}

aclError aclop_NanToNum(const aclTensor* self, float scalar, aclTensor* out, aclrtStream stream) {
    return aclIrregularOpRun(aclnnNanToNumGetWorkspaceSize, aclnnNanToNum, stream,
        self, scalar, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), out);
}

aclError aclop_Cumsum(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    int64_t dim_index = dim->GetData()[0];  // TODO, not sure how to convert
    aclDataType dtype = GetDataType(out, self); // extra parameter from out Tensor, maybe do the conversion outside this func
    return aclReductionOpRun(self, out,
        aclnnCumsumGetWorkspaceSize, aclnnCumsum, stream, dim_index, dtype); 
}

// dim: why it is a aclScalar?
aclError aclop_Cumprod(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    aclScalar* dim_index = nullptr; // TODO, not sure how to convert
    aclDataType dtype = GetDataType(out, self); // TODO extra parameter from out Tensor, maybe do the conversion outside this func
    return aclReductionOpRun(self, out,
        aclnnCumprodGetWorkspaceSize, aclnnCumprod, stream, dim_index, dtype); 
}

aclError aclop_Nansum(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    aclDataType dtype = GetDataType(out, self);
    return aclReductionOpRun(self, out,
        aclnnReduceNansumGetWorkspaceSize, aclnnReduceNansum, stream, dim, keepdim, dtype); 
}

// aclError aclop_Nanprod(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
//     const KwargsType& kwargs, aclrtStream stream) {
//     aclDataType dtype; // self->GetDataType();
//     return aclReductionOpRun(self, out,
//         aclnnReduceNanprodGetWorkspaceSize, aclnnReduceNanprod, stream, dim, keepdim, dtype); 
// }
// CANN has no aclnnNanprod / aclnn_reduce_nanprod.h (the block above is what a
// dedicated op would look like). Compose it instead: NumPy's nanprod treats NaN
// as 1.0 in the product, so substitute 1 for NaN and run a plain prod reduction
// -- the same "NanToNum + plain reduction" composition as aclop_NanMin/NanMax
// above and aclop_Nancumprod below.
aclError aclop_NanProd(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    aclDataType dtype = GetDataType(out, self);
    // 整型/布尔没有 NaN (NumPy 的 nanprod 对它们等价于 prod), 且 aclnnNanToNum
    // 只接受浮点输入 -> 跳过 nan_to_num, 直接 prod (判断见 dtype_has_no_nan)
    if (dtype_has_no_nan(dtype)) {
        return aclReductionOpRun(self, out,
            aclnnProdDimGetWorkspaceSize, aclnnProdDim, stream,
            dim->GetData()[0], keepdim, dtype);
    }
    aclTensor* temp = aclTensorLike(self, dtype);
    if (temp == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    float scalar = 1.0f;
    aclError ret = aclop_NanToNum(self, scalar, temp, stream);
    if (ret == ACL_SUCCESS) {
        // same single-axis limitation as aclop_Prod (aclnnProdDim takes one dim)
        int64_t dim_index = dim->GetData()[0];
        ret = aclReductionOpRun(temp, out,
            aclnnProdDimGetWorkspaceSize, aclnnProdDim, stream, dim_index, keepdim, dtype);
    }
    // aclTensorLike 会 aclrtMalloc 一块显存，必须用 DestroyTensorLike 成对释放
    aclDestroyTensorLike(temp);
    return ret;
}

aclError aclop_Nancumprod(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    aclScalar* dim_index = nullptr; // TODO, not sure how to convert
    aclDataType dtype = GetDataType(out, self);
    aclTensor* temp = aclTensorLike(self, dtype);
    float scalar = 0.0f;
    aclError ret = aclop_NanToNum(self, scalar, temp, stream);

    ret = aclReductionOpRun(temp, out,
        aclnnCumprodGetWorkspaceSize, aclnnCumprod, stream, dim_index, dtype); 
    // aclTensorLike 会 aclrtMalloc 一块显存，必须用 DestroyTensorLike 成对释放
    aclDestroyTensorLike(temp);
    return ret;
}

aclError aclop_Nancumsum(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    int64_t dim_index = dim->GetData()[0];  // TODO, not sure how to convert
    aclDataType dtype = GetDataType(out, self);
    aclTensor* temp = aclTensorLike(self, dtype);
    float scalar = 0.0f;
    aclError ret = aclop_NanToNum(self, scalar, temp, stream);

    ret = aclReductionOpRun(temp, out,
        aclnnCumsumGetWorkspaceSize, aclnnCumsum, stream, dim_index, dtype); 
    // aclTensorLike 会 aclrtMalloc 一块显存，必须用 DestroyTensorLike 成对释放
    aclDestroyTensorLike(temp);
    return ret;
}

// CANN has no aclnnNanmin/Nanmax. Substitute an infinity of the *opposite*
// sign for NaN so the plain min/max reduction skips them. Passing +-inf as
// the posinf/neginf replacements of aclnnNanToNum leaves genuine infinities
// already present in the input untouched, preserving NumPy semantics.
aclError aclop_NanMin(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    aclDataType dtype = GetDataType(out, self);
    aclTensor* temp = aclTensorLike(self, dtype);
    if (temp == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    float scalar = std::numeric_limits<float>::infinity();
    aclError ret = aclop_NanToNum(self, scalar, temp, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclReductionOpRun(temp, out,
            aclnnAminGetWorkspaceSize, aclnnAmin, stream, dim, keepdim);
    }
    // aclTensorLike 会 aclrtMalloc 一块显存，必须用 DestroyTensorLike 成对释放
    aclDestroyTensorLike(temp);
    return ret;
}

aclError aclop_NanMax(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    aclDataType dtype = GetDataType(out, self);
    aclTensor* temp = aclTensorLike(self, dtype);
    if (temp == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    float scalar = -std::numeric_limits<float>::infinity();
    aclError ret = aclop_NanToNum(self, scalar, temp, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclReductionOpRun(temp, out,
            aclnnAmaxGetWorkspaceSize, aclnnAmax, stream, dim, keepdim);
    }
    // aclTensorLike 会 aclrtMalloc 一块显存，必须用 DestroyTensorLike 成对释放
    aclDestroyTensorLike(temp);
    return ret;
}

// CANN has no aclnnNanArgMax/NanArgMin. Same composition as aclop_NanMax /
// aclop_NanMin: substitute an infinity of the opposite sign for NaN, then run
// the plain argmax/argmin. For integer/bool inputs there is no NaN, so the
// plain op is applied directly (aclnnNanToNum only accepts floating point).
aclError aclop_NanArgMax(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    int64_t dim_index = dim->GetData()[0];  // same single-axis limitation as aclop_ArgMax
    aclDataType dtype = ACL_DT_UNDEFINED;
    aclError ret = aclGetDataType(self, &dtype);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    if (dtype_has_no_nan(dtype)) {
        return aclReductionOpRun(self, out,
            aclnnArgMaxGetWorkspaceSize, aclnnArgMax, stream, dim_index, keepdim);
    }
    aclTensor* temp = aclTensorLike(self, dtype);
    if (temp == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    float scalar = -std::numeric_limits<float>::infinity();
    ret = aclop_NanToNum(self, scalar, temp, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclReductionOpRun(temp, out,
            aclnnArgMaxGetWorkspaceSize, aclnnArgMax, stream, dim_index, keepdim);
    }
    // aclTensorLike 会 aclrtMalloc 一块显存，必须用 DestroyTensorLike 成对释放
    aclDestroyTensorLike(temp);
    return ret;
}

aclError aclop_NanArgMin(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    int64_t dim_index = dim->GetData()[0];
    aclDataType dtype = ACL_DT_UNDEFINED;
    aclError ret = aclGetDataType(self, &dtype);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    if (dtype_has_no_nan(dtype)) {
        return aclReductionOpRun(self, out,
            aclnnArgMinGetWorkspaceSize, aclnnArgMin, stream, dim_index, keepdim);
    }
    aclTensor* temp = aclTensorLike(self, dtype);
    if (temp == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    float scalar = std::numeric_limits<float>::infinity();
    ret = aclop_NanToNum(self, scalar, temp, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclReductionOpRun(temp, out,
            aclnnArgMinGetWorkspaceSize, aclnnArgMin, stream, dim_index, keepdim);
    }
    // aclTensorLike 会 aclrtMalloc 一块显存，必须用 DestroyTensorLike 成对释放
    aclDestroyTensorLike(temp);
    return ret;
}

// CANN has no aclnnIsNan and no "count" reduction (see acl_reduction_ops.h
// includes), so NumPy's count_non_nan (= sum of `not isnan(x)`) is composed:
//
//   nan_mask = (x != x)               -> true exactly for NaN (same trick as
//                                       aclop_IsNan, which CANN also lacks)
//   mask     = s_where(nan_mask, 0, 1) -> 0/1 *in self's dtype*
//   out      = reduce_sum(mask, ...)   -> cast to the (integer) output dtype
//
// The 0/1 detour through s_where avoids reducing a bool tensor and avoids a
// bool->int aclnnCast, neither of which is verified on Ascend. Only the
// NaN mask is bool; it is produced by a comparison, exactly like aclop_IsNan
// and aclop_Copysign already do.
//
// NOTE: complex inputs are not supported -- the "1" sentinel and the sum are
// built in self's dtype, and neither aclnnFillScalar nor aclnnReduceSum takes
// complex. The only caller on Ascend is `_nanvar`, whose complex path needs
// `ascend_nanvar_core_complex*` anyway (not registered).
aclError aclop_CountNonNaN(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    aclDataType dtype = ACL_DT_UNDEFINED;
    aclError ret = aclGetDataType(self, &dtype);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    aclDataType out_dtype = ACL_DT_UNDEFINED;
    ret = aclGetDataType(out, &out_dtype);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    aclTensor* nan_mask = aclTensorLike(self, ACL_BOOL);
    aclTensor* zeros = aclTensorLike(self, dtype);
    aclTensor* ones = aclTensorLike(self, dtype);
    aclTensor* mask = aclTensorLike(self, dtype);
    if (nan_mask == nullptr || zeros == nullptr || ones == nullptr || mask == nullptr) {
        aclDestroyTensorLike(nan_mask);
        aclDestroyTensorLike(zeros);
        aclDestroyTensorLike(ones);
        aclDestroyTensorLike(mask);
        return ACL_ERROR_INVALID_PARAM;
    }

    float zero = 0.0f;
    float one = 1.0f;
    const aclScalar* zero_scalar = aclCreateScalar(&zero, ACL_FLOAT);
    const aclScalar* one_scalar = aclCreateScalar(&one, ACL_FLOAT);

    ret = aclBinaryOpRun(self, self, nan_mask,
        aclnnNeTensorGetWorkspaceSize, aclnnNeTensor, stream, false);
    if (ret == ACL_SUCCESS) {
        ret = aclInplaceBinaryOpRun(zeros, zero_scalar,
            aclnnInplaceFillScalarGetWorkspaceSize, aclnnInplaceFillScalar, stream, false);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclInplaceBinaryOpRun(ones, one_scalar,
            aclnnInplaceFillScalarGetWorkspaceSize, aclnnInplaceFillScalar, stream, false);
    }
    if (ret == ACL_SUCCESS) {
        // s_where(cond, self, other): the true branch gets 0 (NaN), the false
        // branch 1 (not NaN) -- i.e. exactly `!isnan(x)` as a 0/1 array.
        ret = aclIrregularOpRun(aclnnSWhereGetWorkspaceSize, aclnnSWhere, stream,
            nan_mask, zeros, ones, mask);
    }
    if (ret == ACL_SUCCESS) {
        // dtype is the *accumulator/output* dtype: passing out_dtype makes the
        // op cast the 0/1 values and accumulate them as integers.
        ret = aclReductionOpRun(mask, out,
            aclnnReduceSumGetWorkspaceSize, aclnnReduceSum, stream, dim, keepdim, out_dtype);
    }

    aclDestroyScalar(zero_scalar);
    aclDestroyScalar(one_scalar);
    // aclTensorLike 会 aclrtMalloc 一块显存，必须用 DestroyTensorLike 成对释放
    aclDestroyTensorLike(nan_mask);
    aclDestroyTensorLike(zeros);
    aclDestroyTensorLike(ones);
    aclDestroyTensorLike(mask);
    return ret;
}

#ifdef __cplusplus
}
#endif

#endif // header