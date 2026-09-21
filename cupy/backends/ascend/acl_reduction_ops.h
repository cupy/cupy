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
// DECLARE_ACL_REDUCTION_OP(Any)
aclError aclop_Any(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out,
    const KwargsType& kwargs, aclrtStream stream) {
    return aclReductionOpRun(self, out,
        aclnnAnyGetWorkspaceSize, aclnnAny, stream, dim, keepdim); 
}
DECLARE_ACL_REDUCTION_OP(All)

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