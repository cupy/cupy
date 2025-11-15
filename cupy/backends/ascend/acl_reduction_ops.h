#ifndef CUPY_ACL_REDUCTION_OPS_HEADER
#define CUPY_ACL_REDUCTION_OPS_HEADER

#include <cmath>

// bool reduction op
#include "aclnnop/aclnn_all.h"
#include "aclnnop/aclnn_any.h"

// statistics, TODO: keyword args
#include "aclnnop/aclnn_mean.h"
#include "aclnnop/aclnn_std.h"
#include "aclnnop/aclnn_var.h"
#include "aclnnop/aclnn_bincount.h"
#include "aclnnop/aclnn_median.h"
#include "aclnnop/aclnn_median.h" // nan version
#include "aclnnop/aclnn_aminmax.h" // ptp :  aminmax
// missing quantile, percentile
#include "aclnnop/aclnn_histc.h"
#include "aclnnop/aclnn_reduce_nansum.h"

#include "./acl_op_template.h"
#include "acl/acl.h"


#ifdef __cplusplus
extern "C" {
#endif

// ================================================================================================================
// DECLARE_ACL_REDUCTION_OP(Any)
aclError aclop_Any(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
    return aclReductionOpRun(self, out,
        aclnnAnyGetWorkspaceSize, aclnnAny, stream, false, dim, keepdim); 
}
DECLARE_ACL_REDUCTION_OP(All)

// why this Min has no dim and keepdim control
aclError aclop_Max(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
    return aclReductionOpRun(self, out,
        aclnnMaxGetWorkspaceSize, aclnnMax, stream, false); 
}
aclError aclop_Min(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
    return aclReductionOpRun(self, out,
        aclnnMinGetWorkspaceSize, aclnnMin, stream, false); 
}
//DECLARE_ACL_REDUCTION_OP(Amin)
//DECLARE_ACL_REDUCTION_OP(Amax)
aclError aclop_ArgMax(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
    int64_t dim_index = dim->GetData()[0];  // TODO, not sure how to convert
    return aclReductionOpRun(self, out,
        aclnnArgMaxGetWorkspaceSize, aclnnArgMax, stream, false, dim_index, keepdim); 
}
aclError aclop_ArgMin(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
    int64_t dim_index = dim->GetData()[0];  // TODO caller will put int64_t dim into aclIntArray
    return aclReductionOpRun(self, out,
        aclnnArgMinGetWorkspaceSize, aclnnArgMin, stream, false, dim_index, keepdim); 
}

aclError aclop_Mean(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
    aclDataType dtype = GetDataType(out, self);
    return aclReductionOpRun(self, out,
        aclnnMeanGetWorkspaceSize, aclnnMean, stream, false, dim, keepdim, dtype); 
}
// aclnn also provide nanMedian version
aclError aclop_Median(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
    return aclReductionOpRun(self, out,
        aclnnMedianGetWorkspaceSize, aclnnMedian, stream, false); 
}
// aclError aclop_Bincount(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
//     // need weights tensor, int64_t minLength 
//     return aclReductionOpRun(self, out,
//         aclnnBincountGetWorkspaceSize, aclnnBincount, stream, false); 
// }

aclError aclop_Std(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
    int64_t correction = 0; // numpy ddof default to 0, correction is added in numpy 2.0
    return aclReductionOpRun(self, out,
        aclnnStdGetWorkspaceSize, aclnnStd, stream, false, dim, correction, keepdim); 
}
aclError aclop_Var(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
    bool unbiased = true;
    // TODO:  ddof/correction, default to zero, if keyword arg ddof is given, add impl here
    return aclReductionOpRun(self, out,
        aclnnVarGetWorkspaceSize, aclnnVar, stream, false, dim, unbiased, keepdim); 
}

aclError aclop_Prod(const aclTensor* self, const aclIntArray* axis, bool keepdim, aclTensor* out, aclrtStream stream) {
    int64_t dim_index = axis->GetData()[0];  // TODO, not sure how to convert
    aclDataType dtype = GetDataType(out, self);
    return aclReductionOpRun(self, out,
        aclnnProdDimGetWorkspaceSize, aclnnProdDim, stream, false, dim_index, keepdim, dtype); 
}

aclError aclop_Sum(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
    aclDataType dtype = GetDataType(out, self);
    return aclReductionOpRun(self, out,
        aclnnReduceSumGetWorkspaceSize, aclnnReduceSum, stream, false, dim, keepdim, dtype); 
}

aclError aclop_NanToNum(const aclTensor* self, float scalar, aclTensor* out, aclrtStream stream) {
    return aclIrregularOpRun(aclnnNanToNumGetWorkspaceSize, aclnnNanToNum, stream,
        self, scalar, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), out);
}

aclError aclop_Cumsum(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
    int64_t dim_index = dim->GetData()[0];  // TODO, not sure how to convert
    aclDataType dtype = GetDataType(out, self); // extra parameter from out Tensor, maybe do the conversion outside this func
    return aclReductionOpRun(self, out,
        aclnnCumsumGetWorkspaceSize, aclnnCumsum, stream, false, dim_index, dtype); 
}
// dim: why it is a aclScalar?
aclError aclop_Cumprod(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
    aclScalar* dim_index = nullptr; // TODO, not sure how to convert
    aclDataType dtype = GetDataType(out, self); // TODO extra parameter from out Tensor, maybe do the conversion outside this func
    return aclReductionOpRun(self, out,
        aclnnCumprodGetWorkspaceSize, aclnnCumprod, stream, false, dim_index, dtype); 
}
aclError aclop_Nansum(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
    aclDataType dtype = GetDataType(out, self);
    return aclReductionOpRun(self, out,
        aclnnReduceNansumGetWorkspaceSize, aclnnReduceNansum, stream, false, dim, keepdim, dtype); 
}
// aclError aclop_Nanprod(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
//     aclDataType dtype; // self->GetDataType();
//     return aclReductionOpRun(self, out,
//         aclnnReduceNanprodGetWorkspaceSize, aclnnReduceNanprod, stream, false, dim, keepdim, dtype); 
// }
aclError aclop_Nancumprod(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
    aclScalar* dim_index = nullptr; // TODO, not sure how to convert
    aclDataType dtype = GetDataType(out, self);
    aclTensor* temp = aclTensorLike(self, dtype);
    float scalar = 0.0f;
    aclError ret = aclop_NanToNum(self, scalar, temp, stream);

    ret = aclReductionOpRun(temp, out,
        aclnnCumprodGetWorkspaceSize, aclnnCumprod, stream, false, dim_index, dtype); 
    aclDestroyTensor(temp);
    return ret;
}

aclError aclop_Nancumsum(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
    int64_t dim_index = dim->GetData()[0];  // TODO, not sure how to convert
    aclDataType dtype = GetDataType(out, self);
    aclTensor* temp = aclTensorLike(self, dtype);
    float scalar = 0.0f;
    aclError ret = aclop_NanToNum(self, scalar, temp, stream);

    ret = aclReductionOpRun(temp, out,
        aclnnCumsumGetWorkspaceSize, aclnnCumsum, stream, false, dim_index, dtype); 
    aclDestroyTensor(temp);
    return ret;
}
    
#ifdef __cplusplus
}
#endif

#endif // header