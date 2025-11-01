#ifndef CUPY_ACL_MATH_HEADER
#define CUPY_ACL_MATH_HEADER

#include "aclnnop/aclnn_cos.h"
#include "aclnnop/aclnn_cosh.h"
#include "aclnnop/aclnn_sin.h"
#include "aclnnop/aclnn_sinh.h"
#include "aclnnop/aclnn_tan.h"
#include "aclnnop/aclnn_tanh.h"
#include "aclnnop/aclnn_acos.h"
#include "aclnnop/aclnn_acosh.h"
#include "aclnnop/aclnn_asin.h"
#include "aclnnop/aclnn_asinh.h"
#include "aclnnop/aclnn_atan.h"
#include "aclnnop/aclnn_atanh.h"
#include "aclnnop/aclnn_atan2.h"

#include "aclnnop/aclnn_erf.h"
#include "aclnnop/aclnn_erfc.h"
#include "aclnnop/aclnn_erfinv.h"

#include "aclnnop/aclnn_exp.h"
#include <aclnnop/aclnn_expm1.h>
#include "aclnnop/aclnn_exp2.h"
#include "aclnnop/aclnn_log.h"
#include "aclnnop/aclnn_log1p.h"
#include "aclnnop/aclnn_log2.h"
#include "aclnnop/aclnn_log10.h"
#include <aclnnop/aclnn_logaddexp.h>
#include <aclnnop/aclnn_logaddexp2.h>
#include "aclnnop/aclnn_sqrt.h"
//#include "aclnnop/aclnn_square.h" // np.pow with scalar 2
#include "aclnnop/aclnn_pow.h"  // np.pow

#include "aclnnop/aclnn_abs.h"
#include "aclnnop/aclnn_neg.h"
#include "aclnnop/aclnn_ceil.h"
#include "aclnnop/aclnn_floor.h"
#include "aclnnop/aclnn_clamp.h"
#include "aclnnop/aclnn_signbit.h"
#include "aclnnop/aclnn_reciprocal.h"
// sign, inverse

// equal scalar, tensor, vector/list is_nan (no such)
#include "aclnnop/aclnn_is_inf.h"
#include "aclnnop/aclnn_isfinite.h"
#include "aclnnop/aclnn_isposinf.h"
#include "aclnnop/aclnn_isneginf.h"
// is_real
#include "aclnnop/aclnn_isclose.h"

// ge, eq, le, gt, lt, 
#include <aclnnop/aclnn_logical_and.h>
#include <aclnnop/aclnn_logical_or.h>
#include <aclnnop/aclnn_logical_not.h>
#include <aclnnop/aclnn_logical_xor.h>
#include <aclnnop/aclnn_gt_tensor.h>
#include <aclnnop/aclnn_gt_scalar.h>
#include <aclnnop/aclnn_ge_tensor.h>
#include <aclnnop/aclnn_ge_scalar.h>
#include <aclnnop/aclnn_lt_scalar.h>
#include <aclnnop/aclnn_lt_tensor.h>
#include <aclnnop/aclnn_le_tensor.h>
#include <aclnnop/aclnn_le_scalar.h>
#include <aclnnop/aclnn_equal.h>
#include <aclnnop/aclnn_ne_scalar.h>
#include <aclnnop/aclnn_ne_tensor.h>


// bool reduction op
#include "aclnnop/aclnn_all.h"
#include "aclnnop/aclnn_any.h"

// bitwise op: and not not xor
#include "aclnnop/aclnn_bitwise_and_tensor.h"
#include "aclnnop/aclnn_bitwise_and_scalar.h"
#include "aclnnop/aclnn_bitwise_or_tensor.h"
#include "aclnnop/aclnn_bitwise_or_scalar.h"
#include "aclnnop/aclnn_bitwise_xor_tensor.h"
#include "aclnnop/aclnn_bitwise_xor_scalar.h"
#include "aclnnop/aclnn_bitwise_not.h" // numpy op: np.invert
// #include "aclnnop/aclnn_shift_left.h"  // numpy op: _left_shift

// binary op
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_sub.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_div.h"
#include "aclnnop/aclnn_gcd.h"
#include "aclnnop/aclnn_remainder.h" // tensor scalar 4 combinations
#include "aclnnop/aclnn_fmod_scalar.h"
#include "aclnnop/aclnn_fmod_tensor.h" 
#include "aclnnop/aclnn_floor_divide.h"
#include <aclnnop/aclnn_maximum.h>  // find the bigger from two tensors
#include <aclnnop/aclnn_minimum.h>

// tertiary op, not numpy op
//#include "aclnnop/aclnn_addcmul.h" // out = self + value * tensor1 * tensor2

// foreach? batch, not numpy op
#include "aclnnop/aclnn_foreach_add_scalar.h"
#include "aclnnop/aclnn_foreach_sub_scalar.h"
#include "aclnnop/aclnn_foreach_mul_scalar.h"  // _v2?

// reduce op, how about dim
#include "aclnnop/aclnn_reduce_sum.h"
#include "aclnnop/aclnn_cumsum.h"
#include "aclnnop/aclnn_cumprod.h"
#include "aclnnop/aclnn_prod.h"
#include "aclnnop/aclnn_max.h"  // nan?
#include "aclnnop/aclnn_min.h"
#include "aclnnop/aclnn_dot.h"
#include "aclnnop/aclnn_einsum.h" 
#include "aclnnop/aclnn_reduce_nansum.h"
#include <aclnnop/aclnn_nan_to_num.h>

#include "aclnnop/aclnn_argmax.h"  // return the index instead of value
#include "aclnnop/aclnn_argmin.h"
#include <aclnnop/aclnn_round.h>

// statistics
#include "aclnnop/aclnn_mean.h"
#include "aclnnop/aclnn_std.h"
#include "aclnnop/aclnn_var.h"
// count

// creation op:  with dim info
// arange, eye, diag, linspace, ones, zeros, 
#include "aclnnop/aclnn_arange.h"
#include "aclnnop/aclnn_eye.h"  //  np.eye == np.identity(N)
#include "aclnnop/aclnn_diag.h"  // UnaryScalarOp   not sure TODO
// fill_scalar to impl   numpy op: no.one

// manipulation op:  sort select take put
#include "aclnnop/aclnn_fill_tensor.h" // fill_scalar, masked_fill
#include "aclnnop/aclnn_flip.h"  // transpose ??
//#include "aclnnop/aclnn_rot.h"
#include "aclnnop/aclnn_stack.h"
#include "aclnnop/aclnn_cat.h"
#include "aclnnop/aclnn_flatten.h"
//#include "aclnnop/aclnn_reshape.h"

// indexing: argsort, unique, sort

// manipulation: transpose, reshape, cast, pad continguous in aclnn_kernels/
// #include "aclnn_kernels/transpose.h"
// #include "aclnn_kernels/cast.h"
// #include "aclnn_kernels/pad.h"
// #include "aclnn_kernels/slice.h"

// linalg matrix op: qr, tril triu, cross, trace, norm, det
#include "aclnnop/aclnn_matmul.h"

#include "./acl_op_template.h"
#include "acl/acl.h"


#ifdef __cplusplus
extern "C" {
#endif

    // aclError aclop_BitwiseAndTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) {
    //     return aclBinaryOpRun(self, other, out,
    //         aclnnBitwiseAndTensorGetWorkspaceSize, aclnnBitwiseAndTensor, stream, false); 
    // }
    // aclError aclop_InplaceBitwiseAndTensor(aclTensor* self, const aclTensor* other, aclrtStream stream) {
    //     return aclBinaryInplaceOpRun(self, other,
    //         aclnnInplaceBitwiseAndTensorGetWorkspaceSize, aclnnBitwiseAndTensor, stream, false); 
    // }
    // aclError aclop_BitwiseAndScalar(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream) {
    //     return aclBinaryOpRun(self, other, out,
    //         aclnnBitwiseAndScalarGetWorkspaceSize, aclnnBitwiseAndScalar, stream, false); 
    // }
    // aclError aclop_InplaceBitwiseAndScalar(aclTensor* self, const aclScalar* other, aclrtStream stream) {
    //     return aclBinaryInplaceOpRun(self, other,
    //         aclnnInplaceBitwiseAndScalarGetWorkspaceSize, aclnnBitwiseAndScalar, stream, false); 
    // }

    DECLARE_ACL_BINARY_OPS_FUNC(BitwiseAndTensor)
    DECLARE_ACL_BINARY_SCALAR_OPS_FUNC(BitwiseAndScalar)

    DECLARE_ACL_BINARY_OPS_FUNC(BitwiseOrTensor)
    DECLARE_ACL_BINARY_SCALAR_OPS_FUNC(BitwiseOrScalar)

    DECLARE_ACL_BINARY_OPS_FUNC(BitwiseXorTensor)
    DECLARE_ACL_BINARY_SCALAR_OPS_FUNC(BitwiseXorScalar)

    // BitwiseNot has no inplace version, so can not use the macro to clear
    aclError aclop_BitwiseNot(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) {
        return aclUnaryOpRun(self, out,
        aclnnBitwiseNotGetWorkspaceSize, aclnnBitwiseNot, stream, false);
    }

    // support double dtype? double is supported except for CubeCore matmul/dot
    DECLARE_ACL_BINARY_OP(LogicalXor)
    DECLARE_ACL_BINARY_OP(LogicalAnd)
    DECLARE_ACL_BINARY_OP(LogicalOr)
    DECLARE_ACL_UNARY_OP(LogicalNot)

    DECLARE_ACL_BINARY_OP(GtTensor)
    DECLARE_ACL_BINARY_SCALAR_OP(GtScalar)
    DECLARE_ACL_BINARY_OP(GeTensor)
    DECLARE_ACL_BINARY_SCALAR_OP(GeScalar)
    DECLARE_ACL_BINARY_OP(LtTensor)
    DECLARE_ACL_BINARY_SCALAR_OP(LtScalar)
    DECLARE_ACL_BINARY_OP(LeTensor)
    DECLARE_ACL_BINARY_SCALAR_OP(LeScalar)

    DECLARE_ACL_BINARY_OP(Equal) // no inplace version
    DECLARE_ACL_BINARY_SCALAR_OP(NeScalar)
    DECLARE_ACL_BINARY_OP(NeTensor)
    // IsClose() has extra args: double rtol, double atol, bool equal_nan
    DECLARE_ACL_UNARY_OP(IsFinite)
    DECLARE_ACL_UNARY_OP(IsInf)
    DECLARE_ACL_UNARY_OP(IsPosInf)
    DECLARE_ACL_UNARY_OP(IsNegInf)
    // TODO: IsNaN() no such op? it depends on ascend env var controlled behavior

    // ==============================================================
    DECLARE_ACL_UNARY_OPS_FUNC(Cos)
    DECLARE_ACL_UNARY_OPS_FUNC(Sin)
    DECLARE_ACL_UNARY_OPS_FUNC(Tan)
    DECLARE_ACL_UNARY_OPS_FUNC(Acos)
    DECLARE_ACL_UNARY_OPS_FUNC(Asin)
    DECLARE_ACL_UNARY_OPS_FUNC(Atan)
    DECLARE_ACL_UNARY_OPS_FUNC(Cosh)
    DECLARE_ACL_UNARY_OPS_FUNC(Sinh)
    DECLARE_ACL_UNARY_OPS_FUNC(Tanh)

    // ascend ADD is ternary op with one extra scalar coeff, so can not use the macro to declare
    aclError aclop_Add(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) {
        float alpha = 1.0f;
        return aclTernaryOpRun(self, other, alpha, out,
        aclnnAddGetWorkspaceSize, aclnnAdd, stream, false);
    }
    aclError aclop_InplaceAdd(aclTensor* self, const aclTensor* other, aclrtStream stream) {
        float alpha = 1.0f;
        return aclTernaryInplaceOpRun(self, other, alpha,
        aclnnInplaceAddGetWorkspaceSize, aclnnInplaceAdd, stream, false);
    }
    aclError aclop_Sub(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) {
        float alpha = 1.0f;
        return aclTernaryOpRun(self, other, alpha, out,
        aclnnSubGetWorkspaceSize, aclnnSub, stream, false);
    }
    aclError aclop_InplaceSub(aclTensor* self, const aclTensor* other, aclrtStream stream) {
        float alpha = 1.0f;
        return aclTernaryInplaceOpRun(self, other, alpha,
        aclnnInplaceSubGetWorkspaceSize, aclnnInplaceSub, stream, false);
    }
    DECLARE_ACL_BINARY_OPS_FUNC(Mul)
    DECLARE_ACL_BINARY_OPS_FUNC(Div)
    DECLARE_ACL_BINARY_OPS_FUNC(FloorDivide)
    DECLARE_ACL_BINARY_OPS_FUNC(FmodTensor)  // for float and ints
    // DivMod  has 3 modes: https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/aolapi/context/aclnnDivMod&aclnnInplaceDivMod.md

    DECLARE_ACL_BINARY_OP(Maximum)
    DECLARE_ACL_BINARY_OP(Minimum)
    // divmod has two outs

    DECLARE_ACL_UNARY_OPS_FUNC(Reciprocal)
    DECLARE_ACL_UNARY_OPS_FUNC(Neg)

    //DECLARE_ACL_UNARY_OPS_FUNC(Signbit)  // no inplace version
    aclError aclop_Signbit(const aclTensor* self, aclTensor* out, aclrtStream stream) {
        return aclUnaryOpRun(self, out,
        aclnnSignbitGetWorkspaceSize, aclnnSignbit, stream, false);
    }
    //DECLARE_ACL_UNARY_OPS_FUNC(Abs) // no inplace version
    aclError aclop_Abs(const aclTensor* self, aclTensor* out, aclrtStream stream) {
        return aclUnaryOpRun(self, out,
        aclnnAbsGetWorkspaceSize, aclnnAbs, stream, false);
    }
    DECLARE_ACL_UNARY_OPS_FUNC(Floor)
    DECLARE_ACL_UNARY_OPS_FUNC(Ceil)

    DECLARE_ACL_UNARY_OPS_FUNC(Exp)
    DECLARE_ACL_UNARY_OPS_FUNC(Expm1)
    DECLARE_ACL_UNARY_OPS_FUNC(Log)
    DECLARE_ACL_UNARY_OPS_FUNC(Log2)
    DECLARE_ACL_UNARY_OPS_FUNC(Log10)
    DECLARE_ACL_UNARY_OPS_FUNC(Log1p)

    // Power has 3 version, Remainder has 4 version
    // DECLARE_ACL_BINARY_OPS_FUNC(PowerTensorTensor)
    // DECLARE_ACL_BINARY_SCALAR_OPS_FUNC(PowerTensorScalar)

    aclError aclop_MatMul(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) {
        uint8_t math_type = 0; // 0 means keeping dtype precision KEEP_DTYPE
        return aclBinaryOpRun(self, other, out,
            aclnnMatmulGetWorkspaceSize, aclnnMatmul, stream, false, math_type); 
    }

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
    //DECLARE_ACL_REDUCTION_OP(ArgMax)
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
        aclDataType dtype; // TODO 指定输出张量的数据类型
        return aclReductionOpRun(self, out,
            aclnnMeanGetWorkspaceSize, aclnnMean, stream, false, dim, keepdim, dtype); 
    }
    aclError aclop_Std(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
        int64_t correction; // TODO ?
        return aclReductionOpRun(self, out,
            aclnnStdGetWorkspaceSize, aclnnStd, stream, false, dim, correction, keepdim); 
    }
    //DECLARE_ACL_REDUCTION_OP(Std)
    aclError aclop_Var(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
        bool unbiased; // TODO ?
        return aclReductionOpRun(self, out,
            aclnnVarGetWorkspaceSize, aclnnVar, stream, false, dim, unbiased, keepdim); 
    }

    aclError aclop_Prod(const aclTensor* self, const aclIntArray* axis, bool keepdim, aclTensor* out, aclrtStream stream) {
        int64_t dim_index = axis->GetData()[0];  // TODO, not sure how to convert
        aclDataType dtype; //  self->GetDataType(); // TODO 指定输出张量的数据类型
        return aclReductionOpRun(self, out,
            aclnnProdDimGetWorkspaceSize, aclnnProdDim, stream, false, dim_index, keepdim, dtype); 
    }

    aclError aclop_Sum(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
        aclDataType dtype; // self->GetDataType();
        return aclReductionOpRun(self, out,
            aclnnReduceSumGetWorkspaceSize, aclnnReduceSum, stream, false, dim, keepdim, dtype); 
    }

    aclError aclop_NanToNum(const aclTensor* self, float scalar, aclTensor* out, aclrtStream stream) {
        return aclIrregularOpRun(aclnnNanToNumGetWorkspaceSize, aclnnNanToNum, stream, false,
            self, scalar, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), out);
    }

    /**
     * 根据源张量创建新张量，保持相同形状但使用指定数据类型
     * 
     * @param source 源张量指针
     * @param dtype 目标数据类型
     * @return 新创建的张量指针，失败返回nullptr
     */
    aclTensor* aclTensorLike(const aclTensor* source, aclDataType dtype) {
        // 参数检查
        if (source == nullptr) {
            std::cerr << "Error: Source tensor is null for aclTensorLike() " << std::endl;
            return nullptr;
        }
        
        aclError ret = ACL_SUCCESS;
        // 获取维度数量
        // 1. 获取并打印逻辑形状 (View Shape)
        int64_t* viewDims = nullptr;
        uint64_t viewDimsNum = 0;
        ret = aclGetViewShape(source, &viewDims, &viewDimsNum);
        CHECK_STATUS(ret);
        int64_t* storageDims = nullptr;
        uint64_t storageDimsNum = 0;
        ret = aclGetStorageShape(source, &storageDims, &storageDimsNum);
        CHECK_STATUS(ret);
        int64_t* strides = nullptr;
        uint64_t stridesNum = 0;
        ret = aclGetViewStrides(source, &strides, &stridesNum);
        CHECK_STATUS(ret);
        // 2. 获取源张量的格式
        aclFormat format;
        ret = aclGetFormat(source, &format);
        CHECK_STATUS(ret);
        aclDataType source_dtype = ACL_DT_UNDEFINED;
        ret = aclGetDataType(source, &source_dtype);
        size_t source_type_size = aclDataTypeSize(source_dtype);
        size_t type_size = aclDataTypeSize(dtype);
        if (type_size == 0 || source_type_size == 0) {
            std::cerr << "Error: Invalid data type size" << std::endl;
            return nullptr;
        }
        float type_size_ratio = (float)type_size / (float)source_type_size;
        // 4. 计算新张量所需内存大小
        size_t element_count = 1;
        for (size_t i = 0; i < storageDimsNum; ++i) {
            element_count *= storageDims[i];
            strides[i] = static_cast<int64_t>(strides[i] * type_size_ratio);
        }
        
        size_t total_size = element_count * type_size;
        
        // 5. 分配设备内存
        void* device_addr = nullptr;
        ret = aclrtMalloc(&device_addr, total_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS || device_addr == nullptr) {
            std::cerr << "Error: Failed to allocate device memory, error code: " << ret << std::endl;
            return nullptr;
        }
        
        aclTensor* new_tensor = aclCreateTensor(viewDims, viewDimsNum, dtype, 
                                            strides, 0, format,
                                            storageDims, storageDimsNum, device_addr);
        
        if (new_tensor == nullptr) {
            std::cerr << "Error: Failed to create new tensor" << std::endl;
            aclrtFree(device_addr);
            return nullptr;
        }
        delete[] viewDims;
        delete[] storageDims;
        delete[] strides;
        return new_tensor;
    }

    aclError aclop_Cumsum(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
        int64_t dim_index = dim->GetData()[0];  // TODO, not sure how to convert
        aclDataType dtype; //  self->GetDataType(); // TODO extra parameter from out Tensor, maybe do the conversion outside this func
        return aclReductionOpRun(self, out,
            aclnnCumsumGetWorkspaceSize, aclnnCumsum, stream, false, dim_index, dtype); 
    }
    // dim: why it is a aclScalar?
    aclError aclop_Cumprod(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
        aclScalar* dim_index = nullptr; // TODO, not sure how to convert
        aclDataType dtype; // self->GetDataType(); // TODO extra parameter from out Tensor, maybe do the conversion outside this func
        return aclReductionOpRun(self, out,
            aclnnCumprodGetWorkspaceSize, aclnnCumprod, stream, false, dim_index, dtype); 
    }
    aclError aclop_Nansum(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, aclrtStream stream) {
        aclDataType dtype; // self->GetDataType();
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
        aclDataType dtype ; // self->GetDataType(); // TODO extra parameter from out Tensor, maybe do the conversion outside this func
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
        aclDataType dtype ; //  self->GetDataType(); // TODO extra parameter from out Tensor, maybe do the conversion outside this func
        aclTensor* temp = aclTensorLike(self, dtype);
        float scalar = 0.0f;
        aclError ret = aclop_NanToNum(self, scalar, temp, stream);

        ret = aclReductionOpRun(temp, out,
            aclnnCumsumGetWorkspaceSize, aclnnCumsum, stream, false, dim_index, dtype); 
        aclDestroyTensor(temp);
        return ret;
    }
    
    // This is a general function, must be launched differently
    aclError aclop_Round(const aclTensor* self, int decimals, aclTensor* out, aclrtStream stream) {
        return aclBinaryOpRun(self, decimals, out,
            aclnnRoundDecimalsGetWorkspaceSize, aclnnRoundDecimals, stream, false);
    }
    // TODO: fix, rint, around
    // numpy.clip -> aclnnClamp
    // aclError aclop_Clamp(const aclTensor* self, aclScalar* amin, aclScalar* amax, aclTensor* out, aclrtStream stream) {
    //     ret = aclTernaryOpRun(self, amin, amax, out,
    //         aclnnClampGetWorkspaceSize, aclnnClamp, stream, false);
    // }

    // arange() 
    // TODO: Outpout with 2 or more output like `divmod`

#ifdef __cplusplus
}
#endif

#endif // header