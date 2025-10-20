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
#include "aclnnop/aclnn_exp2.h"
#include "aclnnop/aclnn_log.h"
#include "aclnnop/aclnn_log1p.h"
#include "aclnnop/aclnn_log2.h"
#include "aclnnop/aclnn_log10.h"
#include "aclnnop/aclnn_sqrt.h"
//#include "aclnnop/aclnn_square.h" // np.pow with scalar 2
#include "aclnnop/aclnn_pow.h"  // np.pow

#include "aclnnop/aclnn_abs.h"
#include "aclnnop/aclnn_neg.h"
#include "aclnnop/aclnn_ceil.h"
#include "aclnnop/aclnn_floor.h"
#include "aclnnop/aclnn_clamp.h"
// sign, inverse, neg, reciprocal

// equal scalar, tensor, vector/list
//is_inf, isclose, is_posinf isfinite, is_nan (no such)
#include "aclnnop/aclnn_is_inf.h"
#include "aclnnop/aclnn_isfinite.h"

#include "aclnnop/aclnn_equal.h"
#include "aclnnop/aclnn_isclose.h"
// ge, eq, le, gt, lt, 
#include "aclnnop/aclnn_ge_tensor.h"
// reduction op
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

// logical op: for bool/cast_to_bool input tensor
#include "aclnnop/aclnn_logical_and.h"
#include "aclnnop/aclnn_logical_or.h"
#include "aclnnop/aclnn_logical_not.h"

// indexing: argsort, unique, sort


// binary op
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_sub.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_div.h"
#include "aclnnop/aclnn_gcd.h"
#include "aclnnop/aclnn_remainder.h"
//#include "aclnnop/aclnn_mod.h"  // no such? fmode
//floor_div

// tertiary op, not numpy op
//#include "aclnnop/aclnn_addcmul.h" // out = self + value * tensor1 * tensor2

// foreach? batch, not numpy op
#include "aclnnop/aclnn_foreach_add_scalar.h"
#include "aclnnop/aclnn_foreach_sub_scalar.h"
#include "aclnnop/aclnn_foreach_mul_scalar.h"  // _v2?

#include "aclnnop/aclnn_div.h"
#include "aclnnop/aclnn_gcd.h"

// reduce op, how about dim
#include "aclnnop/aclnn_cumsum.h"
#include "aclnnop/aclnn_cumprod.h"
#include "aclnnop/aclnn_max.h"  // nan?
#include "aclnnop/aclnn_min.h"
#include "aclnnop/aclnn_dot.h"
#include "aclnnop/aclnn_einsum.h" 

#include "aclnnop/aclnn_argmax.h"  // return the index instead of value
#include "aclnnop/aclnn_argmin.h"

// statistics
#include "aclnnop/aclnn_mean.h"
#include "aclnnop/aclnn_std.h"
#include "aclnnop/aclnn_var.h"
// count

// creation op with dim info
#include "aclnnop/aclnn_matmul.h"
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

// manipulation: transpose, reshape, cast, pad continguous in aclnn_kernels/
// #include "aclnn_kernels/transpose.h"
// #include "aclnn_kernels/cast.h"
// #include "aclnn_kernels/pad.h"
// #include "aclnn_kernels/slice.h"

// linalg matrix op
// qr, tril triu, matmul
// trace, norm, det

#include "./acl_op_template.h"
#include "acl/acl.h"


#ifdef __cplusplus
extern "C" {
#endif

    DECLARE_ACL_BINARY_OPS_FUNC(BitwiseAndTensor)
    //DECLARE_ACL_BINARY_OPS_FUNC(BitwiseAndScalar)

    DECLARE_ACL_BINARY_OPS_FUNC(BitwiseOrTensor)
    //DECLARE_ACL_BINARY_OPS_FUNC(BitwiseOrScalar)

    DECLARE_ACL_BINARY_OPS_FUNC(BitwiseXorTensor)
    //DECLARE_ACL_BINARY_OPS_FUNC(BitwiseXorScalar)

    // BitwiseNot has no inplace version, so can not use the macro to clear
    aclError aclop_BitwiseNot(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) {
        return aclUnaryOpRun(self, out,
        aclnnBitwiseNotGetWorkspaceSize, aclnnBitwiseNot, stream, false);
    }

    DECLARE_ACL_UNARY_OPS_FUNC(Cos)
    DECLARE_ACL_UNARY_OPS_FUNC(Sin)
    DECLARE_ACL_UNARY_OPS_FUNC(Tan)
    DECLARE_ACL_UNARY_OPS_FUNC(Acos)
    DECLARE_ACL_UNARY_OPS_FUNC(Asin)
    DECLARE_ACL_UNARY_OPS_FUNC(Atan)
    DECLARE_ACL_UNARY_OPS_FUNC(Cosh)
    DECLARE_ACL_UNARY_OPS_FUNC(Sinh)
    DECLARE_ACL_UNARY_OPS_FUNC(Tanh)

    // aclError aclop_BitwiseAndTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) {
    //     return aclBinaryOpRun(self, other, out,
    //         aclnnBitwiseAndTensorGetWorkspaceSize, aclnnBitwiseAndTensor, stream, false); 
    // }
    // aclError aclop_InplaceBitwiseAndTensor(aclTensor* self, const aclTensor* other, aclrtStream stream) {
    //     return aclBinaryInplaceOpRun(self, other,
    //         aclnnInplaceBitwiseAndTensorGetWorkspaceSize, aclnnBitwiseAndTensor, stream, false); 
    // }
    aclError aclop_BitwiseAndScalar(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream) {
        return aclBinaryOpRun(self, other, out,
            aclnnBitwiseAndScalarGetWorkspaceSize, aclnnBitwiseAndScalar, stream, false); 
    }
    aclError aclop_InplaceBitwiseAndScalar(aclTensor* self, const aclScalar* other, aclrtStream stream) {
        return aclBinaryInplaceOpRun(self, other,
            aclnnInplaceBitwiseAndScalarGetWorkspaceSize, aclnnBitwiseAndScalar, stream, false); 
    }

    // ascend add op is special with one extra scalar coeff, so can not use the macro to declare
    aclError aclop_Add(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) {
        float alpha = 1.0f;
        return aclTernaryScalarOpRun(self, other, alpha, out,
        aclnnAddGetWorkspaceSize, aclnnAdd, stream, false);
    }
    aclError aclop_InplaceAdd(aclTensor* self, const aclTensor* other, aclrtStream stream) {
        float alpha = 1.0f;
        return aclTernaryInplaceScalarOpRun(self, other, alpha,
        aclnnInplaceAddGetWorkspaceSize, aclnnInplaceAdd, stream, false);
    }

    aclError aclop_MatMul(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) {
        uint8_t math_type = 0; // keep dtype precision KEEP_DTYPE
        return aclBinaryOpRun(self, other, out,
            aclnnMatmulGetWorkspaceSize, aclnnMatmul, stream,  false, math_type); 
    }
    

#ifdef __cplusplus
}
#endif

#endif // header