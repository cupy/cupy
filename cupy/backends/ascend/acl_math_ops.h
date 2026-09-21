#ifndef CUPY_ACL_MATH_HEADER
#define CUPY_ACL_MATH_HEADER

#include <cmath>

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
#include "aclnnop/aclnn_sinc.h"

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
// deg2rad, rad2deg, square is impl by tensor mul scalar
// radians, degrees are alias to deg2rad, rad2deg
#include "aclnnop/aclnn_pow.h"  // np.pow
#include "aclnnop/aclnn_pow_tensor_tensor.h"

#include "aclnnop/aclnn_abs.h"  // numpy has 3 ops: abs, fabs, absolute
#include "aclnnop/aclnn_neg.h"
//#include "aclnnop/aclnn_pos.h" // no such op
#include "aclnnop/aclnn_ceil.h"
#include "aclnnop/aclnn_floor.h"
#include "aclnnop/aclnn_trunc.h"   // numpy.fix / numpy.trunc
#include "aclnnop/aclnn_round.h"   // numpy.rint / numpy.round / around
#include "aclnnop/aclnn_real.h"    // complex related
// numpy.conj / numpy.imag: CANN 8.5 has no dedicated aclnn op, they are
// emulated in Cython (conj = conj-real + (-1)*imag, imag = view of the
// imaginary part of a complex tensor).
#include "aclnnop/aclnn_clamp.h" // numpy.clip
#include "aclnnop/aclnn_s_where.h" // ternary select, used by copysign etc.
#include "aclnnop/aclnn_signbit.h"
#include "aclnnop/aclnn_sign.h"
#include "aclnnop/aclnn_reciprocal.h"
#include "aclnnop/aclnn_cast.h"    // used by the left_shift composition
// TODO: not yet register
#include "aclnnop/aclnn_heaviside.h"
// ldexp Returns x1 * 2**x2, element-wise.

// equal scalar, tensor, vector/list is_nan (no such)
#include "aclnnop/aclnn_is_inf.h"
#include "aclnnop/aclnn_isfinite.h"
#include "aclnnop/aclnn_right_shift.h"
#include "aclnnop/aclnn_ne_tensor.h"
#include "aclnnop/aclnn_isposinf.h"
#include "aclnnop/aclnn_isneginf.h"
#include "aclnnop/aclnn_isclose.h"

// TODO: complex related op
// complex, imag, real, conj, conjugate, angle, absolute(can deal with complex)
#include "aclnnop/aclnn_complex.h"
//#include "aclnnop/aclnn_angle.h"
// #include "aclnnop/aclnn_conjugate.h"
//#include "aclnnop/aclnn_imaginary.h"

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
#include <aclnnop/aclnn_eq_scalar.h>
#include <aclnnop/aclnn_eq_tensor.h>

// bitwise op: and not not xor
#include "aclnnop/aclnn_bitwise_and_tensor.h"
#include "aclnnop/aclnn_bitwise_and_scalar.h"
#include "aclnnop/aclnn_bitwise_or_tensor.h"
#include "aclnnop/aclnn_bitwise_or_scalar.h"
#include "aclnnop/aclnn_bitwise_xor_tensor.h"
#include "aclnnop/aclnn_bitwise_xor_scalar.h"
#include "aclnnop/aclnn_bitwise_not.h" // numpy op name: np.invert
// CANN 9.0 provides aclnn_left_shift; CANN 8.5 has no left-shift op, so
// aclop_LeftShift composes it as x * 2**n (see below).
#if defined(CUPY_CANN_VERSION) && CUPY_CANN_VERSION >= 900
#include "aclnnop/aclnn_left_shift.h"
#endif

// binary op
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_sub.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_div.h"
#include "aclnnop/aclnn_gcd.h"
//#include "aclnnop/aclnn_lcm.h" // no such alcop, we impl
#include "aclnnop/aclnn_remainder.h" // containing tensor scalar 4 combinations
#include "aclnnop/aclnn_fmod_scalar.h"
#include "aclnnop/aclnn_fmod_tensor.h" 
#include "aclnnop/aclnn_floor_divide.h"
// reverse (scalar <op> tensor) 用到的：aclnnRsubs 是唯一的原生「标量在左」算子，
// 其余（div / floor_divide / fmod）没有 ScalarTensor 版本，用 AclScalarTensorGuard
// 把标量物化成 1 元素张量后走原有的 tensor-tensor 接口。
#include "aclnnop/aclnn_rsub.h"
#include "aclnnop/aclnn_fill_scalar.h"
#include <aclnnop/aclnn_maximum.h>  // find the bigger from two tensors
#include <aclnnop/aclnn_minimum.h>

// tertiary op, not numpy op
//#include "aclnnop/aclnn_addcmul.h" // out = self + value * tensor1 * tensor2

// foreach tensor in aclTensorList, there is no such numpy op
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
#include "aclnnop/aclnn_einsum.h" // TODO
#include <aclnnop/aclnn_nan_to_num.h> // TODO, numpy has more control arg

#include "aclnnop/aclnn_argmax.h"  // return the index instead of value
#include "aclnnop/aclnn_argmin.h"
// amin, amax


// linalg matrix op: qr, tril triu, cross, trace, norm, det
#include "aclnnop/aclnn_matmul.h"
#include "aclnnop/aclnn_dot.h"
#include "aclnnop/aclnn_inverse.h"
#include "aclnnop/aclnn_trace.h"
#include "aclnnop/aclnn_diag.h"
#include "aclnnop/aclnn_qr.h"
#include "aclnnop/aclnn_triangular_solve.h"
#include "aclnnop/aclnn_triu.h"
#include "aclnnop/aclnn_tril.h"
//#include "aclnnop/aclnn_solve.h"
// linalg_cross

#include "./acl_op_template.h"
#include "acl/acl.h"


#ifdef __cplusplus
extern "C" {
#endif

    DECLARE_ACL_BINARY_OPS_FUNC(BitwiseAndTensor)
    DECLARE_ACL_BINARY_SCALAR_OPS_FUNC(BitwiseAndScalar)

    DECLARE_ACL_BINARY_OPS_FUNC(BitwiseOrTensor)
    DECLARE_ACL_BINARY_SCALAR_OPS_FUNC(BitwiseOrScalar)

    DECLARE_ACL_BINARY_OPS_FUNC(BitwiseXorTensor)
    DECLARE_ACL_BINARY_SCALAR_OPS_FUNC(BitwiseXorScalar)

    // BitwiseNot has no inplace version, so can not use the macro to clear
    aclError aclop_BitwiseNot(const aclTensor* self, aclTensor* out, aclrtStream stream) {
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

    // Equal is  all(equal(tensor1, tensor2)) -> scalar
    DECLARE_ACL_BINARY_OP(EqTensor)
    DECLARE_ACL_BINARY_SCALAR_OP(EqScalar)
    DECLARE_ACL_BINARY_SCALAR_OP(NeScalar)
    DECLARE_ACL_BINARY_OP(NeTensor)
    // IsClose() has extra args: double rtol, double atol, bool equal_nan
    DECLARE_ACL_UNARY_OP(IsFinite)
    DECLARE_ACL_UNARY_OP(IsInf)
    DECLARE_ACL_UNARY_OP(IsPosInf)
    DECLARE_ACL_UNARY_OP(IsNegInf)

    // CANN 8.5.1 provides only `aclnnRightShift` (there is no left-shift op).
    // `bitwise_right_shift` is an Array API standard function, so this closes
    // a real gap; `bitwise_left_shift` must be composed instead.
    // NB: there is no `aclnnRightShiftScalar`; the scalar form is reached by
    // the dispatcher only if a SCALAR_BINARY_OP registration is added, which
    // would need an explicit aclScalar→tensor promotion here.
    aclError aclop_RightShift(const aclTensor* self, const aclTensor* other,
                              aclTensor* out, aclrtStream stream) {
        return aclBinaryOpRun(self, other, out,
            aclnnRightShiftGetWorkspaceSize, aclnnRightShift, stream, false);
    }

    // numpy.left_shift(x, n): CANN 8.5.1 has no left-shift aclnn op, so it is
    // composed as x * 2**n in double precision, then cast to the output dtype.
    // Exact for 0 <= n while x and x * 2**n fit the double mantissa (int64
    // inputs lose the low bits for |x| > 2**53). CANN 9.0 provides
    // aclnnLeftShift, used via conditional compiling.
#if defined(CUPY_CANN_VERSION) && CUPY_CANN_VERSION >= 900
    aclError aclop_LeftShift(const aclTensor* self, const aclTensor* other,
                             aclTensor* out, aclrtStream stream) {
        return aclBinaryOpRun(self, other, out,
            aclnnLeftShiftGetWorkspaceSize, aclnnLeftShift, stream, false);
    }
#else
    aclError aclop_LeftShift(const aclTensor* self, const aclTensor* other,
                             aclTensor* out, aclrtStream stream) {
        if (self == nullptr || other == nullptr || out == nullptr) {
            return ACL_ERROR_INVALID_PARAM;
        }
        aclDataType out_dtype;
        aclGetDataType(out, &out_dtype);
        // t1 = double(other); t2 = 2**t1; t3 = double(self); t4 = t3 * t2
        aclTensor* t1 = aclTensorLike(out, ACL_DOUBLE);
        aclTensor* t2 = aclTensorLike(out, ACL_DOUBLE);
        aclTensor* t3 = aclTensorLike(out, ACL_DOUBLE);
        aclTensor* t4 = aclTensorLike(out, ACL_DOUBLE);
        aclError ret = aclIrregularOpRun(aclnnCastGetWorkspaceSize, aclnnCast, stream,
            other, ACL_DOUBLE, t1);
        ret = aclUnaryOpRun(t1, t2,
            aclnnExp2GetWorkspaceSize, aclnnExp2, stream, false);
        ret = aclIrregularOpRun(aclnnCastGetWorkspaceSize, aclnnCast, stream,
            self, ACL_DOUBLE, t3);
        ret = aclBinaryOpRun(t3, t2, t4,
            aclnnMulGetWorkspaceSize, aclnnMul, stream, false);
        ret = aclIrregularOpRun(aclnnCastGetWorkspaceSize, aclnnCast, stream,
            t4, out_dtype, out);
        // aclTensorLike 会 aclrtMalloc 一块显存，必须用 DestroyTensorLike 成对释放
        aclDestroyTensorLike(t1);
        aclDestroyTensorLike(t2);
        aclDestroyTensorLike(t3);
        aclDestroyTensorLike(t4);
        return ret;
    }
#endif

    // CANN has no aclnnIsNan, but `x != x` is true exactly for NaN, so compose
    // it from aclnnNeTensor. (`numpy.isnan` is in the Array API standard, so
    // this is a real gap rather than a convenience.)
    aclError aclop_IsNan(const aclTensor* self, aclTensor* out, aclrtStream stream) {
        if (self == nullptr || out == nullptr) {
            return ACL_ERROR_INVALID_PARAM;
        }
        // aclnnNeTensor cannot be called in place (the output would alias an
        // input), so compute into a bool-typed temporary matching `self`'s
        // shape and then cast to the requested output dtype if needed.
        return aclBinaryOpRun(self, self, out,
            aclnnNeTensorGetWorkspaceSize, aclnnNeTensor, stream, false);
    }

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
    // cupy ufunc names: cupy_arccosh / cupy_arcsinh / cupy_arctanh
    DECLARE_ACL_UNARY_OPS_FUNC(Acosh)
    DECLARE_ACL_UNARY_OPS_FUNC(Asinh)
    DECLARE_ACL_UNARY_OPS_FUNC(Atanh)

    DECLARE_ACL_UNARY_OPS_FUNC(Trunc)  // numpy.fix / numpy.trunc
    // numpy.rint(x) == round half to even; aclnnRound is the unary variant.
    // NOTE: named Rint to avoid clashing with the irregular aclop_Round
    // (aclnnRoundDecimals) declared in acl_general_ops.h.
    aclError aclop_Rint(const aclTensor* self, aclTensor* out, aclrtStream stream) {
        return aclUnaryOpRun(self, out,
            aclnnRoundGetWorkspaceSize, aclnnRound, stream, false);
    }
    aclError aclop_InplaceRint(aclTensor* self, aclrtStream stream) {
        return aclInplaceUnaryOpRun(self,
            aclnnInplaceRoundGetWorkspaceSize, aclnnInplaceRound, stream, false);
    }
    // numpy.real: aclnnReal only accepts complex input (complex64/128).
    // NumPy semantics for a real array are the identity (out0 = in0), served
    // by aclnnCast to the out dtype -- same dtype-preserving copy as
    // aclop_Copy (aclnn_copy.h only has the inplace variant, no aclnnCopy).
    // No inplace version.
    aclError aclop_Real(const aclTensor* self, aclTensor* out, aclrtStream stream) {
        aclDataType dtype = ACL_DT_UNDEFINED;
        aclGetDataType(self, &dtype);
        if (dtype == ACL_COMPLEX64 || dtype == ACL_COMPLEX128) {
            return aclUnaryOpRun(self, out,
                aclnnRealGetWorkspaceSize, aclnnReal, stream, false);
        }
        aclDataType out_dtype = ACL_DT_UNDEFINED;
        aclGetDataType(out, &out_dtype);
        return aclIrregularOpRun(aclnnCastGetWorkspaceSize, aclnnCast, stream,
            self, out_dtype, out);
    }

    // numpy.cbrt(x) = x ** (1/3): CANN has no cbrt op, use pow with 1/3.
    aclError aclop_Cbrt(const aclTensor* self, aclTensor* out, aclrtStream stream) {
        double exp = 1.0 / 3.0;
        return aclBinaryOpRun(self, exp, out,
            aclnnPowTensorScalarGetWorkspaceSize, aclnnPowTensorScalar, stream, false);
    }

    DECLARE_ACL_BINARY_OP(Atan2)  // arctan(x1/x2)
    DECLARE_ACL_UNARY_OPS_FUNC(Sinc)
    DECLARE_ACL_UNARY_OPS_FUNC(Erf)
    DECLARE_ACL_UNARY_OPS_FUNC(Erfc)
    DECLARE_ACL_UNARY_OPS_FUNC(Erfinv)

    // ascend ADD is ternary op with one extra scalar coeff, so can not use the macro to declare
    aclError aclop_Add(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) {
        double alpha = 1.0;  // double will be converted to aclScalar of the same dtype of self aclTensor
        return aclTernaryOpRun(self, other, alpha, out,
        aclnnAddGetWorkspaceSize, aclnnAdd, stream, false);
    }
    aclError aclop_InplaceAdd(aclTensor* self, const aclTensor* other, aclrtStream stream) {
        double alpha = 1.0;
        return aclTernaryInplaceOpRun(self, other, alpha,
        aclnnInplaceAddGetWorkspaceSize, aclnnInplaceAdd, stream, false);
    }
    aclError aclop_Sub(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) {
        double alpha = 1.0;
        return aclTernaryOpRun(self, other, alpha, out,
        aclnnSubGetWorkspaceSize, aclnnSub, stream, false);
    }
    aclError aclop_InplaceSub(aclTensor* self, const aclTensor* other, aclrtStream stream) {
        double alpha = 1.0;
        return aclTernaryInplaceOpRun(self, other, alpha,
        aclnnInplaceSubGetWorkspaceSize, aclnnInplaceSub, stream, false);
    }
    DECLARE_ACL_BINARY_OPS_FUNC(Mul)
    DECLARE_ACL_BINARY_OPS_FUNC(Div)
    // Tensor op Scalar
    DECLARE_ACL_BINARY_SCALAR_OP(Muls)
    DECLARE_ACL_BINARY_SCALAR_OP(Divs)
    aclError aclop_Adds(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream) {
        double alpha = 1.0;
        return aclTernaryOpRun(self, other, alpha, out,
        aclnnAddsGetWorkspaceSize, aclnnAdds, stream, false);
    }
    aclError aclop_Subs(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream) {
        double alpha = 1.0;
        return aclTernaryOpRun(self, other, alpha, out,
        aclnnSubsGetWorkspaceSize, aclnnSubs, stream, false);
    }

    // true_divide == divide
    DECLARE_ACL_BINARY_OPS_FUNC(FloorDivide) // python  `//` int div op, output int
    DECLARE_ACL_BINARY_SCALAR_OPS_FUNC(FloorDivides) // tensor // scalar
    DECLARE_ACL_BINARY_OPS_FUNC(FmodTensor)  // for float and ints
    DECLARE_ACL_BINARY_SCALAR_OPS_FUNC(FmodScalar)

    // ======================================================================
    // reverse scalar binary ops: out = scalar <op> tensor
    //
    // 为什么必须有：`1 - x` / `2 / x` / `1 > x` 这些调用的标量在**左**操作数，
    // 而 dispatcher 以前的窄签名只有「tensor <op> scalar」一种入口 ——
    //   * 没有注册的（subtract/floor_divide/...）会抛 NotImplementedError；
    //   * 已经注册的（true_divide/power）会把操作数顺序搞反，**静默算错**
    //     （`2 / x` 被算成 `x / 2`）。
    // 现在 pyx 侧按操作数位置选 OpType（REVERSE_SCALAR_BINARY_OP），C++ 侧在这里
    // 用「原生 ScalarTensor 接口」或「标量物化」给出正确语义。
    // ======================================================================

    // 把 aclScalar 物化成一个 1 元素设备张量（broadcast 语义），供没有
    // ScalarTensor 变体的 aclnn 接口使用。RAII：析构时释放 tensor + 设备内存。
    class AclScalarTensorGuard {
    public:
        AclScalarTensorGuard(const aclScalar* scalar, aclDataType dtype, aclrtStream stream) {
            if (scalar == nullptr || dtype == ACL_DT_UNDEFINED) {
                std::cerr << "ERROR: AclScalarTensorGuard: null scalar or undefined dtype\n";
                return;
            }
            size_t type_size = aclDataTypeSize(dtype);
            if (type_size == 0) {
                std::cerr << "ERROR: AclScalarTensorGuard: unsupported dtype\n";
                return;
            }
            if (aclrtMalloc(&device_addr_, type_size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
                device_addr_ == nullptr) {
                std::cerr << "ERROR: AclScalarTensorGuard: aclrtMalloc failed\n";
                device_addr_ = nullptr;
                return;
            }
            int64_t dims[1] = {1};
            int64_t strides[1] = {1};
            tensor_ = aclCreateTensor(dims, 1, dtype, strides, 0, ACL_FORMAT_ND, dims, 1,
                                      device_addr_);
            if (tensor_ == nullptr) {
                std::cerr << "ERROR: AclScalarTensorGuard: aclCreateTensor failed\n";
                aclrtFree(device_addr_);
                device_addr_ = nullptr;
                return;
            }
            // 标量 dtype 与目标 dtype 不同（例如 `2 / float32_tensor`）时先转一次
            const aclScalar* value = scalar;
            bool owned = false;
            if (static_cast<aclDataType>(scalar->GetDataType()) != dtype) {
                value = CreateAclScalar(AclScalarToDouble(scalar), dtype);
                owned = true;
            }
            aclError ret = aclIrregularOpRun(aclnnInplaceFillScalarGetWorkspaceSize,
                aclnnInplaceFillScalar, stream, tensor_, value);
            if (owned) {
                aclDestroyScalar(value);
            }
            if (ret != ACL_SUCCESS) {
                std::cerr << "ERROR: AclScalarTensorGuard: fill scalar failed\n";
            }
        }

        AclScalarTensorGuard(const AclScalarTensorGuard&) = delete;
        AclScalarTensorGuard& operator=(const AclScalarTensorGuard&) = delete;

        ~AclScalarTensorGuard() {
            if (tensor_ != nullptr) {
                aclDestroyTensor(tensor_);
            }
            if (device_addr_ != nullptr) {
                aclrtFree(device_addr_);
            }
        }

        const aclTensor* get() const {
            return tensor_;
        }

        explicit operator bool() const {
            return tensor_ != nullptr;
        }

    private:
        aclTensor* tensor_ = nullptr;
        void* device_addr_ = nullptr;
    };

    // out = scalar - tensor：CANN 原生 aclnnRsubs 就是 `out = other - self*alpha`
    aclError aclop_Rsubs(const aclScalar* self, const aclTensor* other, aclTensor* out,
                         aclrtStream stream) {
        double alpha = 1.0;
        return aclTernaryOpRun(other, self, alpha, out,
            aclnnRsubsGetWorkspaceSize, aclnnRsubs, stream, false);
    }

    // out = scalar / tensor：无 ScalarTensor 版本，物化后走 aclnnDiv（1 元素 broadcast）
    aclError aclop_RDivs(const aclScalar* self, const aclTensor* other, aclTensor* out,
                         aclrtStream stream) {
        aclDataType dtype = ACL_DT_UNDEFINED;
        aclGetDataType(out, &dtype);
        AclScalarTensorGuard numerator(self, dtype, stream);
        if (!numerator) {
            return ACL_ERROR_INVALID_PARAM;
        }
        return aclBinaryOpRun(numerator.get(), other, out,
            aclnnDivGetWorkspaceSize, aclnnDiv, stream, false);
    }

    // out = scalar // tensor
    aclError aclop_RFloorDivides(const aclScalar* self, const aclTensor* other, aclTensor* out,
                                 aclrtStream stream) {
        aclDataType dtype = ACL_DT_UNDEFINED;
        aclGetDataType(out, &dtype);
        AclScalarTensorGuard numerator(self, dtype, stream);
        if (!numerator) {
            return ACL_ERROR_INVALID_PARAM;
        }
        return aclBinaryOpRun(numerator.get(), other, out,
            aclnnFloorDivideGetWorkspaceSize, aclnnFloorDivide, stream, false);
    }

    // out = scalar % tensor（fmod 语义，符号跟被除数）
    aclError aclop_RFmodScalar(const aclScalar* self, const aclTensor* other, aclTensor* out,
                               aclrtStream stream) {
        aclDataType dtype = ACL_DT_UNDEFINED;
        aclGetDataType(out, &dtype);
        AclScalarTensorGuard dividend(self, dtype, stream);
        if (!dividend) {
            return ACL_ERROR_INVALID_PARAM;
        }
        return aclBinaryOpRun(dividend.get(), other, out,
            aclnnFmodTensorGetWorkspaceSize, aclnnFmodTensor, stream, false);
    }

    // out = scalar ** tensor（CANN 有原生 aclnnPowScalarTensor）
    aclError aclop_RPowScalar(const aclScalar* self, const aclTensor* other, aclTensor* out,
                              aclrtStream stream) {
        return aclIrregularOpRun(aclnnPowScalarTensorGetWorkspaceSize, aclnnPowScalarTensor,
            stream, self, other, out);
    }

    // out = scalar % tensor（remainder 语义，符号跟除数；CANN 原生 ScalarTensor 版本）
    aclError aclop_RRemainderScalar(const aclScalar* self, const aclTensor* other, aclTensor* out,
                                    aclrtStream stream) {
        return aclIrregularOpRun(aclnnRemainderScalarTensorGetWorkspaceSize,
            aclnnRemainderScalarTensor, stream, self, other, out);
    }

    // 比较运算的 reverse 就是「换边」：`scalar > tensor` == `tensor < scalar`
    aclError aclop_RGtScalar(const aclScalar* self, const aclTensor* other, aclTensor* out,
                             aclrtStream stream) {
        return aclop_LtScalar(other, self, out, stream);
    }
    aclError aclop_RGeScalar(const aclScalar* self, const aclTensor* other, aclTensor* out,
                             aclrtStream stream) {
        return aclop_LeScalar(other, self, out, stream);
    }
    aclError aclop_RLtScalar(const aclScalar* self, const aclTensor* other, aclTensor* out,
                             aclrtStream stream) {
        return aclop_GtScalar(other, self, out, stream);
    }
    aclError aclop_RLeScalar(const aclScalar* self, const aclTensor* other, aclTensor* out,
                             aclrtStream stream) {
        return aclop_GeScalar(other, self, out, stream);
    }

    DECLARE_ACL_BINARY_OPS_FUNC(RemainderTensorTensor) // remainder has 4 version
    DECLARE_ACL_BINARY_SCALAR_OP(RemainderTensorScalar) // remainder has 4 version
    // Power has 3 version, Remainder has 4 version
    DECLARE_ACL_BINARY_OP(PowTensorTensor)
    DECLARE_ACL_BINARY_SCALAR_OPS_FUNC(PowTensorScalar)

    DECLARE_ACL_BINARY_OP(Gcd)
    aclError aclop_Lcm(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) {
        // how to deal with minus integer?
        aclDataType dtype;
        aclGetDataType(self, &dtype);
        aclTensor* temp = aclTensorLike(self, dtype);
        auto ret = aclBinaryOpRun(self, other, temp,
            aclnnMulGetWorkspaceSize, aclnnMul, stream, false);
        ret = aclUnaryOpRun(temp, temp, // on inplace version, is that OK?
            aclnnAbsGetWorkspaceSize, aclnnAbs, stream, false);
        aclTensor* gcd = aclTensorLike(self, dtype);
        ret = aclBinaryOpRun(temp, other, gcd,
            aclnnGcdGetWorkspaceSize, aclnnGcd, stream, false); 
        ret = aclBinaryOpRun(temp, gcd, out,
            aclnnDivGetWorkspaceSize, aclnnDiv, stream, false);
        // aclTensorLike 会 aclrtMalloc 一块显存，必须用 DestroyTensorLike 成对释放
        aclDestroyTensorLike(temp);
        aclDestroyTensorLike(gcd);
        return ret;
    }


    DECLARE_ACL_BINARY_OP(Maximum)
    DECLARE_ACL_BINARY_OP(Minimum)
    // divmod has two outs

    // numpy.hypot(x1, x2) = sqrt(x1**2 + x2**2); no aclnn op, compose it.
    aclError aclop_Hypot(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) {
        aclDataType dtype;
        aclGetDataType(self, &dtype);
        aclTensor* sq1 = aclTensorLike(self, dtype);
        aclTensor* sq2 = aclTensorLike(other, dtype);
        auto ret = aclBinaryOpRun(self, self, sq1,
            aclnnMulGetWorkspaceSize, aclnnMul, stream, false);
        ret = aclBinaryOpRun(other, other, sq2,
            aclnnMulGetWorkspaceSize, aclnnMul, stream, false);
        double alpha = 1.0;
        ret = aclTernaryOpRun(sq1, sq2, alpha, out,
            aclnnAddGetWorkspaceSize, aclnnAdd, stream, false);
        ret = aclUnaryOpRun(out, out,
            aclnnSqrtGetWorkspaceSize, aclnnSqrt, stream, false);
        // aclTensorLike 会 aclrtMalloc 一块显存，必须用 DestroyTensorLike 成对释放
        aclDestroyTensorLike(sq1);
        aclDestroyTensorLike(sq2);
        return ret;
    }

    // numpy.copysign(x1, x2) = |x1| with the sign of x2 (sign of +0 / -0 counts).
    // aclnnSign(0) == 0, so the sign bit is taken from `x2 >= 0` instead.
    aclError aclop_Copysign(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) {
        aclDataType dtype;
        aclGetDataType(self, &dtype);
        aclTensor* mag = aclTensorLike(self, dtype);   // |x1|
        aclTensor* neg = aclTensorLike(self, dtype);   // -|x1|
        aclTensor* mask = aclTensorLike(other, ACL_BOOL);  // x2 >= 0
        auto ret = aclUnaryOpRun(self, mag,
            aclnnAbsGetWorkspaceSize, aclnnAbs, stream, false);
        ret = aclUnaryOpRun(mag, neg,
            aclnnNegGetWorkspaceSize, aclnnNeg, stream, false);
        // 0.0 is treated as non-negative, matching the IEEE sign bit of +0
        float zero = 0.0f;
        const aclScalar* zero_scalar = aclCreateScalar(&zero, ACL_FLOAT);
        ret = aclBinaryOpRun(other, zero_scalar, mask,
            aclnnGeScalarGetWorkspaceSize, aclnnGeScalar, stream, false);
        aclDestroyScalar(zero_scalar);
        ret = aclIrregularOpRun(aclnnSWhereGetWorkspaceSize, aclnnSWhere, stream,
            mask, mag, neg, out);
        // aclTensorLike 会 aclrtMalloc 一块显存，必须用 DestroyTensorLike 成对释放
        aclDestroyTensorLike(mag);
        aclDestroyTensorLike(neg);
        aclDestroyTensorLike(mask);
        return ret;
    }

    DECLARE_ACL_UNARY_OPS_FUNC(Reciprocal)
    DECLARE_ACL_UNARY_OPS_FUNC(Neg)

    aclError aclop_Abs(const aclTensor* self, aclTensor* out, aclrtStream stream) {
        return aclUnaryOpRun(self, out,
        aclnnAbsGetWorkspaceSize, aclnnAbs, stream, false);
    }
    // numpy.fabs is abs restricted to real floating point; aclnnAbs handles it.
    aclError aclop_Fabs(const aclTensor* self, aclTensor* out, aclrtStream stream) {
        return aclUnaryOpRun(self, out,
        aclnnAbsGetWorkspaceSize, aclnnAbs, stream, false);
    }

    aclError aclop_Square(const aclTensor* self, aclTensor* out, aclrtStream stream) {
        float power = 2.0f;
        return aclBinaryOpRun(self, power, out,
            aclnnPowTensorScalarGetWorkspaceSize, aclnnPowTensorScalar, stream, false); 
    }
    aclError aclop_Rsqrt(const aclTensor* self, aclTensor* out, aclrtStream stream) {
        float power = -0.5f;
        return aclBinaryOpRun(self, power, out,
            aclnnPowTensorScalarGetWorkspaceSize, aclnnPowTensorScalar, stream, false); 
    }
    aclError aclop_Deg2rad(const aclTensor* self, aclTensor* out, aclrtStream stream) {
        double ratio = M_PI / 180.0;
        return aclBinaryOpRun(self, ratio, out,
            aclnnMulsGetWorkspaceSize, aclnnMuls, stream, false); 
    }
    aclError aclop_Rad2deg(const aclTensor* self, aclTensor* out, aclrtStream stream) {
        double ratio = 180.0 / M_PI;
        return aclBinaryOpRun(self, ratio, out,
            aclnnMulsGetWorkspaceSize, aclnnMuls, stream, false); 
    }

    DECLARE_ACL_BINARY_OP(Complex)

    DECLARE_ACL_UNARY_OP(Signbit)  // no inplace version
    DECLARE_ACL_UNARY_OP(Sign) 

    //DECLARE_ACL_UNARY_OPS_FUNC(Abs) // no inplace version
    // aclError aclop_Abs(const aclTensor* self, aclTensor* out, aclrtStream stream) {
    //     return aclUnaryOpRun(self, out,
    //     aclnnAbsGetWorkspaceSize, aclnnAbs, stream, false);
    // }
    DECLARE_ACL_UNARY_OPS_FUNC(Floor)
    DECLARE_ACL_UNARY_OPS_FUNC(Ceil)
    // NOTE: aclnnSqrt supports real dtypes only (no complex).
    DECLARE_ACL_UNARY_OPS_FUNC(Sqrt)
    DECLARE_ACL_UNARY_OPS_FUNC(Exp)
    DECLARE_ACL_UNARY_OPS_FUNC(Exp2)
    DECLARE_ACL_UNARY_OPS_FUNC(Expm1)
    DECLARE_ACL_UNARY_OPS_FUNC(Log)
    DECLARE_ACL_UNARY_OPS_FUNC(Log2)
    DECLARE_ACL_UNARY_OPS_FUNC(Log10)
    DECLARE_ACL_UNARY_OPS_FUNC(Log1p)
    DECLARE_ACL_BINARY_OP(LogAddExp2)
    DECLARE_ACL_BINARY_OP(LogAddExp)

    // ================================================================================

    aclError aclop_Matmul(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) {
        // aclnnMatmulGetWorkspaceSize(self, mat2, out, int8_t cubeMathType, ...)
        int8_t math_type = 0; // 0 == KEEP_DTYPE, keep input precision
        // row-major A @ B, no transpose trick (unlike cuBLAS column-major)
        return aclBinaryOpRun(self, other, out,
            aclnnMatmulGetWorkspaceSize, aclnnMatmul, stream, false, math_type); 
    }
    // aclnnDot(self, tensor, out): 1-D . 1-D -> 0-D, dtypes FLOAT/BF16/FLOAT16
    DECLARE_ACL_BINARY_OP(Dot)

    DECLARE_ACL_UNARY_OP(Inverse)
    //DECLARE_ACL_UNARY_OP(Diag)  // depends on how cupy_XXX is defined, ufunc/elementwiseKernel
    
    // aclError aclop_Det(const aclTensor* self, aclTensor* out, aclrtStream stream) {
    //     return aclUnaryOpRun(self, out,
    //         aclnnDetGetWorkspaceSize, aclnnDet, stream, false, math_type); 
    // }
    
#ifdef __cplusplus
}
#endif

#endif // header