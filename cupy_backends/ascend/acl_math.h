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
// pow, sqrt

#include "aclnnop/aclnn_abs.h"
#include "aclnnop/aclnn_neg.h"
#include "aclnnop/aclnn_ceil.h"
#include "aclnnop/aclnn_floor.h"
#include "aclnnop/aclnn_clamp.h"
// sign, inverse, neg, reciprocal


#include "aclnnop/aclnn_all.h"
#include "aclnnop/aclnn_any.h"
// equal scalar, tensor, vector/list
//is_inf, isclose, is_posinf isfinite, is_nan (no such)
#include "aclnnop/aclnn_is_inf.h"
#include "aclnnop/aclnn_isfinite.h"

#include "aclnnop/aclnn_equal.h"
#include "aclnnop/aclnn_isclose.h"
// ge, eq, le, gt, lt, 
#include "aclnnop/aclnn_ge_tensor.h"

// binary and not not xor
#include "aclnnop/aclnn_bitwise_and_tensor.h"
#include "aclnnop/aclnn_bitwise_or_tensor.h"
#include "aclnnop/aclnn_bitwise_not.h"
#include "aclnnop/aclnn_bitwise_xor_tensor.h"

// logical for bool
#include "aclnnop/aclnn_logical_and.h"
#include "aclnnop/aclnn_logical_or.h"
#include "aclnnop/aclnn_logical_not.h"

// manipulation: transpose, reshape, cast, pad continguous in aclnn_kernels/
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/slice.h"

// indexing: argsort, unique
#include "aclnnop/aclnn_arange.h"
#include "aclnnop/aclnn_argmax.h"
#include "aclnnop/aclnn_argmin.h"

// binary op
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_sub.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_div.h"
#include "aclnnop/aclnn_gcd.h"
#include "aclnnop/aclnn_remainder.h"
//#include "aclnnop/aclnn_mod.h"  // no such?
// radd ??

#include "aclnnop/aclnn_foreach_add_scalar.h"
#include "aclnnop/aclnn_foreach_sub_scalar.h"
#include "aclnnop/aclnn_foreach_mul_scalar.h"  // _v2?
#include "aclnnop/aclnn_div.h"
#include "aclnnop/aclnn_gcd.h"

// reduce op, how about dim
#include "aclnnop/aclnn_cumsum.h"
#include "aclnnop/aclnn_cumprod.h"
#include "aclnnop/aclnn_max.h"
#include "aclnnop/aclnn_min.h"
#include "aclnnop/aclnn_dot.h"
#include "aclnnop/aclnn_einsum.h" 

// statistics
#include "aclnnop/aclnn_mean.h"
#include "aclnnop/aclnn_std.h"
#include "aclnnop/aclnn_var.h"


// Op with dim info
// arange, eye, diag, linspace, ones, zeros, 
// flip faltten copy stack

// foreach?
// qr, tril triu, matmul
// trace, norm, det

// sort select take put