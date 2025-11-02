#ifndef CUPY_ACL_GENERAL_OPS_HEADER
#define CUPY_ACL_GENERAL_OPS_HEADER

#include <aclnnop/aclnn_round.h>
#include <aclnnop/aclnn_clamp.h>

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

#include "./acl_op_template.h"
#include "./acl_scalar_arg.h"
#include "acl/acl.h"

    // arange() , // TODO: fix, rint(), around,
    // TODO: Outpout with 2 or more output like `divmod`
    // aclError aclop_Divmode(const std::vector<const aclTensor*> ins, const std::vector<aclTensor*> outs,
    //     const std::vector<const aclScalar*> args, aclrtStream stream) {
        
    // }

    // This is a general function, must be launched differently, keyward args?
    aclError aclop_Round(const std::vector<const aclTensor*> ins, const std::vector<aclTensor*> outs,
        const ArgsType& args, const KargsType& kargs, aclrtStream stream) {
        const aclTensor* self = ins[0];
        aclTensor* out = outs[0];
        int decimals = ToScalarArg<int>(args[0]);
        return aclBinaryOpRun(self, decimals, out,
            aclnnRoundDecimalsGetWorkspaceSize, aclnnRoundDecimals, stream, false);
    }
    // numpy.clip -> aclnnClamp
    aclError aclop_Clamp(const std::vector<const aclTensor*> ins, const std::vector<aclTensor*> outs,
        const ArgsType& args, const KargsType& kargs, aclrtStream stream) {
        const aclTensor* self = ins[0];
        aclTensor* out = outs[0];
        const aclScalar* amin = args[0];
        const aclScalar* amax = args[1];
        return aclTernaryOpRun(self, amin, amax, out,
            aclnnClampGetWorkspaceSize, aclnnClamp, stream, false);
    }

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif // header