#ifndef CUPY_ACL_GENERAL_OPS_HEADER
#define CUPY_ACL_GENERAL_OPS_HEADER

// creation op:  with dim info
// arange, eye, diag, linspace, ones, zeros, 
#include "aclnnop/aclnn_arange.h"
#include "aclnnop/aclnn_eye.h"  //  np.eye == np.identity(N)
#include "aclnnop/aclnn_diag.h"  // UnaryScalarOp   not sure TODO
#include "aclnnop/aclnn_trace.h" // UnaryOp

// math ops, but it is irregular ops
#include <aclnnop/aclnn_round.h>
#include <aclnnop/aclnn_clamp.h>

// convolve,  mode='fill'
#include "aclnnop/aclnn_fill_scalar.h"
#include "aclnnop/aclnn_fill_tensor.h"

#include "aclnnop/aclnn_div.h"
#include "aclnnop/aclnn_cast.h"

// indexing: argsort, unique, unique2, sort
// no count() , unique(), unique2() op
#include "aclnnop/aclnn_index.h"

// normal, uniform distributions:

// manipulation op:  sort select take put
// use fill_scalar (zeros) to impl   numpy op: no.one
#include "aclnnop/aclnn_fill_tensor.h" // fill_scalar, masked_fill
#include "aclnnop/aclnn_fill_scalar.h"
#include "aclnnop/aclnn_take.h"
#include "aclnnop/aclnn_put.h"

#include "aclnnop/aclnn_flip.h"
//#include "aclnnop/aclnn_rot.h"
#include "aclnnop/aclnn_stack.h"
#include "aclnnop/aclnn_cat.h"
// split, resize
#include "aclnnop/aclnn_flatten.h"
#include "aclnnop/aclnn_permute.h"
#include "aclnnop/aclnn_copy.h"

// manipulation: transpose, reshape, cast, pad continguous in aclnn_kernels/
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/reshape.h"  // use experiment/platform header

#include "./acl_op_template.h"
#include "./acl_scalar_arg.h"
#include "acl/acl.h"


    // arange(), 
    // aclnnEyeGetWorkspaceSize(int64_t n, int64_t m, aclTensor* out,
    // aclnnLinspaceGetWorkspaceSize(const aclScalar* start, const aclScalar* end, int64_t steps, aclTensor* out,

    // TODO: fix, rint(), around,
    // aclnnTraceGetWorkspaceSize(const aclTensor* self, aclTensor* out     
    // aclnnTrilGetWorkspaceSize(const aclTensor* self, int64_t diagonal, aclTensor* out,  // set upper as zeros
    // aclnnPermuteGetWorkspaceSize(const aclTensor* self, const aclIntArray* dims, aclTensor* out,

    // two output tensor
    // aclError aclop_Sort(){
    //     ACLNN_API aclnnStatus aclnnSortGetWorkspaceSize(const aclTensor* self, bool stable, int64_t dim, bool descending,
    //                                             aclTensor* valuesOut, aclTensor* indicesOut, uint64_t* workspaceSize,
    //                                             aclOpExecutor** executor);
    // }

    // frexp() Decompose the elements of x into mantissa and twos exponent.
    // aclError aclop_Frexp(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
    //     const ArgsType& args, const KargsType& kargs, aclrtStream stream) {
    //     const aclTensor* self = ins[0];
    //     aclTensor* out = outs[0];
    //     int decimals = ToScalarArg<int>(args[0]);
    //     return aclIrregularOpRun(aclnnFrexpGetWorkspaceSize, aclnnFrexp, stream,
    //         self, outs);
    // }

    // aclnnTopkGetWorkspaceSize(const aclTensor* self, int64_t k, int64_t dim, bool largest,
    //                                             bool sorted, aclTensor* valuesOut, aclTensor* indicesOut,


    // aclnnStackGetWorkspaceSize(const aclTensorList* tensors, int64_t dim, aclTensor* out,
    aclError aclop_Stack(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KargsType& kargs, aclrtStream stream) {
        if (ins.size() >= 1) {
            auto tl = ToAclTensorList(ins);
            int64_t dim = GetScalarArg<int64_t>(args, 0, kargs, "dim");
            return aclIrregularOpRun(aclnnStackGetWorkspaceSize, aclnnStack, stream,
                tl, dim, outs[0]);
        } else {
            std::cout << "Error:" <<  __FUNCTION__  << " take args: tensorList, axis, out) \n";
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    aclError aclop_Concat(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KargsType& kargs, aclrtStream stream) {
        if (ins.size() >= 1) {
            auto tl = ToAclTensorList(ins);
            // TODO: default dim value
            int64_t dim = GetScalarArg<int64_t>(args, 0, kargs, "dim");
            return aclIrregularOpRun(aclnnCatGetWorkspaceSize, aclnnCat, stream,
                tl, dim, outs[0]);
        } else {
            std::cout << "Error:" <<  __FUNCTION__  << " take args: tensorList, axis, out) \n";
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    // numpy has no such op resize
    // aclnnResizeGetWorkspaceSize(const aclTensor* self, const aclFloatArray* scales, const char* mode, aclTensor* out,

    // aclnnFlattenGetWorkspaceSize(const aclTensor* self, int64_t axis, aclTensor* out,
    aclError aclop_Flatten(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KargsType& kargs, aclrtStream stream) {
        if (outs.size() == 1) {
            int64_t axis = GetScalarArg<int64_t>(args, 0, kargs, "axis");
            return aclIrregularOpRun(aclnnFlattenGetWorkspaceSize, aclnnFlatten, stream,
                ins[0], axis, outs[0]);
        } else {
            std::cout << "Error:" <<  __FUNCTION__  << " take input tensors (self), arg axis, and out tensor \n";
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    // numpy.take_along_axis(arr, indices, axis=-1)
    DECLARE_ACL_BINARY_OP(Take)  // axis=None, out=None, mode='raise'
    // numpy.put(a, ind, v, mode='raise')
    // aclError aclop_Put(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
    //     const ArgsType& args, const KargsType& kargs, aclrtStream stream) {
    //     aclTensor* self = ins[0];
    //     if (args.size() == 3) {
    //         return aclInplaceBinaryOpRun(self, ins[1], ins[2],
    //             aclnnInplacePutGetWorkspaceSize, aclnnInplacePut, stream);
    //     } else {
    //         std::cout << "Error:" <<  __FUNCTION__  << " take 3 input tensors (self, index, value) \n";
    //         return ACL_ERROR_INVALID_PARAM;
    //     }
    // }

    aclError aclop_Cast(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KargsType& kargs, aclrtStream stream) {
        if (outs.size() == 1) {
            const aclTensor* self = ins[0];
            aclDataType dtype;
            aclGetDataType(outs[0], &dtype);
            return aclIrregularOpRun(aclnnCastGetWorkspaceSize, aclnnCast, stream,
                self, dtype, outs[0]);
        } else {
            std::cout << "Error:" <<  __FUNCTION__  << " take 3 input tensors (self, index, value) \n";
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    //DECLARE_ACL_BINARY_OP(InplaceCopy)
    // aclnnTakeGetWorkspaceSize(const aclTensor* self, const aclTensor* index, aclTensor* out, ...);
    // aclnnInplacePutGetWorkspaceSize(aclTensor* selfRef, const aclTensor* index,
    //                                                 const aclTensor* source, bool accumulate,
    // aclnnInplaceCopyGetWorkspaceSize(aclTensor* selfRef, const aclTensor* src,

    
    // astype():  casting UnaryOp with dtype
    // aclnnCastGetWorkspaceSize(const aclTensor* self, const aclDataType dtype, aclTensor* out,
    // fill_kernel = ElementwiseKernel('T x', 'T y', 'y = x', 'cupy_fill')
    aclError aclop_Fill(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KargsType& kargs, aclrtStream stream) {
        aclTensor* self = outs[0];
        if (args.size()) {
            return aclInplaceBinaryOpRun(self, args[0],
                aclnnInplaceFillScalarGetWorkspaceSize, aclnnInplaceFillScalar, stream, false);
        } else if (ins.size() >= 1) {
            return aclInplaceBinaryOpRun(self, ins[0],
                aclnnInplaceFillTensorGetWorkspaceSize, aclnnInplaceFillTensor, stream, false);
        } else {
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    aclError aclop_fill(const aclTensorList* ins, const aclTensorList* outs,
        const ArgsType& args, const KargsType& kargs, aclrtStream stream) {
            // aclTensor* self = aclGetTensorList(tensorList, 0);  // no such api
            // if (self) {
            //     return 0;
            // }
    }

    // This is a general function, must be launched differently, keyward args?
    aclError aclop_Round(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KargsType& kargs, aclrtStream stream) {
        const aclTensor* self = ins[0];
        aclTensor* out = outs[0];
        int decimals = ToScalarArg<int>(args[0]);
        return aclIrregularOpRun(aclnnRoundDecimalsGetWorkspaceSize, aclnnRoundDecimals, stream,
            self, decimals, out);
    }
    // numpy.clip -> aclnnClamp
    aclError aclop_Clamp(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
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