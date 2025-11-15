#ifndef CUPY_ACL_GENERAL_OPS_HEADER
#define CUPY_ACL_GENERAL_OPS_HEADER

// creation op:  with dim info
// arange, eye, diag, linspace (no such) ones, zeros, 
#include "aclnnop/aclnn_arange.h"
#include "aclnnop/aclnn_eye.h"  //  np.eye == np.identity(N)
#include "aclnnop/aclnn_diag.h"  // UnaryScalarOp   not sure TODO
#include "aclnnop/aclnn_trace.h" // UnaryOp

// math ops, but it is irregular ops
#include <aclnnop/aclnn_round.h>
#include <aclnnop/aclnn_isclose.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_nonzero.h>

// convolve,  mode='fill'
#include "aclnnop/aclnn_fill_scalar.h"
#include "aclnnop/aclnn_fill_tensor.h"
// masked_fill
// use fill_scalar (zeros) to impl   numpy op: np.zeros, np.ones

#include "aclnnop/aclnn_div.h"
#include "aclnnop/aclnn_remainder.h"
#include "aclnnop/aclnn_copy.h"

// indexing: argsort, unique, unique2, sort
// no count() , unique(), unique2() op
#include "aclnnop/aclnn_unique2.h"
#include "aclnnop/aclnn_index.h"
#include "aclnnop/aclnn_sort.h"

// normal, uniform distributions:

// manipulation op:  sort select take put
#include "aclnnop/aclnn_take.h"
#include "aclnnop/aclnn_put.h"

#include "aclnnop/aclnn_flip.h"
//#include "aclnnop/aclnn_rot.h"
#include "aclnnop/aclnn_stack.h"
#include "aclnnop/aclnn_cat.h" // concatenate
// split, resize
#include "aclnnop/aclnn_flatten.h"
#include "aclnnop/aclnn_permute.h"
#include "aclnnop/aclnn_cast.h"

// manipulation: transpose, reshape, cast, pad continguous in aclnn_kernels/
// including these experiment/platform headers can cause `segmentation fault`
// #include "aclnn_kernels/transpose.h"
// #include "aclnn_kernels/cast.h"
// #include "aclnn_kernels/pad.h"
// #include "aclnn_kernels/slice.h"
// #include "aclnn_kernels/reshape.h"

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
    //     const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
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
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.size() >= 1) {
            auto tl = ToAclTensorList(ins);
            int64_t dim = GetScalarArg<int64_t>(args, 0, kwargs, "dim");
            return aclIrregularOpRun(aclnnStackGetWorkspaceSize, aclnnStack, stream,
                tl, dim, outs[0]);
        } else {
            std::cout << "Error:" <<  __FUNCTION__  << " take args: tensorList, axis, out) \n";
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    aclError aclop_Concat(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.size() >= 1) {
            auto tl = ToAclTensorList(ins);
            // TODO: default dim value
            int64_t dim = GetScalarArg<int64_t>(args, 0, kwargs, "dim");
            return aclIrregularOpRun(aclnnCatGetWorkspaceSize, aclnnCat, stream,
                tl, dim, outs[0]);
        } else {
            std::cout << "Error:" <<  __FUNCTION__  << " take args: tensorList, axis, out) \n";
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    // dims is a int/tuple of int/None
    aclError aclop_Flip(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
            const aclIntArray* dims = nullptr; // default to axis = None
            return aclIrregularOpRun(aclnnFlipGetWorkspaceSize, aclnnFlip, stream,
                ins[0], dims, outs[0]);
    }

    // numpy has op resize, but diff from the scaling
    // aclnnResizeGetWorkspaceSize(const aclTensor* self, const aclFloatArray* scales, const char* mode, aclTensor* out,

    // aclnnFlattenGetWorkspaceSize(const aclTensor* self, int64_t axis, aclTensor* out,
    aclError aclop_Flatten(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (outs.size() == 1) {
            int64_t axis = GetScalarArg<int64_t>(args, 0, kwargs, "axis");
            return aclIrregularOpRun(aclnnFlattenGetWorkspaceSize, aclnnFlatten, stream,
                ins[0], axis, outs[0]);
        } else {
            std::cout << "Error:" <<  __FUNCTION__  << " take input tensors (self), arg axis, and out tensor \n";
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    // numpy.take_along_axis(arr, indices, axis=-1)
    // ElementwiseKernel('raw T a, S indices, uint32 ldim, uint32 cdim, uint32 rdim, int64 index_range', 'T out'
    DECLARE_ACL_BINARY_OP(Take)  // axis=None, out=None, mode='raise'
    // numpy.put(a, ind, v, mode='raise')
    // aclError aclop_Put(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
    //     const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
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
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
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

    // aclnnTakeGetWorkspaceSize(const aclTensor* self, const aclTensor* index, aclTensor* out, ...);
    // aclnnInplacePutGetWorkspaceSize(aclTensor* selfRef, const aclTensor* index,
    //                                                 const aclTensor* source, bool accumulate,

    // aclError aclop_Copy(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
    //     const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
    //     const aclTensor* other = ins[0];
    //     aclTensor* out = outs[0];
    //     int decimals = ToScalarArg<int>(args[0]);
    //     return aclIrregularOpRun(aclnnInplaceCopyGetWorkspaceSize, aclnnInplaceCopy, stream,
    //         out, other);
    // }

    // experimental api style
    aclError aclop_fill(const aclTensorList* ins, const aclTensorList* outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
            // aclTensor* self = aclGetTensorList(tensorList, 0);  // no such api
            // if (self) {
            //     return 0;
            // }
    }
    
    // astype():  casting UnaryOp with dtype
    // aclnnCastGetWorkspaceSize(const aclTensor* self, const aclDataType dtype, aclTensor* out,
    // fill_kernel = ElementwiseKernel('T x', 'T y', 'y = x', 'cupy_fill')
    aclError aclop_Fill(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
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

    // args such as `device, like, dtype` will be dealt by caller
    // numpy.arange(numpy.arange([start, ]stop, [step, ]) 
    // cupy.arange() kernel is so diff, can not reuse the cupy_arange kernel name to register?
    // cupy_arange  take  start and step as input parameter
    aclError aclop_Arange(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        const aclScalar* step = nullptr;
        const aclScalar* start = args[0];
        const aclScalar* stop = args[1];
        if (args.size() >= 3 ) {
            step = args.at(2);
        } else if (HasScalarKwarg(kwargs, "step")) {
            step = kwargs.at("step");
        } else {
            // TODO: calc step? or aclop accept nullptr for step?
        }
        return aclIrregularOpRun(aclnnArangeGetWorkspaceSize, aclnnArange, stream,
            start, stop, step, outs[0]);
    }

    // cupy conforms to numpy's API: linspace(start, stop, num=50, endpoint=True, retstep=False)
    // aclnn does not have such op, so use arange to mimic
    aclError aclop_Linspace(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (args.size() < 3 ) {
            std::cout << "ASCEND Error: linspace must have 3 arg, start, stop, count\n";
            return ACL_ERROR_INVALID_PARAM;
        }
        double dstart = GetScalarArg<double>(args, 0, kwargs, "start", 0);
        double dstop = GetScalarArg<double>(args, 1, kwargs, "stop", 0);
        double dcount = GetScalarArg<double>(args, 1, kwargs, "stop", 0);
        double dstep = (dstop - dstart ) / dcount;
        const aclScalar* step = nullptr;  // TODO: create scalar of start same type?
        const aclScalar* start = args[0];
        const aclScalar* stop = args[1];

        return aclIrregularOpRun(aclnnArangeGetWorkspaceSize, aclnnArange, stream,
            start, stop, step, outs[0]);
    }

    // numpy using `kind` to specify method, always in ascending order, `order` for sort objects
    // cupy_sort: support only stable, as cupy does not support string scalar as arg
    // ArrayAPI standard: not yet checked, TODO
    aclError aclop_Sort(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        const aclTensor* self = ins[0];
        int64_t axis = GetScalarArg<int64_t>(args, 0, kwargs, "dim", -1); // -1 means last axis
        bool stable = GetScalarArg<bool>(args, 1, kwargs, "stable", true);
        bool descending = GetScalarArg<bool>(args, 1, kwargs, "order", false);
        aclTensor* indices = nullptr;  // alcop sort can accept nullptr for indexOut
        if (outs.size() > 1) {
            indices = outs[1];  // int64 tensor
        }
        return aclIrregularOpRun(aclnnSortGetWorkspaceSize, aclnnSort, stream,
            self, stable, axis, descending, outs[1], indices); // value and index out arrays
    }

    // This is a general function, must be launched differently, keyward args?
    aclError aclop_Round(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (args.size() && ins.size()) {
            const aclTensor* self = ins[0];
            aclTensor* out = outs[0];
            int64_t decimals = ToScalarArg<int64_t>(args[0]); // will arithmetic scalar do static_cast?
            return aclIrregularOpRun(aclnnRoundDecimalsGetWorkspaceSize, aclnnRoundDecimals, stream,
                self, decimals, out);
        } else {
            std::cout << "ASCEND Error: Round() take a tensor and a int as input parameters\n";
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    // cupy_clip -> aclnnClamp  'ddd->d'
    aclError aclop_Clamp(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        const aclTensor* self = ins[0];
        aclTensor* out = outs[0];
        if (args.size() >=2) {
            const aclScalar* amin = args[0];
            const aclScalar* amax = args[1];
            return aclTernaryOpRun(self, amin, amax, out,
                aclnnClampGetWorkspaceSize, aclnnClamp, stream, false);
        } else {
            std::cout << "ASCEND: cupy/numpy support both amax and amin can be array/tensor, yet impl \n";
        }
    }

    // Remainder has TT, ST, TS , inplace version, aclnnRemainderTensorScalar&aclnnInplaceRemainderTensorScalar
    aclError aclop_Divmod(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        const aclTensor* self = ins[0];
        int mode = 2; // TODO numpy mode -> aclop mode
        // 0-对应None：默认不执行舍入。
        // 1-对应trunc：将除法的小数部分舍入为零。
        // 2-对应floor：向下舍入除法的结果。
        auto ret = aclIrregularOpRun(aclnnDivModGetWorkspaceSize, aclnnDivMod, stream,
            self, ins[0], mode, outs[0]);
        ret = aclIrregularOpRun(aclnnRemainderTensorTensorGetWorkspaceSize, aclnnRemainderTensorTensor, stream,
            self, ins[0], outs[1]);
        return ret;
    }

    // this aclnn api perfectly match cupy's, while `cupy_is_close`, `cupy_is_close_complex`
    // TODO: dtype
    aclError aclop_IsClose(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        const aclTensor* self = ins[0];
        double atol = GetScalarArg<double>(args, 0, kwargs, "rtol", 1e-5); 
        double rtol = GetScalarArg<double>(args, 1, kwargs, "atol", 1e-8);
        bool equal_nan = GetScalarArg<bool>(args, 1, kwargs, "order", false);
        aclTensor* indices = nullptr;  // alcop sort can accept nullptr for indexOut
        if (outs.size() > 1) {
            indices = outs[1];  // int64 tensor
        }
        return aclIrregularOpRun(aclnnIsCloseGetWorkspaceSize, aclnnIsClose, stream,
            self, ins[1], rtol, atol, equal_nan, outs[0]); // value and index out arrays
    }

    // `cupy_copy` register it as ufunc,  numpy has extra order=K args
    aclError aclop_Copy(const aclTensor* src, aclTensor* out, aclrtStream stream) {
        return aclIrregularOpRun(aclnnInplaceCopyGetWorkspaceSize, aclnnInplaceCopy, stream,
            out, src);
    }
    // `argwhere` find nonzero index, similar as `nonzero`
    aclError aclop_Nonzero(const aclTensor* self, aclTensor* out, aclrtStream stream) {
        return aclIrregularOpRun(aclnnNonzeroGetWorkspaceSize, aclnnNonzero, stream,
            self, out);
    }
    

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif // header