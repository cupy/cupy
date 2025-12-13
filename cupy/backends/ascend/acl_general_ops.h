#ifndef CUPY_ACL_GENERAL_OPS_HEADER
#define CUPY_ACL_GENERAL_OPS_HEADER

// creation op:  with dim info
// arange, eye, diag, linspace (no such) 
// ones(), zeros() are done by fill(), so does not need to call kernel
#include "aclnnop/aclnn_arange.h"
#include "aclnnop/aclnn_eye.h"  //  np.eye == np.identity(N)
#include "aclnnop/aclnn_diag.h"  // UnaryScalarOp   not sure TODO
#include "aclnnop/aclnn_trace.h" // UnaryOp

// math ops, but it is irregular ops
#include <aclnnop/aclnn_round.h>
#include <aclnnop/aclnn_isclose.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_nonzero.h>
#include <aclnnop/aclnn_heaviside.h>

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
#include "aclnnop/aclnn_argsort.h"

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

    // aclnnEyeGetWorkspaceSize(int64_t n, int64_t m, aclTensor* out,
    // _creation.basic.py eye() use ndarray_base.diagnal() ->  _indexing._ndarray_diagonal -> _diagnal
    // no kernel is needed, but _transpose() used

    // TODO: fix, rint(), around,
    // aclnnTraceGetWorkspaceSize(const aclTensor* self, aclTensor* out     
    // aclnnTrilGetWorkspaceSize(const aclTensor* self, int64_t diagonal, aclTensor* out,  // set upper as zeros
    // aclnnPermuteGetWorkspaceSize(const aclTensor* self, const aclIntArray* dims, aclTensor* out,


    // frexp() Decompose the elements of x into mantissa and twos exponent,  no such in aclnn ??
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
    // numpy.stack()
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
            PrintArgs(__func__, args, kwargs, std::cout);
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
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    // astype():  casting UnaryOp with dtype
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
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    aclError aclop_Heaviside(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        aclTensor* self = outs[0];
        if (args.size()) {
            std::cout << "ASCEND: scaler version is yet impl for heaviside op\n";
            return ACL_ERROR_INVALID_PARAM;
        } else if (ins.size() >= 1) {
            // return aclBinaryOpRun(self, ins[0], outs[0],
            //     aclnnHeavisideGetWorkspaceSize, aclnnHeaviside, stream, false);
            return ACL_ERROR_INVALID_PARAM;  // TODO: need link a new so file?
        } else {
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    // numpy.arange(numpy.arange([start = 0, ]stop, [step = , ]) 
    // `cupy_arange` kernel takes only start and step as input parameter, num is output tensor's elem_count
    // cupy.arange() python code will deal with output tensor creation, so `stop` value can be inferred from elem count
    aclError aclop_Arange(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        const aclScalar* step = nullptr;
        const aclScalar* start = nullptr;
        aclDataType dtype;
        aclGetDataType(outs[0], &dtype);
        auto numel = GetAclTensorElementCount(outs[0]);
        PrintArgs(__func__, args, kwargs, std::cout);
        if (args.size() >= 2) {
            start = args[0];
            step = args[1];
            // stop can keep it as nullptr? or must have the same dtype as output tensor?
            double dstart = GetScalarArg<double>(args, 0, kwargs, "start", 0.0);
            double dstep = GetScalarArg<double>(args, 1, kwargs, "step", 1.0);
            const aclScalar* stop = CreateAclScalar(dstart + dstep * (numel - 1), dtype);
            return aclIrregularOpRun(aclnnArangeGetWorkspaceSize, aclnnArange, stream,
                start, stop, step, outs[0]);
        } else {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    // cupy conforms to numpy's API: linspace(start, stop, num=50, endpoint=True, retstep=False)
    // numpy/Array API standard
    // cupy_linspace kernel has 2 variants, one is actually the same as cupy_arange()
    // aclnnLinspaceGetWorkspaceSize(const aclScalar* start, const aclScalar* end, int64_t steps, aclTensor* out,
    aclError aclop_Linspace(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (args.size() < 2 ) {
            std::cout << "ASCEND Error: linspace must have 3 arg, start, stop, count\n";
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        } else {
            aclop_Arange(ins, outs, args, kwargs, stream);
        }

        // double dstart = GetScalarArg<double>(args, 0, kwargs, "start", 0);
        // double dstop = GetScalarArg<double>(args, 1, kwargs, "stop", 0);
        // double dcount = GetScalarArg<double>(args, 1, kwargs, "stop", 0);
        // double dstep = (dstop - dstart ) / dcount;
        // const aclScalar* step = nullptr;  // TODO: create scalar of start same type?
        // const aclScalar* start = args[0];
        // const aclScalar* stop = args[1];

        // return aclIrregularOpRun(aclnnArangeGetWorkspaceSize, aclnnArange, stream,
        //     start, stop, step, outs[0]);
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
        PrintArgs(__func__, args, kwargs, std::cout);
        return aclIrregularOpRun(aclnnSortGetWorkspaceSize, aclnnSort, stream,
            self, stable, axis, descending, outs[0], indices); // value and index out arrays
    }

    aclError aclop_Argsort(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        const aclTensor* self = ins[0];
        int64_t dim = GetScalarArg<int64_t>(args, 0, kwargs, "dim", -1); // -1 means last axis
        bool descending = GetScalarArg<bool>(args, 1, kwargs, "order", false);
        aclTensor* indices = nullptr;  // alcop sort can accept nullptr for indexOut
        if (outs.size() >= 1) {
            indices = outs[0];  // int64 tensor output
        } else {

        }
        return aclIrregularOpRun(aclnnArgsortGetWorkspaceSize, aclnnArgsort, stream,
            self, dim, descending, indices); // index out arrays
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
            PrintArgs(__func__, args, kwargs, std::cout);
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
            PrintArgs(__func__, args, kwargs, std::cout);
            std::cout << "ASCEND: cupy/numpy support both amax and amin can be array/tensor, yet impl \n";
            return ACL_ERROR_INVALID_PARAM;
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
    // TODO: dtype check
    aclError aclop_IsClose(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        const aclTensor* self = ins[0];
        double atol = GetScalarArg<double>(args, 0, kwargs, "rtol", 1e-5); 
        double rtol = GetScalarArg<double>(args, 1, kwargs, "atol", 1e-8);
        bool equal_nan = GetScalarArg<bool>(args, 1, kwargs, "order", false);
        aclTensor* indices = nullptr;
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

    // choose
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


    // random.normal(loc=0.0, scale=1.0, size=None), normal distribution
    aclError aclop_Normal(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        // return aclIrregularOpRun(aclnnInplaceNormalGetWorkspaceSize, 
            // const aclTensor* selfRef, float mean, float std, int64_t seed,
            //                                              int64_t offset, uint64_t* workspaceSize,
            //                                              aclOpExecutor** executor);
    }
    
    // random.rand() uniform distribution
    aclError aclop_Uniform(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
    }

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif // header