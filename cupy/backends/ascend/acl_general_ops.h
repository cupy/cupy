#ifndef CUPY_ACL_GENERAL_OPS_HEADER
#define CUPY_ACL_GENERAL_OPS_HEADER

// creation op:  with dim info
// arange, eye, diag, linspace (no such) 
// ones(), zeros() are done by fill(), so does not need to call kernel
#include "aclnnop/aclnn_arange.h"
#include "aclnnop/aclnn_eye.h"  //  np.eye == np.identity(N)
#include "aclnnop/aclnn_diag.h"  // UnaryScalarOp   not sure TODO
#include "aclnnop/aclnn_trace.h" // UnaryOp

// linalg: qr / svd / inverse (trace is a UnaryOp, declared above)
#include "aclnnop/aclnn_qr.h"
#include "aclnnop/aclnn_svd.h"
#include "aclnnop/aclnn_inverse.h"

// triangular part
#include "aclnnop/aclnn_tril.h"
#include "aclnnop/aclnn_triu.h"

// history: ptp -> aminmax (min/max pair)
#include "aclnnop/aclnn_aminmax.h"
#include "aclnnop/aclnn_aminmax_all.h"
#include "aclnnop/aclnn_aminmax_dim.h"
#include "aclnnop/aclnn_histc.h"

// complex construction: complex(real, imag)
#include "aclnnop/aclnn_complex.h"

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
#include "aclnnop/aclnn_roll.h"
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
            AclTensorListGuard tl(ins);
            // numpy.stack() defaults to axis=0 -> a real default, keep it
            int64_t dim = GetScalarArg<int64_t>(args, 0, kwargs, "dim", 0);
            return aclIrregularOpRun(aclnnStackGetWorkspaceSize, aclnnStack, stream,
                tl.get(), dim, outs[0]);
        } else {
            std::cout << "Error:" <<  __FUNCTION__  << " take args: tensorList, axis, out) \n";
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    aclError aclop_Concat(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.size() >= 1) {
            AclTensorListGuard tl(ins);
            int64_t dim = GetScalarArg<int64_t>(args, 0, kwargs, "dim", 0);
            return aclIrregularOpRun(aclnnCatGetWorkspaceSize, aclnnCat, stream,
                tl.get(), dim, outs[0]);
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

    // numpy.permute(x, dims) -> aclnnPermute(self, dims, out)
    aclError aclop_Permute(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.empty() || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        // `dims` is passed either as an aclIntArray argument or as a python
        // sequence stored positionally in `args`.
        aclIntArray* dims = nullptr;
        if (!args.empty() && op::IsBasicType(args.back()->GetDataType())) {
            // unlikely path: a single scalar axis
            int64_t axis = ToScalarArg<int64_t>(args.back());
            dims = aclCreateIntArray(&axis, 1);
        }
        aclError ret = aclIrregularOpRun(aclnnPermuteGetWorkspaceSize, aclnnPermute, stream,
            ins[0], dims, outs[0]);
        if (dims != nullptr) {
            aclDestroyIntArray(dims);
        }
        return ret;
    }

    // numpy.roll(x, shift, axis) -> aclnnRoll(x, shifts, dims, out)
    aclError aclop_Roll(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.empty() || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        const aclTensor* self = ins[0];
        int64_t shift = GetScalarArg<int64_t>(args, 0, kwargs, "shift", 0);
        int64_t axis = GetScalarArg<int64_t>(args, 1, kwargs, "axis", 0);
        aclIntArray* shifts = aclCreateIntArray(&shift, 1);
        aclIntArray* dims = aclCreateIntArray(&axis, 1);
        aclError ret = aclIrregularOpRun(aclnnRollGetWorkspaceSize, aclnnRoll, stream,
            self, shifts, dims, outs[0]);
        aclDestroyIntArray(shifts);
        aclDestroyIntArray(dims);
        return ret;
    }

    // numpy has op resize, but diff from the scaling
    // aclnnResizeGetWorkspaceSize(const aclTensor* self, const aclFloatArray* scales, const char* mode, aclTensor* out,

    // aclnnFlattenGetWorkspaceSize(const aclTensor* self, int64_t axis, aclTensor* out,
    aclError aclop_Flatten(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (outs.size() == 1) {
            // ndarray.flatten()/ravel() flatten from axis 0
            int64_t axis = GetScalarArg<int64_t>(args, 0, kwargs, "axis", 0);
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
            // aclnnArange treats steop as the exclusive bound (half open), matching numpy, otherwise the last elem uninitialized
            double dstart = GetScalarArg<double>(args, 0, kwargs, "start", 0.0);
            double dstep = GetScalarArg<double>(args, 1, kwargs, "step", 1.0);
            const aclScalar* stop = CreateAclScalar(dstart + dstep * numel, dtype);
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
        int64_t axis = GetScalarArg<int64_t>(args, 0, kwargs, "axis", -1); // -1 means last axis
        bool stable = GetScalarArg<bool>(args, 1, kwargs, "stable", true);
        bool descending = GetScalarArg<bool>(args, 2, kwargs, "descending", false);
        // aclnnSort 的 indexOut 允许传 nullptr：只排序取值时不需要索引。
        // 注意必须由调用方提供 indexOut（outs[1]），因为这里临时造出来的
        // tensor 在 aclIrregularOpRun 内部就被销毁了。
        aclTensor* indices = nullptr;
        if (outs.size() > 1) {
            indices = outs[1];
        }
        return aclIrregularOpRun(aclnnSortGetWorkspaceSize, aclnnSort, stream,
            self, stable, axis, descending, outs[0], indices); // value and index out arrays
    }

    aclError aclop_Argsort(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        const aclTensor* self = ins[0];
        int64_t dim = GetScalarArg<int64_t>(args, 0, kwargs, "dim", -1); // -1 means last axis
        bool descending = GetScalarArg<bool>(args, 1, kwargs, "descending", false);
        if (outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        aclTensor* indices = outs[0];  // int64 tensor output
        return aclIrregularOpRun(aclnnArgsortGetWorkspaceSize, aclnnArgsort, stream,
            self, dim, descending, indices); // index out arrays
    }

    // This is a general function, aclnnRoundDecimals() still round to 0 decimal, why?
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

    // numpy.nan_to_num(x, nan=0.0, posinf=None, neginf=None)
    // aclnnNanToNum(self, float nan, float posinf, float neginf, out)
    aclError aclop_NanToNum(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        const aclTensor* self = ins[0];
        aclTensor* out = outs[0];
        // defaults are the NumPy ones: nan=0, posinf=FLT_MAX, neginf=-FLT_MAX
        float nan = GetScalarArg<float>(args, 0, kwargs, "nan", 0.0f);
        float posinf = GetScalarArg<float>(args, 1, kwargs, "posinf", 0.0f);
        float neginf = GetScalarArg<float>(args, 2, kwargs, "neginf", 0.0f);
        if (posinf == 0.0f) {
            posinf = std::numeric_limits<float>::max();
        }
        if (neginf == 0.0f) {
            neginf = std::numeric_limits<float>::lowest();
        }
        return aclIrregularOpRun(aclnnNanToNumGetWorkspaceSize, aclnnNanToNum, stream,
            self, nan, posinf, neginf, out);
    }

    // ufunc: cupy_clip -> aclnnClamp  'ddd->d'
    aclError aclop_Clamp(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        const aclTensor* self = ins[0];
        aclTensor* out = outs[0];
        if (args.size() >= 2) {
            const aclScalar* amin = args[0];
            const aclScalar* amax = args[1];
            return aclTernaryOpRun(self, amin, amax, out,
                aclnnClampGetWorkspaceSize, aclnnClamp, stream, false);
        } else if (ins.size() >= 3) {
            std::cout << "ASCEND: cupy/numpy support both amax and amin can be array/tensor, yet impl \n";
            return ACL_ERROR_INVALID_PARAM;
        } else {
            PrintArgs(__func__, args, kwargs, std::cout);
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
    // cupy_copy / elementwise_copy: `out = src`.
    // `aclnnInplaceCopy` requires both tensors to share a dtype, so when the
    // dtypes differ (e.g. `ndarray.astype`) fall back to `aclnnCast`.
    aclError aclop_Copy(const aclTensor* src, aclTensor* out, aclrtStream stream) {
        aclDataType src_dtype, out_dtype;
        aclGetDataType(src, &src_dtype);
        aclGetDataType(out, &out_dtype);
        if (src_dtype != out_dtype) {
            return aclIrregularOpRun(aclnnCastGetWorkspaceSize, aclnnCast, stream,
                src, out_dtype, out);
        }
        return aclIrregularOpRun(aclnnInplaceCopyGetWorkspaceSize, aclnnInplaceCopy, stream,
            out, src);
    }
    // `argwhere` find nonzero index, similar as `nonzero`
    aclError aclop_Nonzero(const aclTensor* self, aclTensor* out, aclrtStream stream) {
        return aclIrregularOpRun(aclnnNonzeroGetWorkspaceSize, aclnnNonzero, stream,
            self, out);
    }

    // numpy.unique(ar, return_index, return_inverse, return_counts)
    //   -> aclnnUnique2(self, sorted, returnInverse, returnCounts, valueOut, inverseOut, countsOut)
    // A `nullptr` output slot means "not requested"; at least `valueOut` is required.
    aclError aclop_Unique2(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.empty() || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        bool sorted = GetScalarArg<bool>(args, 0, kwargs, "sorted", true);
        aclTensor* values = outs[0];
        aclTensor* inverse = (outs.size() > 1) ? outs[1] : nullptr;
        aclTensor* counts = (outs.size() > 2) ? outs[2] : nullptr;
        bool returnInverse = (inverse != nullptr);
        bool returnCounts = (counts != nullptr);
        return aclIrregularOpRun(aclnnUnique2GetWorkspaceSize, aclnnUnique2, stream,
            ins[0], sorted, returnInverse, returnCounts, values, inverse, counts);
    }

    // ------------------------------------------------------------------
    // linalg: trace / tril / triu / qr / svd / inverse
    // ------------------------------------------------------------------

    // numpy.trace(a, offset=0) -> aclnnTrace(self, out)  (sum of diagonal)
    aclError aclop_Trace(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.empty() || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        // NB: aclnnTrace has no `offset`; NumPy's nonzero `offset` is not covered.
        return aclIrregularOpRun(aclnnTraceGetWorkspaceSize, aclnnTrace, stream,
            ins[0], outs[0]);
    }

    // numpy.tril(m, k=0) -> aclnnTril(self, diagonal, out)
    aclError aclop_Tril(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.empty() || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        int64_t diagonal = GetScalarArg<int64_t>(args, 0, kwargs, "k", 0);
        return aclIrregularOpRun(aclnnTrilGetWorkspaceSize, aclnnTril, stream,
            ins[0], diagonal, outs[0]);
    }

    // numpy.triu(m, k=0) -> aclnnTriu(self, diagonal, out)
    aclError aclop_Triu(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.empty() || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        int64_t diagonal = GetScalarArg<int64_t>(args, 0, kwargs, "k", 0);
        return aclIrregularOpRun(aclnnTriuGetWorkspaceSize, aclnnTriu, stream,
            ins[0], diagonal, outs[0]);
    }

    // numpy.linalg.qr(a, mode='reduced') -> aclnnQr(self, some, Q, R)
    // `some=true` is the reduced (default) mode; `some=false` is complete.
    aclError aclop_Qr(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.empty() || outs.size() < 2) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        bool some = GetScalarArg<bool>(args, 0, kwargs, "some", true);
        return aclIrregularOpRun(aclnnQrGetWorkspaceSize, aclnnQr, stream,
            ins[0], some, outs[0], outs[1]);
    }

    // numpy.linalg.svd(a, full_matrices=True) -> aclnnSvd(input, fullMatrices, computeUV, sigma, u, v)
    // When `computeUV` is false only `sigma` is produced (numpy.linalg.svdvals).
    aclError aclop_Svd(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.empty() || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        bool full_matrices = GetScalarArg<bool>(args, 0, kwargs, "full_matrices", true);
        aclTensor* sigma = outs[0];
        aclTensor* u = (outs.size() > 1) ? outs[1] : nullptr;
        aclTensor* v = (outs.size() > 2) ? outs[2] : nullptr;
        bool computeUV = (u != nullptr && v != nullptr);
        return aclIrregularOpRun(aclnnSvdGetWorkspaceSize, aclnnSvd, stream,
            ins[0], full_matrices, computeUV, sigma, u, v);
    }

    // numpy.linalg.inv(a) -> aclnnInverse(self, out)
    aclError aclop_Inverse(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.empty() || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        return aclIrregularOpRun(aclnnInverseGetWorkspaceSize, aclnnInverse, stream,
            ins[0], outs[0]);
    }

    // ------------------------------------------------------------------
    // statistics: ptp -> aminmax, and histogram -> histc
    // ------------------------------------------------------------------

    // numpy.ptp(a) == max - min; exported through aclnnAminmax.
    // `outs` = [minOut, maxOut] when both are wanted, else [minOut, maxOut]
    // with only the requested side consumed by the caller.
    aclError aclop_Aminmax(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.empty() || outs.size() < 2) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        bool keepdim = GetScalarArg<bool>(args, 1, kwargs, "keepdim", false);
        // `dim` may be given as a scalar axis or a sequence.
        if (!args.empty() && args[0] != nullptr && op::IsBasicType(args[0]->GetDataType())) {
            int64_t dim = ToScalarArg<int64_t>(args[0]);
            return aclIrregularOpRun(aclnnAminmaxDimGetWorkspaceSize, aclnnAminmaxDim, stream,
                ins[0], dim, keepdim, outs[0], outs[1]);
        }
        return aclIrregularOpRun(aclnnAminmaxAllGetWorkspaceSize, aclnnAminmaxAll, stream,
            ins[0], outs[0], outs[1]);
    }

    // numpy.histogram(a, bins, range) -> aclnnHistc(self, bins, min, max, out)
    aclError aclop_Histc(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.empty() || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        int64_t bins = GetScalarArg<int64_t>(args, 0, kwargs, "bins", 10);
        aclDataType dtype;
        aclGetDataType(ins[0], &dtype);
        double dmin = GetScalarArg<double>(args, 1, kwargs, "min", 0.0);
        double dmax = GetScalarArg<double>(args, 2, kwargs, "max", 0.0);
        const aclScalar* min = CreateAclScalar(dmin, dtype);
        const aclScalar* max = CreateAclScalar(dmax, dtype);
        return aclIrregularOpRun(aclnnHistcGetWorkspaceSize, aclnnHistc, stream,
            ins[0], bins, min, max, outs[0]);
    }

    // numpy.complex()/asarray from real+imag -> aclnnComplex(real, imag, out)
    aclError aclop_Complex(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.size() < 2 || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        return aclIrregularOpRun(aclnnComplexGetWorkspaceSize, aclnnComplex, stream,
            ins[0], ins[1], outs[0]);
    }

    // choose
    // numpy.take_along_axis(arr, indices, axis=-1)

    // ElementwiseKernel('raw T a, S indices, uint32 ldim, uint32 cdim, uint32 rdim, int64 index_range', 'T out'
    // axis=None, out=None, mode='raise',  there is _take_scalar_kernel, will not be supported
    aclError aclop_Take(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        const aclTensor* self = ins[0];
        if (ins.size() == 2) {
            return aclIrregularOpRun(aclnnTakeGetWorkspaceSize, aclnnTake, stream,
                self, ins[1], outs[0]);
        } else {
            std::cout << "Error:" <<  __FUNCTION__  << " take 3 input tensors take(self, index, out) \n";
            return ACL_ERROR_INVALID_PARAM;
        }
    }
    // aclnnTakeGetWorkspaceSize(const aclTensor* self, const aclTensor* index, aclTensor* out, ...);

    // numpy.put(a, ind, v, mode='raise'), while ACLOP has accumulate arg
    // cupy support ('raise', 'wrap', 'clip') mode, by diff kernel
    // cdef _put_raise_kernel = ElementwiseKernel('S ind, raw T vals, int64 n_vals, int64 n',
    aclError aclop_Put(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        bool accumulate = false;
        if (ins.size() + outs.size() == 3) {
            aclTensor* self = outs[0];
            return aclIrregularOpRun(aclnnInplacePutGetWorkspaceSize, aclnnInplacePut, stream,
                self, ins[0], ins[1], accumulate);
        } else {
            std::cout << "Error:" <<  __FUNCTION__  << " put 3 input tensors put(self, index, value) \n";
            return ACL_ERROR_INVALID_PARAM;
        }
    }

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