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
#include <aclnnop/aclnn_einsum.h>  // ascend_einsum: ARG_STRING 的第一个真实消费者
#include <aclnnop/aclnn_gather.h>   // scatter_max/min 的组合实现用
#include <aclnnop/aclnn_maximum.h>  // scatter_max 的组合实现用
#include <aclnnop/aclnn_minimum.h>  // scatter_min 的组合实现用
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
#include "aclnnop/aclnn_add.h"   // aclnnInplaceAdd (masked scatter accumulate)

// indexing: argsort, unique, unique2, sort
// no count() , unique(), unique2() op
#include "aclnnop/aclnn_unique2.h"
#include "aclnnop/aclnn_index.h"
#include "aclnnop/aclnn_sort.h"
#include "aclnnop/aclnn_argsort.h"
// prefix scan (cupy.cumsum / cupy.cumprod and the boolean-index prefix sums)
#include "aclnnop/aclnn_cumsum.h"
#include "aclnnop/aclnn_cumprod.h"
// setitem / boolean indexing: a[idx] = v, a[idx] += v, a[mask] = v, a[mask]
#include "aclnnop/aclnn_scatter_update.h"
#include "aclnnop/aclnn_scatter_add.h"
#include "aclnnop/aclnn_masked_scatter.h"
#include "aclnnop/aclnn_masked_select.h"
// numpy.where(cond, x, y)
#include "aclnnop/aclnn_s_where.h"
// numpy.searchsorted(sortedSequence, self, ...)
#include "aclnnop/aclnn_searchsorted.h"
// cupy.bincount (ElementwiseKernel cupy_bincount_kernel -> aclop_Bincount)
#include "aclnnop/aclnn_bincount.h"

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

#include <algorithm>
#include <sstream>
#include <string>

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
    // Fix (review D8): `dims` used to be hardcoded to nullptr, so a single-axis
    // `flip(a, 0)` flipped *every* axis. `dims == nullptr` is reserved for the
    // `axis=None` case, which is what NumPy means by "flip all axes".
    //
    // M3 (docs/ascend/arg_passing_plan.md): `axis` 现在既可以是一个 int，也可以
    // 是 int 序列 —— 统一参数通道（TryGetInt64List + AclIntArrayGuard）让
    // `flip(a, (0, 2))` 这类 multi-axis 调用终于可达（以前在 pyx 侧就被拒绝）。
    aclError aclop_Flip(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.empty() || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        std::vector<int64_t> axis;
        if (!TryGetInt64List(args, 0, kwargs, "axis", &axis)) {
            // axis=None -> dims == nullptr -> flip all axes（NumPy 语义）
            axis.clear();
        }
        AclIntArrayGuard dims(axis);
        return aclIrregularOpRun(aclnnFlipGetWorkspaceSize, aclnnFlip, stream,
            ins[0], dims.get(), outs[0]);
    }

    // numpy.permute(x, dims) -> aclnnPermute(self, dims, out)
    aclError aclop_Permute(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.empty() || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        // `dims` 既可以是 int 序列（numpy.permute），也可以是单个 int
        std::vector<int64_t> dims_values;
        if (!TryGetInt64List(args, 0, kwargs, "dims", &dims_values)) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        AclIntArrayGuard dims(dims_values);
        return aclIrregularOpRun(aclnnPermuteGetWorkspaceSize, aclnnPermute, stream,
            ins[0], dims.get(), outs[0]);
    }

    // numpy.roll(x, shift, axis) -> aclnnRoll(x, shifts, dims, out)
    // Fix (review D8): `axis=None` means "flatten, roll, reshape back", which is
    // what aclnn expresses with `dims == nullptr` — it used to be turned into
    // `axis=0` and therefore rolled along the wrong axis.
    // M3: `shift` / `axis` 支持 int 序列（`roll(a, (1, 2), axis=(0, 1))`），
    // 两侧长度由 aclnn 自己校验（numpy 是广播，长度不一致时这里交给 aclnn 报错）。
    aclError aclop_Roll(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.empty() || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        std::vector<int64_t> shifts_vec;
        if (!TryGetInt64List(args, 0, kwargs, "shift", &shifts_vec)) {
            shifts_vec.push_back(0);
        }
        std::vector<int64_t> dims_vec;
        // axis=None -> dims == nullptr（先展平再 roll）
        TryGetInt64List(args, 1, kwargs, "axis", &dims_vec);
        AclIntArrayGuard shifts(shifts_vec);
        AclIntArrayGuard dims(dims_vec);
        return aclIrregularOpRun(aclnnRollGetWorkspaceSize, aclnnRoll, stream,
            ins[0], shifts.get(), dims.get(), outs[0]);
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
        if (args.size() && args[0].scalar != nullptr) {
            return aclInplaceBinaryOpRun(self, args[0].scalar,
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
            start = args[0].scalar;
            step = args[1].scalar;
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
            return aclop_Arange(ins, outs, args, kwargs, stream);
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
        if (args.size() && args[0].scalar != nullptr && ins.size()) {
            const aclTensor* self = ins[0];
            aclTensor* out = outs[0];
            int64_t decimals = ToScalarArg<int64_t>(args[0].scalar); // will arithmetic scalar do static_cast?
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
            const aclScalar* amin = args[0].scalar;
            const aclScalar* amax = args[1].scalar;
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

    // cupy 的 ufunc 是 `_is_close(a, b, rtol, atol, equal_nan)`（nin=5，
    // cupy/_logic/comparison.py:132），所以标量操作数按序落在 args[0..2]。
    // aclnnIsClose 的签名是 (self, other, rtol, atol, equal_nan, out)。
    // NOTE: CANN 头文件里 rtol/atol 的中文描述写反了（rtol 标成「绝对宽容」），
    // 这里按参数名对齐 numpy 语义（rtol=相对、atol=绝对）。
    // TODO: dtype check（CANN 白名单：self/other 整型+浮点+bool，out 只收 BOOL）
    aclError aclop_IsClose(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        const aclTensor* self = ins[0];
        double rtol = GetScalarArg<double>(args, 0, kwargs, "rtol", 1e-5);
        double atol = GetScalarArg<double>(args, 1, kwargs, "atol", 1e-8);
        bool equal_nan = GetScalarArg<bool>(args, 2, kwargs, "equal_nan", false);
        return aclIrregularOpRun(aclnnIsCloseGetWorkspaceSize, aclnnIsClose, stream,
            self, ins[1], rtol, atol, equal_nan, outs[0]);
    }

    // cupy.linalg.einsum(subscripts, *operands) 的 Ascend 原生路径：
    //   ins  = operands（>=1 个张量，统一参数通道之外的操作数照旧走 intensors）
    //   args = [subscripts] —— 统一参数通道 ARG_STRING 的**第一个真实消费者**
    //          （arg_passing_plan.md A3 / §2.3 字符串策略）
    //   outs = [out]（shape/dtype 由 python 侧 host 推导后分配，见
    //          cupy/linalg/_einsum.py 的 Ascend 快速路径）
    // NOTE: CANN 白名单只有 FLOAT16/FLOAT/INT16/UINT16/INT32/UINT32/INT64/UINT64
    //（没有 DOUBLE/INT8/BOOL），dtype 检查与回退在 python 侧做。
    aclError aclop_Einsum(const std::vector<const aclTensor*>& ins,
                          const std::vector<aclTensor*>& outs,
                          const ArgsType& args, const KwargsType& kwargs,
                          aclrtStream stream) {
        if (ins.empty() || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        const char* equation = GetStringArg(args, 0, kwargs, "subscripts");
        if (equation == nullptr || equation[0] == '\0') {
            std::cerr << "ERROR: aclop_Einsum: subscripts (ARG_STRING) missing\n";
            return ACL_ERROR_INVALID_PARAM;
        }
        AclTensorListGuard tensors(ins);
        if (!tensors) {
            return ACL_ERROR_INVALID_PARAM;
        }
        return aclIrregularOpRun(aclnnEinsumGetWorkspaceSize, aclnnEinsum, stream,
            tensors.get(), equation, outs[0]);
    }

    // `cupy_copy` register it as ufunc,  numpy has extra order=K args
    // cupy_copy / elementwise_copy: `out = src`.
    // NPU 实测 `aclnnInplaceCopy` 不可靠（同 dtype 拷贝也会失败），统一改用
    // `aclnnCast`（同 dtype 时等价于纯拷贝，异 dtype 时完成转换）。
    // src/out 任一 dtype 元数据未定义（ACL_DT_UNDEFINED）时直接拒绝，
    // 否则 aclnnCast 第一段接口会以 EL0003 Invalid_Argument 深层报错。
    aclError aclop_Copy(const aclTensor* src, aclTensor* out, aclrtStream stream) {
        aclDataType src_dtype, out_dtype;
        aclGetDataType(src, &src_dtype);
        aclGetDataType(out, &out_dtype);
        if (src_dtype == ACL_DT_UNDEFINED || out_dtype == ACL_DT_UNDEFINED) {
            std::cout << "Error:" << __FUNCTION__
                      << " src/out dtype must be defined (src=" << src_dtype
                      << ", out=" << out_dtype << ")" << std::endl;
            return ACL_ERROR_INVALID_PARAM;
        }
        return aclIrregularOpRun(aclnnCastGetWorkspaceSize, aclnnCast, stream,
            src, out_dtype, out);
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
        // `dim` may be given as a scalar axis or (via the unified arg channel) a
        // sequence; aclnnAminmaxDim itself only takes a single int64_t, so a
        // sequence is reduced to its first axis here (documented in the plan).
        std::vector<int64_t> dims;
        if (TryGetInt64List(args, 0, kwargs, "dim", &dims) && !dims.empty()) {
            return aclIrregularOpRun(aclnnAminmaxDimGetWorkspaceSize, aclnnAminmaxDim, stream,
                ins[0], dims[0], keepdim, outs[0], outs[1]);
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

    // numpy.put(a, ind, v, mode='raise')
    //
    // `_put_raise_kernel(indices, values, values.size, n, self, err)`
    //   ins  = [indices, values]      args = [n_vals, n]      outs = [self, err]
    //
    // NOTE: aclnnInplacePut() has no way to report an out-of-range index back to
    // the host, so the `err` output of the CuPy kernel (which drives the
    // `IndexError` of mode='raise') is intentionally left untouched. Indices out
    // of range therefore do not raise any more, matching torch.put_ semantics.
    // mode='wrap'/'clip' have no aclnn equivalent and stay unregistered.
    aclError aclop_PutRaise(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.size() < 2 || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        bool accumulate = GetScalarArg<bool>(args, 2, kwargs, "accumulate", false);
        return aclIrregularOpRun(aclnnInplacePutGetWorkspaceSize, aclnnInplacePut, stream,
            outs[0], ins[0], ins[1], accumulate);
    }

    // ------------------------------------------------------------------
    // prefix scan: cupy.cumsum / cupy.cumprod
    // (also the per-element mask prefix sum used by boolean indexing)
    // ------------------------------------------------------------------
    // Called from `cupy/_core/_routines_math.pyx:scan_core()`; args=[axis].
    // NB: `aclnnCumprod` takes `dim` as an aclScalar while `aclnnCumsum` takes
    // an int64_t, and both need the accumulation dtype explicitly.
    aclError aclop_Cumsum(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.empty() || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        int64_t dim = GetScalarArg<int64_t>(args, 0, kwargs, "dim", 0);
        aclDataType dtype = ACL_DT_UNDEFINED;
        if (aclGetDataType(outs[0], &dtype) != ACL_SUCCESS) {
            return ACL_ERROR_INVALID_PARAM;
        }
        return aclIrregularOpRun(aclnnCumsumGetWorkspaceSize, aclnnCumsum, stream,
            ins[0], dim, dtype, outs[0]);
    }

    aclError aclop_Cumprod(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.empty() || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        int64_t dim = GetScalarArg<int64_t>(args, 0, kwargs, "dim", 0);
        aclDataType dtype = ACL_DT_UNDEFINED;
        if (aclGetDataType(outs[0], &dtype) != ACL_SUCCESS) {
            return ACL_ERROR_INVALID_PARAM;
        }
        const aclScalar* dim_scalar = CreateAclScalar(dim, ACL_INT64);
        aclError ret = aclIrregularOpRun(aclnnCumprodGetWorkspaceSize, aclnnCumprod, stream,
            ins[0], dim_scalar, dtype, outs[0]);
        aclDestroyScalar(dim_scalar);
        return ret;
    }

    // ------------------------------------------------------------------
    // setitem / scatter:  a[idx] = v   (cupy_scatter_update)
    //                     a[idx] += v  (cupy_scatter_add)
    //   ins  = [values, indices]   args = [cdim, rdim, adim]   outs = [target]
    //
    // The CuPy kernel indexes `target[(li * adim + idx) * rdim + ri]` where
    // `i = ((li * cdim + ci) * rdim + ri)` runs over the broadcast shape of
    // `values`, i.e. numpy's `target[idx] = values` with `idx` starting at axis
    // `len(lshape)`. `aclnnInplaceScatterUpdate(data, indices, updates, axis)`
    // (torch `scatter_`) uses exactly the same convention.
    //
    // `axis` is not passed explicitly, but it is recoverable from
    // `values.numel() == prod(lshape) * cdim * rdim`.
    static int64_t ScatterAxis(const std::vector<const aclTensor*>& ins,
        const ArgsType& args, const KwargsType& kwargs) {
        int64_t cdim = GetScalarArg<int64_t>(args, 0, kwargs, "cdim", 0);
        int64_t rdim = GetScalarArg<int64_t>(args, 1, kwargs, "rdim", 1);
        if (cdim <= 0 || rdim <= 0) {
            throw std::invalid_argument(
                "aclop_Scatter*: cdim/rdim were not supplied, cannot derive axis");
        }
        return GetAclTensorElementCount(ins[0]) / (cdim * rdim);
    }

    aclError aclop_ScatterUpdate(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.size() < 2 || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        int64_t axis = ScatterAxis(ins, args, kwargs);
        return aclIrregularOpRun(aclnnInplaceScatterUpdateGetWorkspaceSize, aclnnInplaceScatterUpdate, stream,
            outs[0], ins[1], ins[0], axis);
    }

    aclError aclop_ScatterAdd(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.size() < 2 || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        int64_t axis = ScatterAxis(ins, args, kwargs);
        // aclnnScatterAdd has no inplace variant; `self` and `out` deliberately
        // alias so that the added result lands back in `a` (the CUDA kernel is
        // `atomicAdd(&a[...], v)`).
        return aclIrregularOpRun(aclnnScatterAddGetWorkspaceSize, aclnnScatterAdd, stream,
            outs[0], axis, ins[1], ins[0], outs[0]);
    }

    // scatter_max / scatter_min（`cupy.maximum.at` / `cupy.minimum.at`）：
    // CANN 没有原生的 scatter reduce=max/min —— aclnnScatter 的 reduce 只有
    // (add,1)/(mul,2)/(none,0)，aclnnIndexPutImpl 只有 accumulate/replace，
    // 也没有 ScatterMax/ScatterMin/ScatterElements —— 所以用三段组合：
    //     existing = aclnnGather(a, axis, index)        # 被索引位置的现值
    //     merged   = aclnnMaximum(existing, src)        # min 用 aclnnMinimum
    //     a       &= aclnnInplaceScatterUpdate(a, index, merged)
    // max/min 的合并幂等、与顺序无关，因此不需要原子性：同一位置被写多次时
    // 最终值就是最大/最小，与 CUDA atomicMax/atomicMin 语义一致。
    // 中间 buffer 用 aclTensorLike（连续新分配），析构成对释放。
    static aclError ScatterMaxMin(const std::vector<const aclTensor*>& ins,
        const std::vector<aclTensor*>& outs, const ArgsType& args,
        const KwargsType& kwargs, aclrtStream stream, bool is_max) {
        if (ins.size() < 2 || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        int64_t axis = ScatterAxis(ins, args, kwargs);
        aclDataType dtype = ACL_DT_UNDEFINED;
        aclGetDataType(outs[0], &dtype);
        aclTensor* existing = aclTensorLike(ins[0], dtype);
        aclTensor* merged = aclTensorLike(ins[0], dtype);
        if (existing == nullptr || merged == nullptr) {
            aclDestroyTensorLike(existing);
            aclDestroyTensorLike(merged);
            return ACL_ERROR_INVALID_PARAM;
        }
        aclError ret = aclIrregularOpRun(aclnnGatherGetWorkspaceSize, aclnnGather,
            stream, outs[0], axis, ins[1], existing);
        if (ret == ACL_SUCCESS) {
            if (is_max) {
                ret = aclIrregularOpRun(aclnnMaximumGetWorkspaceSize, aclnnMaximum,
                    stream, existing, ins[0], merged);
            } else {
                ret = aclIrregularOpRun(aclnnMinimumGetWorkspaceSize, aclnnMinimum,
                    stream, existing, ins[0], merged);
            }
        }
        if (ret == ACL_SUCCESS) {
            ret = aclIrregularOpRun(aclnnInplaceScatterUpdateGetWorkspaceSize,
                aclnnInplaceScatterUpdate, stream, outs[0], ins[1], merged, axis);
        }
        aclDestroyTensorLike(existing);
        aclDestroyTensorLike(merged);
        return ret;
    }

    aclError aclop_ScatterMax(const std::vector<const aclTensor*>& ins,
        const std::vector<aclTensor*>& outs, const ArgsType& args,
        const KwargsType& kwargs, aclrtStream stream) {
        return ScatterMaxMin(ins, outs, args, kwargs, stream, true);
    }

    aclError aclop_ScatterMin(const std::vector<const aclTensor*>& ins,
        const std::vector<aclTensor*>& outs, const ArgsType& args,
        const KwargsType& kwargs, aclrtStream stream) {
        return ScatterMaxMin(ins, outs, args, kwargs, stream, false);
    }

    // `_scatter_update_mask_kernel(src, mask, mask_scanned, a)` -> a[mask] = src
    // `mask_scanned` (the mask prefix sum) is not needed: aclnnMaskedScatter
    // already consumes source elements in row-major order.
    aclError aclop_ScatterUpdateMask(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.size() < 2 || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        return aclIrregularOpRun(aclnnInplaceMaskedScatterGetWorkspaceSize, aclnnInplaceMaskedScatter, stream,
            outs[0], ins[1], ins[0]);
    }

    // `_scatter_add_mask_kernel(src, mask, mask_scanned, a)` -> a[mask] += src
    // There is no masked-add in aclnn, so it is composed as
    //     tmp = 0 ; tmp[mask] = src ; a += tmp
    // (`tmp` is zero elsewhere, hence the add leaves the other entries alone).
    aclError aclop_ScatterAddMask(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.size() < 2 || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        aclTensor* self = outs[0];
        aclDataType dtype = ACL_DT_UNDEFINED;
        if (aclGetDataType(self, &dtype) != ACL_SUCCESS) {
            return ACL_ERROR_INVALID_PARAM;
        }
        aclTensor* tmp = aclTensorLike(self, dtype);
        if (tmp == nullptr) {
            return ACL_ERROR_INVALID_PARAM;
        }
        const aclScalar* zero = CreateAclScalar(0.0, dtype);
        const aclScalar* one = CreateAclScalar(1.0, dtype);
        aclError ret = aclIrregularOpRun(aclnnInplaceFillScalarGetWorkspaceSize, aclnnInplaceFillScalar,
            stream, tmp, zero);
        if (ret == ACL_SUCCESS) {
            ret = aclIrregularOpRun(aclnnInplaceMaskedScatterGetWorkspaceSize, aclnnInplaceMaskedScatter,
                stream, tmp, ins[1], ins[0]);
        }
        if (ret == ACL_SUCCESS) {
            ret = aclIrregularOpRun(aclnnInplaceAddGetWorkspaceSize, aclnnInplaceAdd,
                stream, self, tmp, one);
        }
        aclDestroyScalar(zero);
        aclDestroyScalar(one);
        // aclDestroyTensorLike also frees the device buffer that aclTensorLike
        // allocated (aclDestroyTensor alone would leak it).
        aclDestroyTensorLike(tmp);
        return ret;
    }

    // `_getitem_mask_kernel(a, mask, mask_scanned, out)` -> out = a[mask]
    // `aclnnMaskedSelect` emits the selected values in row-major order, which is
    // exactly what the CUDA kernel writes at `mask_scanned - 1`; `out` is passed
    // as a 1-D view by `_getitem_mask_single` for that reason.
    aclError aclop_GetitemMask(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.size() < 2 || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        return aclIrregularOpRun(aclnnMaskedSelectGetWorkspaceSize, aclnnMaskedSelect, stream,
            ins[0], ins[1], outs[0]);
    }

    // `cupy_searchsorted_kernel(v, a, a.size, side_is_right, assume_increasing, y)`
    //   ins = [v, a]  args = [n_bins, side_is_right, assume_increasing]  outs = [y]
    // `assume_increasing` is irrelevant here: aclnnSearchSorted always performs a
    // real binary search (NumPy's documented undefined behavior for
    // non-monotonic input is not something we need to reproduce).
    aclError aclop_SearchSorted(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.size() < 2 || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        bool right = GetScalarArg<bool>(args, 1, kwargs, "side_is_right", false);
        return aclIrregularOpRun(aclnnSearchSortedGetWorkspaceSize, aclnnSearchSorted, stream,
            ins[1], ins[0], false /* outInt32 */, right, nullptr /* sorter */, outs[0]);
    }

    // `cupy_bincount_kernel(x, b)` / `cupy_bincount_with_weight_kernel(x, w, b)`
    //   ins = [x] or [x, weights]  outs = [b]
    // cupy's Python `bincount` (histogram.py) folds `minlength` into the output
    // size host-side and pre-fills `b` with zeros, so minlength=0 is correct
    // here; it is still read from args/kwargs for future-proofing.
    aclError aclop_Bincount(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.empty() || outs.empty()) {
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        const aclTensor* weights = (ins.size() > 1) ? ins[1] : nullptr;
        int64_t minlength = GetScalarArg<int64_t>(args, 0, kwargs, "minlength", 0);
        return aclIrregularOpRun(aclnnBincountGetWorkspaceSize, aclnnBincount, stream,
            ins[0], weights, minlength, outs[0]);
    }

    // ufunc `cupy_where`: out = condition ? self : other
    //   ins = [condition, self, other]  (upscalar operands are not supported yet:
    //   the positional-scalar arg channel cannot rebuild a 0-d aclTensor, see
    //   arg_passing_plan.md M3)
    aclError aclop_Where(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        if (ins.size() < 3 || outs.empty()) {
            std::cout << "Error: " << __FUNCTION__
                      << " needs 3 tensor operands (condition, x, y); a scalar operand is"
                         " not supported by the Ascend backend yet\n";
            PrintArgs(__func__, args, kwargs, std::cout);
            return ACL_ERROR_INVALID_PARAM;
        }
        return aclIrregularOpRun(aclnnSWhereGetWorkspaceSize, aclnnSWhere, stream,
            ins[0], ins[1], ins[2], outs[0]);
    }

    // random.normal(loc=0.0, scale=1.0, size=None), normal distribution
    //
    // Review D8: these two bodies were commented out entirely, i.e. a non-void
    // function fell off the end (undefined behaviour) — a landmine waiting for
    // the first caller. They are not registered, so return an error instead.
    aclError aclop_Normal(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        // return aclIrregularOpRun(aclnnInplaceNormalGetWorkspaceSize, 
            // const aclTensor* selfRef, float mean, float std, int64_t seed,
            //                                              int64_t offset, uint64_t* workspaceSize,
            //                                              aclOpExecutor** executor);
        PrintArgs(__func__, args, kwargs, std::cout);
        return ACL_ERROR_INVALID_PARAM;
    }
    
    // random.rand() uniform distribution
    aclError aclop_Uniform(const std::vector<const aclTensor*>& ins, const std::vector<aclTensor*>& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream) {
        PrintArgs(__func__, args, kwargs, std::cout);
        return ACL_ERROR_INVALID_PARAM;
    }

    // -----------------------------------------------------------------------
    // 参数通道探针（`ascend_dump_args`）：只记录收到的参数，不做任何计算。
    //
    // 用途：在没有 NPU 的环境里验证「统一参数通道」真的把
    // scalar / int 序列 / str / None 按 tag 送达了 C++ 侧
    // （tests/ascend/test_unified_args.py 通过 py_last_dump_args() 读取）。
    // 生产路径不注册它给任何 numpy API。
    // -----------------------------------------------------------------------
    inline std::string& AclArgDumpBuffer() {
        static std::string buffer;
        return buffer;
    }

    aclError aclop_DumpArgs(const std::vector<const aclTensor*>& ins,
        const std::vector<aclTensor*>& outs, const ArgsType& args, const KwargsType& kwargs,
        aclrtStream stream) {
        std::ostringstream oss;
        oss << "ins=" << ins.size() << ",outs=" << outs.size();
        for (size_t i = 0; i < args.size(); ++i) {
            oss << ";arg[" << i << "]=";
            PrintArg(args[i], oss);
        }
        // 按 key 排序，便于测试里做字符串断言
        std::vector<std::string> keys;
        for (const auto& pair : kwargs) {
            keys.push_back(pair.first);
        }
        std::sort(keys.begin(), keys.end());
        for (const std::string& key : keys) {
            oss << ";kwarg[" << key << "]=";
            PrintArg(kwargs.at(key), oss);
        }
        AclArgDumpBuffer() = oss.str();
        return ACL_SUCCESS;
    }

    inline const char* aclop_GetLastDumpArgs() {
        return AclArgDumpBuffer().c_str();
    }

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif // header