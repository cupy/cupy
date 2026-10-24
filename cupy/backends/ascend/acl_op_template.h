#ifndef CUPY_ASCEND_OP_TEMPLATE_HEADER
#define CUPY_ASCEND_OP_TEMPLATE_HEADER

#include <iostream>
#include <utility> // for std::forward

#include "acl/acl.h"
#include "aclnn/opdev/common_types.h"
#include "acl_type_traits.h"

using AclnnKernelFunc = aclnnStatus (*)(void* workspace, uint64_t workspaceSize, 
                                       aclOpExecutor* executor, aclrtStream stream);
// ArgsType / KwargsType（带 tag 的统一参数通道）定义在 acl_scalar_arg.h 里；
// 这里只 include 一次，避免两处定义漂移（以前两处各写了一份）。
#include "acl_scalar_arg.h"

#define CHECK_STATUS(status) \
do { \
    if (status != ACL_SUCCESS) { \
        std::cerr << "Failed to run acl function in " << __FUNCTION__ << ": " << \
        __FILE__ << ":" <<__LINE__ << "," << aclGetRecentErrMsg() << std::endl; \
    } \
} while (0)

// throw std::runtime_error(oss.str()); // may help to locate error, message may be buried

inline aclDataType GetDataType(const aclTensor* out, const aclTensor* self = nullptr) {
    aclDataType dtype = ACL_DT_UNDEFINED;
    if (out) {
        auto ret = aclGetDataType(self, &dtype);
        CHECK_STATUS(ret);
    }
    else if (self) {
        auto ret = aclGetDataType(self, &dtype);
        CHECK_STATUS(ret);
    }
    return dtype;
}

// dtype 分类工具：整数/布尔没有 NaN。
//
// 为什么存在：CANN 没有原生的 nan* 归约算子，nanmax/nanmin/nanprod/
// nanargmax 等（见 acl_reduction_ops.h 的 aclop_Nan* 系列）都用
// "aclnnNanToNum 把 NaN 替换成 ±inf + 普通归约" 的组合来模拟。但
//   1) NumPy 的 nan* 系列对整数/布尔输入等价于普通归约（根本没有 NaN）；
//   2) aclnnNanToNum 只接受浮点输入，整型会在第一段 GetWorkspaceSize 报
//      EL0003 Invalid_Argument。
// 所以这类组合算子对整数/布尔必须跳过 NanToNum 步骤直接跑普通算子。
// 该 switch 原来在 aclop_NanProd / aclop_NanArgMax / aclop_NanArgMin 三处
// 逐字重复，抽到这里统一维护。以后再出现同类 dtype 分流需求（例如某
// aclnn 算子不支持无符号类型需要转 int 的判断），也加成这里的工具函数，
// 不要再抄 switch。
inline bool dtype_has_no_nan(aclDataType dtype) {
    switch (dtype) {
        case ACL_INT8: case ACL_UINT8:
        case ACL_INT16: case ACL_UINT16:
        case ACL_INT32: case ACL_UINT32:
        case ACL_INT64: case ACL_UINT64:
        case ACL_BOOL:
            return true;
        default:
            return false;  // 浮点/复数可能含 NaN，需要走 NanToNum 组合
    }
}

int64_t GetAclTensorElementCount(const aclTensor* tensor) {
    std::vector<int64_t> shape_vec;
    int64_t numel = 0;
    if (tensor == nullptr) {
        return numel;
    }
    
    int64_t* shape = nullptr;
    uint64_t dim_count = 0;
    
    aclError ret = aclGetViewShape(tensor, &shape, &dim_count);
    if (ret == ACL_SUCCESS) {
        shape_vec.assign(shape, shape + dim_count);
        delete[] shape;
    }
    
    if (!shape_vec.empty()) {
        int64_t element_count = 1;
        for (auto dim : shape_vec) {
            element_count *= dim;
        }
        numel = element_count;
    }
    return numel;
}

/**
 * 根据源张量创建新张量，保持相同形状但使用指定数据类型, numpy.empty_like()
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
        delete[] viewDims;
        delete[] storageDims;
        delete[] strides;
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
        delete[] viewDims;
        delete[] storageDims;
        delete[] strides;
        return nullptr;
    }
    
    aclTensor* new_tensor = aclCreateTensor(viewDims, viewDimsNum, dtype, 
                                        strides, 0, format,
                                        storageDims, storageDimsNum, device_addr);
    
    if (new_tensor == nullptr) {
        std::cerr << "Error: Failed to create new tensor" << std::endl;
        aclrtFree(device_addr);
        delete[] viewDims;
        delete[] storageDims;
        delete[] strides;
        return nullptr;
    }
    delete[] viewDims;
    delete[] storageDims;
    delete[] strides;
    return new_tensor;
}

/**
 * 释放由 aclTensorLike() 创建的临时张量。
 *
 * aclTensorLike() 在内部 aclrtMalloc 了一块设备内存，而 aclDestroyTensor() 只释放
 * aclTensor 自身（shape/stride 元数据），不会释放这块设备内存。因此两者必须成对使用，
 * 否则每次调用这类复合算子（lcm/hypot/copysign/nanmin/nanmax/nancumsum/nancumprod）
 * 都会泄漏一块显存。
 */
inline void aclDestroyTensorLike(aclTensor* tensor) {
    if (tensor == nullptr) {
        return;
    }
    void* data = tensor->GetData();
    aclDestroyTensor(tensor);
    if (data != nullptr) {
        aclError ret = aclrtFree(data);
        if (ret != ACL_SUCCESS) {
            std::cerr << "Error: aclrtFree failed for aclTensorLike tensor, "
                      << "error code: " << ret << std::endl;
        }
    }
}


/* inplace use another func (unary)
op变种有 aclScalar(binary only), Foreach(TensorList), inplace(unary/binary)
op的操作数有unary和binary, multiple (这个很少,暂不处理)
GetWorkspaceSize 缺乏规则, 可以写一个std::forward函数, 变得有规律
ret = aclnnMatmulGetWorkspaceSize(a_tensor, b_tensor, out_tensor, math_type, &workspace_size, &executor);
ret aclnnAddGetWorkspaceSize(selfTensor, otherTensor, alpha, outTensor, &workspaceSize, &executor);
*/
template<typename WsFunc, typename Operand, typename... Args>
aclError aclBinaryOpRun(
    const aclTensor* selfTensor,
    Operand other, // operand can be aclScalar* or aclTensor*
    aclTensor* outTensor,
    WsFunc wsfunc, AclnnKernelFunc kfunc,
    aclrtStream stream, bool sync,
    Args&&... args)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclError ret = 0;

    const aclScalar* alpha = nullptr;
    if constexpr (std::is_scalar_v<Operand>  && ! std::is_pointer_v<Operand>) {
        float alphaValue = other;
        alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
        // 第一段: 获取所需Workspace大小
        ret = wsfunc(selfTensor, alpha, outTensor, std::forward<Args>(args)...,
            &workspaceSize, &executor);
    } else {
        ret = wsfunc(selfTensor, other, outTensor, std::forward<Args>(args)...,
            &workspaceSize, &executor);
    }

    // e.g. ret = aclnnMatmulGetWorkspaceSize(a_tensor, b_tensor, out_tensor, math_type, &workspace_size, &executor);
    if (ret != ACL_SUCCESS) {
        std::cout << "Failed to allocate workspace \n";
        CHECK_STATUS(ret);
        if (alpha != nullptr) {
            aclDestroyScalar(alpha);
        }
        return ACL_ERROR_RT_FAILURE;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 第二段: 在指定的Stream上执行算子, this is fixed func type
    ret = kfunc(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::cout << "Failed to run the kernel\n";
        CHECK_STATUS(ret);
        if (workspaceSize > 0 && workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        if (alpha != nullptr) {
            aclDestroyScalar(alpha);
        }
        return ACL_ERROR_RT_FAILURE;
    }

    if(sync) {
        aclrtSynchronizeStream(stream);
    }
    if (workspaceSize > 0) {
        ret = aclrtFree(workspaceAddr);
    }
    // alpha 只在标量操作数分支由本函数创建，必须由本函数释放
    if (alpha != nullptr) {
        aclDestroyScalar(alpha);
    }
    return ACL_SUCCESS;
}

template<typename WsFunc, typename Operand, typename... Args>
aclError aclInplaceBinaryOpRun(
    aclTensor* selfTensor,
    Operand otherTensor,
    WsFunc wsfunc, AclnnKernelFunc kfunc,
    aclrtStream stream, bool sync,
    Args&&... args)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 第一段: 获取所需Workspace大小
    aclError ret = wsfunc(selfTensor, otherTensor, std::forward<Args>(args)..., &workspaceSize, &executor);
    // e.g.
    if (ret != ACL_SUCCESS) {
        CHECK_STATUS(ret);
        std::cout << "Failed to allocate workspace \n";
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 第二段: 在指定的Stream上执行算子, this is fixed func type
    ret = kfunc(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::cout << "Failed to run the kernel\n";
        CHECK_STATUS(ret);
        if (workspaceSize > 0 && workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        return ACL_ERROR_RT_FAILURE;
    }

    if(sync) {
        aclrtSynchronizeStream(stream);
    }
    if (workspaceSize > 0) {
        ret = aclrtFree(workspaceAddr);
    }
    return ACL_SUCCESS;
}

// output = self <op> other * scalar,  3 operands here scalar is one operand
template<typename WsFunc, typename Operand, typename Scalar, typename... Args>
aclError aclTernaryOpRun(
    const aclTensor* selfTensor, Operand otherTensor, Scalar scalar, aclTensor* outTensor,
    WsFunc wsfunc, AclnnKernelFunc kfunc, aclrtStream stream, bool sync,
    Args&&... args)
{
    const aclScalar* alpha = nullptr;
    if constexpr (std::is_scalar_v<Scalar>  && ! std::is_pointer_v<Scalar>) {
        aclDataType dtype = ACL_DT_UNDEFINED;
        aclGetDataType(selfTensor, &dtype);
        alpha = CreateAclScalar(scalar, dtype);
    } else {
        alpha = scalar;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 第一段: 获取所需Workspace大小
    aclError ret = wsfunc(selfTensor, otherTensor, alpha, outTensor, std::forward<Args>(args)...,
        &workspaceSize, &executor);
    //ret = aclnnAddGetWorkspaceSize(selfTensor, otherTensor, alpha, outTensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        CHECK_STATUS(ret);
        std::cout << "Failed to allocate workspace \n";
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 第二段: 在指定的Stream上执行算子, this is fixed func type
    ret = kfunc(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::cout << "Failed to run the kernel\n";
        CHECK_STATUS(ret);
        if (workspaceSize > 0 && workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        if constexpr (std::is_scalar_v<Scalar> && !std::is_pointer_v<Scalar>) {
            if (alpha != nullptr) {
                aclDestroyScalar(alpha);
            }
        }
        return ACL_ERROR_RT_FAILURE;
    }

    if(sync) {
        aclrtSynchronizeStream(stream);
    }
    if (workspaceSize > 0) {
        ret = aclrtFree(workspaceAddr);
    }
    // alpha 只在标量操作数分支由本函数创建（原来被错误地嵌在 workspaceSize>0 里，
    // workspaceSize==0 时会泄漏）
    if constexpr (std::is_scalar_v<Scalar> && !std::is_pointer_v<Scalar>) {
        if (alpha != nullptr) {
            aclDestroyScalar(alpha);
        }
    }
    return ACL_SUCCESS;
}

// output = self <op> other * scalar,  3 operands here `scalar` is one operand
template<typename WsFunc, typename Operand, typename Scalar, typename... Args>
aclError aclTernaryInplaceOpRun(
    aclTensor* selfTensor, Operand otherTensor, Scalar scalar,
    WsFunc wsfunc, AclnnKernelFunc kfunc, aclrtStream stream, bool sync,
    Args&&... args)
{
    const aclScalar* alpha = nullptr;
    if constexpr (std::is_scalar_v<Scalar>  && ! std::is_pointer_v<Scalar>) {
        aclDataType dtype = ACL_DT_UNDEFINED;
        aclGetDataType(selfTensor, &dtype);
        alpha = CreateAclScalar(scalar, dtype);
    } else {
        alpha = scalar;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 第一段: 获取所需Workspace大小
    aclError ret = wsfunc(selfTensor, otherTensor, alpha, std::forward<Args>(args)...,
        &workspaceSize, &executor);
    //ret = aclnnInplaceAddGetWorkspaceSize(selfTensor, otherTensor, alpha,  &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        CHECK_STATUS(ret);
        std::cout << "Failed to run WorkspaceSize for a kernel\n";
        return ACL_ERROR_RT_FAILURE;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 第二段: 在指定的Stream上执行算子, this is fixed func type
    ret = kfunc(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::cout << "Failed to run the kernel\n";
        CHECK_STATUS(ret);
        if (workspaceSize > 0 && workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        if constexpr (std::is_scalar_v<Scalar> && !std::is_pointer_v<Scalar>) {
            if (alpha != nullptr) {
                aclDestroyScalar(alpha);
            }
        }
        return ACL_ERROR_RT_FAILURE;
    }

    if(sync) {
        aclrtSynchronizeStream(stream);
    }
    if (workspaceSize > 0) {
        ret = aclrtFree(workspaceAddr);
    }
    // 原来这里从不释放 alpha（标量操作数分支创建）
    if constexpr (std::is_scalar_v<Scalar> && !std::is_pointer_v<Scalar>) {
        if (alpha != nullptr) {
            aclDestroyScalar(alpha);
        }
    }
    return ACL_SUCCESS;
}

// // ForeachOp group small aclTensor input list, numpy/cupy does not have such ufunc
// template<typename WsFunc, typename... Args>
// aclError aclBinaryForeachOpRun(aclTensorList inputs, aclTensorList others, aclTensorList outputs,
//     WsFunc wsfunc, AclnnKernelFunc kfunc, aclrtStream stream, bool sync,
//     Args&&... args)

template<typename WsFunc, typename Operand, typename... Args>
aclError aclUnaryOpRun(
    const aclTensor* selfTensor,
    Operand outTensor,
    WsFunc wsfunc, AclnnKernelFunc kfunc,
    aclrtStream stream, bool sync,
    Args&&... args)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 第一段: 获取所需Workspace大小
    aclError ret = wsfunc(selfTensor, outTensor, std::forward<Args>(args)..., &workspaceSize, &executor);
    // e.g. aclnnStatus aclnnAsinGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);
    if (ret != ACL_SUCCESS) {
        std::cout << "Failed to run WorkspaceSize for a kernel\n";
        CHECK_STATUS(ret);
        return ACL_ERROR_RT_FAILURE;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 第二段: 在指定的Stream上执行算子, this is fixed func type
    ret = kfunc(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::cout << "Failed to run the kernel\n";
        CHECK_STATUS(ret);
        if (workspaceSize > 0 && workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        return ACL_ERROR_RT_FAILURE;
    }

    if(sync) {
        aclrtSynchronizeStream(stream);
    }
    if (workspaceSize > 0) {
        ret = aclrtFree(workspaceAddr);
    }
    return ACL_SUCCESS;
}


template<typename WsFunc, typename... Args>
aclError aclInplaceUnaryOpRun(
    aclTensor* selfTensor,
    WsFunc wsfunc, AclnnKernelFunc kfunc,
    aclrtStream stream, bool sync,
    Args&&... args)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 第一段: 获取所需Workspace大小
    aclError ret = wsfunc(selfTensor, std::forward<Args>(args)..., &workspaceSize, &executor);
    // e.g. ret = aclnnMatmulGetWorkspaceSize(a_tensor, b_tensor, out_tensor, math_type, &workspace_size, &executor);
    if (ret != ACL_SUCCESS) {
        std::cout << "Failed to run WorkspaceSize for a kernel\n";
        CHECK_STATUS(ret);
        return ACL_ERROR_RT_FAILURE;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 第二段: 在指定的Stream上执行算子, this is fixed func type
    ret = kfunc(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::cout << "Failed to run the kernel";
        CHECK_STATUS(ret);
        if (workspaceSize > 0 && workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        return ACL_ERROR_RT_FAILURE;
    }

    if(sync) {
        aclrtSynchronizeStream(stream);
    }
    if (workspaceSize > 0) {
        ret = aclrtFree(workspaceAddr);
    }
    return ACL_SUCCESS;
}

// for irregular aclnn op, which may does not have self or out tensor
// or there is non-fixed arg between self and out tensor
template<typename WsFunc, typename... Args>
aclError aclIrregularOpRun(
    WsFunc wsfunc, AclnnKernelFunc kfunc,
    aclrtStream stream,
    Args&&... args)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 第一段: 获取所需Workspace大小
    aclError ret = wsfunc(std::forward<Args>(args)..., &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        CHECK_STATUS(ret);
        std::cout << "Failed to run WorkspaceSize for a irregular op kernel\n";
        return ACL_ERROR_RT_FAILURE;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 第二段: 在指定的Stream上执行算子, this is fixed func type
    ret = kfunc(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        CHECK_STATUS(ret);
        std::cout << "Failed to run the irregular op kernel";
        if (workspaceSize > 0 && workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        return ACL_ERROR_RT_FAILURE;
    }

    if (workspaceSize > 0) {
        ret = aclrtFree(workspaceAddr);
    }
    return ACL_SUCCESS;
}


template<typename WsFunc, typename OutType, typename... Args>
aclError aclReductionOpRun(
    const aclTensor* selfTensor,
    //DimType dim, bool keepdim,
    OutType outTensor,
    WsFunc wsfunc, AclnnKernelFunc kfunc,
    aclrtStream stream,
    Args&&... args)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 第一段: 获取所需Workspace大小
    aclError ret = wsfunc(selfTensor, std::forward<Args>(args)..., outTensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::cout << "Failed to run WorkspaceSize for a kernel\n";
        CHECK_STATUS(ret);
        return ACL_ERROR_RT_FAILURE;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 第二段: 在指定的Stream上执行算子, this is fixed func type
    ret = kfunc(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::cout << "Failed to run the kernel";
        CHECK_STATUS(ret);
        if (workspaceSize > 0 && workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        return ACL_ERROR_RT_FAILURE;
    }

    if (workspaceSize > 0) {
        ret = aclrtFree(workspaceAddr);
    }
    return ACL_SUCCESS;
}

// op without inplace version
#define DECLARE_ACL_BINARY_OP(opname) \
    aclError aclop_##opname(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) { \
        return aclBinaryOpRun(self, other, out, \
            aclnn##opname##GetWorkspaceSize, aclnn##opname, stream, false); \
    }

// declare the op and its inplace version
#define DECLARE_ACL_BINARY_OPS_FUNC(opname) \
    aclError aclop_##opname(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) { \
        return aclBinaryOpRun(self, other, out, \
            aclnn##opname##GetWorkspaceSize, aclnn##opname, stream, false); \
    } \
    aclError aclop_Inplace##opname(aclTensor* self, const aclTensor* other, aclrtStream stream) { \
        return aclInplaceBinaryOpRun(self, other, \
            aclnnInplace##opname##GetWorkspaceSize, aclnnInplace##opname, stream, false); \
    }

// op without inplace version
#define DECLARE_ACL_BINARY_SCALAR_OP(opname) \
    aclError aclop_##opname(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream) { \
        return aclBinaryOpRun(self, other, out, \
            aclnn##opname##GetWorkspaceSize, aclnn##opname, stream, false); \
    }

// declare the out = self + sclar binary op and its inplace version
#define DECLARE_ACL_BINARY_SCALAR_OPS_FUNC(opname) \
    aclError aclop_##opname(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream) { \
        return aclBinaryOpRun(self, other, out, \
            aclnn##opname##GetWorkspaceSize, aclnn##opname, stream, false); \
    } \
    aclError aclop_Inplace##opname(aclTensor* self, const aclScalar* other, aclrtStream stream) { \
        return aclInplaceBinaryOpRun(self, other, \
            aclnnInplace##opname##GetWorkspaceSize, aclnnInplace##opname, stream, false); \
    }    

// declare the unary op
#define DECLARE_ACL_UNARY_OP(opname) \
aclError aclop_##opname(const aclTensor* self, aclTensor* out, aclrtStream stream) { \
    return aclUnaryOpRun(self, out, \
        aclnn##opname##GetWorkspaceSize, aclnn##opname, stream, false); \
}

// declare the unary op and its inplace version
#define DECLARE_ACL_UNARY_OPS_FUNC(opname) \
    aclError aclop_##opname(const aclTensor* self, aclTensor* out, aclrtStream stream) { \
        return aclUnaryOpRun(self, out, \
            aclnn##opname##GetWorkspaceSize, aclnn##opname, stream, false); \
    } \
    aclError aclop_Inplace##opname(aclTensor* self, aclrtStream stream) { \
        return aclInplaceUnaryOpRun(self, \
            aclnnInplace##opname##GetWorkspaceSize, aclnnInplace##opname, stream, false); \
    }

// declare the reduction op (sum, prod, any, all), dim may have diff type
#define DECLARE_ACL_REDUCTION_OP(opname) \
    aclError aclop_##opname(const aclTensor* self, const aclIntArray* dim, bool keepdim, \
        aclTensor* out, const KwargsType& kwargs, aclrtStream stream) { \
        return aclReductionOpRun(self, out, \
            aclnn##opname##GetWorkspaceSize, aclnn##opname, stream, dim, keepdim); \
    } \

#endif // end of header file