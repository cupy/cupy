#ifndef CUPY_ASCEND_OP_TEMPLATE_HEADER
#define CUPY_ASCEND_OP_TEMPLATE_HEADER

#include <iostream>
#include <utility> // for std::forward

#include "acl/acl.h"
#include "aclnn/opdev/common_types.h"
#include "acl_type_traits.h"

using AclnnKernelFunc = aclnnStatus (*)(void* workspace, uint64_t workspaceSize, 
                                       aclOpExecutor* executor, aclrtStream stream);
using KwargsType = std::unordered_map<std::string, const aclScalar*>;

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
        return nullptr;
    }
    
    aclTensor* new_tensor = aclCreateTensor(viewDims, viewDimsNum, dtype, 
                                        strides, 0, format,
                                        storageDims, storageDimsNum, device_addr);
    
    if (new_tensor == nullptr) {
        std::cerr << "Error: Failed to create new tensor" << std::endl;
        aclrtFree(device_addr);
        return nullptr;
    }
    delete[] viewDims;
    delete[] storageDims;
    delete[] strides;
    return new_tensor;
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
        return ACL_ERROR_RT_FAILURE;
    }

    if(sync) {
        aclrtSynchronizeStream(stream);
    }
    if (workspaceSize > 0) {
        ret = aclrtFree(workspaceAddr);
        aclDestroyScalar(alpha);
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