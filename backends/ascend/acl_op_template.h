#ifndef CUPY_ASCEND_OP_TEMPLATE_HEADER
#define CUPY_ASCEND_OP_TEMPLATE_HEADER

#include <iostream>
#include <utility> // for std::forward

#include "acl/acl.h"


using AclnnKernelFunc = aclnnStatus (*)(void* workspace, uint64_t workspaceSize, 
                                       aclOpExecutor* executor, aclrtStream stream);

// if template function not working, then use macro func

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
    Operand otherTensor, // operand can be aclScalar* or aclTensor*
    aclTensor* outTensor,
    WsFunc wsfunc, AclnnKernelFunc kfunc,
    aclrtStream stream, bool sync,
    Args&&... args)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 第一段: 获取所需Workspace大小
    aclError ret = wsfunc(selfTensor, otherTensor, outTensor, std::forward<Args>(args)...,
        &workspaceSize, &executor);
    // e.g. ret = aclnnMatmulGetWorkspaceSize(a_tensor, b_tensor, out_tensor, math_type, &workspace_size, &executor);
    if (ret != ACL_SUCCESS) {
        std::cout << "Failed to allocate workspace";
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
aclError aclBinaryInplaceOpRun(
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
    if (ret != ACL_SUCCESS) { /* 错误处理 */ }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 第二段: 在指定的Stream上执行算子, this is fixed func type
    ret = kfunc(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::cout << "Failed to run the kernel";
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
aclError aclBinaryScalarOpRun(
    const aclTensor* selfTensor, Operand otherTensor, Scalar scalar, aclTensor* outTensor,
    WsFunc wsfunc, AclnnKernelFunc kfunc, aclrtStream stream, bool sync,
    Args&&... args)
{
    float alphaValue = scalar; // TODO
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 第一段: 获取所需Workspace大小
    aclError ret = wsfunc(selfTensor, otherTensor, alpha, outTensor, std::forward<Args>(args)...,
        &workspaceSize, &executor);
    //ret = aclnnAddGetWorkspaceSize(selfTensor, otherTensor, alpha, outTensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) { /* 错误处理 */ }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 第二段: 在指定的Stream上执行算子, this is fixed func type
    ret = kfunc(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::cout << "Failed to run the kernel";
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
    if (ret != ACL_SUCCESS) { /* 错误处理 */ }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 第二段: 在指定的Stream上执行算子, this is fixed func type
    ret = kfunc(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::cout << "Failed to run the kernel";
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
aclError aclUnaryInplaceOpRun(
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
    if (ret != ACL_SUCCESS) { /* 错误处理 */ }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 第二段: 在指定的Stream上执行算子, this is fixed func type
    ret = kfunc(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::cout << "Failed to run the kernel";
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

// declare the op and its inplace version
#define DECLARE_ACL_BINARY_OPS_FUNC(opname) \
    aclError aclop_##opname(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) { \
        return aclBinaryOpRun(self, other, out, \
            aclnn##opname##GetWorkspaceSize, aclnn##opname, stream, false); \
    } \
    aclError aclop_Inplace##opname(aclTensor* self, const aclTensor* other, aclrtStream stream) { \
        return aclBinaryInplaceOpRun(self, other, \
            aclnnInplace##opname##GetWorkspaceSize, aclnnInplace##opname, stream, false); \
    }


// declare the out = self + sclar binary op and its inplace version
#define DECLARE_ACL_BINARY_SCALAR_OPS_FUNC(opname) \
    aclError aclop_##opname(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream) { \
        return aclBinaryOpRun(self, other, out, \
            aclnn##opname##GetWorkspaceSize, aclnn##opname, stream, false); \
    } \
    aclError aclop_Inplace##opname(aclTensor* self, const aclScalar* other, aclrtStream stream) { \
        return aclBinaryInplaceOpRun(self, other, \
            aclnnInplace##opname##GetWorkspaceSize, aclnnInplace##opname, stream, false); \
    }    

// declare the op and its inplace version
#define DECLARE_ACL_UNARY_OPS_FUNC(opname) \
    aclError aclop_##opname(const aclTensor* self, aclTensor* out, aclrtStream stream) { \
        return aclUnaryOpRun(self, out, \
            aclnn##opname##GetWorkspaceSize, aclnn##opname, stream, false); \
    } \
    aclError aclop_Inplace##opname(aclTensor* self, aclrtStream stream) { \
        return aclUnaryInplaceOpRun(self, \
            aclnnInplace##opname##GetWorkspaceSize, aclnnInplace##opname, stream, false); \
    }


#endif // end of header file