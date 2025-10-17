#include <iostream>
#include <vector>
#include <random>
#include "acl/acl.h"
#include "aclnnop/aclnn_matmul.h" // 确保包含MatMul算子的头文件
#include "aclnnop/aclnn_add.h" // 确保包含MatMul算子的头文件
#include "aclnnop/aclnn_cos.h" // 确保包含MatMul算子的头文件

#include "../aclOpTemplate.h"

/*
g++ -std=c++11 aclnn_*_demo.cpp \
  -I/usr/local/Ascend/ascend-toolkit/latest/include \
  -L/usr/local/Ascend/ascend-toolkit/latest/lib64 \
  -lascendcl -lnnopbase -laclnn_math -laclnn_rand -llibacl_op_compiler -lopapi \
  -o aclnn_matmul_demo
*/

#define CHECK_RET(cond, msg, ret_code) \
    do { \
        if (!(cond)) { \
            std::cerr << "Error: " << (msg) << ", ret=" << (ret_code) << std::endl; \
            return ret_code; \
        } \
    } while (0)

// 辅助函数：生成指定形状和范围的随机浮点数据
std::vector<float> generate_random_data(const std::vector<int64_t>& shape, float min_val = -1.0f, float max_val = 1.0f) {
    size_t total_size = 1;
    for (auto dim : shape) total_size *= dim;
    
    std::vector<float> data(total_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (size_t i = 0; i < total_size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

// 辅助函数：创建ACL张量
int create_acl_tensor(const std::vector<float>& host_data, const std::vector<int64_t>& shape, 
                      void** device_addr, aclDataType data_type, aclTensor** tensor) {
    size_t size_in_bytes = host_data.size() * sizeof(float);
    
    // 1. 在Device上申请内存
    aclError ret = aclrtMalloc(device_addr, size_in_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "Failed to allocate device memory", ret);
    
    // 2. 将主机数据拷贝到Device
    ret = aclrtMemcpy(*device_addr, size_in_bytes, host_data.data(), size_in_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, "Failed to copy data to device", ret);
    
    // 3. 计算连续张量的步幅（Strides）
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = shape[i+1] * strides[i+1];
    }
    
    // 4. 创建ACL张量对象
    *tensor = aclCreateTensor(shape.data(), shape.size(), data_type,
                              strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *device_addr);
    CHECK_RET(*tensor != nullptr, "Failed to create ACL tensor", -1);
    
    return 0;
}

int main() {
    aclError ret;
    const int64_t matrix_size = 4096; // 4K矩阵
    std::vector<int64_t> shape = {matrix_size, matrix_size};
    
    // 1. 初始化AscendCL环境
    ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, "aclInit failed", ret);
    
    ret = aclrtSetDevice(0); // 使用设备0
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSetDevice failed", ret);
    
    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtCreateStream failed", ret);
    
    // 2. 生成随机数据并创建输入/输出张量
    std::vector<float> a_data = generate_random_data(shape);
    std::vector<float> b_data = generate_random_data(shape);
    std::vector<float> out_data(shape[0] * shape[1], 0.0f);
    
    void *a_device_addr = nullptr, *b_device_addr = nullptr, *out_device_addr = nullptr;
    aclTensor *a_tensor = nullptr, *b_tensor = nullptr, *out_tensor = nullptr;
    
    ret = create_acl_tensor(a_data, shape, &a_device_addr, ACL_FLOAT, &a_tensor);
    CHECK_RET(ret == 0, "Create tensor A failed", ret);
    
    ret = create_acl_tensor(b_data, shape, &b_device_addr, ACL_FLOAT, &b_tensor);
    CHECK_RET(ret == 0, "Create tensor B failed", ret);
    
    ret = create_acl_tensor(out_data, shape, &out_device_addr, ACL_FLOAT, &out_tensor);
    CHECK_RET(ret == 0, "Create output tensor failed", ret);
    
    // ================================================
#if 0
    // 3. 获取MatMul算子所需工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    uint8_t math_type = 0; // keep dtype precision KEPP_DTYPE
    ret = aclnnMatmulGetWorkspaceSize(a_tensor, b_tensor, out_tensor, math_type, &workspace_size, &executor);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnMatmulGetWorkspaceSize failed", ret);
    
    // 4. 申请工作空间内存
    void* workspace_addr = nullptr;
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, "Failed to allocate workspace memory", ret);
    }
    
    // 5. 执行MatMul计算
    ret = aclnnMatmul(workspace_addr, workspace_size, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnMatmul failed", ret);
    
    // 6. 同步流以确保计算完成
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed", ret);
    if (workspace_addr) aclrtFree(workspace_addr);
#else
    uint8_t math_type = 0; // keep dtype precision KEPP_DTYPE
    bool sync = true;
    ret = aclBinaryOpRun(
        a_tensor, b_tensor, out_tensor,
        aclnnMatmulGetWorkspaceSize, aclnnMatmul, stream, sync,
        math_type);  // extra option arg

    ret = aclBinaryScalarOpRun(
        a_tensor, b_tensor, 1.0, out_tensor,
        aclnnAddGetWorkspaceSize, aclnnAdd, stream, sync);

    ret = aclUnaryOpRun(
        a_tensor, out_tensor,
        aclnnCosGetWorkspaceSize, aclnnCos, stream, sync
        );

    ret = aclUnaryInplaceOpRun(
        a_tensor,
        aclnnInplaceCosGetWorkspaceSize, aclnnInplaceCos, stream, sync
        );
#endif
    // ===================================================
    
    // 7. (可选) 将结果拷贝回主机进行验证或使用
    std::vector<float> host_result(out_data.size());
    ret = aclrtMemcpy(host_result.data(), host_result.size() * sizeof(float), 
                     out_device_addr, out_data.size() * sizeof(float), 
                     ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, "Failed to copy result back to host", ret);
    
    std::cout << "MatMul computation completed successfully!" << std::endl;
    
    // 8. 资源清理
    if (a_tensor) aclDestroyTensor(a_tensor);
    if (b_tensor) aclDestroyTensor(b_tensor);
    if (out_tensor) aclDestroyTensor(out_tensor);

    if (a_device_addr) aclrtFree(a_device_addr);
    if (b_device_addr) aclrtFree(b_device_addr);
    if (out_device_addr) aclrtFree(out_device_addr);
    
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    
    return 0;
}