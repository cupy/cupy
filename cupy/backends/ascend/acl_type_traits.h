#ifndef CUPY_ACL_TYPE_TRAITS
#define CUPY_ACL_TYPE_TRAITS

#include <iostream>
#include <type_traits>
#include <string>

#include "acl/acl.h"
#include "aclnn/opdev/common_types.h"

// 辅助模板：将C++类型映射为aclDataType
template<typename T>
struct TypeToAclDataType {};

// 模板特化，定义常用类型的映射
template<> struct TypeToAclDataType<float> {
    static constexpr aclDataType value = ACL_FLOAT;
};
template<> struct TypeToAclDataType<double> {
    // 注意：CANN算子对double支持可能有限，通常使用float
    static constexpr aclDataType value = ACL_DOUBLE;
};
template<> struct TypeToAclDataType<int32_t> {
    static constexpr aclDataType value = ACL_INT32;
};
template<> struct TypeToAclDataType<int64_t> {
    static constexpr aclDataType value = ACL_INT64;
};
template<> struct TypeToAclDataType<uint8_t> {
    static constexpr aclDataType value = ACL_UINT8;
};
template<> struct TypeToAclDataType<bool> {
    static constexpr aclDataType value = ACL_BOOL;
};
// ... 可根据需要添加其他类型的特化，如half、bool等

// 辅助函数：将数据类型转换为字符串
std::string aclDtypeToString(aclDataType dtype) {
    switch(dtype) {
        case ACL_FLOAT: return "ACL_FLOAT";
        case ACL_INT32: return "ACL_INT32";
        case ACL_INT64: return "ACL_INT64";
        case ACL_FLOAT16: return "ACL_FLOAT16";
        case ACL_BOOL: return "ACL_BOOL";
        case ACL_UINT8: return "ACL_UINT8";
        default: return "UNKNOWN";
    }
}

// 辅助函数：打印单个 aclScalar 的值
void PrintScalarValue(const aclScalar* opscalar, std::ostream& os) {
    if (!opscalar) {
        os << "NULL";
        return;
    }
    // op::DataType is ge::DataType is aclDataType enum
    op::DataType dtype = opscalar->GetDataType();
    const void* vdata = opscalar->GetData();
    switch(dtype) {
        case ACL_FLOAT:
            os << opscalar->ToFloat(); // *static_cast<const float*>(vdata);
            break;
        case ACL_DOUBLE:
            os << *static_cast<const double*>(vdata);
            break;
        case ACL_INT32:
            os << *static_cast<const int32_t*>(vdata);
            break;
        case ACL_INT64:
            os << *static_cast<const int64_t*>(vdata);
            break;
        case ACL_BOOL:
            os << *static_cast<const bool*>(vdata);
            break;
        default:
            os << "[Unprintable Type]";
            break;
    }
}

template<typename ScalarT>
aclScalar* CreateAclScalar(ScalarT s) {
    constexpr aclDataType dtype = TypeToAclDataType<ScalarT>::value;
    // 注意：传入宿主侧数据的指针
    return aclCreateScalar(static_cast<void*>(&s), dtype);
}

aclScalar* CreateAclScalar(double value, aclDataType dtype) {
    // 通常建议将double转换为float以匹配NPU常用精度
    if (dtype == ACL_FLOAT) {
        float float_value = static_cast<float>(value);
        return aclCreateScalar(static_cast<void*>(&float_value), dtype);
    }
    // 如果确实需要double，则直接使用（请确保硬件和算子支持ACL_DOUBLE）
    else if (dtype == ACL_DOUBLE) {
        return aclCreateScalar(static_cast<void*>(&value), dtype);
    }
    // 处理不期望的数据类型
    else {
        // 在实际代码中，可能需要更复杂的错误处理逻辑，如记录日志或抛出异常
        return nullptr;
    }
}


// 获取类型名称的辅助函数
// template<typename T>
// constexpr const char* type_name() {
//     if constexpr (std::is_same_v<T, int32_t>) return "int32_t";
//     else if constexpr (std::is_same_v<T, float>) return "float";
//     else if constexpr (std::is_same_v<T, double>) return "double";
//     else if constexpr (std::is_same_v<T, bool>) return "bool";
//     else return "unknown";
// }

#endif