#ifndef CUPY_ASCEND_SCALAR_ARG_HEADER
#define CUPY_ASCEND_SCALAR_ARG_HEADER

#include <iostream>
#include <type_traits>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "aclnn/opdev/common_types.h"
#include "aclnn/opdev/data_type_utils.h"

using KargsType = std::unordered_map<std::string, const aclScalar*>;
using ArgsType = std::vector<const aclScalar*>;


inline aclTensorList* ToAclTensorList(const std::vector<const aclTensor*>& tempVector) {
    aclTensorList* tensorList = aclCreateTensorList(
        tempVector.data(),  // 指向aclTensor指针数组的指针
        static_cast<uint64_t>(tempVector.size())  // 张量数量
    );
}

bool HasScalarArg(const ArgsType& args, int argIndex, const KargsType& kargs, std::string key)
{
    if (kargs.find(key) != kargs.end()) {
        return true;
    } else if (argIndex < args.size()) {
        return true;
    } else {
        return false;
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

// // 获取aclScalar类型名称
// const char* get_acl_scalar_type_name(aclScalar::DataType type) {
//     switch (type) {
//         case aclScalar::DataType::INT32: return "int32_t";
//         case aclScalar::DataType::FLOAT: return "float";
//         case aclScalar::DataType::DOUBLE: return "double";
//         case aclScalar::DataType::BOOL: return "bool";
//         default: return "unknown";
//     }
// }

// 整数类型转换
template<typename ToScalarType> ToScalarType CheckIntegerArg(double source_value, op::DataType dtype,
    bool throw_on_error = true, bool warn_on_precision_loss = true) {

    const char* target_type_name = typeid(ToScalarType).name();
    if constexpr (std::is_same_v<ToScalarType, bool>) {
        // 布尔类型特殊处理
        return source_value != 0.0;
    } else {
        // 其他整数类型
        constexpr ToScalarType min_val = std::numeric_limits<ToScalarType>::min();
        constexpr ToScalarType max_val = std::numeric_limits<ToScalarType>::max();
        
        if (source_value < static_cast<double>(min_val) || 
            source_value > static_cast<double>(max_val)) {
            if (throw_on_error) {
                throw std::out_of_range("Value " + std::to_string(source_value) + 
                                        " out of range for " + target_type_name + 
                                        " [" + std::to_string(min_val) + 
                                        ", " + std::to_string(max_val) + "]");
            }
            std::cerr << "WARNING: Value " << source_value << " out of range for " 
                        << target_type_name << ", clamping to bounds" << std::endl;
            
            // 钳制到边界值
            if (source_value < static_cast<double>(min_val)) {
                return min_val;
            } else {
                return max_val;
            }
        }
        
        // 检查精度损失（浮点数转整数）
        if (dtype == op::DataType::DT_FLOAT || dtype == op::DataType::DT_DOUBLE) {
            double integer_part;
            double fractional_part = std::modf(source_value, &integer_part);
            
            if (std::abs(fractional_part) > std::numeric_limits<double>::epsilon() * 100) {
                if (warn_on_precision_loss) {
                    std::cerr << "WARNING: Precision loss when converting " << source_value 
                                << " to " << target_type_name 
                                << ", fractional part " << fractional_part << " will be truncated" << std::endl;
                }
            }
        }
        return static_cast<ToScalarType>(source_value);
    }
}

template<typename ToScalarType> ToScalarType CheckFloatArg(double source_value, op::DataType dtype,
    bool throw_on_error = true, bool warn_on_precision_loss = true) {

    const char* target_type_name = typeid(ToScalarType).name();
    // 浮点数类型转换
    constexpr ToScalarType min_val = std::numeric_limits<ToScalarType>::lowest();
    constexpr ToScalarType max_val = std::numeric_limits<ToScalarType>::max();
    
    if (source_value < static_cast<double>(min_val) || 
        source_value > static_cast<double>(max_val)) {
        if (throw_on_error) {
            throw std::out_of_range("Value " + std::to_string(source_value) + 
                                    " out of range for " + target_type_name + 
                                    " [" + std::to_string(min_val) + 
                                    ", " + std::to_string(max_val) + "]");
        }
        std::cerr << "WARNING: Value " << source_value << " out of range for " 
                    << target_type_name << ", clamping to bounds" << std::endl;
        
        // 钳制到边界值
        if (source_value < static_cast<double>(min_val)) {
            return min_val;
        } else {
            return max_val;
        }
    }
    
    // 检查精度损失（高精度浮点数转低精度）
    if constexpr (std::is_same_v<ToScalarType, float> && 
                    (dtype == op::DataType::DT_DOUBLE)) {
        float converted = static_cast<float>(source_value);
        double round_trip = static_cast<double>(converted);
        
        if (std::abs(source_value - round_trip) > std::numeric_limits<double>::epsilon() * 1000) {
            if (warn_on_precision_loss) {
                std::cerr << "WARNING: Precision loss when converting " << source_value 
                            << " from double to float, difference: " 
                            << (source_value - round_trip) << std::endl;
            }
        }
    }
    // TODO: float16, complex
    return static_cast<ToScalarType>(source_value);
}

template<typename ToScalarType>
ToScalarType ToScalarArg(const aclScalar* s, bool throw_on_error = true) {
    
    if (s == nullptr) {
        if (throw_on_error) {
            throw std::invalid_argument("aclScalar pointer is null");
        }
        std::cerr << "WARNING: aclScalar pointer is null, returning default value" << std::endl;
        return ToScalarType{};
    }
    
    op::DataType dtype = s->GetDataType();
    const char* target_type_name = typeid(ToScalarType).name();
    
    // 用于存储源值的中间变量
    double source_value = 0.0;
    
    // 提取源值并转换为double进行统一处理
    if (op::IsBasicType(dtype)) {
        // source_value = dtype.ToDouble();  // TODO
    } else {
        if (throw_on_error) {
            throw std::runtime_error("Unsupported aclScalar data type");
        }
        std::cerr << "WARNING: Unsupported aclScalar data type" << std::endl;
        return ToScalarType{};
    }
    
    // 检查NaN和无穷大
    if (std::isnan(source_value)) {
        if (throw_on_error) {
            throw std::runtime_error("Cannot convert NaN to " + std::string(target_type_name));
        }
        std::cerr << "WARNING: Attempting to convert NaN to " << target_type_name << std::endl;
        return ToScalarType{};
    }
    
    if (std::isinf(source_value)) {
        if (throw_on_error) {
            throw std::runtime_error("Cannot convert infinity to " + std::string(target_type_name));
        }
        std::cerr << "WARNING: Attempting to convert infinity to " << target_type_name << std::endl;
        // 对于浮点类型，可以尝试转换无穷大
        if constexpr (std::is_floating_point_v<ToScalarType>) {
            return source_value > 0 ? std::numeric_limits<ToScalarType>::infinity() 
                                   : -std::numeric_limits<ToScalarType>::infinity();
        }
        return ToScalarType{};
    }
    
    // 范围检查和转换
    if constexpr (std::is_integral_v<ToScalarType>) {
        return CheckIntegerArg<ToScalarType>(source_value, dtype);
    } else if constexpr (std::is_floating_point_v<ToScalarType>) {
        return CheckFloatArg<ToScalarType>(source_value, dtype);
    } // TODO: complex, string?
}

// dict kargs has priority than the list unnamed arg
template<typename ToScalarType>
ToScalarType GetScalarArg(const ArgsType& args, int argIndex, const KargsType& kargs, std::string key)
{
    ToScalarType scalar;
    const aclScalar* arg = nullptr;
    if (kargs.find(key) != kargs.end()) {
        arg = kargs.at(key);
    } else if (argIndex < args.size()) {
        arg = args.at(argIndex);
    } else {
        std::cerr << "WARNING: Failed to get argument from args list or kargs dict, use the default scalar value\n";
    }
    if (arg) {
        scalar = ToScalarArg<ToScalarType>(arg);
    }
    return scalar;
}

#endif // header file