#ifndef CUPY_ASCEND_SCALAR_ARG_HEADER
#define CUPY_ASCEND_SCALAR_ARG_HEADER

#include <iostream>
#include <type_traits>
#include <limits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "aclnn/opdev/common_types.h"
#include "aclnn/opdev/data_type_utils.h"
#include "acl_type_traits.h"

// ---------------------------------------------------------------------------
// 统一参数通道：跨边界的参数不再是裸 `const aclScalar*`，而是**带 tag 的值**。
//
// 背景（docs/ascend/arg_passing_plan.md §2.2/§2.3）：
//   * Python 侧的 args/kwargs 以前只能表达「aclScalar」，于是 sequence（IntArray）、
//     str、None、ndarray 全部落到「无法转换 → 抛错 / 静默丢弃」；
//   * C++ 侧也无从知道收到的指针到底是什么类型。
// 现在所有参数都走同一条通路，并用 kind 明确标注：
//
//   ARG_NONE        None                      （例如 axis=None 表示「全部轴」）
//   ARG_SCALAR      int/float/bool/np 标量    -> aclScalar*
//   ARG_INT_ARRAY   list/tuple[int]           -> std::vector<int64_t>（axis/dims/shape/shift...）
//   ARG_STRING      str                       -> std::string（自持所有权，见下）
//   ARG_TENSOR      ndarray                   （预留：where= 等；pyx 侧暂未产出）
//
// 取值只用 GetScalarArg / GetInt64List / GetStringArg / GetTensorArg，
// 类型不匹配时返回缺省值（或按调用方要求报错），**不做指针重新解释**。
//
// 所有权（重要）：
//   * ARG_SCALAR 的 aclScalar 由 pyx 创建，必须由派发层在 finally 里销毁（_destroy_acl_arg）；
//   * ARG_INT_ARRAY / ARG_STRING **按值持有**（vector/string 在 C++ 侧）。
//     这里刻意不传 `aclIntArray*`：CANN 头文件里 aclIntArray 是不透明类型
//     （只有 aclCreateIntArray/aclDestroyIntArray），C++ 侧读不出内容；
//     需要 aclIntArray 的 aclnn 接口用 AclIntArrayGuard 现场构造（见下）。
// ---------------------------------------------------------------------------
enum AclArgKind {
    ARG_NONE = 0,
    ARG_SCALAR = 1,
    ARG_INT_ARRAY = 2,
    ARG_STRING = 3,
    ARG_TENSOR = 4,
};

struct AclArg {
    AclArgKind kind = ARG_NONE;
    const aclScalar* scalar = nullptr;
    const aclTensor* tensor = nullptr;
    // 字符串自带所有权：aclScalar 的 ACL_STRING 只有指针语义，需要 host 侧保活，
    // 这里直接存一份 std::string，把生命周期交给容器（避免 pointer-keyed keepalive）。
    std::string str;
    // int 序列（axis/dims/shape/shift...）按值保存，见文件头注释
    std::vector<int64_t> ints;

    AclArg() = default;
};

inline AclArg MakeNoneArg() {
    return AclArg();
}

inline AclArg MakeScalarArg(const aclScalar* scalar) {
    AclArg arg;
    arg.kind = ARG_SCALAR;
    arg.scalar = scalar;
    return arg;
}

inline AclArg MakeIntArrayArg(const std::vector<int64_t>& values) {
    AclArg arg;
    arg.kind = ARG_INT_ARRAY;
    arg.ints = values;
    return arg;
}

inline AclArg MakeStringArg(const std::string& value) {
    AclArg arg;
    arg.kind = ARG_STRING;
    arg.str = value;
    return arg;
}

inline AclArg MakeTensorArg(const aclTensor* tensor) {
    AclArg arg;
    arg.kind = ARG_TENSOR;
    arg.tensor = tensor;
    return arg;
}

// 现场把 int 序列变成 aclnn 需要的 aclIntArray（values 为空 -> 保持 nullptr，
// 例如 `axis=None` 表示「全部轴」，Flip/Roll 依赖这个语义）。
class AclIntArrayGuard {
public:
    explicit AclIntArrayGuard(const std::vector<int64_t>& values) {
        if (!values.empty()) {
            array_ = aclCreateIntArray(values.data(), static_cast<uint64_t>(values.size()));
        }
    }

    AclIntArrayGuard(const AclIntArrayGuard&) = delete;
    AclIntArrayGuard& operator=(const AclIntArrayGuard&) = delete;

    ~AclIntArrayGuard() {
        if (array_ != nullptr) {
            aclDestroyIntArray(array_);
        }
    }

    const aclIntArray* get() const {
        return array_;
    }

    explicit operator bool() const {
        return array_ != nullptr;
    }

private:
    aclIntArray* array_ = nullptr;
};

inline const char* AclArgKindToString(AclArgKind kind) {
    switch (kind) {
        case ARG_NONE: return "none";
        case ARG_SCALAR: return "scalar";
        case ARG_INT_ARRAY: return "int_array";
        case ARG_STRING: return "string";
        case ARG_TENSOR: return "tensor";
        default: return "unknown";
    }
}

using KwargsType = std::unordered_map<std::string, AclArg>;
using ArgsType = std::vector<AclArg>;


inline aclTensorList* ToAclTensorList(const std::vector<const aclTensor*>& tempVector) {
    aclTensorList* tensorList = aclCreateTensorList(
        tempVector.data(),  // 指向aclTensor指针数组的指针
        static_cast<uint64_t>(tempVector.size())  // 张量数量
    );
    return tensorList;
}

/**
 * aclCreateTensorList() 出来的 list 必须 aclDestroyTensorList()（review D5：
 * 全仓库先前没有一处释放，concatenate/stack 每次调用泄漏一个 list 对象）。
 * 这里用 RAII 包一层，把「创建 + 释放」绑在一起，调用点只需 get()。
 */
class AclTensorListGuard {
public:
    explicit AclTensorListGuard(const std::vector<const aclTensor*>& tensors)
        : list_(ToAclTensorList(tensors)) {}

    AclTensorListGuard(const AclTensorListGuard&) = delete;
    AclTensorListGuard& operator=(const AclTensorListGuard&) = delete;

    ~AclTensorListGuard() {
        if (list_ != nullptr) {
            aclDestroyTensorList(list_);
        }
    }

    aclTensorList* get() const {
        return list_;
    }

    explicit operator bool() const {
        return list_ != nullptr;
    }

private:
    aclTensorList* list_;
};

// ---------------------------------------------------------------------------
// 统一参数通道的读取接口
//
// kwargs 优先于位置参数（语义与旧实现一致）；返回的指针指向容器内部，调用方不要保存。
// ---------------------------------------------------------------------------
// 前置声明：TryGetInt64List 需要把 aclScalar 转成 int64，而 ToScalarArg 的定义在
// 本文件后面（它依赖 acl_type_traits.h 的转换工具）。默认值只能声明一次，放在这里。
template<typename ToScalarType> ToScalarType ToScalarArg(const aclScalar* s,
                                                         bool throw_on_error = true);

inline const AclArg* FindArg(const ArgsType& args, int argIndex, const KwargsType& kargs,
                             const std::string& key)
{
    KwargsType::const_iterator it = kargs.find(key);
    if (it != kargs.end()) {
        return &it->second;
    }
    if (argIndex >= 0 && argIndex < static_cast<int>(args.size())) {
        return &args[argIndex];
    }
    return nullptr;
}

inline bool HasArg(const ArgsType& args, int argIndex, const KwargsType& kargs,
                   const std::string& key)
{
    const AclArg* arg = FindArg(args, argIndex, kargs, key);
    return arg != nullptr && arg->kind != ARG_NONE;
}

// 是否是「标量」参数（旧 API 的名字保留：Flip/Roll 等调用点仍用它做存在性判断）
inline bool HasScalarArg(const ArgsType& args, int argIndex, const KwargsType& kargs,
                         std::string key)
{
    const AclArg* arg = FindArg(args, argIndex, kargs, key);
    return arg != nullptr && arg->kind == ARG_SCALAR && arg->scalar != nullptr;
}

inline bool HasScalarKwarg(const KwargsType& kargs, std::string key)
{
    KwargsType::const_iterator it = kargs.find(key);
    return it != kargs.end() && it->second.kind == ARG_SCALAR && it->second.scalar != nullptr;
}

inline bool HasIntArrayArg(const ArgsType& args, int argIndex, const KwargsType& kargs,
                           const std::string& key)
{
    const AclArg* arg = FindArg(args, argIndex, kargs, key);
    return arg != nullptr && arg->kind == ARG_INT_ARRAY;
}

inline bool HasStringArg(const ArgsType& args, int argIndex, const KwargsType& kargs,
                         const std::string& key)
{
    const AclArg* arg = FindArg(args, argIndex, kargs, key);
    return arg != nullptr && arg->kind == ARG_STRING;
}

inline bool HasTensorArg(const ArgsType& args, int argIndex, const KwargsType& kargs,
                         const std::string& key)
{
    const AclArg* arg = FindArg(args, argIndex, kargs, key);
    return arg != nullptr && arg->kind == ARG_TENSOR && arg->tensor != nullptr;
}

inline const char* GetStringArg(const ArgsType& args, int argIndex, const KwargsType& kargs,
                               const std::string& key, const char* defaultValue = nullptr)
{
    const AclArg* arg = FindArg(args, argIndex, kargs, key);
    if (arg == nullptr || arg->kind == ARG_NONE) {
        return defaultValue;
    }
    if (arg->kind != ARG_STRING) {
        std::cerr << "WARNING: GetStringArg: '" << key << "' is a "
                  << AclArgKindToString(arg->kind) << ", not a string\n";
        return defaultValue;
    }
    return arg->str.c_str();
}

inline const aclTensor* GetTensorArg(const ArgsType& args, int argIndex,
                                     const KwargsType& kargs, const std::string& key,
                                     const aclTensor* defaultValue = nullptr)
{
    const AclArg* arg = FindArg(args, argIndex, kargs, key);
    if (arg == nullptr || arg->kind == ARG_NONE) {
        return defaultValue;
    }
    if (arg->kind != ARG_TENSOR) {
        std::cerr << "WARNING: GetTensorArg: '" << key << "' is a "
                  << AclArgKindToString(arg->kind) << ", not a tensor\n";
        return defaultValue;
    }
    return arg->tensor;
}

// 「int 标量 / int 序列 / None」三态的通用读取：
//   * ARG_SCALAR     -> out = [value]
//   * ARG_INT_ARRAY  -> out = 全部元素
//   * ARG_NONE/缺省  -> 返回 false（例如 axis=None 表示「全部轴」）
// 这是 multi-axis 参数（flip/roll/permute/aminmax 的 dim ...）接入 IntArray 的入口。
inline bool TryGetInt64List(const ArgsType& args, int argIndex, const KwargsType& kargs,
                            const std::string& key, std::vector<int64_t>* out)
{
    if (out == nullptr) {
        return false;
    }
    out->clear();
    const AclArg* arg = FindArg(args, argIndex, kargs, key);
    if (arg == nullptr || arg->kind == ARG_NONE) {
        return false;
    }
    if (arg->kind == ARG_SCALAR) {
        if (arg->scalar == nullptr) {
            return false;
        }
        out->push_back(ToScalarArg<int64_t>(arg->scalar, true));
        return true;
    }
    if (arg->kind == ARG_INT_ARRAY) {
        *out = arg->ints;
        return !out->empty();
    }
    std::cerr << "WARNING: TryGetInt64List: '" << key << "' is a "
              << AclArgKindToString(arg->kind) << ", expected int / sequence[int] / None\n";
    return false;
}


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

// IEEE-754 binary16 位模式 -> float（aclnn 侧没有可直接使用的半精度转换工具）
inline float HalfBitsToFloat(uint16_t bits) {
    auto sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    uint32_t exp = (bits >> 10) & 0x1fu;
    uint32_t frac = bits & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (frac == 0) {
            out = sign;  // +/-0
        } else {
            // subnormal: normalize
            exp = 127 - 15 + 1;
            while ((frac & 0x400u) == 0) {
                frac <<= 1;
                --exp;
            }
            frac &= 0x3ffu;
            out = sign | (exp << 23) | (frac << 13);
        }
    } else if (exp == 0x1fu) {
        out = sign | 0x7f800000u | (frac << 13);  // Inf / NaN
    } else {
        out = sign | ((exp + (127 - 15)) << 23) | (frac << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

// 提取 aclScalar 的值并统一转换成 double
inline double AclScalarToDouble(const aclScalar* s) {
    if (s == nullptr) {
        throw std::invalid_argument("AclScalarToDouble: aclScalar pointer is null");
    }
    op::DataType dtype = s->GetDataType();
    const void* vdata = s->GetData();
    switch (dtype) {
        case op::DataType::DT_BOOL:
            return *static_cast<const bool*>(vdata) ? 1.0 : 0.0;
        case op::DataType::DT_INT8:
            return static_cast<double>(*static_cast<const int8_t*>(vdata));
        case op::DataType::DT_UINT8:
            return static_cast<double>(*static_cast<const uint8_t*>(vdata));
        case op::DataType::DT_INT16:
            return static_cast<double>(*static_cast<const int16_t*>(vdata));
        case op::DataType::DT_UINT16:
            return static_cast<double>(*static_cast<const uint16_t*>(vdata));
        case op::DataType::DT_INT32:
            return static_cast<double>(*static_cast<const int32_t*>(vdata));
        case op::DataType::DT_UINT32:
            return static_cast<double>(*static_cast<const uint32_t*>(vdata));
        case op::DataType::DT_INT64:
            return static_cast<double>(*static_cast<const int64_t*>(vdata));
        case op::DataType::DT_UINT64:
            return static_cast<double>(*static_cast<const uint64_t*>(vdata));
        case op::DataType::DT_FLOAT:
            return static_cast<double>(*static_cast<const float*>(vdata));
        case op::DataType::DT_DOUBLE:
            return *static_cast<const double*>(vdata);
        case op::DataType::DT_FLOAT16:
            // IEEE-754 binary16 -> float
            return static_cast<double>(
                HalfBitsToFloat(*static_cast<const uint16_t*>(vdata)));
        default:
            throw std::runtime_error("AclScalarToDouble: unsupported aclScalar dtype");
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
    // NOTE: `dtype` is a runtime value, so this can not be `if constexpr`.
    if (std::is_same_v<ToScalarType, float> &&
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
ToScalarType ToScalarArg(const aclScalar* s, bool throw_on_error) {
    
    if (s == nullptr) {
        if (throw_on_error) {
            throw std::invalid_argument("aclScalar pointer is null");
        }
        std::cerr << "WARNING: aclScalar pointer is null, returning default value" << std::endl;
        return ToScalarType{};
    }
    
    op::DataType dtype = s->GetDataType();
    const char* target_type_name = typeid(ToScalarType).name();
    
    // 提取源值并转换为double进行统一处理
    double source_value = 0.0;
    if (op::IsBasicType(dtype)) {
        source_value = AclScalarToDouble(s);
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
//
// `defaultValue` is deliberately an `std::optional` (review D9): a call site that
// supplies a fallback (e.g. `..., "stable", true`) may silently use it, but one
// that does *not* now **throws** when the argument is missing. Previously every
// missing argument was reported with a single std::cerr line and the op then ran
// with a default-constructed value, which turned "the caller forgot to pass the
// parameter" into a silently wrong result.
template<typename ToScalarType>
ToScalarType GetScalarArg(const ArgsType& args, int argIndex, const KwargsType& kargs,
    const std::string& key, std::optional<ToScalarType> defaultValue = std::nullopt)
{
    const AclArg* arg = FindArg(args, argIndex, kargs, key);

    if (arg == nullptr || arg->kind == ARG_NONE || arg->scalar == nullptr) {
        if (!defaultValue.has_value()) {
            throw std::invalid_argument(
                "GetScalarArg: required argument '" + key + "' (positional #" +
                std::to_string(argIndex) + ") was not supplied to the aclnn op; "
                "refusing to continue with a silent default");
        }
        return defaultValue.value();
    }
    if (arg->kind != ARG_SCALAR) {
        // 以前这里会把一个非 aclScalar 的指针当成 aclScalar 用（UB）；现在显式报错
        const std::string msg =
            "GetScalarArg: argument '" + key + "' is a " +
            std::string(AclArgKindToString(arg->kind)) + ", not a scalar";
        if (defaultValue.has_value()) {
            std::cerr << "WARNING: " << msg << ", using the default value\n";
            return defaultValue.value();
        }
        throw std::invalid_argument(msg);
    }
    return ToScalarArg<ToScalarType>(arg->scalar);
}

// 打印一个参数（调试用，PrintArgs 与测试 op 共用）
inline void PrintArg(const AclArg& arg, std::ostream& os) {
    os << "kind=" << AclArgKindToString(arg.kind);
    switch (arg.kind) {
        case ARG_SCALAR:
            os << ", type=";
            PrintScalarType(arg.scalar, os);
            os << ", value=";
            PrintScalarValue(arg.scalar, os);
            break;
        case ARG_INT_ARRAY:
            os << ", values=[";
            for (size_t i = 0; i < arg.ints.size(); ++i) {
                if (i) {
                    os << ", ";
                }
                os << arg.ints[i];
            }
            os << "]";
            break;
        case ARG_STRING:
            os << ", value=\"" << arg.str << "\"";
            break;
        case ARG_TENSOR:
            os << ", tensor=" << static_cast<const void*>(arg.tensor);
            break;
        default:
            break;
    }
}

inline void PrintArgs(const char* func, const ArgsType& args, const KwargsType& kwargs,
                      std::ostream& os) {
    // 1. 打印位置参数信息
    os << "=== function name: " << func << " ===" << std::endl;
    os << "=== Positional Arguments (Args) ===" << "Count: " << args.size() << std::endl;

    for (size_t i = 0; i < args.size(); ++i) {
        os << "  Args[" << i << "]: ";
        PrintArg(args[i], os);
        os << std::endl;
    }

    // 2. 打印关键字参数信息
    os << "\n=== Keyword Arguments (Kwargs) ===" << "Count: " << kwargs.size() << std::endl;
    for (const auto& pair : kwargs) {
        os << "  Kwargs['" << pair.first << "']: ";
        PrintArg(pair.second, os);
        os << std::endl;
    }
}

#endif // header file