#ifndef CUPY_ACL_TYPE_TRAITS
#define CUPY_ACL_TYPE_TRAITS

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

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
template<> struct TypeToAclDataType<int8_t> {
    static constexpr aclDataType value = ACL_INT8;
};
template<> struct TypeToAclDataType<int16_t> {
    static constexpr aclDataType value = ACL_INT16;
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
template<> struct TypeToAclDataType<uint16_t> {
    static constexpr aclDataType value = ACL_UINT16;
};
template<> struct TypeToAclDataType<uint32_t> {
    static constexpr aclDataType value = ACL_UINT32;
};
template<> struct TypeToAclDataType<uint64_t> {
    static constexpr aclDataType value = ACL_UINT64;
};
template<> struct TypeToAclDataType<bool> {
    static constexpr aclDataType value = ACL_BOOL;
};
// 未特化的类型保持空实现：实例化 TypeToAclDataType<T>::value 会在编译期报错
// （比运行期拿到一个错误 dtype 好）。

// 辅助函数：将数据类型转换为字符串
std::string aclDtypeToString(aclDataType dtype) {
    switch(dtype) {
        case ACL_DT_UNDEFINED: return "ACL_DT_UNDEFINED";
        case ACL_FLOAT: return "ACL_FLOAT";
        case ACL_FLOAT16: return "ACL_FLOAT16";
        case ACL_DOUBLE: return "ACL_DOUBLE";
        case ACL_INT8: return "ACL_INT8";
        case ACL_INT16: return "ACL_INT16";
        case ACL_INT32: return "ACL_INT32";
        case ACL_INT64: return "ACL_INT64";
        case ACL_UINT8: return "ACL_UINT8";
        case ACL_UINT16: return "ACL_UINT16";
        case ACL_UINT32: return "ACL_UINT32";
        case ACL_UINT64: return "ACL_UINT64";
        case ACL_BOOL: return "ACL_BOOL";
        case ACL_STRING: return "ACL_STRING";
        case ACL_BF16: return "ACL_BF16";
        case ACL_COMPLEX32: return "ACL_COMPLEX32";
        case ACL_COMPLEX64: return "ACL_COMPLEX64";
        case ACL_COMPLEX128: return "ACL_COMPLEX128";
        default: return "UNKNOWN";
    }
}

void PrintScalarType(const aclScalar* opscalar, std::ostream& os) {
    if (!opscalar) {
        os << "NULL";
        return;
    }
    // op::DataType is ge::DataType is aclDataType enum
    op::DataType dtype = opscalar->GetDataType();
    os << aclDtypeToString(static_cast<aclDataType>(dtype));
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
        case ACL_FLOAT16:
        case ACL_BF16:
            // aclScalar 对半精度/bf16 存的是 16bit 位模式（common_types.h 的
            // v_t 只有 uint16_t ui16，见 aclScalar::BFloat16()）；这里不调用
            // op::fp16_t 的 out-of-line 转换函数，直接打印位模式。
            os << "bits=0x" << std::hex << *static_cast<const uint16_t*>(vdata)
               << std::dec;
            break;
        case ACL_DOUBLE:
            os << *static_cast<const double*>(vdata);
            break;
        case ACL_INT8:
            os << *static_cast<const int8_t*>(vdata);
            break;
        case ACL_UINT8:
            os << *static_cast<const uint8_t*>(vdata);
            break;
        case ACL_INT16:
            os << *static_cast<const int16_t*>(vdata);
            break;
        case ACL_UINT16:
            os << *static_cast<const uint16_t*>(vdata);
            break;
        case ACL_INT32:
            os << *static_cast<const int32_t*>(vdata);
            break;
        case ACL_UINT32:
            os << *static_cast<const uint32_t*>(vdata);
            break;
        case ACL_INT64:
            os << *static_cast<const int64_t*>(vdata);
            break;
        case ACL_UINT64:
            os << *static_cast<const uint64_t*>(vdata);
            break;
        case ACL_BOOL:
            os << *static_cast<const bool*>(vdata);
            break;
        case ACL_COMPLEX64: {
            // aclScalar 内部是 std::complex<float>（common_types.h:346），
            // 用 ToComplex64() 取值而不是直接解引用 GetData()。
            const std::complex<float> v = opscalar->ToComplex64();
            os << '(' << v.real() << (v.imag() < 0 ? '-' : '+')
               << std::abs(v.imag()) << "j)";
            break;
        }
        case ACL_COMPLEX128: {
            const std::complex<double> v = opscalar->ToComplex128();
            os << '(' << v.real() << (v.imag() < 0 ? '-' : '+')
               << std::abs(v.imag()) << "j)";
            break;
        }
        case ACL_STRING:
            os << opscalar->ToString().GetString();
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

// ---------------------------------------------------------------------------
// double -> 指定 dtype 的 aclScalar 创建
//
// 调用点（Arange / Histc / Fill / MaskedScatter / ternary 的 alpha / Clamp ...）
// 都是「用 tensor 的 dtype 造 scalar」，所以这里**必须**覆盖全部基础类型：
// 少一个分支就会返回 nullptr，而 nullptr 交给 aclnn 轻则
// ACLNN_ERR_PARAM_INVALID（以前只剩 stdout 一行 WARNING），重则崩溃。
// uint16/uint32 原先就落在这个坑里 —— 注意 aclScalar 本身是支持它们的
// （common_types.h: v_t::ui16/ui32 + ToUint16()/ToUint32()，
// opdev/data_type_utils.h 的 TypeSize/IsBasicType 也包含 DT_UINT16/32），
// 缺的只是本函数。
//
// 另一条硬约束：本函数在「没有 `except +` 的 extern 边界」内被调用
// （见 docs/ascend/refactor_exception.md），因此**不能 throw**：C++ 异常不会
// 被翻译成 Python 异常，只会穿过 Cython 栈导致 std::terminate。越界/NaN
// 一律饱和 + stderr 警告。
// 返回值仍可能是 nullptr（dtype 真的无法用 double 表达，或 aclCreateScalar
// 分配失败），调用方需要检查 —— 派发层已把 nullptr 引发的 aclnn 失败升级为
// Python 异常。
// ---------------------------------------------------------------------------
namespace acl_type_traits_detail {

// float -> IEEE-754 binary16 位模式（round-to-nearest-even）。
// 与 acl_scalar_arg.h 的 HalfBitsToFloat() 对称；不依赖 op::fp16_t 的
// out-of-line 成员函数。
inline uint16_t FloatToHalfBits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t exp_field = (bits >> 23) & 0xffu;
    uint32_t frac = bits & 0x7fffffu;

    if (exp_field == 0xffu) {  // Inf / NaN
        return static_cast<uint16_t>(sign | 0x7c00u | (frac ? 0x200u : 0x0u));
    }

    const int32_t exp = static_cast<int32_t>(exp_field) - 112;  // -127 + 15
    if (exp >= 0x1f) {  // overflow -> Inf
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    if (exp <= 0) {  // subnormal 或 0
        if (exp < -10) {  // 太小，直接下溢为 +-0
            return sign;
        }
        frac |= 0x800000u;  // 补回隐含的 1
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half = frac >> shift;
        const uint32_t round_bit = 1u << (shift - 1);
        if ((frac & round_bit) && ((frac & (round_bit - 1u)) || (half & 1u))) {
            ++half;
        }
        return static_cast<uint16_t>(sign | half);
    }

    uint16_t half = static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(exp) << 10) | (frac >> 13));
    // round-to-nearest-even；进位可能把 exponent 推到 0x1f -> Inf，正是期望行为
    if ((frac & 0x1000u) && ((frac & 0xfffu) || (half & 1u))) {
        ++half;
    }
    return half;
}

// float -> bfloat16 位模式（op::bfloat16 是 header-only 实现）
inline uint16_t FloatToBfloat16Bits(float value) {
    return op::bfloat16(value).value;
}

// 整数：越界饱和（不 throw，见上方注释）；NaN 退化为 0。
template<typename ToInt>
inline ToInt SaturateInteger(double value, aclDataType dtype) {
    constexpr double lo = static_cast<double>(std::numeric_limits<ToInt>::lowest());
    constexpr double hi = static_cast<double>(std::numeric_limits<ToInt>::max());
    if (std::isnan(value)) {
        std::cerr << "WARNING: aclScalar: NaN is not representable as "
                  << aclDtypeToString(dtype) << ", using 0\n";
        return static_cast<ToInt>(0);
    }
    if (value < lo) {
        std::cerr << "WARNING: aclScalar: " << value << " underflows "
                  << aclDtypeToString(dtype) << ", clamped to " << lo << "\n";
        return std::numeric_limits<ToInt>::lowest();
    }
    if (value > hi) {
        std::cerr << "WARNING: aclScalar: " << value << " overflows "
                  << aclDtypeToString(dtype) << ", clamped to " << hi << "\n";
        return std::numeric_limits<ToInt>::max();
    }
    return static_cast<ToInt>(value);
}

// aclCreateScalar 会把值拷贝进 aclScalar 内联 union，所以调用方的临时变量
// 出了作用域没有关系（同 api/acl_utils.pyx 的说明）。
inline aclScalar* CreateScalarChecked(void* value, aclDataType dtype) {
    aclScalar* scalar = aclCreateScalar(value, dtype);
    if (scalar == nullptr) {
        const char* msg = aclGetRecentErrMsg();
        std::cerr << "ERROR: aclCreateScalar failed for dtype "
                  << aclDtypeToString(dtype) << ": "
                  << (msg != nullptr ? msg : "(aclGetRecentErrMsg is empty)")
                  << "\n";
    }
    return scalar;
}

}  // namespace acl_type_traits_detail

aclScalar* CreateAclScalar(double value, aclDataType dtype) {
    using namespace acl_type_traits_detail;
    switch (dtype) {
        case ACL_FLOAT: {
            float converted = static_cast<float>(value);
            return CreateScalarChecked(&converted, dtype);
        }
        case ACL_DOUBLE:
            // 如果确实需要double，则直接使用（请确保硬件和算子支持ACL_DOUBLE）
            return CreateScalarChecked(&value, dtype);
        case ACL_FLOAT16: {
            uint16_t bits = FloatToHalfBits(static_cast<float>(value));
            return CreateScalarChecked(&bits, dtype);
        }
        case ACL_BF16: {
            uint16_t bits = FloatToBfloat16Bits(static_cast<float>(value));
            return CreateScalarChecked(&bits, dtype);
        }
        case ACL_INT8: {
            int8_t converted = SaturateInteger<int8_t>(value, dtype);
            return CreateScalarChecked(&converted, dtype);
        }
        case ACL_UINT8: {
            uint8_t converted = SaturateInteger<uint8_t>(value, dtype);
            return CreateScalarChecked(&converted, dtype);
        }
        case ACL_INT16: {
            int16_t converted = SaturateInteger<int16_t>(value, dtype);
            return CreateScalarChecked(&converted, dtype);
        }
        case ACL_UINT16: {
            uint16_t converted = SaturateInteger<uint16_t>(value, dtype);
            return CreateScalarChecked(&converted, dtype);
        }
        case ACL_INT32: {
            int32_t converted = SaturateInteger<int32_t>(value, dtype);
            return CreateScalarChecked(&converted, dtype);
        }
        case ACL_UINT32: {
            uint32_t converted = SaturateInteger<uint32_t>(value, dtype);
            return CreateScalarChecked(&converted, dtype);
        }
        case ACL_INT64: {
            int64_t converted = SaturateInteger<int64_t>(value, dtype);
            return CreateScalarChecked(&converted, dtype);
        }
        case ACL_UINT64: {
            uint64_t converted = SaturateInteger<uint64_t>(value, dtype);
            return CreateScalarChecked(&converted, dtype);
        }
        case ACL_BOOL: {
            bool converted = value != 0.0;
            return CreateScalarChecked(&converted, dtype);
        }
        case ACL_COMPLEX64: {
            // 本函数的输入是 double，虚部只能是 0（调用点传的都是实数）
            std::complex<float> converted(static_cast<float>(value), 0.0f);
            return CreateScalarChecked(&converted, dtype);
        }
        case ACL_COMPLEX128: {
            std::complex<double> converted(value, 0.0);
            return CreateScalarChecked(&converted, dtype);
        }
        default:
            // 字符串 / qint / float8 等无法由 double 表达：明确报错，不用默认值
            std::cerr << "ERROR: aclScalar: dtype " << aclDtypeToString(dtype)
                      << " cannot be created from a double, returning nullptr\n";
            return nullptr;
    }
}

#endif
