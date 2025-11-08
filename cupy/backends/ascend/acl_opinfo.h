#ifndef ASCEND_OP_INFO_H
#define ASCEND_OP_INFO_H

#include <string>
#include <iostream>


enum OpType {
    INVALID_OP = -1,
    GENERAL_OP = 0,
    UNARY_OP = 1,
    INPLACE_UNARY_OP = 2,
    REDUCTION_OP = 3,
    BINARY_OP = 4,
    INPLACE_BINARY_OP = 5,
    SCALAR_BINARY_OP = 6,
    INPLACE_SCALAR_BINARY_OP = 7,
    TRI_OP = 8,
    INPLACE_TRI_OP = 9,
};

// OpInfo结构体定义
struct OpInfo {
    std::string op_name;
    OpType op_type;
    
    // 默认构造函数
    OpInfo() : op_name("invalid"), op_type(OpType::INVALID_OP) {}
    
    // 参数化构造函数
    OpInfo(const std::string& name, OpType type) : op_name(name), op_type(type) {}
    
    // 拷贝构造函数
    OpInfo(const OpInfo& other) : op_name(other.op_name), op_type(other.op_type) {}
    
    // 赋值运算符
    OpInfo& operator=(const OpInfo& other) {
        if (this != &other) {
            op_name = other.op_name;
            op_type = other.op_type;
        }
        return *this;
    }
    
    // C++17及以下版本的比较运算符
    bool operator==(const OpInfo& other) const {
        return op_name == other.op_name && op_type == other.op_type;
    }
    
    bool operator!=(const OpInfo& other) const {
        return !(*this == other);
    }
    
    // 小于运算符（用于std::map排序）
    bool operator<(const OpInfo& other) const {
        if (op_name != other.op_name) {
            return op_name < other.op_name;
        }
        return static_cast<int>(op_type) < static_cast<int>(other.op_type);
    }
    
    bool operator>(const OpInfo& other) const {
        return other < *this;
    }
    
    bool operator<=(const OpInfo& other) const {
        return !(other < *this);
    }
    
    bool operator>=(const OpInfo& other) const {
        return !(*this < other);
    }
    
    // 输出运算符（用于调试）
    friend std::ostream& operator<<(std::ostream& os, const OpInfo& op) {
        os << "OpInfo{name: " << op.op_name << ", type: " << static_cast<int>(op.op_type) << "}";
        return os;
    }
    
    // 哈希函数支持（用于std::unordered_map）
    struct Hash {
        std::size_t operator()(const OpInfo& op) const {
            std::size_t h1 = std::hash<std::string>{}(op.op_name);
            std::size_t h2 = std::hash<int>{}(static_cast<int>(op.op_type));
            return h1 ^ (h2 << 1);
        }
    };
};

#endif // OP_INFO_H