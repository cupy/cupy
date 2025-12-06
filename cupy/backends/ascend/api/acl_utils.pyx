import cython
cimport cpython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from collections import namedtuple

from cupy._core import _dtype
from cupy._core.core import _ndarray_base
from libc.stdint cimport int32_t, int16_t
from libcpp.unordered_map cimport unordered_map as cpp_map
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.vector cimport vector
from libcpp.complex cimport complex

import threading as _threading

from cupy.xpu import stream as stream_module

ASCEND_OP_PREFIX = "ascend_"

# 为vector[const aclScalar*]&创建类型别名
ctypedef vector[const aclScalar*] ArgsType
ctypedef cpp_map[string, const aclScalar*] KwargsType

# 4. 为迭代器创建别名（便于遍历）
ctypedef vector[const aclScalar*].iterator ArgsIterator
ctypedef cpp_map[string, const aclScalar*].const_iterator KargsConstIterator
#ctypedef pair[const string, const aclScalar*] KwargsItem

#include "backends/ascend/api/acl_types.pxi" # already included in pxd file

cdef extern from "aclnn/opdev/common_types.h" nogil:
    cdef cppclass aclTensor # declare/import externally declared C++ class
    aclTensor* aclCreateTensor(
        const int64_t* viewDims,
        uint64_t viewDimsNum,
        aclDataType dataType,
        const int64_t* stride,
        int64_t offset,
        aclFormat format,
        const int64_t* storageDims,
        uint64_t storageDimsNum,
        void* tensorData
    )
    aclScalar* aclCreateScalar(void* value, aclDataType dataType)
    aclIntArray* aclCreateIntArray(const int64_t *value, uint64_t size)

    aclnnStatus aclDestroyTensor(const aclTensor *tensor)
    aclnnStatus aclDestroyScalar(const aclScalar *scalar)
    aclnnStatus aclDestroyIntArray(const aclIntArray *array)
    const char *aclGetRecentErrMsg()

cdef aclDataType numpy_to_acl_dtype(dtype,
    bint is_half_allowed=True, bint is_double_supported=True):
    # double and complex128 is not supported on ASCEND910
    cdef str dtype_char
    try:
        dtype_char = dtype.char
    except AttributeError:
        dtype_char = dtype

    if dtype_char == 'e':
        return aclDataType.ACL_FLOAT16
    elif dtype_char == 'E' and is_half_allowed:
        # complex32, bfloat16 not supported in NumPy
        return aclDataType.ACL_COMPLEX32
    elif dtype_char == 'f':
        return aclDataType.ACL_FLOAT # float32
    elif dtype_char == 'F':
        return aclDataType.ACL_COMPLEX64
    elif dtype_char == 'd' and is_double_supported:
        return aclDataType.ACL_DOUBLE
    elif dtype_char == 'D' and is_double_supported:
        return aclDataType.ACL_COMPLEX128
    elif dtype_char == 'b':
        return aclDataType.ACL_INT8
    elif dtype_char == 'B':
        return aclDataType.ACL_UINT8
    elif dtype_char == 'h':
        return aclDataType.ACL_INT16
    elif dtype_char == 'H':
        return aclDataType.ACL_UINT16
    elif dtype_char == 'i':
        return aclDataType.ACL_INT32
    elif dtype_char == 'I':
        return aclDataType.ACL_UINT32
    elif (dtype_char == 'q' or dtype_char == 'l') and is_double_supported:
        return aclDataType.ACL_INT64
    elif dtype_char == 'Q' and is_double_supported:
        return aclDataType.ACL_UINT64
    elif dtype_char == '?':
        return aclDataType.ACL_BOOL
    else:
        print('ASCEND: DEBUG dtype is not supported: {}'.format(dtype))
        return aclDataType.ACL_DT_UNDEFINED


cdef aclScalar* cupy_scalar_to_acl_scalar(_cupy_scalar s) except*:
    """
    将 CuPy 标量对象转换为 aclScalar。
    参数:
        s: CuPy 标量对象，应具有 `dtype` 属性和数据指针访问方式。
    返回:
        aclScalar: 转换后的 ACL 标量。
    异常:
        TypeError: 如果输入不是预期的 CuPy 标量类型。
        ValueError: 如果 dtype 转换失败或数据指针无效。
    """
    cdef void* value_ptr = NULL
    cdef aclScalar* acl_scalar = NULL
    cdef aclDataType dtype
    cdef string msg
    
    try:
        # 根据数据类型分配内存并复制值
        #TODO: "S" for string, "O" for object, "C" for complex
        if s.kind == 'i' and s.size == 8:  # 整数类型
            value_ptr = PyMem_Malloc(sizeof(int64_t))
            if value_ptr == NULL:
                raise MemoryError("Failed to allocate memory for integer64 scalar")
            (<int64_t*>value_ptr)[0] = (<int64_t*>s.ptr)[0]
            dtype = ACL_INT64
        elif s.kind == 'i' and s.size == 4:  # 整数类型
            value_ptr = PyMem_Malloc(sizeof(int32_t))
            if value_ptr == NULL:
                raise MemoryError("Failed to allocate memory for integer32 scalar")
            (<int32_t*>value_ptr)[0] = (<int32_t*>s.ptr)[0]
            dtype = ACL_INT32
        elif s.kind == 'i' and s.size == 2:  # 整数类型
            value_ptr = PyMem_Malloc(sizeof(int16_t))
            if value_ptr == NULL:
                raise MemoryError("Failed to allocate memory for integer32 scalar")
            (<int16_t*>value_ptr)[0] = (<int16_t*>s.ptr)[0]
            dtype = ACL_INT16
        elif s.kind == 'u':  # unsign 整数类型
            raise TypeError("Unsigned integer scalar is not supported yet, TODO")
        elif s.kind == 'f' and s.size == 8:  # 浮点类型
            value_ptr = PyMem_Malloc(sizeof(double))
            if value_ptr == NULL:
                raise MemoryError("Failed to allocate memory for float64 scalar")
            (<double*>value_ptr)[0] = (<double*>s.ptr)[0]
            dtype = ACL_DOUBLE
        elif s.kind == 'f' and s.size == 4:  # 浮点类型
            value_ptr = PyMem_Malloc(sizeof(float))
            if value_ptr == NULL:
                raise MemoryError("Failed to allocate memory for float32 scalar")
            (<float*>value_ptr)[0] = (<float*>s.ptr)[0]
            dtype = ACL_FLOAT
        elif s.kind == 'f' and s.size == 2:  # 浮点类型
            value_ptr = PyMem_Malloc(sizeof(unsigned short))
            if value_ptr == NULL:
                raise MemoryError("Failed to allocate memory for float16 scalar")
            (<float*>value_ptr)[0] = (<unsigned short*>s.ptr)[0]
            dtype = ACL_FLOAT16
        elif s.kind == 'C':  # complex
            if s.size == 8:
                value_ptr = PyMem_Malloc(s.size)
                if value_ptr == NULL:
                    raise MemoryError("Failed to allocate memory for complex64 scalar")
                (<complex[float]*>value_ptr)[0] = (<complex[float]*>s.ptr)[0]
                dtype = ACL_COMPLEX64
            elif s.size == 16:
                value_ptr = PyMem_Malloc(s.size)
                if value_ptr == NULL:
                    raise MemoryError("Failed to allocate memory for complex128 scalar")
                (<complex[double]*>value_ptr)[0] = (<complex[double]*>s.ptr)[0]
                dtype = ACL_COMPLEX128
            else:
                raise TypeError("Complex scalar is not supported yet, TODO")
        elif s.kind == 'S':  # string type
            raise TypeError("string scalar is not supported yet, TODO")
        elif s.kind == 'b':  # bool
            value_ptr = PyMem_Malloc(sizeof(bint))
            if value_ptr == NULL:
                raise MemoryError("Failed to allocate memory for bool scalar")
            (<bint*>value_ptr)[0] = (<bint*>s.ptr)[0]
            dtype = ACL_BOOL
        else:
            raise TypeError(f"Unsupported dtype kind: {s.kind}")
        
        acl_scalar = aclCreateScalar(value_ptr, dtype)
        if acl_scalar == NULL:
            msg = aclGetRecentErrMsg()
            raise RuntimeError("Failed to create aclScalar with error: %s", msg)
        return acl_scalar
    except Exception as e:
        # 异常处理：确保资源清理
        if value_ptr != NULL:
            PyMem_Free(value_ptr)
        if acl_scalar != NULL:
            aclDestroyScalar(acl_scalar)
        raise MemoryError("Failed to create aclScalar with error %s" % e)

cdef KwargsType _create_keyword_args(dict kwargs) except *:
    cdef KwargsType acl_kwargs
    if kwargs:
        for key, value in kwargs.items():
            acl_kwargs[key] = cupy_scalar_to_acl_scalar(value)
    return acl_kwargs

cdef void _delete_keyword_args(KwargsType& my_map) except *:
    cdef:
        # 使用非常量迭代器，因为我们需要修改map（删除元素）
        cpp_map[string, const aclScalar*].iterator it = my_map.begin()
        cpp_map[string, const aclScalar*].iterator end = my_map.end()
        const aclScalar* scalar_ptr

    # 安全遍历并删除
    while it != end:
        scalar_ptr = deref(it).second
        if scalar_ptr != NULL:
            aclDestroyScalar(<const aclScalar*>scalar_ptr)

        # 3. 将迭代器指向下一个元素，并擦除当前元素。
        #    it = my_map.erase(it) 会返回指向下一个有效元素的迭代器，这是安全的方法。
        it = my_map.erase(it)

cdef aclTensor* cupy_ndarray_to_acl_tensor(_ndarray_base cupy_array) except *:
    """
    将CuPy _ndarray_base转换为ACL Tensor
    
    Args:
        cupy_array: CuPy数组基类对象
        
    Returns:
        aclTensor*: 指向创建的ACL Tensor的指针
    """
    cdef:
        int64_t* view_dims = NULL
        int64_t* strides = NULL
        int64_t* storage_dims = NULL
        aclTensor* acl_tensor = NULL
        aclDataType data_type
        aclFormat format = ACL_FORMAT_ND
        int64_t offset = 0
        void* tensor_data = NULL
        int i
        int64_t ndim
    
    try:
        # 1. 获取CuPy数组的形状和维度
        ndim = len(cupy_array._shape) # len() works for cython 3.1+ only
        
        # 分配内存用于存储维度信息
        view_dims = <int64_t*>PyMem_Malloc(ndim * sizeof(int64_t))
        strides = <int64_t*>PyMem_Malloc(ndim * sizeof(int64_t))
        storage_dims = <int64_t*>PyMem_Malloc(ndim * sizeof(int64_t))
        
        if view_dims == NULL or strides == NULL or storage_dims == NULL:
            raise MemoryError("Failed to allocate memory for dimension arrays")
        
        # 填充维度信息
        for i in range(ndim):
            view_dims[i] = cupy_array._shape[i]
            # aclTensor strids using element size, not the byte size
            strides[i] = cupy_array._strides[i] / cupy_array.dtype.itemsize
            storage_dims[i] = cupy_array._shape[i]  # 假设存储形状与视图形状相同
        
        # 2. 映射数据类型
        data_type = numpy_to_acl_dtype(cupy_array.dtype)
        if data_type == ACL_DT_UNDEFINED:
            raise ValueError(f"Unsupported dtype: {cupy_array.dtype}")
        
        # 3. 获取数据指针
        tensor_data = <void*>cupy_array.data.ptr
        
        # 4. 根据内存布局选择合适的格式
        if cupy_array._c_contiguous:
            format = ACL_FORMAT_ND
        elif cupy_array._f_contiguous:
            format = ACL_FORMAT_NHWC  # 假设Fortran连续对应NHWC格式
        else:
            format = ACL_FORMAT_ND  # 非连续数组使用默认格式
        
        # 5. 创建ACL Tensor
        acl_tensor = aclCreateTensor(
            view_dims,      # 逻辑形状
            ndim,           # 维度数量
            data_type,      # 数据类型
            strides,        # 步长
            offset,         # 偏移量
            format,         # 数据布局格式
            storage_dims,   # 物理存储形状
            ndim,           # 存储维度数量
            tensor_data     # 数据指针（直接使用CuPy内存）
        )
        
        if acl_tensor == NULL:
            raise RuntimeError("Failed to create ACL tensor")
        
        return acl_tensor
        
    except Exception as e:
        # 清理分配的内存
        if view_dims != NULL:
            PyMem_Free(view_dims)
        if strides != NULL:
            PyMem_Free(strides)
        if storage_dims != NULL:
            PyMem_Free(storage_dims)
        if acl_tensor != NULL:
            aclDestroyTensor(acl_tensor)
        raise e


cdef extern from "../acl_opinfo.h":
    # 操作类型枚举
    cdef enum OpType:
        INVALID_OP = -1
        GENERAL_OP = 0
        UNARY_OP = 1
        INPLACE_UNARY_OP = 2
        REDUCTION_OP = 3
        BINARY_OP = 4
        INPLACE_BINARY_OP = 5
        SCALAR_BINARY_OP = 6
        INPLACE_SCALAR_BINARY_OP = 7
        TERNARY_OP = 8
        INPLACE_TERNARY_OP = 9

    cdef cppclass OpInfo:
        # 构造函数
        OpInfo() except +
        OpInfo(string op_name, OpType op_type) except +
        
        # 成员变量
        string op_name
        OpType op_type
        
        # 比较运算符
        bint operator==(const OpInfo& other) const
        bint operator!=(const OpInfo& other) const
        bint operator<(const OpInfo& other) const
        bint operator>(const OpInfo& other) const
        bint operator<=(const OpInfo& other) const
        bint operator>=(const OpInfo& other) const

        cppclass Hash:
            size_t operator()(const OpInfo& op) const

# TODO: thread safety?
# operator_function_ptr registry:
#cdef cpp_map[OpInfo, FuncPtrUnion] _builtin_operators
# unordered map for better performance
cdef cpp_map[OpInfo, FuncPtrUnion, OpInfo.Hash] _builtin_operators

cdef extern from "<cstdbool>" namespace "std":
    ctypedef bint bool "bool"  # 将C++的bool映射到Cython的bint

############################## 定义函数指针类型 ###########################################
ctypedef aclError (*TernaryOpFunc)(const aclTensor* self, const aclTensor* other,
    const aclTensor* other2, aclTensor* out, aclrtStream stream)
ctypedef aclError (*InplaceTernaryOpFunc)(aclTensor* self, const aclTensor* other, aclrtStream stream)

ctypedef aclError (*BinaryOpFunc)(const aclTensor* self, const aclTensor* other,
    aclTensor* out, aclrtStream stream)
ctypedef aclError (*InplaceBinaryOpFunc)(aclTensor* self, const aclTensor* other, aclrtStream stream)
ctypedef aclError (*ScalarBinaryOpFunc)(const aclTensor* self, const aclScalar* other,
    aclTensor* out, aclrtStream stream) 
ctypedef aclError (*InplaceScalarBinaryOpFunc)(aclTensor* self, const aclScalar* other, aclrtStream stream)

ctypedef aclError (*UnaryOpFunc)(const aclTensor* self, aclTensor* out, aclrtStream stream)
ctypedef aclError (*InplaceUnaryOpFunc)(aclTensor* self, aclrtStream stream)


###########################################################################################


# aclTensorList is not convenient to use in C++, so use std::vector directly
ctypedef aclError (*GeneralOpFunc)(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
    const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)

ctypedef aclError (*ReductionOpFunc)(const aclTensor* self, const aclIntArray* dim, bool keepdim,
    aclTensor* out, const KwargsType& kwargs, aclrtStream stream)

# 函数指针联合体，用于存储不同类型的操作
ctypedef union FuncPtrUnion:
    UnaryOpFunc unary_op
    InplaceUnaryOpFunc inplace_unary_op
    BinaryOpFunc binary_op
    InplaceBinaryOpFunc inplace_binary_op
    ScalarBinaryOpFunc scalar_binary_op
    InplaceScalarBinaryOpFunc inplace_scalar_binary_op
    TernaryOpFunc tri_op
    InplaceTernaryOpFunc inplace_tri_op
    ReductionOpFunc reduction_op
    GeneralOpFunc general_op

cdef aclError register_acl_ufunc(string opname, OpType op_type, FuncPtrUnion func_ptr) except * nogil:
    cdef OpInfo op_info
    op_info.op_name = opname
    op_info.op_type = op_type
    
    if _builtin_operators.find(op_info) != _builtin_operators.end():
        # 操作已存在，可以选择覆盖或报错, 这里我们选择覆盖
        _builtin_operators[op_info] = func_ptr
        return 0 # ACL_SUCCESS
    else:
        _builtin_operators[op_info] = func_ptr
        return 0

cdef OpType get_op_type(object ops, bint inplace, bint has_scalar = False):
    # TODO: Ternary op, has_scalar
    if has_scalar:
        if len(ops) == 3 and not inplace:  # 二元操作
            return SCALAR_BINARY_OP
        elif len(ops) == 2 and inplace:  # 原地二元操作
            return INPLACE_SCALAR_BINARY_OP  
    else:
        if len(ops) == 3 and not inplace:  # 二元操作
            return BINARY_OP
        elif len(ops) == 2 and inplace:  # 原地二元操作
            return INPLACE_BINARY_OP  
        elif len(ops) == 2 and not inplace:  # 一元操作
            return UNARY_OP
        elif len(ops) == 1 and inplace:  # 原地一元操作
            return INPLACE_UNARY_OP
        raise RuntimeError("Operator type can not be decided")
    return INVALID_OP

cdef aclError launch_general_func(str opname, sequence ins, sequence outs, list args, dict kwargs, intptr_t stream_ptr) except *:
    if opname.startswith("cupy_"):
        opname = ASCEND_OP_PREFIX + opname[5:]
    cdef OpInfo op_info
    cdef FuncPtrUnion func_ptr
    op_info.op_name = opname.encode("utf-8")
    op_info.op_type = OpType.GENERAL_OP
    if _builtin_operators.find(op_info) == _builtin_operators.end():
        return launch_acl_func(opname, ins, outs, args, kwargs, stream_ptr)
    func_ptr = _builtin_operators[op_info]

    cdef ArgsType acl_args
    cdef KwargsType acl_kwargs = _create_keyword_args(kwargs)
    cdef const aclTensor* ct
    cdef vector[const aclTensor*] intensors
    for op in ins:
        typ = type(op)
        if issubclass(typ, _ndarray_base):
            intensors.push_back(cupy_ndarray_to_acl_tensor(op))
        elif issubclass(typ, _ndarray_base):
            acl_args.push_back(cupy_scalar_to_acl_scalar(op))
    for cupy_scalar in args:
        acl_args.push_back(cupy_scalar_to_acl_scalar(cupy_scalar))

    cdef vector[aclTensor*] outtensors
    for op in outs:
        typ = type(op)
        if issubclass(typ, _ndarray_base):
            outtensors.push_back(cupy_ndarray_to_acl_tensor(op)) 

    cdef aclError ret = 0
    cdef aclrtStream stream = <aclrtStream>NULL  # default stream always working
    if stream_ptr != <intptr_t>0:
        stream = <aclrtStream>stream_ptr

    try:
        ret = func_ptr.general_op(intensors, outtensors, acl_args, acl_kwargs, stream)
    finally:
        # aclDestroyTensor does not deallocate array buffer, but shapes, strides
        for i in range(intensors.size()):
            ct = <const aclTensor*>intensors.at(i)
            aclDestroyTensor(ct)
        for t in outtensors:
            aclDestroyTensor(t)
        for acl_scalar in acl_args:
            aclDestroyScalar(acl_scalar)
        _delete_keyword_args(acl_kwargs)
        if ret != 0:
            print("Failed to run the operator: ", opname)
    return ret

cdef vector[aclTensor*] _create_ops_vector(sequence ins, sequence outs) except *:
    cdef vector[aclTensor*] tensors

    for op in ins:
        typ = type(op)
        if issubclass(typ, _ndarray_base):
            tensors.push_back(cupy_ndarray_to_acl_tensor(op))
        elif typ is _cupy_scalar:
            pass # scalar_ptr has been processed above
        else:
            raise RuntimeError("Operand is not ndarray or scalar: ", op)

    for op in outs: # out is tensor
        typ = type(op)
        if issubclass(typ, _ndarray_base):
            tensors.push_back(cupy_ndarray_to_acl_tensor(op))
        else:
            raise RuntimeError("Operand is not ndarray: ", op)

    return tensors

cdef aclError launch_acl_func(str opname, sequence ins, sequence outs, list args, dict kwargs, intptr_t stream_ptr) except *:
    # 
    cdef aclScalar* scalar_ptr = NULL
    cdef OpInfo op_info
    cdef FuncPtrUnion func_ptr
    op_info.op_name = opname.encode("utf-8")
    op_info.op_type = OpType.GENERAL_OP

    # 区分scalar 和tensor 操作数, 应该是Broadcast应该处理的事情
    # inplace op 是ASCEND引入的?
    for op in ins:
        typ = type(op)
        if typ is _cupy_scalar:
            scalar_ptr = cupy_scalar_to_acl_scalar(op)

    cdef has_scalar = scalar_ptr != NULL
    # cupy inplace op does not generate a new op, but make self == out
    cdef bint inplace = ("inplace" in opname) or not outs
    cdef list ops = ins + outs

    op_info.op_type = get_op_type(ops, inplace, has_scalar)
    if _builtin_operators.find(op_info) == _builtin_operators.end():
        raise KeyError(f"Operator {opname} len(ops) = {len(ops)} not registered {inplace} {has_scalar}, {op_info.op_type}")
    
    func_ptr = _builtin_operators[op_info]
    cdef aclError ret = 0
    cdef aclrtStream stream = <aclrtStream>NULL  # default stream always working
    if stream_ptr != <intptr_t>0:
        stream = <aclrtStream>stream_ptr

    # 转换为ACL张量列表
    tensors = _create_ops_vector(ops, outs)

    try:
        if len(ops) == 3 and not has_scalar and not inplace:  # 二元操作
            if op_info.op_type != BINARY_OP:
                raise RuntimeError(f"Operator {opname} is not a binary operator")
            ret = func_ptr.binary_op(tensors[0], tensors[1], tensors[2], stream)
        elif len(ops) == 2 and inplace:  # 原地二元操作
            print(f"ASCEND DEBUG: inplace operator {opname} called")
            if op_info.op_type != INPLACE_BINARY_OP:
                raise RuntimeError(f"Operator {opname} is not an inplace binary operator")
            ret = func_ptr.inplace_binary_op(tensors[0], tensors[1], stream)

        elif len(ops) == 3 and has_scalar and not inplace:  #  out = self <biop> scalar
            if op_info.op_type != SCALAR_BINARY_OP:
                raise RuntimeError(f"Operator {opname} is not an scalar binary operator")
            ret = func_ptr.scalar_binary_op(tensors[0], scalar_ptr, tensors[1], stream)
        elif len(ops) == 2 and has_scalar:  #  out = self <biop> scalar
            if op_info.op_type != INPLACE_SCALAR_BINARY_OP:
                raise RuntimeError(f"Operator {opname} is not an inplace scalar binary operator")
            ret = func_ptr.inplace_scalar_binary_op(tensors[0], scalar_ptr, stream)
        
        elif len(ops) == 2 and not inplace:  # 一元操作
            if op_info.op_type != UNARY_OP:
                raise RuntimeError(f"Operator {opname} and is not a unary operator")
            ret = func_ptr.unary_op(tensors[0], tensors[1], stream)
        
        elif len(ops) == 1 and inplace:  # 原地一元操作
            if op_info.op_type != INPLACE_UNARY_OP:
                raise RuntimeError(f"Operator {opname.decode('utf-8')} is not an inplace unary operator")
            ret = func_ptr.inplace_unary_op(tensors[0], stream)
        else:
            raise RuntimeError("Invalid number of operands or inplace flag")
            # TODO:  std::runtime_error() with nogil
    finally:
        # does not deallocate array buffer, but shapes, strides
        for t in tensors:
            aclDestroyTensor(t)  # 假设destroyAclTensor函数已存在
        if scalar_ptr:
            aclDestroyScalar(scalar_ptr)

        if ret != 0:
            print("Failed to run the operator ", opname)
    return ret


cdef aclError launch_reduction_op(str opname, sequence ins, sequence outs, object axes, bint keepdims, dict kwargs, intptr_t stream_ptr) except *:
    # 检查操作是否已注册
    if opname.startswith("cupy_"):
        opname = ASCEND_OP_PREFIX + opname[5:]

    cdef OpInfo op_info
    op_info.op_name = opname.encode("utf-8")
    op_info.op_type = REDUCTION_OP
    if _builtin_operators.find(op_info) == _builtin_operators.end():
        raise KeyError(f"Operator {opname} not registered")

    cdef FuncPtrUnion func_ptr = _builtin_operators[op_info]
    cdef aclError ret = 0
    cdef aclrtStream stream = <aclrtStream>NULL  # default stream always working
    if stream_ptr != <intptr_t>0:
        stream = <aclrtStream>stream_ptr

    cdef vector[int64_t] shape
    cdef aclIntArray* dim = NULL
    cdef vector[aclTensor*] tensors

    tensors = _create_ops_vector(ins, outs)

    typ = type(axes) 
    if hasattr(axes, 'size') and hasattr(axes, 'push_back'):
        # dim/axes info from `shape_t` which is `vector.vector[Py_ssize_t]`
        for i in range(axes.size()):
            shape.push_back(axes[i])
    elif typ is _cupy_scalar:
        pass
    elif axes is None:
        shape.push_back(0)
    elif typ is int: # TODO, not sure if it works/compilable
        shape.push_back(axes)
    else:
        raise RuntimeError("axis/axis is not tuple, shape, int type", axes)

    if not shape.size():
        shape.push_back(0)
    dim = aclCreateIntArray(shape.data(), shape.size())
    cdef KwargsType acl_kwargs = _create_keyword_args(kwargs)
    try:
        ret = func_ptr.reduction_op(tensors[0], dim, keepdims, tensors[1], acl_kwargs, stream)
        if ret != 0:
            print("Failed to run the reduction operator ", opname)
    finally:
        # does not deallocate array buffer, but shapes, strides
        for t in tensors:
            aclDestroyTensor(t)  # 假设destroyAclTensor函数已存在
        if dim:
            aclDestroyIntArray(dim)
        _delete_keyword_args(acl_kwargs)
    return ret

cdef extern from "../acl_math_ops.h" nogil:
    aclError aclop_BitwiseAndTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceBitwiseAndTensor(aclTensor* self, const aclTensor* other, aclrtStream stream)
    aclError aclop_BitwiseAndScalar(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceBitwiseAndScalar(aclTensor* self, const aclScalar* other, aclrtStream stream)

    aclError aclop_BitwiseOrTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceBitwiseOrTensor(aclTensor* self, const aclTensor* other, aclrtStream stream)
    aclError aclop_BitwiseXorTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceBitwiseXorTensor(aclTensor* self, const aclTensor* other, aclrtStream stream)
    aclError aclop_BitwiseNot(const aclTensor* self, aclTensor* out, aclrtStream stream) # no inplace version

    aclError aclop_LogicalAnd(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_LogicalXor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_LogicalOr(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_LogicalNot(const aclTensor* self, aclTensor* out, aclrtStream stream)

    aclError aclop_GeTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_LeTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_GtTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_LtTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_NeTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_EqTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    # scalar operand
    aclError aclop_GeScalar(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_LeScalar(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_GtScalar(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_LtScalar(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_EqScalar(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_NeScalar(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)

    aclError aclop_IsInf(const aclTensor* self, aclTensor* out, aclrtStream stream)
    aclError aclop_IsPosInf(const aclTensor* self, aclTensor* out, aclrtStream stream)
    aclError aclop_IsNegInf(const aclTensor* self, aclTensor* out, aclrtStream stream)
    aclError aclop_IsFinite(const aclTensor* self, aclTensor* out, aclrtStream stream)
    #############################################################################
    aclError aclop_Add(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceAdd(aclTensor* self, const aclTensor* other, aclrtStream stream)
    aclError aclop_Sub(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceSub(aclTensor* self, const aclTensor* other, aclrtStream stream)
    aclError aclop_Mul(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceMul(aclTensor* self, const aclTensor* other, aclrtStream stream)
    aclError aclop_Div(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceDiv(aclTensor* self, const aclTensor* other, aclrtStream stream)
    aclError aclop_FloorDivide(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_FmodTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)

    aclError aclop_Maximum(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_Minimum(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_Gcd(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_lcm(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_PowTensorTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_RemainderTensorTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)

    aclError aclop_Adds(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_Subs(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_Muls(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_Divs(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_PowTensorScalar(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_RemainderTensorScalar(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_FmodScalar(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)

    aclError aclop_Neg(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceNeg(aclTensor* self,  aclrtStream stream)
    aclError aclop_Reciprocal(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceReciprocal(aclTensor* self,  aclrtStream stream)
    aclError aclop_Signbit(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Sign(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Abs(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Floor(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceFloor(aclTensor* self,  aclrtStream stream)
    aclError aclop_Ceil(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceCeil(aclTensor* self,  aclrtStream stream)

    aclError aclop_Square(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Rsqrt(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Deg2rad(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Rad2deg(const aclTensor* self,  aclTensor* out, aclrtStream stream)

    aclError aclop_Exp(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Exp2(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Expm1(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Log(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Log2(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Log10(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Log1p(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_LogAddExp(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_LogAddExp2(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)

    aclError aclop_Matmul(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_Dot(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)

    aclError aclop_Cos(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceCos(aclTensor* self,  aclrtStream stream)
    aclError aclop_Sin(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceSin(aclTensor* self,  aclrtStream stream)
    aclError aclop_Tan(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceTan(aclTensor* self,  aclrtStream stream)

    aclError aclop_Acos(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceAcos(aclTensor* self,  aclrtStream stream)
    aclError aclop_Asin(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceAsin(aclTensor* self,  aclrtStream stream)
    aclError aclop_Atan(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceAtan(aclTensor* self,  aclrtStream stream)

    aclError aclop_Cosh(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceCosh(aclTensor* self,  aclrtStream stream)
    aclError aclop_Sinh(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceSinh(aclTensor* self,  aclrtStream stream)
    aclError aclop_Tanh(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceTanh(aclTensor* self,  aclrtStream stream)

    aclError aclop_Atan2(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_Sinc(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Erf(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Erfc(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Erfinv(const aclTensor* self,  aclTensor* out, aclrtStream stream)

# 初始化函数，注册内置操作
cdef void register_math_operators():
    cdef FuncPtrUnion func_union

    ###################################
    # 注册aclop_BitwiseAnd作为二元操作
    func_union.binary_op = aclop_BitwiseAndTensor
    register_acl_ufunc("ascend_bitwise_and", BINARY_OP, func_union)
    func_union.inplace_binary_op = aclop_InplaceBitwiseAndTensor
    register_acl_ufunc("ascend_inplace_bitwise_and", INPLACE_BINARY_OP, func_union)

    func_union.binary_op = aclop_BitwiseOrTensor
    register_acl_ufunc("ascend_bitwise_or", BINARY_OP, func_union)
    func_union.inplace_binary_op = aclop_InplaceBitwiseOrTensor
    register_acl_ufunc("ascend_inplace_bitwise_or", INPLACE_BINARY_OP, func_union)

    func_union.binary_op = aclop_BitwiseXorTensor
    register_acl_ufunc("ascend_bitwise_xor", BINARY_OP, func_union)
    func_union.inplace_binary_op = aclop_InplaceBitwiseXorTensor
    register_acl_ufunc("ascend_inplace_bitwise_xor", INPLACE_BINARY_OP, func_union)

    func_union.unary_op = aclop_BitwiseNot
    register_acl_ufunc("ascend_bitwise_not", UNARY_OP, func_union)
    # func_union.inplace_unary_op = aclop_InplaceBitwiseNotTensor
    # register_acl_ufunc("ascend_inplace_bitwise_not", INPLACE_UNARY_OP, func_union)

    # 注册aclop_BitwiseAndScalar作为原地二元操作
    func_union.scalar_binary_op = aclop_BitwiseAndScalar
    register_acl_ufunc("ascend_bitwise_and", SCALAR_BINARY_OP, func_union)
    func_union.inplace_scalar_binary_op = aclop_InplaceBitwiseAndScalar
    register_acl_ufunc("ascend_inplace_bitwise_and", INPLACE_SCALAR_BINARY_OP, func_union)

    func_union.binary_op = aclop_LogicalAnd
    register_acl_ufunc("ascend_logical_and", BINARY_OP, func_union)
    func_union.binary_op = aclop_LogicalOr
    register_acl_ufunc("ascend_logical_or", BINARY_OP, func_union)
    func_union.binary_op = aclop_LogicalXor
    register_acl_ufunc("ascend_logical_xor", BINARY_OP, func_union)
    func_union.unary_op = aclop_LogicalNot
    register_acl_ufunc("ascend_logical_not", UNARY_OP, func_union)

    func_union.binary_op = aclop_GeTensor
    register_acl_ufunc("ascend_greater_equal", BINARY_OP, func_union)
    func_union.scalar_binary_op = aclop_GeScalar
    register_acl_ufunc("ascend_greater_equal", SCALAR_BINARY_OP, func_union)
    func_union.binary_op = aclop_LeTensor
    register_acl_ufunc("ascend_less_equal", BINARY_OP, func_union)
    func_union.scalar_binary_op = aclop_LeScalar
    register_acl_ufunc("ascend_less_equal", SCALAR_BINARY_OP, func_union)
    func_union.binary_op = aclop_GtTensor
    register_acl_ufunc("ascend_greater", BINARY_OP, func_union)
    func_union.scalar_binary_op = aclop_GtScalar
    register_acl_ufunc("ascend_greater", SCALAR_BINARY_OP, func_union)
    func_union.binary_op = aclop_LtTensor
    register_acl_ufunc("ascend_less", BINARY_OP, func_union)
    func_union.scalar_binary_op = aclop_LtScalar
    register_acl_ufunc("ascend_less", SCALAR_BINARY_OP, func_union)
    func_union.binary_op = aclop_EqTensor
    register_acl_ufunc("ascend_equal", BINARY_OP, func_union)
    func_union.scalar_binary_op = aclop_EqScalar
    register_acl_ufunc("ascend_equal", SCALAR_BINARY_OP, func_union)
    func_union.binary_op = aclop_NeTensor
    register_acl_ufunc("ascend_not_equal", BINARY_OP, func_union)
    func_union.scalar_binary_op = aclop_NeScalar
    register_acl_ufunc("ascend_not_equal", SCALAR_BINARY_OP, func_union)
    #############################################
    # 注册aclop_Add作为二元操作
    func_union.binary_op = aclop_Add
    register_acl_ufunc("ascend_add", BINARY_OP, func_union)
    func_union.inplace_binary_op = aclop_InplaceAdd
    register_acl_ufunc("ascend_inplace_add", INPLACE_BINARY_OP, func_union)
    func_union.binary_op = aclop_Sub
    register_acl_ufunc("ascend_subtract", BINARY_OP, func_union)
    func_union.inplace_binary_op = aclop_InplaceSub
    register_acl_ufunc("ascend_inplace_substract", INPLACE_BINARY_OP, func_union)
    func_union.binary_op = aclop_Mul
    register_acl_ufunc("ascend_multiply", BINARY_OP, func_union)
    func_union.inplace_binary_op = aclop_InplaceMul
    register_acl_ufunc("ascend_inplace_multiply", INPLACE_BINARY_OP, func_union)
    func_union.binary_op = aclop_Div
    register_acl_ufunc("ascend_true_divide", BINARY_OP, func_union)
    func_union.binary_op = aclop_FloorDivide
    register_acl_ufunc("ascend_floor_divide", BINARY_OP, func_union)
    func_union.inplace_binary_op = aclop_InplaceDiv
    register_acl_ufunc("ascend_inplace_divide", INPLACE_BINARY_OP, func_union)
    func_union.binary_op = aclop_FmodTensor
    register_acl_ufunc("ascend_fmod", BINARY_OP, func_union)

    func_union.scalar_binary_op = aclop_Adds
    register_acl_ufunc("ascend_add", SCALAR_BINARY_OP, func_union)
    func_union.scalar_binary_op = aclop_Subs
    register_acl_ufunc("ascend_sub", SCALAR_BINARY_OP, func_union)
    func_union.scalar_binary_op = aclop_Muls
    register_acl_ufunc("ascend_multiply", SCALAR_BINARY_OP, func_union)
    func_union.scalar_binary_op = aclop_Divs
    register_acl_ufunc("ascend_true_divide", SCALAR_BINARY_OP, func_union)
    func_union.scalar_binary_op = aclop_FmodScalar
    register_acl_ufunc("ascend_fmod", SCALAR_BINARY_OP, func_union)

    func_union.binary_op = aclop_Maximum
    register_acl_ufunc("ascend_maximum", BINARY_OP, func_union)
    func_union.binary_op = aclop_Minimum
    register_acl_ufunc("ascend_minimum", BINARY_OP, func_union)
    func_union.binary_op = aclop_Gcd
    register_acl_ufunc("ascend_gcd", BINARY_OP, func_union)
    func_union.binary_op = aclop_RemainderTensorTensor
    register_acl_ufunc("ascend_remainder", BINARY_OP, func_union)
    func_union.binary_op = aclop_PowTensorTensor
    register_acl_ufunc("ascend_pow", BINARY_OP, func_union)
    func_union.scalar_binary_op = aclop_RemainderTensorScalar
    register_acl_ufunc("ascend_remainder", SCALAR_BINARY_OP, func_union)
    func_union.scalar_binary_op = aclop_PowTensorScalar
    register_acl_ufunc("ascend_pow", SCALAR_BINARY_OP, func_union)

    func_union.unary_op = aclop_Reciprocal
    register_acl_ufunc("ascend_reciprocal", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceReciprocal
    register_acl_ufunc("ascend_inplace_reciprocal", INPLACE_UNARY_OP, func_union)
    func_union.unary_op = aclop_Neg
    register_acl_ufunc("ascend_negative", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceNeg
    register_acl_ufunc("ascend_inplace_negative", INPLACE_UNARY_OP, func_union)
    func_union.unary_op = aclop_Abs
    register_acl_ufunc("ascend_absolute", UNARY_OP, func_union)
    func_union.unary_op = aclop_Signbit
    register_acl_ufunc("ascend_signbit", UNARY_OP, func_union)
    func_union.unary_op = aclop_Sign
    register_acl_ufunc("ascend_sign", UNARY_OP, func_union)

    func_union.unary_op = aclop_Square
    register_acl_ufunc("ascend_square", UNARY_OP, func_union)
    func_union.unary_op = aclop_Rsqrt
    register_acl_ufunc("ascend_rsqrt", UNARY_OP, func_union)
    func_union.unary_op = aclop_Deg2rad
    register_acl_ufunc("ascend_deg2rad", UNARY_OP, func_union)
    func_union.unary_op = aclop_Rad2deg
    register_acl_ufunc("ascend_rad2deg", UNARY_OP, func_union)

    func_union.unary_op = aclop_IsFinite
    register_acl_ufunc("ascend_is_finite", UNARY_OP, func_union)
    func_union.unary_op = aclop_IsInf
    register_acl_ufunc("ascend_is_inf", UNARY_OP, func_union)
    func_union.unary_op = aclop_IsNegInf
    register_acl_ufunc("ascend_is_negnative_inf", UNARY_OP, func_union)
    func_union.unary_op = aclop_IsPosInf
    register_acl_ufunc("ascend_is_positive_inf", UNARY_OP, func_union)

    func_union.unary_op = aclop_Floor
    register_acl_ufunc("ascend_floor", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceFloor
    register_acl_ufunc("ascend_inplace_floor", INPLACE_UNARY_OP, func_union)
    func_union.unary_op = aclop_Ceil
    register_acl_ufunc("ascend_ceil", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceCeil
    register_acl_ufunc("ascend_inplace_ceil", INPLACE_UNARY_OP, func_union)

    func_union.unary_op = aclop_Exp
    register_acl_ufunc("ascend_exp", UNARY_OP, func_union)
    func_union.unary_op = aclop_Expm1
    register_acl_ufunc("ascend_expm1", UNARY_OP, func_union)
    func_union.unary_op = aclop_Log
    register_acl_ufunc("ascend_log", UNARY_OP, func_union)
    func_union.unary_op = aclop_Log2
    register_acl_ufunc("ascend_log2", UNARY_OP, func_union)
    func_union.unary_op = aclop_Log10
    register_acl_ufunc("ascend_log10", UNARY_OP, func_union)
    func_union.unary_op = aclop_Log1p
    register_acl_ufunc("ascend_log1p", UNARY_OP, func_union)

    func_union.binary_op = aclop_LogAddExp2
    register_acl_ufunc("ascend_logaddexp2", BINARY_OP, func_union)
    func_union.binary_op = aclop_LogAddExp
    register_acl_ufunc("ascend_logaddexp", BINARY_OP, func_union)

    ###################################
    # 注册aclop_Matmul作为二元操作
    func_union.binary_op = aclop_Matmul
    register_acl_ufunc("ascend_matmul", BINARY_OP, func_union)
    func_union.binary_op = aclop_Dot
    register_acl_ufunc("ascend_dot", BINARY_OP, func_union)

    ###############################################
    # 注册aclop_Cos操作, 注册aclop_InplaceCos作为原地操作
    func_union.unary_op = aclop_Cos
    register_acl_ufunc("ascend_cos", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceCos
    register_acl_ufunc("ascend_inplace_cos", INPLACE_UNARY_OP, func_union)

    func_union.unary_op = aclop_Sin
    register_acl_ufunc("ascend_sin", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceSin
    register_acl_ufunc("ascend_inplace_sin", INPLACE_UNARY_OP, func_union)

    func_union.unary_op = aclop_Tan
    register_acl_ufunc("ascend_tan", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceTan
    register_acl_ufunc("ascend_inplace_tan", INPLACE_UNARY_OP, func_union)
    ##################### arcXXX op ######################
    func_union.unary_op = aclop_Acos
    register_acl_ufunc("ascend_acos", UNARY_OP, func_union)
    # 注册aclop_InplaceCos作为原地操作
    func_union.inplace_unary_op = aclop_InplaceAcos
    register_acl_ufunc("ascend_inplace_acos", INPLACE_UNARY_OP, func_union)

    func_union.unary_op = aclop_Asin
    register_acl_ufunc("ascend_asin", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceAsin
    register_acl_ufunc("ascend_inplace_asin", INPLACE_UNARY_OP, func_union)

    func_union.unary_op = aclop_Atan
    register_acl_ufunc("ascend_atan", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceAtan
    register_acl_ufunc("ascend_inplace_atan", INPLACE_UNARY_OP, func_union)
    ###################### cosh op #####################
    func_union.unary_op = aclop_Cosh
    register_acl_ufunc("ascend_cosh", UNARY_OP, func_union)
    # 注册aclop_InplaceCos作为原地操作
    func_union.inplace_unary_op = aclop_InplaceCosh
    register_acl_ufunc("ascend_inplace_cosh", INPLACE_UNARY_OP, func_union)

    func_union.unary_op = aclop_Sinh
    register_acl_ufunc("ascend_sinh", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceSinh
    register_acl_ufunc("ascend_inplace_sinh", INPLACE_UNARY_OP, func_union)

    func_union.unary_op = aclop_Tanh
    register_acl_ufunc("ascend_tanh", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceTanh
    register_acl_ufunc("ascend_inplace_tanh", INPLACE_UNARY_OP, func_union)

    func_union.binary_op = aclop_Atan2
    register_acl_ufunc("ascend_tan2", BINARY_OP, func_union)
    func_union.unary_op = aclop_Sinc
    register_acl_ufunc("ascend_sinc", UNARY_OP, func_union)
    func_union.unary_op = aclop_Erf
    register_acl_ufunc("ascend_erf", UNARY_OP, func_union)
    func_union.unary_op = aclop_Erfc
    register_acl_ufunc("ascend_erfc", UNARY_OP, func_union)
    func_union.unary_op = aclop_Erfinv
    register_acl_ufunc("ascend_erfinv", UNARY_OP, func_union)

#################### reduction ops ####################
cdef extern from "../acl_reduction_ops.h" nogil:

    aclError aclop_Any(const aclTensor* self, const aclIntArray* dim, bint keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_All(const aclTensor* self, const aclIntArray* dim, bint keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Max(const aclTensor* self, const aclIntArray* dim, bint keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Min(const aclTensor* self, const aclIntArray* dim, bint keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_ArgMax(const aclTensor* self, const aclIntArray* dim, bint keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_ArgMin(const aclTensor* self, const aclIntArray* dim, bint keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Mean(const aclTensor* self, const aclIntArray* dim, bint keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Sum(const aclTensor* self, const aclIntArray* dim, bint keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Prod(const aclTensor* self, const aclIntArray* dim, bint keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Nansum(const aclTensor* self, const aclIntArray* dim, bint keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)
    #aclError aclop_Nanprod(const aclTensor* self, const aclIntArray* dim, bint keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Nancumprod(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Nancumsum(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)

cdef void register_reduction_operators():
    cdef FuncPtrUnion func_union
    func_union.reduction_op = aclop_Any
    register_acl_ufunc("ascend_any", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_All
    register_acl_ufunc("ascend_all", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_Max
    register_acl_ufunc("ascend_max", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_Min
    register_acl_ufunc("ascend_min", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_ArgMin
    register_acl_ufunc("ascend_argmax", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_Mean
    register_acl_ufunc("ascend_argmin", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_ArgMax
    register_acl_ufunc("ascend_mean", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_Sum
    register_acl_ufunc("ascend_sum", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_Prod
    register_acl_ufunc("ascend_prod", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_Nansum
    register_acl_ufunc("ascend_nansum", REDUCTION_OP, func_union)
    #func_union.reduction_op = aclop_Nanprod
    #register_acl_ufunc("ascend_nanprod", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_Nancumsum
    register_acl_ufunc("ascend_nancumsum", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_Nancumprod
    register_acl_ufunc("ascend_nancumprod", REDUCTION_OP, func_union)


# general ops
cdef extern from "../acl_general_ops.h" nogil:
    aclError aclop_Copy(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Nonzero(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Fill(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)

    aclError aclop_Arange(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Concat(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Stack(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Flip(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Sort(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)

    # special math ops
    aclError aclop_Round(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Divmod(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Clamp(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_IsClose(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Heaviside(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)

cdef void register_irregular_operators():
    cdef FuncPtrUnion func_union
    func_union.general_op = aclop_Concat
    register_acl_ufunc("ascend_concatenate", GENERAL_OP, func_union)
    func_union.general_op = aclop_Stack # cupy has no such kernel
    register_acl_ufunc("ascend_stack", GENERAL_OP, func_union)
    func_union.general_op = aclop_Flip # cupy has no such kernel
    register_acl_ufunc("ascend_flip", GENERAL_OP, func_union)

    func_union.general_op = aclop_Sort
    register_acl_ufunc("ascend_sort", GENERAL_OP, func_union)
    func_union.general_op = aclop_Arange
    register_acl_ufunc("ascend_arange", GENERAL_OP, func_union)

    func_union.general_op = aclop_Round
    register_acl_ufunc("ascend_round", GENERAL_OP, func_union)
    func_union.general_op = aclop_Divmod
    register_acl_ufunc("ascend_divmod", GENERAL_OP, func_union)
    func_union.general_op = aclop_Clamp
    register_acl_ufunc("ascend_clip", GENERAL_OP, func_union)
    func_union.general_op = aclop_IsClose
    register_acl_ufunc("ascend_is_close", GENERAL_OP, func_union)
    func_union.general_op = aclop_Heaviside
    register_acl_ufunc("ascend_heaviside", GENERAL_OP, func_union)

    func_union.unary_op = aclop_Copy
    register_acl_ufunc("ascend_copy", UNARY_OP, func_union)
    func_union.general_op  = aclop_Fill
    register_acl_ufunc("ascend_fill", GENERAL_OP, func_union)
    func_union.unary_op = aclop_Nonzero
    register_acl_ufunc("ascend_nonzero", UNARY_OP, func_union)

def py_register_acl_ufunc(str opname, int func_type, long func_ptr):
    """Python层级的操作注册函数, func_type is OpType enum value"""
    cdef string c_opname = opname.encode('utf-8')
    cdef FuncPtrUnion func_union
    cdef OpType op_type
    
    op_type = <OpType>func_type
    if op_type == BINARY_OP:
        func_union.binary_op = <BinaryOpFunc>func_ptr
    elif op_type == INPLACE_BINARY_OP:
        func_union.inplace_binary_op = <InplaceBinaryOpFunc>func_ptr
    elif op_type == UNARY_OP:
        func_union.unary_op = <UnaryOpFunc>func_ptr
    elif op_type == INPLACE_UNARY_OP:
        func_union.inplace_unary_op = <InplaceUnaryOpFunc>func_ptr
    else:
        raise ValueError("Invalid function type")
    
    return register_acl_ufunc(c_opname, op_type, func_union)

'''
# TODO: passing stream by intptr_t
def py_launch_acl_func(str opname, tuple ops, bint inplace=False):
    """Python层级的ACL函数启动器"""
    cdef string c_opname = opname.encode('utf-8')
    return launch_acl_func(c_opname, ops, inplace)
'''

cdef void init_builtin_operators():
    register_math_operators()
    register_reduction_operators()
    register_irregular_operators()

# only one function can be run during module init
init_builtin_operators()
