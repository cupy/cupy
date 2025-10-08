import cython
cimport cpython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from collections import namedtuple

#from cupy_backends.ascend.api.acl_types cimport *
from cupy._core import _dtype
from cupy._core.core import _ndarray_base
from libcpp.map cimport map as cpp_map
from libcpp.vector cimport vector


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

    aclnnStatus aclDestroyTensor(const aclTensor *tensor)


cdef aclDataType numpy_to_acl_dtype(dtype,
    bint is_half_allowed=False, bint is_double_supported=False):
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
    elif dtype_char == 'I':
        return aclDataType.ACL_INT32
    elif dtype_char == 'I':
        return aclDataType.ACL_UINT32
    elif dtype_char == 'q' and is_double_supported:
        return aclDataType.ACL_INT64
    elif dtype_char == 'Q' and is_double_supported:
        return aclDataType.ACL_UINT64
    elif dtype_char == '?':
        return aclDataType.ACL_BOOL
    else:
        #raise TypeError('dtype is not supported: {}'.format(dtype)) # TODO: consider throw?
        return aclDataType.ACL_DT_UNDEFINED

    


# 主转换函数
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
        ndim = len(cupy_array._shape)
        
        # 分配内存用于存储维度信息
        view_dims = <int64_t*>PyMem_Malloc(ndim * sizeof(int64_t))
        strides = <int64_t*>PyMem_Malloc(ndim * sizeof(int64_t))
        storage_dims = <int64_t*>PyMem_Malloc(ndim * sizeof(int64_t))
        
        if view_dims == NULL or strides == NULL or storage_dims == NULL:
            raise MemoryError("Failed to allocate memory for dimension arrays")
        
        # 填充维度信息
        for i in range(ndim):
            view_dims[i] = cupy_array._shape[i]
            strides[i] = cupy_array._strides[i]
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
        UNARY_OP = 0
        INPLACE_UNARY_OP = 1
        BINARY_OP = 4
        INPLACE_BINARY_OP = 5
        SCALAR_BINARY_OP = 6
        INPLACE_SCALAR_BINARY_OP = 7
        TRI_OP = 8
        INPLACE_TRI_OP = 9

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

# operator_function_ptr registry: TODO: thread safety
#cdef cpp_map[OpInfo, FuncPtrUnion] _builtin_operators # unordered map for better performance
cdef cpp_map[OpInfo, FuncPtrUnion] _builtin_operators
#cdef _builtin_operators = {}  

# 定义函数指针类型
ctypedef aclError (*TriOpFunc)(const aclTensor* self, const aclTensor* other,
    const aclTensor* other2, aclTensor* out, aclrtStream stream)
ctypedef aclError (*InplaceTriOpFunc)(aclTensor* self, const aclTensor* other, aclrtStream stream)
# TODO: BinaryScalarOpFunc
ctypedef aclError (*BinaryOpFunc)(const aclTensor* self, const aclTensor* other,
    aclTensor* out, aclrtStream stream)
ctypedef aclError (*InplaceBinaryOpFunc)(aclTensor* self, const aclTensor* other, aclrtStream stream)
# TODO: out = self + scalar
ctypedef aclError (*UnaryOpFunc)(const aclTensor* self, aclTensor* out, aclrtStream stream)
ctypedef aclError (*InplaceUnaryOpFunc)(aclTensor* self, aclrtStream stream)

# 函数指针联合体，用于存储不同类型的操作
ctypedef union FuncPtrUnion:
    # TODO: failed nullptr is that a choice?
    UnaryOpFunc unary_op
    InplaceUnaryOpFunc inplace_unary_op
    BinaryOpFunc binary_op
    InplaceBinaryOpFunc inplace_binary_op
    TriOpFunc tri_op
    InplaceTriOpFunc inplace_tri_op

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

cdef OpType get_op_type(tuple ops, bint inplace):
    cdef bint has_scalar = False # TODO: detect and return op_type
    if len(ops) == 3 and not inplace:  # 二元操作
        return BINARY_OP
    elif len(ops) == 2 and inplace:  # 原地二元操作
        return INPLACE_BINARY_OP  
    elif len(ops) == 2 and not inplace:  # 一元操作
        return UNARY_OP
    elif len(ops) == 1 and inplace:  # 原地一元操作
        return INPLACE_UNARY_OP
    else:
        raise RuntimeError("Operator type can not be decided")
    return INVALID_OP

cdef aclError launch_acl_func(string opname, tuple ops, bint inplace, size_t stream_ptr) except *:
    # 检查操作是否已注册
    cdef OpInfo op_info
    op_info.op_name = opname
    op_info.op_type = get_op_type(ops, inplace)
    if _builtin_operators.find(op_info) == _builtin_operators.end():
        raise KeyError(f"Operator {opname} not registered")
    
    cdef FuncPtrUnion func_ptr = _builtin_operators[op_info]
    cdef aclError ret = 0
    cdef aclrtStream stream = <aclrtStream>stream_ptr
    # 转换为ACL张量列表
    cdef vector[aclTensor*] tensors
    for op in ops:
        tensors.push_back(cupy_ndarray_to_acl_tensor(op))
    try:
        if len(ops) == 3:  # 二元操作
            if op_info.op_type != BINARY_OP:
                raise RuntimeError(f"Operator {opname} is not a binary operator")
            ret = func_ptr.binary_op(tensors[0], tensors[1], tensors[2], stream)
        
        elif len(ops) == 2 and inplace:  # 原地二元操作
            if op_info.op_type != INPLACE_BINARY_OP:
                raise RuntimeError(f"Operator {opname} is not an inplace binary operator")
            ret = func_ptr.inplace_binary_op(tensors[0], tensors[1], stream)
        
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
            # TODO: deal with aclScalar
            aclDestroyTensor(t)  # 假设destroyAclTensor函数已存在
    return ret

cdef extern from "../acl_math.h" nogil:
    aclError aclop_BitwiseAndTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceBitwiseAndTensor(aclTensor* self, const aclTensor* other, aclrtStream stream)
    aclError aclop_BitwiseOrTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceBitwiseOrTensor(aclTensor* self, const aclTensor* other, aclrtStream stream)
    aclError aclop_BitwiseXorTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceBitwiseXorTensor(aclTensor* self, const aclTensor* other, aclrtStream stream)
    aclError aclop_BitwiseNotTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceBitwiseNotTensor(aclTensor* self, const aclTensor* other, aclrtStream stream)

    aclError aclop_Add(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    #aclError aclop_InplaceAdd(aclTensor* self, const aclTensor* other, aclrtStream stream)

    aclError aclop_Cos(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceCos(aclTensor* self,  aclrtStream stream)

# 初始化函数，注册内置操作
cdef void init_builtin_operators():
    cdef FuncPtrUnion func_union
    
    # 注册aclop_Add作为二元操作
    func_union.binary_op = aclop_Add
    register_acl_ufunc("ascend_add", BINARY_OP, func_union)
    
    # 注册aclop_InplaceAnd作为原地二元操作
    #func_union.inplace_binary_op = aclop_InplaceAdd
    #register_acl_ufunc("ascend_inplace_add", INPLACE_BINARY_OP, func_union)

    # 注册aclop_Add作为二元操作
    func_union.binary_op = aclop_BitwiseAndTensor
    register_acl_ufunc("ascend_bitwise_and", BINARY_OP, func_union)
    
    # 注册aclop_InplaceAnd作为原地二元操作
    func_union.inplace_binary_op = aclop_InplaceBitwiseAndTensor
    register_acl_ufunc("ascend_inplace_bitwise_and", INPLACE_BINARY_OP, func_union)

    # 注册aclop_Cos操作
    func_union.unary_op = aclop_Cos
    register_acl_ufunc("ascend_cos", UNARY_OP, func_union)
    
    # 注册aclop_InplaceCos作为原地操作
    func_union.inplace_unary_op = aclop_InplaceCos
    register_acl_ufunc("ascend_inplace_cos", INPLACE_UNARY_OP, func_union)


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
# TODO: passing stream, how?
def py_launch_acl_func(str opname, tuple ops, bint inplace=False):
    """Python层级的ACL函数启动器"""
    cdef string c_opname = opname.encode('utf-8')
    return launch_acl_func(c_opname, ops, inplace)
'''

# TODO: not sure if it is possible to run during import 模块初始化时注册内置操作
init_builtin_operators()