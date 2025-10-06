import cython
cimport cpython

from cupy_backends.ascend.api.acl_types cimport *
from cupy._core import _dtype
from cupy._core.core import _ndarray_base

cdef extern from "aclnn/opdev/common_types.h":  
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

    void aclDestroyTensor(aclTensor* tensor)


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

# TODO not impl yet
cdef object register_acl_ufunc(str opname, object opcfunc) except*:
    return None
