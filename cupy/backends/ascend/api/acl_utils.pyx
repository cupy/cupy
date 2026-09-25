import cython
import os
import operator as _operator
cimport cpython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from collections import namedtuple

from cupy._core import _dtype
from cupy._core.core import _ndarray_base
from cupy._core._scalar cimport CScalar, scalar_to_c_scalar
from libc.stdint cimport (
    int8_t, int16_t, int32_t, int64_t,
    uint8_t, uint16_t, uint32_t, uint64_t,
    uintptr_t,
)
from libc.string cimport memcpy
from libcpp.unordered_map cimport unordered_map as cpp_map
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.set cimport set as cpp_set
from libcpp.vector cimport vector
from libcpp.complex cimport complex

import threading as _threading

from cupy.xpu import stream as stream_module

ASCEND_OP_PREFIX = "ascend_"

#include "backends/ascend/api/acl_types.pxi" # already included in pxd file

# ---------------------------------------------------------------------------
# 统一参数通道（docs/ascend/arg_passing_plan.md §2.2/§2.3）
#
# args/kwargs 的每个元素都是带 tag 的 AclArg：scalar / int 序列 / string /
# tensor / none，定义在 cupy/backends/ascend/acl_scalar_arg.h。
# ---------------------------------------------------------------------------
cdef extern from "../acl_scalar_arg.h":
    ctypedef enum AclArgKind:
        ARG_NONE
        ARG_SCALAR
        ARG_INT_ARRAY
        ARG_STRING
        ARG_TENSOR

    cdef cppclass AclArg:
        AclArg()
        AclArgKind kind
        const aclScalar* scalar
        const aclTensor* tensor
        string str
        vector[int64_t] ints

    AclArg MakeNoneArg()
    AclArg MakeScalarArg(const aclScalar* scalar)
    AclArg MakeIntArrayArg(const vector[int64_t]& values)
    AclArg MakeStringArg(const string& value)
    AclArg MakeTensorArg(const aclTensor* tensor)

# 为vector[AclArg]&创建类型别名
ctypedef vector[AclArg] ArgsType
ctypedef cpp_map[string, AclArg] KwargsType

# 4. 为迭代器创建别名（便于遍历）
ctypedef cpp_map[string, AclArg].const_iterator KargsConstIterator
ctypedef cpp_map[string, AclArg].iterator KargsIterator

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

# aclScalar 需要一起保活的 host 侧资源：aclScalar* -> Python 对象。
# 目前只有 ACL_STRING 用得上（见 create_acl_scalar_from_py_str）。
# 注意：销毁 aclScalar 必须走 _destroy_acl_scalar()，否则这里的强引用会一直挂着。
cdef dict _acl_scalar_owners = {}


cdef void _destroy_acl_scalar(const aclScalar* scalar):
    """aclDestroyScalar + 释放为它保活的 host 资源。

    AscendCL 的 aclScalar 只拷贝标量值本身（common_types.h 里值存放在内联
    union v_t 中，析构为空）；但字符串标量只能是指针语义，所以顺序必须是
    「先销毁 aclScalar，再放开保活引用」。
    """
    if scalar == NULL:
        return
    aclDestroyScalar(scalar)
    _acl_scalar_owners.pop(<uintptr_t>scalar, None)


cdef aclDataType numpy_dtype_to_acl_dtype(dtype,
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
    elif (dtype_char == 'Q' or dtype_char == 'L') and is_double_supported:
        return aclDataType.ACL_UINT64 # numpy 2.x change geh dtype char for uint64 to `L`
    elif dtype_char == '?':
        return aclDataType.ACL_BOOL
    else:
        print('ASCEND: DEBUG dtype is not supported: {}'.format(dtype))
        return aclDataType.ACL_DT_UNDEFINED

cdef aclScalar* create_acl_scalar_from_py_str(str py_str):
    """将 Python 字符串转换为 aclScalar。

    ACL_STRING 的 aclScalar 内部没有任何存放字符串内容的位置（见
    aclnn/opdev/common_types.h 的 `union v_t`），所以它保存的只能是指针
    本身：aclScalar 的整个生命周期内字符串 buffer 必须保持有效。

    这里直接使用 CPython bytes 对象内部那块「地址稳定且以 NUL 结尾」的
    buffer，并把 bytes 对象登记到 `_acl_scalar_owners` 保活，销毁时由
    `_destroy_acl_scalar` 放开引用。

    （原实现把临时 `py_bytes` 的指针交给了 aclScalar，函数返回后该 bytes
    被回收 -> aclScalar 里是野指针；同时 PyMem_Malloc 出来的 `buffer`
    从头到尾没被使用也没被释放 -> 泄漏。）
    """
    cdef bytes py_bytes = py_str.encode('utf-8')
    cdef aclScalar* scalar_ptr = aclCreateScalar(
        <void*><const char*>py_bytes, aclDataType.ACL_STRING)
    if scalar_ptr == NULL:
        raise MemoryError("Failed to create string aclScalar")
    # aclScalar 内部可能持有该指针，必须保活到 aclScalar 被销毁
    _acl_scalar_owners[<uintptr_t>scalar_ptr] = py_bytes
    return scalar_ptr

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
        # 剩余：'S' 字符串（create_acl_scalar_from_py_str，仅白名单算子）、
        # 'O' object（不支持，也不该静默支持）。
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
                raise MemoryError("Failed to allocate memory for integer16 scalar")
            (<int16_t*>value_ptr)[0] = (<int16_t*>s.ptr)[0]
            dtype = ACL_INT16
        elif s.kind == 'i' and s.size == 1:  # 整数类型int8
            value_ptr = PyMem_Malloc(sizeof(signed char))
            if value_ptr == NULL:
                raise MemoryError("Failed to allocate memory for integer8 scalar")
            (<signed char*>value_ptr)[0] = (<signed char*>s.ptr)[0]
            dtype = ACL_INT8
        elif s.kind == 'u':  # unsigned 整数类型
            value_ptr = PyMem_Malloc(s.size)
            if value_ptr == NULL:
                raise MemoryError("Failed to allocate memory for unsigned scalar")
            if s.size == 1:
                (<uint8_t*>value_ptr)[0] = (<uint8_t*>s.ptr)[0]
                dtype = ACL_UINT8
            elif s.size == 2:
                (<uint16_t*>value_ptr)[0] = (<uint16_t*>s.ptr)[0]
                dtype = ACL_UINT16
            elif s.size == 4:
                (<uint32_t*>value_ptr)[0] = (<uint32_t*>s.ptr)[0]
                dtype = ACL_UINT32
            elif s.size == 8:
                (<uint64_t*>value_ptr)[0] = (<uint64_t*>s.ptr)[0]
                dtype = ACL_UINT64
            else:
                raise TypeError(f"Unsigned scalar of size {s.size} is not supported")
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
        elif s.kind == 'f' and s.size == 2:  # float16: stored as raw bits
            value_ptr = PyMem_Malloc(sizeof(uint16_t))
            if value_ptr == NULL:
                raise MemoryError("Failed to allocate memory for float16 scalar")
            (<uint16_t*>value_ptr)[0] = (<uint16_t*>s.ptr)[0]
            dtype = ACL_FLOAT16
        elif s.kind == 'c' or s.kind == 'C':  # complex
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
                raise TypeError(f"Complex scalar of size {s.size} is not supported")
        elif s.kind == 'S':  # string type is not supported by cupy.CScalar
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
            raise RuntimeError(
                "Failed to create aclScalar with error: {}".format(msg))
        # aclCreateScalar 会把值拷贝进 aclScalar 内部的内联 union
        # (common_types.h: aclScalar 的析构为空)，所以这里必须立刻释放我们
        # 自己的临时 buffer，否则每次标量算子调用都会泄漏一块。
        PyMem_Free(value_ptr)
        value_ptr = NULL
        return acl_scalar
    except Exception as e:
        # 异常处理：确保资源清理
        if value_ptr != NULL:
            PyMem_Free(value_ptr)
        _destroy_acl_scalar(acl_scalar)
        raise MemoryError("Failed to create aclScalar with error %s" % e)

cdef aclScalar* _convert_arg_to_acl_scalar(arg):
    typ = type(arg)
    if issubclass(typ, CScalar):
        return cupy_scalar_to_acl_scalar(arg)
    elif typ is str:
        return create_acl_scalar_from_py_str(arg)
    else:
        pyarg = scalar_to_c_scalar(arg)
        if pyarg:
            return cupy_scalar_to_acl_scalar(pyarg)
        else:
            print("ASCEND: arg can not be converted to aclScalar: ", arg)
            return NULL

# ---------------------------------------------------------------------------
# 参数（args/kwargs）校验 —— 见 docs/ascend/arg_passing_plan.md 阶段 1
#
# 背景（code_review_ascend_backend.md §2.5）：无法转换的参数原来只是 print 一行
# 然后静默丢弃 -> 算子照常执行但参数没生效 = 静默错误结果。
# 现在：不可转换 / 未知 key 一律抛错；CUPY_ASCEND_LENIENT_ARGS=1 可临时恢复旧行为。
# ---------------------------------------------------------------------------

#: C++ 侧 `GetScalarArg` 实际消费的参数 key（cupy/backends/ascend/*.h 全量提取）
_KNOWN_SCALAR_KEYS = frozenset((
    'atol', 'axis', 'bins', 'descending', 'dim', 'full_matrices', 'k',
    'keepdim', 'max', 'min', 'nan', 'neginf', 'order', 'posinf', 'rtol',
    'shift', 'some', 'sorted', 'stable', 'start', 'step', 'stop',
    # ufunc 路径注入的 key（_kernel.pyx:913）：语义实现见 arg_passing_plan.md M2，
    # 在实现之前必须显式报错而不是静默丢弃（否则 where=mask 会算错）。
    'where',
))

#: 声明允许透传 ACL_STRING 标量的算子（字符串应优先在 host 侧解析成 int/bool）
#: `ascend_dump_args` 是参数通道探针（tests/ascend/test_unified_args.py），
#: 用来证明 str 参数能按 ARG_STRING 送达 C++ 侧。
_STRING_ARG_OPS = frozenset((
    'ascend_dump_args',
    # einsum 的 equation：aclnnEinsum 的签名就是 (tensor list, const char*, out)，
    # 字符串无法在 host 侧降为数值 —— ARG_STRING 的第一个真实消费者。
    'ascend_einsum',
))


cdef inline bint _lenient_args():
    """CUPY_ASCEND_LENIENT_ARGS=1 -> 恢复「打印并丢弃」的旧行为（迁移期用）。"""
    return os.environ.get('CUPY_ASCEND_LENIENT_ARGS') == '1'


cdef void _destroy_arg(AclArg& arg) except *:
    """释放一个参数持有的 host 资源。

    只有 ARG_SCALAR 需要在 pyx 侧销毁（aclScalar 由这里创建）；
    ARG_INT_ARRAY / ARG_STRING 是 C++ 侧按值持有的（vector/string），随容器析构。
    """
    if arg.kind == ARG_SCALAR:
        _destroy_acl_scalar(arg.scalar)
        arg.scalar = NULL
    arg.kind = ARG_NONE


cdef void _delete_args(ArgsType& args) except *:
    """销毁位置参数列表里的全部参数。"""
    cdef Py_ssize_t i
    for i in range(args.size()):
        _destroy_arg(args[i])


cdef AclArg _convert_arg(str opname, str name, object arg) except *:
    """把 Python 参数转成带 tag 的 AclArg —— 统一参数通道的唯一入口。

    类型驱动，不做猜测：
      * ``None``                    -> ARG_NONE（例如 axis=None 表示「全部轴」）
      * ``str``                     -> ARG_STRING（仅 `_STRING_ARG_OPS` 白名单内的算子）
      * ``list`` / ``tuple[int]``   -> ARG_INT_ARRAY（axis/dims/shape/shift...）
      * 数值/布尔/numpy 标量         -> ARG_SCALAR
    其余类型仍然响亮失败（ndarray 见 _convert_arg_strict 的说明）。
    """
    cdef aclScalar* s
    cdef vector[int64_t] values
    cdef list items
    cdef object item
    cdef bytes py_bytes

    if arg is None:
        return MakeNoneArg()

    if type(arg) is str:
        if opname in _STRING_ARG_OPS:
            py_bytes = (<str>arg).encode('utf-8')
            return MakeStringArg(py_bytes)
        if _lenient_args():
            return MakeNoneArg()
        raise NotImplementedError(
            f"{opname}: 字符串参数 {name}={arg!r} 不支持直接透传到 aclnn；"
            f"请在 host 侧解析为数值/布尔（当前算子未声明 ARG_STRING）")

    if isinstance(arg, (list, tuple)):
        items = list(arg)
        values.resize(len(items))
        for i in range(len(items)):
            item = items[i]
            try:
                # operator.index：只接受整数（含 numpy 整数），float/str 不静默截断
                values[i] = <int64_t>_operator.index(item)
            except TypeError:
                raise NotImplementedError(
                    f"{opname}: 参数 {name} 是序列，但第 {i} 个元素 {item!r} "
                    f"(type {type(item).__name__}) 不是整数；"
                    f"int 序列（ARG_INT_ARRAY）只接受整数元素")
        return MakeIntArrayArg(values)

    s = _convert_arg_to_acl_scalar(arg)
    if s != NULL:
        return MakeScalarArg(s)
    if _lenient_args():
        return MakeNoneArg()
    raise NotImplementedError(
        f"{opname}: 参数 {name} = {arg!r} (type {type(arg).__name__}) 无法转换为 "
        f"aclScalar，当前不支持；为避免静默错误结果这里直接报错。"
        f"（迁移期可设 CUPY_ASCEND_LENIENT_ARGS=1 恢复旧的丢弃行为）")


cdef aclScalar* _convert_arg_strict(str opname, str name, arg) except *:
    """转换单个参数为 aclScalar；不支持则抛错（而非静默丢弃）。

    ``opname``/``name`` 只用于错误信息（name 形如 ``'axis'`` 或 ``'#0'``）。
    """
    cdef aclScalar* s
    if type(arg) is str and opname not in _STRING_ARG_OPS:
        # 字符串应优先在 host 侧解析为 int/bool（见 arg_passing_plan.md §2.3）；
        # ACL_STRING 仅对显式声明的算子开放。
        if _lenient_args():
            return NULL
        raise NotImplementedError(
            f"{opname}: 字符串参数 {name}={arg!r} 不支持直接透传到 aclnn；"
            f"请在 host 侧解析为数值/布尔（当前算子未声明 ARG_STRING）")
    s = _convert_arg_to_acl_scalar(arg)
    if s != NULL:
        return s
    if _lenient_args():
        return NULL
    raise NotImplementedError(
        f"{opname}: 参数 {name} = {arg!r} (type {type(arg).__name__}) 无法转换为 "
        f"aclScalar，当前不支持；为避免静默错误结果这里直接报错。"
        f"（迁移期可设 CUPY_ASCEND_LENIENT_ARGS=1 恢复旧的丢弃行为）")


cdef KwargsType _create_keyword_args(dict kwargs, str opname="<unknown>") except *:
    cdef KwargsType acl_kwargs
    cdef string cpp_str
    cdef const char* c_str
    cdef bytes py_bytes
    cdef AclArg sarg
    if kwargs:
        try:
            for key, value in kwargs.items():
                if key not in _KNOWN_SCALAR_KEYS and not _lenient_args():
                    raise ValueError(
                        f"{opname}: 未知参数 key {key!r}（不在 C++ 侧消费的 key "
                        f"白名单内）——它会被静默丢弃，所以这里直接报错。"
                        f"已支持的 key 见 _KNOWN_SCALAR_KEYS")
                sarg = _convert_arg(opname, key, value)
                py_bytes = key.encode("utf-8")
                c_str = py_bytes
                cpp_str = c_str
                acl_kwargs[cpp_str] = sarg
        except Exception:
            # 中途失败时，已放进局部 map 的参数既不会随返回值交出去，
            # 也没有别人还能拿到它的指针，必须就地回收，否则泄漏。
            _delete_keyword_args(acl_kwargs)
            raise
    return acl_kwargs

cdef void _delete_keyword_args(KwargsType& my_map) except *:
    cdef:
        # 使用非常量迭代器，因为我们需要修改map（删除元素）
        cpp_map[string, AclArg].iterator it = my_map.begin()
        cpp_map[string, AclArg].iterator end = my_map.end()

    # 安全遍历并删除
    while it != end:
        _destroy_arg(deref(it).second)

        # 3. 将迭代器指向下一个元素，并擦除当前元素。
        #    it = my_map.erase(it) 会返回指向下一个有效元素的迭代器，这是安全的方法。
        it = my_map.erase(it)


cdef class _AclTensorOwner:
    """aclTensor 生命周期内需要保活/释放的所有 host 侧资源。

    * ``ref``                    —— 持有 aclTensor 数据 buffer 的 Python 对象
      （视图无法用 offset 表达时物化出来的临时数组，或原数组本身）；
    * ``view_dims``/``strides``/``storage_dims`` —— 传给 aclCreateTensor 的
      三个维度数组。

    aclDestroyTensor 只释放 aclTensor 自身的 shape/stride 元数据：
    既不会释放设备数据，也不会释放我们用 PyMem_Malloc 出来的维度数组，
    所以两者都必须由本对象负责回收（原实现在成功路径上完全没释放这三个数组）。
    """

    cdef object ref
    cdef int64_t* view_dims
    cdef int64_t* strides
    cdef int64_t* storage_dims

    # 必须用 cdef 方法而不是 __cinit__：__cinit__ 会生成 Python 调用入口，
    # 裸指针参数会被当成「要转换成 Python 对象」而报错。
    cdef void _init(self, object ref, int64_t* view_dims, int64_t* strides,
                    int64_t* storage_dims):
        self.ref = ref
        self.view_dims = view_dims
        self.strides = strides
        self.storage_dims = storage_dims

    def __dealloc__(self):
        if self.view_dims != NULL:
            PyMem_Free(self.view_dims)
        if self.strides != NULL:
            PyMem_Free(self.strides)
        if self.storage_dims != NULL:
            PyMem_Free(self.storage_dims)


cdef object _materialize_host(_ndarray_base cupy_array):
    """把视图/非连续数组物化成独立 C-连续新数组（host transfer 路径）。

    为什么不能用 ``cupy_array.copy()`` / ``.get()``：它们会走 ElementwiseKernel/
    ufunc 派发（``ascend_copy`` -> ``launch_general_func`` ->
    ``cupy_ndarray_to_acl_tensor``），对同一个问题视图再次进入本函数，
    无限递归。这里全程只用底层 memcpy：``MemoryPointer.copy_to_host`` 是
    ``runtime.memcpy`` (D2H) 的裸封装；最后的 ``cupy.array(numpy数组)`` 走
    创建路径的 H2D memcpy，同样不经过 ufunc。

    步骤（host_view.copy() 后交给 cupy.array(...)，见调用方）：
      1. 计算视图实际覆盖的字节区间 [data.ptr+min_off, data.ptr+max_off+itemsize)，
         只拷这一段（负步长轴会把区间基址拉低）；
      2. ``copy_to_host`` 把这段显存裸拷进 host 缓冲；
      3. 在 host 缓冲的**元素 dtype 域**上用 ``numpy.as_strided`` 按原
         shape / 元素步长重建视图（零拷贝）；
      4. ``host_view.copy()`` 得到 C-连续 host 数组。

    注意必须在元素 dtype 域而不是 uint8 域做 as_strided：uint8 域里每个
    逻辑元素只占 1 字节，copy() 会丢掉其余 itemsize-1 字节，且随后
    .view(dtype) 要求末轴字节数整除而失败。dtype 域要求缓冲偏移是
    itemsize 的倍数——这里恒成立：分配 512B 对齐、切片偏移与步长都是
    元素倍数，故 min_off 与 -min_off 都是 itemsize 的倍数。
    """
    import numpy
    import cupy as _cupy_mod

    cdef:
        Py_ssize_t i, ndim = len(cupy_array._shape)
        Py_ssize_t itemsize = cupy_array.dtype.itemsize
        Py_ssize_t min_off = 0, max_off = 0, off
        Py_ssize_t span, first_byte

    # 各轴对字节区间的贡献：k in [0, n-1]，步长 s -> 最小/最大偏移
    for i in range(ndim):
        if cupy_array._shape[i] > 1:
            off = (cupy_array._shape[i] - 1) * cupy_array._strides[i]
            if off < 0:
                min_off += off
            else:
                max_off += off
    span = max_off - min_off + itemsize

    # 2. 裸 memcpy D2H：只拷视图覆盖的区间（不含底层分配的其余部分）
    # copy_to_host() not available in PooledMemory, but MemoryPointer
    # msut copy from data.ptr, offset is the byte count from view starting addr
    # e.g. part[5:7], ptr - mem.ptr =20, span is the stride inside view
    # offset 0 will copy part[5:7] materialized into [0,1], not [5,6]
    host = numpy.empty(span, dtype=numpy.uint8)
    from cupy.xpu.memory import MemoryPointer as _MemoryPointer
    _mp = _MemoryPointer(cupy_array.data.meme,
        (cupy_array.data.ptr - cupy_array.data.mem.ptr) + min_off)
    _mp.copy_to_host(host.ctypes.data, span)

    # 3. 元素域视图：host 起点 = data.ptr + min_off，即物化区间第一个逻辑元素
    #    所以 frombuffer 的offset为0
    flat = numpy.frombuffer(host, dtype=cupy_array.dtype,
                            offset=0, count=span // itemsize)
    # as_strided 的 strides 恒为字节单位（与 dtype 无关），cupy 的
    # strides 也是字节单位，直接沿用，不能再除以 itemsize。
    host_view = numpy.lib.stride_tricks.as_strided(
        flat,
        shape=tuple(cupy_array.shape),
        strides=tuple(cupy_array.strides))

    # 4. C-连续 host 拷贝，交给 cupy.array 走 H2D memcpy（不经 ufunc）
    return _cupy_mod.array(host_view.copy(), dtype=cupy_array.dtype)


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
        object owner_ref = None
        Py_ssize_t remaining
        _AclTensorOwner owner

    try:
        # 1. 获取CuPy数组的形状和维度
        ndim = len(cupy_array._shape) # len() works for cython 3.1+ only

        # aclnnTensorData 的偏移必须以元素为单位从 buffer 起点计算, 而 CuPy
        # 的 nbytes 是从 data.ptr 起算的剩余字节数（视图不包含前面的数据）,
        # 因此视图无法用 offset 表达, 这里先物化成从 0 开始的新数组。
        if (cupy_array.data.ptr != cupy_array.data.mem.ptr or not cupy_array._c_contiguous):
            if cupy_array.size:
                # 视图无法用 offset 表达：物化成独立 C-连续数组。
                # 必须走 _materialize_host（裸 memcpy），不能用 .copy()——那会
                # 经过 ufunc 派发（ascend_copy），对本视图再次进入本函数，
                # 无限递归（见 _materialize_host 文档）。
                cupy_array = _materialize_host(cupy_array)
                owner_ref = cupy_array
        else:
            owner_ref = cupy_array
            cupy_array = cupy_array

        # 分配内存用于存储维度信息
        view_dims = <int64_t*>PyMem_Malloc(ndim * sizeof(int64_t))
        strides = <int64_t*>PyMem_Malloc(ndim * sizeof(int64_t))
        storage_dims = <int64_t*>PyMem_Malloc(ndim * sizeof(int64_t))

        if view_dims == NULL or strides == NULL or storage_dims == NULL:
            raise MemoryError("Failed to allocate memory for dimension arrays")

        # 填充维度信息
        item_size = cupy_array.dtype.itemsize
        for i in range(ndim):
            view_dims[i] = cupy_array._shape[i]
            # aclTensor strides use element size, not the byte size
            strides[i] = cupy_array._strides[i] // item_size
            storage_dims[i] = cupy_array._shape[i]  # 假设存储形状与视图形状相同

        # 2. 映射数据类型
        data_type = numpy_dtype_to_acl_dtype(cupy_array.dtype)
        if data_type == ACL_DT_UNDEFINED:
            raise ValueError(f"Unsupported dtype: {cupy_array.dtype}")

        # 3. 获取数据指针
        tensor_data = <void*>cupy_array.data.ptr

        # 4. 根据内存布局选择合适的格式
        if cupy_array._f_contiguous and ndim > 1 and not cupy_array._c_contiguous:
            raise NotImplementedError(
                'Ascend does not support Fortran-contiguous arrays: '
                'call cupy.ascontiguousarray() first')
        format = ACL_FORMAT_ND
        
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

        # 所有权转移：数据保活引用 + 三个维度数组都交给 owner，在
        # cupy_destroy_acl_tensor 里统一释放，避免每次转换泄漏 3*ndim*8 字节。
        owner = _AclTensorOwner()
        owner._init(owner_ref, view_dims, strides, storage_dims)
        cupy_acl_tensor_owners[<uintptr_t>acl_tensor] = owner
        # 置空以免（将来）异常路径重复释放
        view_dims = NULL
        strides = NULL
        storage_dims = NULL
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


# aclTensor* -> _AclTensorOwner（数据保活引用 + 维度数组）
# aclDestroyTensor 只释放 shape/stride 元数据，不释放数据，所以必须在这里保活。
cdef dict cupy_acl_tensor_owners = {}


cdef aclError cupy_destroy_acl_tensor(const aclTensor* tensor) except *:
    # 顺序很重要：先销毁 aclTensor（它可能仍引用我们传进去的维度数组/数据），
    # 再放开 Python 侧的保活引用。
    cdef aclError ret = aclDestroyTensor(tensor)
    cupy_acl_tensor_owners.pop(<uintptr_t>tensor, None)
    return ret


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
        TRI_OP = 8
        INPLACE_TRI_OP = 9
        # out = scalar <op> tensor（标量在左操作数），见 acl_opinfo.h
        REVERSE_SCALAR_BINARY_OP = 10

cdef extern from "../acl_custom_kernels.h":
    # custom AscendC kernel launcher (binary load + aclrtLaunchKernelWithConfig)
    aclError aclop_LaunchCustomKernel(
        const char* bin_path, const char* func_name,
        void* out0, void* out1, void* in0, void* in1,
        uint64_t n, aclrtStream stream) nogil

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

# Every op name registered under *any* OpType (the map above is keyed by
# (name, op_type)). Used by the "is this kernel callable on Ascend?" check in
# `cupy/_core/_ascend/_kernel.pyx`; a std::set keeps the query O(log n) without
# needing the GIL inside `register_acl_ufunc` (which is `nogil`).
cdef cpp_set[string] _registered_op_names

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
# out = scalar <op> tensor（标量在左）：参数顺序刻意「标量在前」，避免和上面混淆
ctypedef aclError (*ReverseScalarBinaryOpFunc)(const aclScalar* self, const aclTensor* other,
    aclTensor* out, aclrtStream stream)

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
    ReverseScalarBinaryOpFunc reverse_scalar_binary_op
    TernaryOpFunc tri_op
    InplaceTernaryOpFunc inplace_tri_op
    ReductionOpFunc reduction_op
    GeneralOpFunc general_op

cdef aclError register_acl_ufunc(string opname, OpType op_type, FuncPtrUnion func_ptr) except * nogil:
    cdef OpInfo op_info
    op_info.op_name = opname
    op_info.op_type = op_type
    _registered_op_names.insert(opname)

    if _builtin_operators.find(op_info) != _builtin_operators.end():
        # 操作已存在，可以选择覆盖或报错, 这里我们选择覆盖
        _builtin_operators[op_info] = func_ptr
        return 0 # ACL_SUCCESS
    else:
        _builtin_operators[op_info] = func_ptr
        return 0

cdef OpType get_op_type(object ops, bint inplace, bint has_scalar = False,
                        bint scalar_is_lhs = False):
    # TODO: Ternary op, has_scalar
    if has_scalar:
        if len(ops) == 3 and not inplace:  # 二元操作
            # 标量在左还是右决定用哪个槽位：`x - 1`(SCALAR) vs `1 - x`(REVERSE)
            if scalar_is_lhs:
                return REVERSE_SCALAR_BINARY_OP
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

#: 交换律成立的算子：`scalar <op> tensor` 与 `tensor <op> scalar` 结果相同，
#: 因此没有注册 REVERSE 变体时可以直接回退到 SCALAR_BINARY_OP。
#: 不在这张表里的（sub / div / floor_divide / fmod / remainder / pow /
#: greater / less / ...）必须显式注册 reverse 实现，否则报错而不是算错。
_COMMUTATIVE_OPS = frozenset((
    'add', 'multiply', 'maximum', 'minimum', 'fmax', 'fmin',
    'logical_and', 'logical_or', 'logical_xor',
    'bitwise_and', 'bitwise_or', 'bitwise_xor',
    'hypot', 'logaddexp', 'logaddexp2', 'gcd', 'lcm',
    'equal', 'not_equal',
))

# ---------------------------------------------------------------------------
# 错误传递：aclnn/acl 的非零返回码 -> Python 异常
#
# 以前三个派发函数只 print 一行就 return ret，调用方（`_core/_ascend/_kernel.pyx`）
# 连返回值都丢掉，于是「算子没跑成」和「结果是对的」在 Python 侧完全一样 ——
# 典型的静默错误结果（例：`CreateAclScalar` 造不出 uint16 scalar 时把 nullptr
# 交给 aclnn，只剩一段 stderr）。
# 现在统一升级成 RuntimeError，消息带 opname、返回码和 aclGetRecentErrMsg()
# （CANN 侧最近的错误描述，通常是「dtype 不支持 / 参数非法」）。
# 完整的跨语言异常方案见 docs/ascend/refactor_exception.md。
# ---------------------------------------------------------------------------
cdef str _acl_recent_errmsg():
    """aclGetRecentErrMsg() 的安全包装：取不到就返回空串，绝不抛。"""
    cdef const char* msg = aclGetRecentErrMsg()
    if msg == NULL:
        return ''
    try:
        return (<bytes>msg).decode('utf-8', 'replace')
    except Exception:
        return ''


cdef void raise_acl_op_error(str opname, long ret) except *:
    """把算子失败转成 Python 异常（资源回收由调用方的 finally 负责）。"""
    cdef str detail = _acl_recent_errmsg()
    if detail:
        raise RuntimeError(
            '{}: aclnn/acl op failed with ret={} '
            '(aclGetRecentErrMsg: {})'.format(opname, ret, detail))
    raise RuntimeError(
        '{}: aclnn/acl op failed with ret={} '
        '(aclGetRecentErrMsg returned no detail)'.format(opname, ret))


# ---------------------------------------------------------------------------
# 无符号整型 I/O 提升（docs/ascend/AscendSpecialization.md §1）
#
# 部分 aclnn 算子不支持无符号整型（UINT8/16/32/64）。三个 launch_*_raw 派发
# 入口在这里统一拦截：
#   1. _promote_io_dtype —— uint 输入 astype 成有符号、uint 输出新建有符号
#      临时数组；
#   2. 算子在有符号 dtype 上执行；
#   3. _cast_back_outs —— 把有符号临时 out 的结果 cast 回调用方原来的 uint
#      数组（走已注册的 ascend_cast）。
# 开销是每次调用多两次 cast kernel：migration analyzer 应建议用户直接用
# 有符号 dtype 规避（见 AscendSpecialization.md）。
# ---------------------------------------------------------------------------

# 豁免拦截的算子：本身原生支持 uint（或作为本机制的实现载体），提升反而
# 多余/递归。后续发现新的原生支持 uint 的算子，直接往这个 set 里加名字。
_UINT_PROMOTE_EXEMPT_OPS = {
    # 本机制的实现载体：_promote_io_dtype 的 astype 与 _cast_back_outs 都
    # 落到它，aclnnCast 原生支持 uint —— 不豁免会无限递归。
    'ascend_cast',
    # aclop_Copy 同样是 aclnnCast 实现（见 acl_general_ops.h），原生支持 uint
    'ascend_copy',
    # 参数通道探针：只记录不计算，无需提升
    'ascend_dump_args',
}

cdef dict _ASCEND_DTYPE_PROMOTE = {
    'B': 'i',   # uint8 -> 平台 int；unsigned char 的算子支持性不确定
    'H': 'i',   # uint16 -> int；部分算子连 int16 都不支持，不提升到 'h'
    'I': 'i',   # uint32 -> int
    'Q': 'q',   # uint64 -> int64
}


cdef bint _has_promotable_uint(sequence arrs):
    cdef object a
    for a in arrs:
        if isinstance(a, _ndarray_base) and a.dtype.char in _ASCEND_DTYPE_PROMOTE:
            return True
    return False


cdef tuple _promote_io_dtype(str opname, sequence ins, sequence outs):
    """uint 输入提升为有符号、uint 输出新建有符号临时数组。

    返回 ``(p_ins, p_outs, orig_outs, cast_src)``：

    * ``p_ins``/``p_outs`` —— 提升后的 ins/outs（位置与原列表一一对应，
      非 ndarray 操作数如标量原样保留）；
    * ``orig_outs`` —— ``[(下标, 原 uint 数组), ...]``，供 _cast_back_outs
      把结果写回调用方数组；
    * ``cast_src`` —— cast-back 的来源列表：有 out 用提升后的 outs；
      无 out 且算子名含 ``inplace``（如 ``ascend_inplace_add`` 的
      ``a += b`` 形式）时结果写在提升后的 ``ins[0]`` 里，来源是 ins。

    astype/empty 都走创建或 ascend_cast 路径（有符号目标），不会被本
    拦截再次提升，无递归。
    """
    import cupy as _cupy_mod
    cdef list p_ins = list(ins)
    cdef list p_outs = list(outs)
    cdef list orig_outs = []
    cdef Py_ssize_t i
    cdef object a, c, orig_in0 = None
    # in-place 约定：outs 为空且算子名含 inplace 时，ins[0] 既是入参也是出参
    cdef bint inplace = (not p_outs) and ('inplace' in opname) and (len(p_ins) > 0)
    if inplace:
        orig_in0 = p_ins[0]
    for i in range(len(p_ins)):
        a = p_ins[i]
        if isinstance(a, _ndarray_base):
            c = a.dtype.char
            if c in _ASCEND_DTYPE_PROMOTE:
                p_ins[i] = a.astype(_ASCEND_DTYPE_PROMOTE[c])
    for i in range(len(p_outs)):
        a = p_outs[i]
        if isinstance(a, _ndarray_base):
            c = a.dtype.char
            if c in _ASCEND_DTYPE_PROMOTE:
                orig_outs.append((i, a))
                p_outs[i] = _cupy_mod.empty(a.shape, dtype=_ASCEND_DTYPE_PROMOTE[c])
    if inplace and p_ins[0] is not orig_in0:
        orig_outs.append((0, orig_in0))
        return p_ins, p_outs, orig_outs, p_ins
    return p_ins, p_outs, orig_outs, p_outs


cdef aclError _cast_back_outs(list orig_outs, list cast_src, intptr_t stream_ptr) except *:
    """把有符号临时结果 cast 回调用方原来的 uint 数组（ascend_cast）。"""
    cdef aclError ret
    cdef Py_ssize_t idx
    cdef object orig, pout
    for idx, orig in orig_outs:
        pout = cast_src[idx]
        ret = launch_general_func_raw('ascend_cast', [pout], [orig], [], {}, stream_ptr)
        if ret != 0:
            return ret
    return 0


#: 标量操作数按 0-d 张量物化的算子（M3「标量无法重建 0-d aclTensor」限制的
#: 派发层豁免，见 launch_general_func_raw 的 ins 循环）。cupy.where 的标量
#: x/y（如 where(mask, 0, 1)）由此可达 aclnnSWhere。需要同样处理的算子往
#: 这个 set 加名字。
_SCALAR_AS_TENSOR_OPS = {
    'ascend_where',
}


cdef aclError launch_general_func_raw(str opname, sequence ins, sequence outs, list args, dict kwargs, intptr_t stream_ptr) except *:
    if opname.startswith("cupy_"):
        opname = ASCEND_OP_PREFIX + opname[5:]
    # custom AscendC kernel registry takes precedence (B-tier ops with no
    # aclnn counterpart; see cupy/backends/ascend/kernels/__init__.py)
    if _custom_kernel_specs:
        cspec = _custom_kernel_specs.get(opname)
        if cspec is not None:
            return _launch_custom_ufunc(opname, cspec, ins, outs, stream_ptr)
    cdef OpInfo op_info
    cdef FuncPtrUnion func_ptr
    op_info.op_name = opname.encode("utf-8")
    op_info.op_type = OpType.GENERAL_OP
    if _builtin_operators.find(op_info) == _builtin_operators.end():
        # 窄签名路径：同样只传错误码（检查在外层 launch_general_func 做）
        return launch_acl_func_raw(opname, ins, outs, args, kwargs, stream_ptr)
    func_ptr = _builtin_operators[op_info]

    # 无符号整型拦截（AscendSpecialization.md §1）：豁免算子见
    # _UINT_PROMOTE_EXEMPT_OPS。放在 fall-through 之后：promote 后 ins/outs
    # 已无 uint，即使落到下面的分支路径再被拦截也是 no-op，cast-back 责任
    # 只属于发起 promote 的这一层。
    cdef list _orig_outs
    cdef list _cast_src
    cdef aclError _cret
    cdef bint _promoted = False
    if opname not in _UINT_PROMOTE_EXEMPT_OPS and (_has_promotable_uint(ins) or _has_promotable_uint(outs)):
        ins, outs, _orig_outs, _cast_src = _promote_io_dtype(opname, ins, outs)
        _promoted = True

    cdef ArgsType acl_args
    cdef KwargsType acl_kwargs
    cdef const aclTensor* ct
    cdef vector[const aclTensor*] intensors
    cdef vector[aclTensor*] outtensors
    cdef aclError ret = 0
    cdef aclrtStream stream = <aclrtStream>NULL  # default stream always working
    if stream_ptr != <intptr_t>0:
        stream = <aclrtStream>stream_ptr
    # NOTE: 所有资源获取都必须在 try 内，任何一步失败都由 finally 回收；
    # 否则已创建的 tensor 会滞留在 cupy_acl_tensor_owners 里，连同 ndarray
    # 一起永久泄漏（原来的 tensor 创建循环在 try 之外）。
    try:
        acl_kwargs = _create_keyword_args(kwargs, opname)
        for op in ins:
            typ = type(op)
            if issubclass(typ, _ndarray_base):
                intensors.push_back(cupy_ndarray_to_acl_tensor(op))
            elif opname in _SCALAR_AS_TENSOR_OPS and typ is _cupy_scalar:
                # 标量操作数物化成 0-d 设备数组：CScalar 已按 loop dtype
                # 物化（M-D1），取回 numpy 标量后 cupy.array 走创建路径的
                # H2D memcpy（不经 ufunc，无递归）。已知局限：混合 dtype
                # （如 float 标量 + float32 数组）时 aclnnSWhere 仍会因
                # x/y/out dtype 不一致拒绝 —— 与 CUDA 路径的 in-kernel cast
                # 不同，这是 Ascend 派发层的已知差距。
                s = <_cupy_scalar>op
                import numpy as _numpy_mod
                import cupy as _cupy_mod
                buf = _numpy_mod.empty(s.size, dtype=_numpy_mod.uint8)
                memcpy(<void*>buf.ctypes.data, <const void*>s.ptr, s.size)
                ns = _numpy_mod.frombuffer(buf, dtype=s.get_numpy_type())[0]
                intensors.push_back(cupy_ndarray_to_acl_tensor(
                    _cupy_mod.array(ns)))
            else:
                # 操作数里的非 ndarray 只能是标量（-1 * x 之类），
                # 其余类型仍然是响亮失败（_convert_arg_strict）
                ascalar = _convert_arg_strict(opname, <str>('operand %s' % type(op).__name__), op)
                if ascalar:
                    acl_args.push_back(MakeScalarArg(ascalar))
        for i, pos_arg in enumerate(args):
            # 统一参数通道：scalar / int 序列 / str / None（见 _convert_arg）
            acl_args.push_back(_convert_arg(opname, <str>('#%d' % i), pos_arg))
        for op in outs:
            typ = type(op)
            if issubclass(typ, _ndarray_base):
                outtensors.push_back(cupy_ndarray_to_acl_tensor(op))

        ret = func_ptr.general_op(intensors, outtensors, acl_args, acl_kwargs, stream)
    finally:
        # aclDestroyTensor does not deallocate array buffer, but shapes, strides
        for i in range(intensors.size()):
            ct = <const aclTensor*>intensors.at(i)
            cupy_destroy_acl_tensor(ct)
        for t in outtensors:
            cupy_destroy_acl_tensor(t)
        _delete_args(acl_args)
        _delete_keyword_args(acl_kwargs)
    # NOTE: acl/aclnn 的失败**不在这里抛**，而是把错误码原样返回给 caller
    # （见本文件 §错误传递 与 docs/ascend/refactor_exception.md S2）：
    #   * 这一层因此可以是 noexcept 的（C/C++ 调用方不承担跨语言异常）；
    #   * Python 路径用「默认入口」launch_general_func（检查 + 抛，见本文件末
    #     的派发入口说明）。
    # 仍然会抛的是**调用方 bug**（参数不可转换、未知 key、op 未注册），
    # 那些是 Cython 侧的参数校验，不属于 acl 错误码体系。
    # uint 提升：算子成功后把有符号临时结果 cast 回调用方的 uint 数组
    if _promoted and ret == 0:
        _cret = _cast_back_outs(_orig_outs, _cast_src, stream_ptr)
        if _cret != 0:
            ret = _cret
    return ret

cdef vector[aclTensor*] _create_ops_vector(sequence ins, sequence outs) except *:
    cdef vector[aclTensor*] tensors
    cdef aclTensor* t

    # 中途失败时，已经建好的 aclTensor 只存在于这个局部 vector 里（C++ vector
    # 的析构不会销毁 aclTensor），必须就地回收，否则它们会滞留在
    # cupy_acl_tensor_owners 里，连同 ndarray 一起泄漏。
    try:
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
    except Exception:
        for t in tensors:
            cupy_destroy_acl_tensor(t)
        raise

    return tensors

# ---------------------------------------------------------------------------
# custom AscendC kernel ufuncs (plan.md B-tier ops without an aclnn op)
# spec: {'bin': str, 'entry': str, 'n_out': int, 'n_in': int, 'dtypes': tuple}
# ---------------------------------------------------------------------------
cdef dict _custom_kernel_specs = {}


def py_register_custom_kernel(str opname, str bin_path, str entry,
                              int n_out, int n_in, tuple dtypes=()):
    """Register a custom AscendC kernel as an ``ascend_<name>`` ufunc impl."""
    _custom_kernel_specs[opname] = {
        'bin': bin_path, 'entry': entry,
        'n_out': n_out, 'n_in': n_in, 'dtypes': dtypes,
    }


def py_list_custom_kernels() -> list:
    return sorted(_custom_kernel_specs)


cdef str _arg_kind_name(AclArg& arg, object value):
    """把 AclArg 的 tag 映射成测试可见的字符串。

    宽松模式（CUPY_ASCEND_LENIENT_ARGS=1）下不可转换的值会退化成 ARG_NONE，
    这里用 ``value is not None`` 把它和真正的 ``None`` 区分开（保留 'unsupported'）。
    """
    if arg.kind == ARG_NONE:
        return 'none' if value is None else 'unsupported'
    if arg.kind == ARG_SCALAR:
        return 'scalar'
    if arg.kind == ARG_INT_ARRAY:
        return 'int_array'
    if arg.kind == ARG_STRING:
        return 'string'
    if arg.kind == ARG_TENSOR:
        return 'tensor'
    return 'unknown'


def py_describe_args(str opname, tuple args=(), dict kwargs=None) -> list:
    """按 dispatch 路径的同一套规则校验/描述参数（测试与调试用，无需 NPU）。

    返回 ``[(name, kind, repr), ...]``，其中 ``kind`` 为统一参数通道的 tag：

    * ``'scalar'``      —— 数值/布尔/numpy 标量（ARG_SCALAR）
    * ``'int_array'``   —— int 序列（ARG_INT_ARRAY，例如 axis=(0, 1)）
    * ``'string'``      —— 字符串且该算子已在 `_STRING_ARG_OPS` 白名单里（ARG_STRING）
    * ``'none'``        —— 显式 None（ARG_NONE）
    * ``'unsupported'`` —— 仅当 ``CUPY_ASCEND_LENIENT_ARGS=1`` 时才会出现

    与真实路径一致：不可转换的参数、未知 key、未声明的字符串参数都会**抛错**。
    """
    cdef list out = []
    cdef AclArg arg
    if kwargs is None:
        kwargs = {}
    for i, value in enumerate(args):
        arg = _convert_arg(opname, <str>('#%d' % i), value)
        out.append(('#%d' % i, _arg_kind_name(arg, value), repr(value)))
        _destroy_arg(arg)
    for key, value in kwargs.items():
        if key not in _KNOWN_SCALAR_KEYS and not _lenient_args():
            raise ValueError(
                f"{opname}: 未知参数 key {key!r}（不在 C++ 侧消费的 key 白名单内）")
        arg = _convert_arg(opname, key, value)
        out.append((key, _arg_kind_name(arg, value), repr(value)))
        _destroy_arg(arg)
    return out


def py_dump_args(tuple args=(), dict kwargs=None) -> str:
    """测试用：调用参数通道探针 ``ascend_dump_args``，返回 C++ 侧记录到的参数描述。

    无 NPU 可跑（没有任何 aclnn 计算），用来证明 scalar / int 序列 / str / None
    真的按 tag 送达了 C++ 层。
    """
    if kwargs is None:
        kwargs = {}
    launch_general_func("ascend_dump_args", [], [], list(args), dict(kwargs), 0)
    return py_last_dump_args()


def py_last_dump_args() -> str:
    """测试用：上一次 ``ascend_dump_args`` 记录的内容（C++ 侧全局缓冲）。"""
    cdef const char* msg = aclop_GetLastDumpArgs()
    if msg == NULL:
        return ''
    return (<bytes>msg).decode('utf-8', 'replace')


cdef aclError _launch_custom_ufunc(str opname, dict spec, sequence ins,
                                   sequence outs, intptr_t stream_ptr) except *:
    cdef bytes bin_path = spec['bin'].encode('utf-8')
    cdef bytes entry = spec['entry'].encode('utf-8')
    cdef int n_out = spec['n_out']
    cdef int n_in = spec['n_in']
    cdef tuple dtypes = spec['dtypes']

    # v1 limitation: array operands only; scalar promotion is not supported yet
    cdef list a_ins = list(ins)[:n_in]
    cdef list a_outs = list(outs)[:n_out]
    for op in a_ins + a_outs:
        if not isinstance(op, _ndarray_base):
            raise NotImplementedError(
                f'custom AscendC kernel {opname!r} supports array operands '
                f'only (got {type(op).__name__}); wrap scalars with '
                'cupy.asarray(...) explicitly')
    # 实数输入分支：complex 专用 AscendC 内核按 complex64 的 f32 交错布局取
    # 实/虚部，实数输入走不进来；但 NumPy 对实数输入的语义是平凡的，用已注册
    # 的 aclnn 算子组合即可（与 cupy/_core/_routines_math.pyx 的 ufunc 体一致）：
    #   conjugate(x) = x                       -> ascend_copy (aclnnCopy)
    #   imag(x)      = 0                       -> ascend_fill (aclnnInplaceFillScalar)
    #   angle(x)     = arctan2(0, x)           -> ascend_arctan2 (x>=0 -> 0,
    #                  否则 pi, 与 'in0 >= 0 ? 0 : M_PI' 等价)
    # 复数输入继续走 AscendC 内核（本分支不触发）。
    # 注意必须在下面的 out-dtype 检查之前：实数输入的 out dtype 不在
    # dtypes（complex/f32）白名单里，会被先拒掉。
    if a_ins and a_ins[0].dtype.kind != 'c':
        if opname == 'ascend_conjugate':
            return launch_acl_func_raw('ascend_copy', ins, outs, [], {}, stream_ptr)
        if opname == 'ascend_imag':
            return launch_general_func_raw('ascend_fill', ins, outs, [0], {}, stream_ptr)
        if opname == 'ascend_angle':
            import cupy as _cupy_mod
            zeros = _cupy_mod.zeros(a_ins[0].shape, a_ins[0].dtype)
            return launch_acl_func_raw(
                'ascend_arctan2', [zeros] + list(a_ins), outs, [], {}, stream_ptr)
    if dtypes and a_outs and a_outs[0].dtype.char not in dtypes:
        raise NotImplementedError(
            f'custom AscendC kernel {opname!r} supports out dtype '
            f'{dtypes} (got {a_outs[0].dtype})')

    cdef uint64_t n = 1
    if a_outs:
        n = <uint64_t> a_outs[0].size
    elif a_ins:
        n = <uint64_t> a_ins[0].size
    if n == 0:
        return 0

    cdef void* out0 = NULL
    cdef void* out1 = NULL
    cdef void* in0 = NULL
    cdef void* in1 = NULL
    if n_out > 0:
        out0 = <void*><uintptr_t>a_outs[0].data.ptr
    if n_out > 1:
        out1 = <void*><uintptr_t>a_outs[1].data.ptr
    if n_in > 0:
        in0 = <void*><uintptr_t>a_ins[0].data.ptr
    if n_in > 1:
        in1 = <void*><uintptr_t>a_ins[1].data.ptr

    cdef aclrtStream stream = <aclrtStream>NULL
    if stream_ptr != <intptr_t>0:
        stream = <aclrtStream>stream_ptr

    cdef aclError ret = 0
    cdef const char* bin_c = bin_path
    cdef const char* entry_c = entry
    with nogil:
        ret = aclop_LaunchCustomKernel(
            bin_c, entry_c, out0, out1, in0, in1, n, stream)
    if ret != 0:
        raise RuntimeError(f'custom AscendC kernel {opname!r} launch failed: {ret}')
    return 0


cdef aclError launch_acl_func_raw(str opname, sequence ins, sequence outs, list args, dict kwargs, intptr_t stream_ptr) except *:
    # M1 止血：这条路径（UNARY/BINARY/SCALAR/INPLACE 注册表）目前**没有参数通道**，
    # 原来 args/kwargs 被完全丢弃（生成代码里是 CYTHON_UNUSED，见 review §2.2）。
    # 丢弃 = 参数没生效但算子照跑 = 静默错误结果，所以先显式报错；
    # 统一参数通道（unified_op）在 arg_passing_plan.md M2 实现。
    if (args or kwargs) and not _lenient_args():
        raise NotImplementedError(
            f"{opname}: 该算子走窄签名派发（无参数通道），但收到 args={list(args)!r} / "
            f"kwargs={dict(kwargs)!r}；这些参数会被丢弃导致结果错误，故直接报错。"
            f"（迁移期可设 CUPY_ASCEND_LENIENT_ARGS=1 恢复旧行为）")
    #
    # 无符号整型拦截（AscendSpecialization.md §1）：豁免算子见
    # _UINT_PROMOTE_EXEMPT_OPS。
    cdef list _orig_outs
    cdef list _cast_src
    cdef aclError _cret
    cdef bint _promoted = False
    if opname not in _UINT_PROMOTE_EXEMPT_OPS and (_has_promotable_uint(ins) or _has_promotable_uint(outs)):
        ins, outs, _orig_outs, _cast_src = _promote_io_dtype(opname, ins, outs)
        _promoted = True
    cdef aclScalar* scalar_ptr = NULL
    cdef OpInfo op_info
    cdef FuncPtrUnion func_ptr
    cdef Py_ssize_t scalar_index = -1
    cdef Py_ssize_t n_scalars = 0
    cdef bint scalar_is_lhs
    cdef OpType fallback_op_type
    op_info.op_name = opname.encode("utf-8")
    op_info.op_type = OpType.GENERAL_OP

    # 区分scalar 和tensor 操作数, 应该是Broadcast应该处理的事情
    # inplace op 是ASCEND引入的?
    # NOTE: 同时记录标量的**位置**：`1 - x` 与 `x - 1` 都是「二元 + 标量」，
    # 但方向相反。以前只看「有没有标量」，于是 `2 / x` 被算成 `x / 2`（静默算错）。
    for i, op in enumerate(ins):
        typ = type(op)
        if typ is _cupy_scalar:
            if scalar_index < 0:
                scalar_index = i
            n_scalars += 1
            scalar_ptr = cupy_scalar_to_acl_scalar(op)

    cdef has_scalar = scalar_ptr != NULL
    # cupy inplace op does not generate a new op, but make self == out
    cdef bint inplace = ("inplace" in opname) or not outs
    cdef list ops = ins + outs
    # 标量在左操作数（ins[0]），且是标准的 2-in/1-out 形式
    scalar_is_lhs = has_scalar and n_scalars == 1 and scalar_index == 0

    op_info.op_type = get_op_type(ops, inplace, has_scalar, scalar_is_lhs)
    if _builtin_operators.find(op_info) == _builtin_operators.end():
        # 交换律算子（add/multiply/maximum...）的 reverse 与正向等价，回退即可；
        # 其余算子必须先注册 REVERSE 变体（C++ 侧 aclop_R*），否则宁可报错也不能算错。
        if (scalar_is_lhs and opname[len(ASCEND_OP_PREFIX):] in _COMMUTATIVE_OPS):
            fallback_op_type = SCALAR_BINARY_OP
            op_info.op_type = fallback_op_type
        if _builtin_operators.find(op_info) == _builtin_operators.end():
            # scalar 已经创建出来了，抛错前必须回收，否则泄漏
            _destroy_acl_scalar(scalar_ptr)
            if scalar_is_lhs:
                raise NotImplementedError(
                    _no_ascend_impl_msg(opname)
                    + f" (scalar-operand-on-the-left form: '{opname}' has no "
                      f"REVERSE_SCALAR_BINARY_OP implementation; register one "
                      f"with register_acl_ufunc(\"{opname}\", "
                      f"REVERSE_SCALAR_BINARY_OP, ...) and an aclop_R* wrapper)")
            raise NotImplementedError(
                _no_ascend_impl_msg(opname)
                + f" (looked up {op_info.op_type} with len(ops)={len(ops)}, "
                  f"inplace={inplace}, has_scalar={has_scalar})")
    
    func_ptr = _builtin_operators[op_info]
    cdef aclError ret = 0
    cdef aclrtStream stream = <aclrtStream>NULL  # default stream always working
    if stream_ptr != <intptr_t>0:
        stream = <aclrtStream>stream_ptr

    # 转换为ACL张量列表
    # NOTE: 传 (ins, outs) 而不是 (ops, outs) —— ops 已经等于 ins+outs，
    # 传 ops 会让每个 out 被创建两个 aclTensor（review §2.3）。
    tensors = _create_ops_vector(ins, outs)

    try:
        if len(ops) == 3 and not has_scalar and not inplace:  # 二元操作
            if op_info.op_type != BINARY_OP:
                raise RuntimeError(f"Operator {opname} is not a binary operator")
            ret = func_ptr.binary_op(tensors[0], tensors[1], tensors[2], stream)
        elif len(ops) == 2 and inplace:  # 原地二元操作
            if op_info.op_type != INPLACE_BINARY_OP:
                raise RuntimeError(f"Operator {opname} is not an inplace binary operator")
            ret = func_ptr.inplace_binary_op(tensors[0], tensors[1], stream)

        elif len(ops) == 3 and has_scalar and not inplace:
            # tensors = [tensor_operand, out]（标量不建 tensor），顺序与 ins 中的
            # ndarray 顺序一致，所以两种方向都能直接取用。
            if op_info.op_type == REVERSE_SCALAR_BINARY_OP:  # out = scalar <biop> tensor
                ret = func_ptr.reverse_scalar_binary_op(scalar_ptr, tensors[0], tensors[1], stream)
            elif op_info.op_type == SCALAR_BINARY_OP:  # out = tensor <biop> scalar
                ret = func_ptr.scalar_binary_op(tensors[0], scalar_ptr, tensors[1], stream)
            else:
                raise RuntimeError(f"Operator {opname} is not a scalar binary operator")
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
            cupy_destroy_acl_tensor(t)
        _destroy_acl_scalar(scalar_ptr)

    # NOTE: 同 launch_general_func —— 返回错误码，不抛（Python 路径用 checked 版本）
    # uint 提升：算子成功后把有符号临时结果 cast 回调用方的 uint 数组
    if _promoted and ret == 0:
        _cret = _cast_back_outs(_orig_outs, _cast_src, stream_ptr)
        if _cret != 0:
            ret = _cret
    return ret


cdef void _parse_reduction_axes(object axes, object in0, vector[int64_t]& shape) except *:
    """把 reduction 的 ``axis`` 实参解析成 aclnn ``dim`` 用的整数列表（写入 shape）。

    接受：shape_t（vector[Py_ssize_t]）/ 整型 CScalar / None（= 沿全部轴，按
    ``in0`` 的 ndim 展开）/ int / tuple|list[int] / 0-d numpy 标量；其余
    （float/bool/complex 轴、多元素 ndarray 等）响亮报错。

    shape 可能为空：仅当 axis=None 且 in0 是 0-d（没有轴可归约）。
    从 launch_reduction_op_raw 抽出，供其他需要 axis->dim 解析的路径共享
    （测试入口见 py_parse_reduction_axes）。
    """
    cdef _cupy_scalar ax_scalar
    cdef int64_t ax_val = 0
    cdef Py_ssize_t n_dim = 0
    cdef Py_ssize_t i
    typ = type(axes)
    if hasattr(axes, 'size') and hasattr(axes, 'push_back'):
        # dim/axes info from `shape_t` which is `vector.vector[Py_ssize_t]`
        for i in range(axes.size()):
            shape.push_back(axes[i])
    elif typ is _cupy_scalar:
        # reduction 的 axis 以 CScalar 传入。旧实现是 `pass`：shape 落空后走
        # 兜底 push_back(0) —— 无论请求哪个轴都按 0 归约（静默错误结果）。
        # numpy 语义要求 axis 是整数：从 CScalar 的 kind/size 读出整数值；
        # float/bool/complex 轴显式报错（cupy gpu 侧在 python 层就归一成 int，
        # 见 _reduction.pyx::_get_axis -> internal._normalize_axis_index）。
        ax_scalar = <_cupy_scalar>axes
        ax_val = 0
        if ax_scalar.kind == 'i':
            if ax_scalar.size == 8:
                ax_val = (<int64_t*>ax_scalar.ptr)[0]
            elif ax_scalar.size == 4:
                ax_val = (<int32_t*>ax_scalar.ptr)[0]
            elif ax_scalar.size == 2:
                ax_val = (<int16_t*>ax_scalar.ptr)[0]
            elif ax_scalar.size == 1:
                ax_val = (<int8_t*>ax_scalar.ptr)[0]
            else:
                raise TypeError(
                    f'reduction axis: unsupported int width {ax_scalar.size}')
        elif ax_scalar.kind == 'u':
            if ax_scalar.size == 8:
                ax_val = <int64_t>(<uint64_t*>ax_scalar.ptr)[0]
            elif ax_scalar.size == 4:
                ax_val = <int64_t>(<uint32_t*>ax_scalar.ptr)[0]
            elif ax_scalar.size == 2:
                ax_val = <int64_t>(<uint16_t*>ax_scalar.ptr)[0]
            elif ax_scalar.size == 1:
                ax_val = <int64_t>(<uint8_t*>ax_scalar.ptr)[0]
            else:
                raise TypeError(
                    f'reduction axis: unsupported uint width {ax_scalar.size}')
        else:
            raise TypeError(
                'reduction axis must be an integer scalar, got kind '
                f'{ax_scalar.kind!r} ({type(axes).__name__})')
        shape.push_back(ax_val)
    elif axes is None:
        # axis=None：numpy 语义 = 沿**全部轴**归约（_reduction._get_axis(None,
        # ndim) 亦然）。旧实现 push_back(0) 只归约第 0 轴 —— ndim>=2 的
        # `a.sum()` 会算错或被 aclnn 以 out 形状不符拒绝。展开成 range(ndim)；
        # 0-d 输入留空 shape（由调用方决定兜底，通常 dim=[0] 会被 aclnn
        # 响亮拒绝）。
        n_dim = getattr(in0, 'ndim', 0)
        for i in range(n_dim):
            shape.push_back(<int64_t>i)
    elif typ is int: # python integer object
        shape.push_back(axes) # auto converstion from python int to c int64_t
    elif isinstance(axes, (tuple, list)):  # TODO: not sure if numpy.ndarray/cupy.ndarray should be supported
        for ax in axes:
            shape.push_back(<int64_t>ax)
    elif hasattr(axes, 'item') and hasattr(axes, 'dtype') and getattr(axes, 'ndim', 1) == 0:
        # numpy 标量轴（np.int64(1) 等）：原先落到 else 的 RuntimeError。
        # 仅接受整数（bool/float 拒绝，与 numpy 对 axis 的要求一致）；
        # 多元素 ndarray 轴仍走 else 响亮失败。
        ax_item = axes.item()
        if not isinstance(ax_item, int):
            raise TypeError(
                f'reduction axis must be an integer, got {type(axes).__name__}')
        shape.push_back(<int64_t>ax_item)
    else:
        raise TypeError(
            f'reduction axis must be int / tuple[int] / None / shape_t, '
            f'got {type(axes).__name__}: {axes!r}')


def py_parse_reduction_axes(object axes, object in0=None) -> list:
    """测试/调试用：暴露 reduction 的 axes -> dim 解析（无需 NPU）。

    返回解析出的 dim 列表（list[int]）；解析失败抛 TypeError/RuntimeError。
    """
    cdef vector[int64_t] shape
    _parse_reduction_axes(axes, in0, shape)
    return [shape[i] for i in range(shape.size())]


#: aclnn 支持性补丁（见 launch_reduction_op_raw 的插入点）：aclnnAny/aclnnAll
#: 不支持 complex64/128 与 float64 输入，派发前把输入 astype('?')。
#: 语义等价（any/all 只关心是否为零）。需要同样处理的算子往这个 set 加名字。
_BOOL_CAST_INPUT_OPS = frozenset((
    'ascend_any',
    'ascend_all',
))


cdef aclError launch_reduction_op_raw(str opname, sequence ins, sequence outs, object axes, bint keepdims, dict kwargs, intptr_t stream_ptr) except *:
    # 检查操作是否已注册
    if opname.startswith("cupy_"):
        opname = ASCEND_OP_PREFIX + opname[5:]

    cdef OpInfo op_info
    op_info.op_name = opname.encode("utf-8")
    op_info.op_type = REDUCTION_OP
    if _builtin_operators.find(op_info) == _builtin_operators.end():
        raise NotImplementedError(_no_ascend_impl_msg(opname))

    cdef FuncPtrUnion func_ptr = _builtin_operators[op_info]
    cdef aclError ret = 0
    cdef aclrtStream stream = <aclrtStream>NULL  # default stream always working
    if stream_ptr != <intptr_t>0:
        stream = <aclrtStream>stream_ptr

    cdef vector[int64_t] shape
    cdef aclIntArray* dim = NULL
    cdef vector[aclTensor*] tensors

    # REDUCTION_OP 的 C++ 签名固定为 (self, dim, keepdim, out, kwargs, stream)，
    # 即恰好 1 输入 1 输出。多输入/多输出的 ReductionKernel 若放行，tensors[1]
    # 会拿到第二个输入（而非输出）并静默算错，必须在这里显式拒绝。
    # NOTE: 守卫与 axes 解析都放在 _create_ops_vector 之前 —— 本函数里任何
    # raise 都不能发生在 aclTensor 创建之后（否则泄漏）。
    if len(ins) != 1 or len(outs) != 1:
        raise NotImplementedError(
            _no_ascend_impl_msg(opname)
            + f" (reduction requires exactly 1 input and 1 output, "
              f"got {len(ins)} input(s) / {len(outs)} output(s))")

    # NOTE: 特殊处理的插入点（dtype 能力补丁）—— 未来重构时注意。
    # aclnnAny/aclnnAll 不支持 complex64/128 与 float64 输入，这里在**派发层**
    # 把输入 astype('?') 再派发（cast 走 ascend_cast，不经过本函数，无递归）。
    # 语义等价：any/all 只关心元素是否为零，非零即真。
    #   为什么放在这里：所有 reduction 入口都会经过本函数，补丁一处即全覆盖；
    #   重构方向：这类「算子 dtype 能力补丁」更适合放进
    #   _ascend/_reduction.pyx 的 _call（离 ufunc 语义更近、能拿到 ufunc 解析
    #   出的 out dtype），或做成算子声明表在 C++ 侧统一处理。
    if opname in _BOOL_CAST_INPUT_OPS and ins:
        a0 = ins[0]
        if isinstance(a0, _ndarray_base) and a0.dtype.kind in 'fc':
            ins = [a0.astype('?')] + list(ins[1:])


    # 无符号整型拦截（AscendSpecialization.md §1）：豁免算子见
    # _UINT_PROMOTE_EXEMPT_OPS
    cdef list _orig_outs
    cdef list _cast_src
    cdef aclError _cret
    cdef bint _promoted = False
    if opname not in _UINT_PROMOTE_EXEMPT_OPS and (_has_promotable_uint(ins) or _has_promotable_uint(outs)):
        ins, outs, _orig_outs, _cast_src = _promote_io_dtype(opname, ins, outs)
        _promoted = True

    # axes -> dim(IntArray) 解析：所有分支要么填 shape、要么响亮报错。
    # （抽成 _parse_reduction_axes 供其他 launch 路径共享；in0 只在 axis=None
    # 时用于取 ndim。）
    _parse_reduction_axes(axes, ins[0], shape)

    # 兜底只应服务于 0-d 输入的 axis=None（空 shape）：dim=[0] 对 0-d 越界，
    # aclnn 会响亮拒绝；其余分支的 shape 至少有一个元素。
    if not shape.size():
        shape.push_back(0)
    dim = aclCreateIntArray(shape.data(), shape.size())

    tensors = _create_ops_vector(ins, outs)
    cdef KwargsType acl_kwargs
    try:
        # 放进 try 内, 失败时由 finally 回收已创建的 dim/scalar
        acl_kwargs = _create_keyword_args(kwargs, opname)
        ret = func_ptr.reduction_op(tensors[0], dim, keepdims, tensors[1], acl_kwargs, stream)
    finally:
        # does not deallocate array buffer, but shapes, strides
        for t in tensors:
            cupy_destroy_acl_tensor(t)
        if dim:
            aclDestroyIntArray(dim)
        _delete_keyword_args(acl_kwargs)
    # NOTE: 同 launch_general_func —— 返回错误码，不抛（Python 路径用 checked 版本）
    # uint 提升：归约成功后把有符号临时结果 cast 回调用方的 uint 数组
    if _promoted and ret == 0:
        _cret = _cast_back_outs(_orig_outs, _cast_src, stream_ptr)
        if _cret != 0:
            ret = _cret
    return ret


# ---------------------------------------------------------------------------
# 派发的两层 API（见 docs/ascend/arg_passing_plan.md §2.5）
#
#   `launch_*`（**默认、推荐**）
#       返回 `aclError`，acl/aclnn 失败时抛 RuntimeError（带 `aclGetRecentErrMsg()`）。
#       「检查」是默认行为，所以**没有后缀** —— Python 路径直接用这个，
#       不会出现「算子没跑成但结果照用」的静默错误。
#
#   `launch_*_raw`（原语，noexcept 方向）
#       返回 `aclError`，失败时**不抛**，把 Ascend 错误码原样交给 caller。
#       给 C/C++ 侧（将来 noexcept 的边界）与「想自己决定怎么处理」的调用方用。
#
# 也就是：**错误码一定返回给 caller；抛不抛由选哪个入口决定**。
# ---------------------------------------------------------------------------
cdef aclError launch_general_func(str opname, sequence ins, sequence outs,
                                  list args, dict kwargs, intptr_t stream_ptr) except *:
    cdef aclError ret = launch_general_func_raw(opname, ins, outs, args, kwargs, stream_ptr)
    if ret != 0:
        raise_acl_op_error(opname, ret)
    return ret


cdef aclError launch_acl_func(str opname, sequence ins, sequence outs,
                              list args, dict kwargs, intptr_t stream_ptr) except *:
    cdef aclError ret = launch_acl_func_raw(opname, ins, outs, args, kwargs, stream_ptr)
    if ret != 0:
        raise_acl_op_error(opname, ret)
    return ret


cdef aclError launch_reduction_op(str opname, sequence ins, sequence outs,
                                  object axes, bint keepdims, dict kwargs,
                                  intptr_t stream_ptr) except *:
    cdef aclError ret = launch_reduction_op_raw(
        opname, ins, outs, axes, keepdims, kwargs, stream_ptr)
    if ret != 0:
        raise_acl_op_error(opname, ret)
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
    aclError aclop_RightShift(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_LeftShift(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)

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
    aclError aclop_IsNan(const aclTensor* self, aclTensor* out, aclrtStream stream)
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
    aclError aclop_Hypot(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_Copysign(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_Gcd(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_Lcm(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_PowTensorTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_RemainderTensorTensor(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)

    aclError aclop_Adds(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_Subs(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_Muls(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_Divs(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_PowTensorScalar(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_RemainderTensorScalar(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_FmodScalar(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)
    aclError aclop_FloorDivides(const aclTensor* self, const aclScalar* other, aclTensor* out, aclrtStream stream)

    # reverse scalar binary: out = scalar <op> tensor（标量在左操作数）
    aclError aclop_Rsubs(const aclScalar* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_RDivs(const aclScalar* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_RFloorDivides(const aclScalar* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_RFmodScalar(const aclScalar* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_RPowScalar(const aclScalar* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_RRemainderScalar(const aclScalar* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_RGtScalar(const aclScalar* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_RGeScalar(const aclScalar* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_RLtScalar(const aclScalar* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
    aclError aclop_RLeScalar(const aclScalar* self, const aclTensor* other, aclTensor* out, aclrtStream stream)

    aclError aclop_Neg(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceNeg(aclTensor* self,  aclrtStream stream)
    aclError aclop_Reciprocal(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceReciprocal(aclTensor* self,  aclrtStream stream)
    aclError aclop_Signbit(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Sign(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Abs(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Fabs(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Floor(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceFloor(aclTensor* self,  aclrtStream stream)
    aclError aclop_Ceil(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceCeil(aclTensor* self,  aclrtStream stream)

    aclError aclop_Acosh(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceAcosh(aclTensor* self,  aclrtStream stream)
    aclError aclop_Asinh(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceAsinh(aclTensor* self,  aclrtStream stream)
    aclError aclop_Atanh(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceAtanh(aclTensor* self,  aclrtStream stream)
    aclError aclop_Exp2(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceExp2(aclTensor* self,  aclrtStream stream)
    aclError aclop_Sqrt(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceSqrt(aclTensor* self,  aclrtStream stream)

    aclError aclop_Trunc(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceTrunc(aclTensor* self,  aclrtStream stream)
    aclError aclop_Rint(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_InplaceRint(aclTensor* self,  aclrtStream stream)
    aclError aclop_Real(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Cbrt(const aclTensor* self,  aclTensor* out, aclrtStream stream)

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

    # CANN has aclnnRightShift but no left-shift op.
    func_union.binary_op = aclop_RightShift
    register_acl_ufunc("ascend_right_shift", BINARY_OP, func_union)

    # CANN 9.0 provides aclnnLeftShift; on 8.5 aclop_LeftShift composes
    # x * 2**n from aclnnCast/Exp2/Mul (see acl_math_ops.h).
    func_union.binary_op = aclop_LeftShift
    register_acl_ufunc("ascend_left_shift", BINARY_OP, func_union)

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
    # `cupy_subtract` 才是 ufunc 名（create_arithmetic('subtract', ...)）；
    # `ascend_sub` 是历史拼写，两个都注册，避免再出现「名字对不上 -> 静默不派发」。
    register_acl_ufunc("ascend_subtract", SCALAR_BINARY_OP, func_union)
    register_acl_ufunc("ascend_sub", SCALAR_BINARY_OP, func_union)
    func_union.scalar_binary_op = aclop_Muls
    register_acl_ufunc("ascend_multiply", SCALAR_BINARY_OP, func_union)
    func_union.scalar_binary_op = aclop_Divs
    register_acl_ufunc("ascend_true_divide", SCALAR_BINARY_OP, func_union)
    func_union.scalar_binary_op = aclop_FloorDivides
    register_acl_ufunc("ascend_floor_divide", SCALAR_BINARY_OP, func_union)
    func_union.scalar_binary_op = aclop_FmodScalar
    register_acl_ufunc("ascend_fmod", SCALAR_BINARY_OP, func_union)

    # -----------------------------------------------------------------------
    # reverse scalar binary: out = scalar <op> tensor（标量在左操作数）
    #
    # dispatch 按操作数位置选 REVERSE_SCALAR_BINARY_OP（get_op_type），所以
    # `x - 1` 走上面的 SCALAR_BINARY_OP，`1 - x` 走这里。以前 `2 / x` 会被算成
    # `x / 2`（静默错误结果），`1 - x` 则直接报「未注册」。
    # C++ 实现：aclnn 原生 ScalarTensor 接口（Rsubs/PowScalarTensor/
    # RemainderScalarTensor）或标量物化（AclScalarTensorGuard）。
    # -----------------------------------------------------------------------
    func_union.reverse_scalar_binary_op = aclop_Rsubs
    register_acl_ufunc("ascend_subtract", REVERSE_SCALAR_BINARY_OP, func_union)
    register_acl_ufunc("ascend_sub", REVERSE_SCALAR_BINARY_OP, func_union)
    func_union.reverse_scalar_binary_op = aclop_RDivs
    register_acl_ufunc("ascend_true_divide", REVERSE_SCALAR_BINARY_OP, func_union)
    func_union.reverse_scalar_binary_op = aclop_RFloorDivides
    register_acl_ufunc("ascend_floor_divide", REVERSE_SCALAR_BINARY_OP, func_union)
    func_union.reverse_scalar_binary_op = aclop_RFmodScalar
    register_acl_ufunc("ascend_fmod", REVERSE_SCALAR_BINARY_OP, func_union)
    func_union.reverse_scalar_binary_op = aclop_RPowScalar
    register_acl_ufunc("ascend_power", REVERSE_SCALAR_BINARY_OP, func_union)
    register_acl_ufunc("ascend_float_power", REVERSE_SCALAR_BINARY_OP, func_union)
    func_union.reverse_scalar_binary_op = aclop_RRemainderScalar
    register_acl_ufunc("ascend_remainder", REVERSE_SCALAR_BINARY_OP, func_union)
    # 比较运算的 reverse 是「换边」：scalar > tensor == tensor < scalar
    func_union.reverse_scalar_binary_op = aclop_RGtScalar
    register_acl_ufunc("ascend_greater", REVERSE_SCALAR_BINARY_OP, func_union)
    func_union.reverse_scalar_binary_op = aclop_RGeScalar
    register_acl_ufunc("ascend_greater_equal", REVERSE_SCALAR_BINARY_OP, func_union)
    func_union.reverse_scalar_binary_op = aclop_RLtScalar
    register_acl_ufunc("ascend_less", REVERSE_SCALAR_BINARY_OP, func_union)
    func_union.reverse_scalar_binary_op = aclop_RLeScalar
    register_acl_ufunc("ascend_less_equal", REVERSE_SCALAR_BINARY_OP, func_union)

    func_union.binary_op = aclop_Maximum
    register_acl_ufunc("ascend_maximum", BINARY_OP, func_union)
    func_union.binary_op = aclop_Minimum
    register_acl_ufunc("ascend_minimum", BINARY_OP, func_union)
    func_union.binary_op = aclop_Gcd
    register_acl_ufunc("ascend_gcd", BINARY_OP, func_union)
    func_union.binary_op = aclop_Lcm
    register_acl_ufunc("ascend_lcm", BINARY_OP, func_union)
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
    func_union.unary_op = aclop_Fabs
    register_acl_ufunc("ascend_fabs", UNARY_OP, func_union)
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

    # The ufuncs created by `_logic/content._create_float_test_ufunc` are named
    # `cupy_isfinite` / `cupy_isinf`, so the dispatcher looks up
    # `ascend_isfinite` / `ascend_isinf` (no underscore). Register both the
    # correct spelling and the historical one so neither dispatch path breaks.
    func_union.unary_op = aclop_IsFinite
    register_acl_ufunc("ascend_isfinite", UNARY_OP, func_union)
    register_acl_ufunc("ascend_is_finite", UNARY_OP, func_union)
    func_union.unary_op = aclop_IsInf
    register_acl_ufunc("ascend_isinf", UNARY_OP, func_union)
    register_acl_ufunc("ascend_is_inf", UNARY_OP, func_union)
    func_union.unary_op = aclop_IsNegInf
    register_acl_ufunc("ascend_isneginf", UNARY_OP, func_union)
    register_acl_ufunc("ascend_is_negnative_inf", UNARY_OP, func_union)
    func_union.unary_op = aclop_IsPosInf
    register_acl_ufunc("ascend_isposinf", UNARY_OP, func_union)
    register_acl_ufunc("ascend_is_positive_inf", UNARY_OP, func_union)
    func_union.unary_op = aclop_IsNan
    register_acl_ufunc("ascend_isnan", UNARY_OP, func_union)
    register_acl_ufunc("ascend_is_nan", UNARY_OP, func_union)

    func_union.unary_op = aclop_Floor
    register_acl_ufunc("ascend_floor", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceFloor
    register_acl_ufunc("ascend_inplace_floor", INPLACE_UNARY_OP, func_union)
    func_union.unary_op = aclop_Ceil
    register_acl_ufunc("ascend_ceil", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceCeil
    register_acl_ufunc("ascend_inplace_ceil", INPLACE_UNARY_OP, func_union)

    func_union.unary_op = aclop_Trunc
    register_acl_ufunc("ascend_trunc", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceTrunc
    register_acl_ufunc("ascend_inplace_trunc", INPLACE_UNARY_OP, func_union)
    func_union.unary_op = aclop_Rint
    register_acl_ufunc("ascend_rint", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceRint
    register_acl_ufunc("ascend_inplace_rint", INPLACE_UNARY_OP, func_union)
    func_union.unary_op = aclop_Cbrt
    register_acl_ufunc("ascend_cbrt", UNARY_OP, func_union)
    func_union.unary_op = aclop_Real
    register_acl_ufunc("ascend_real", UNARY_OP, func_union)
    func_union.unary_op = aclop_Sqrt
    register_acl_ufunc("ascend_sqrt", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceSqrt
    register_acl_ufunc("ascend_inplace_sqrt", INPLACE_UNARY_OP, func_union)
    func_union.unary_op = aclop_Exp2
    register_acl_ufunc("ascend_exp2", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceExp2
    register_acl_ufunc("ascend_inplace_exp2", INPLACE_UNARY_OP, func_union)
    # cupy_arccosh / cupy_arcsinh / cupy_arctanh
    func_union.unary_op = aclop_Acosh
    register_acl_ufunc("ascend_arccosh", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceAcosh
    register_acl_ufunc("ascend_inplace_arccosh", INPLACE_UNARY_OP, func_union)
    func_union.unary_op = aclop_Asinh
    register_acl_ufunc("ascend_arcsinh", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceAsinh
    register_acl_ufunc("ascend_inplace_arcsinh", INPLACE_UNARY_OP, func_union)
    func_union.unary_op = aclop_Atanh
    register_acl_ufunc("ascend_arctanh", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceAtanh
    register_acl_ufunc("ascend_inplace_arctanh", INPLACE_UNARY_OP, func_union)

    # numpy.power / numpy.float_power reuse the aclnn pow kernels.
    func_union.binary_op = aclop_PowTensorTensor
    register_acl_ufunc("ascend_power", BINARY_OP, func_union)
    func_union.scalar_binary_op = aclop_PowTensorScalar
    register_acl_ufunc("ascend_power", SCALAR_BINARY_OP, func_union)
    func_union.binary_op = aclop_PowTensorTensor
    register_acl_ufunc("ascend_float_power", BINARY_OP, func_union)
    func_union.scalar_binary_op = aclop_PowTensorScalar
    register_acl_ufunc("ascend_float_power", SCALAR_BINARY_OP, func_union)

    # numpy.fmax / numpy.fmin ignore NaN, which matches aclnnMaximum/Minimum.
    func_union.binary_op = aclop_Maximum
    register_acl_ufunc("ascend_fmax", BINARY_OP, func_union)
    func_union.binary_op = aclop_Minimum
    register_acl_ufunc("ascend_fmin", BINARY_OP, func_union)

    # numpy.invert is the ufunc name for bitwise_not (alias in cupy).
    func_union.unary_op = aclop_BitwiseNot
    register_acl_ufunc("ascend_invert", UNARY_OP, func_union)

    func_union.binary_op = aclop_Hypot
    register_acl_ufunc("ascend_hypot", BINARY_OP, func_union)
    func_union.binary_op = aclop_Copysign
    register_acl_ufunc("ascend_copysign", BINARY_OP, func_union)

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
    # NOTE: cupy ufunc names use the `arc` prefix (cupy_arccos, cupy_arcsin,
    # cupy_arctan, cupy_arctan2), so the registered opnames must match.
    func_union.unary_op = aclop_Acos
    register_acl_ufunc("ascend_arccos", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceAcos
    register_acl_ufunc("ascend_inplace_arccos", INPLACE_UNARY_OP, func_union)

    func_union.unary_op = aclop_Asin
    register_acl_ufunc("ascend_arcsin", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceAsin
    register_acl_ufunc("ascend_inplace_arcsin", INPLACE_UNARY_OP, func_union)

    func_union.unary_op = aclop_Atan
    register_acl_ufunc("ascend_arctan", UNARY_OP, func_union)
    func_union.inplace_unary_op = aclop_InplaceAtan
    register_acl_ufunc("ascend_inplace_arctan", INPLACE_UNARY_OP, func_union)

    # NOTE: aclnn has no arcsinh / arccosh / arctanh; emulated elsewhere.
    ###################### cosh op #####################
    func_union.unary_op = aclop_Cosh
    register_acl_ufunc("ascend_cosh", UNARY_OP, func_union)
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
    register_acl_ufunc("ascend_arctan2", BINARY_OP, func_union)
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
    aclError aclop_NanMin(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_NanMax(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_NanProd(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_CountNonNaN(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_NanArgMax(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_NanArgMin(const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, const KwargsType& kwargs, aclrtStream stream)

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
    func_union.reduction_op = aclop_ArgMax
    register_acl_ufunc("ascend_argmax", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_ArgMin
    register_acl_ufunc("ascend_argmin", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_Mean
    register_acl_ufunc("ascend_mean", REDUCTION_OP, func_union)
    # `cupy_mean_empty` is `_mean_core_empty`, used by `cupy.mean` for zero-size
    # input (it differs from `cupy_mean` only by having identity 0, so that the
    # reduction is not rejected before it runs and 0/0 -> nan is produced).
    # Same aclnn op as `ascend_mean`: aclnnMean sums and divides by the element
    # count, which is 0/0 -> nan for an empty input, matching NumPy.
    register_acl_ufunc("ascend_mean_empty", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_Sum
    register_acl_ufunc("ascend_sum", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_Prod
    register_acl_ufunc("ascend_prod", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_Nansum
    register_acl_ufunc("ascend_nansum", REDUCTION_OP, func_union)
    # composed: nan_to_num(nan=1) followed by a plain prod reduction
    # (CANN has no aclnnNanprod; same pattern as aclop_NanMin/NanMax)
    func_union.reduction_op = aclop_NanProd
    register_acl_ufunc("ascend_nanprod", REDUCTION_OP, func_union)
    register_acl_ufunc("ascend_nanprod_with_dtype", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_Nancumsum
    register_acl_ufunc("ascend_nancumsum", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_Nancumprod
    register_acl_ufunc("ascend_nancumprod", REDUCTION_OP, func_union)
    # composed: nan_to_num(+/-inf) followed by a plain min/max reduction
    func_union.reduction_op = aclop_NanMin
    register_acl_ufunc("ascend_nanmin", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_NanMax
    register_acl_ufunc("ascend_nanmax", REDUCTION_OP, func_union)
    # composed: (x != x) -> s_where(., 0, 1) -> sum, i.e. NumPy's count_non_nan
    # (kernel name: cupy_count_non_nan -> ascend_count_non_nan; used by _nanvar)
    func_union.reduction_op = aclop_CountNonNaN
    register_acl_ufunc("ascend_count_non_nan", REDUCTION_OP, func_union)
    # `*_with_dtype` / `*_complex_dtype` kernels share the same aclnn op as the
    # auto-dtype variants: the accumulator/output dtype is taken from the out
    # tensor inside aclop_Sum/aclop_Prod/aclop_Nansum via GetDataType(out, self).
    # (kernel names: cupy_sum_with_dtype / cupy_prod_with_dtype /
    # cupy_nansum_with_dtype / cupy_nansum_complex_dtype /
    # cupy_nanprod_complex_dtype -> ascend_*)
    func_union.reduction_op = aclop_Sum
    register_acl_ufunc("ascend_sum_with_dtype", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_Prod
    register_acl_ufunc("ascend_prod_with_dtype", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_Nansum
    register_acl_ufunc("ascend_nansum_with_dtype", REDUCTION_OP, func_union)
    register_acl_ufunc("ascend_nansum_complex_dtype", REDUCTION_OP, func_union)
    func_union.reduction_op = aclop_NanProd
    register_acl_ufunc("ascend_nanprod_complex_dtype", REDUCTION_OP, func_union)
    # composed: nan_to_num(-inf) -> argmax, i.e. NumPy's nanargmax (CANN has no
    # aclnnNanArgMax); integer/bool inputs skip the substitution (no NaN).
    func_union.reduction_op = aclop_NanArgMax
    register_acl_ufunc("ascend_nanargmax", REDUCTION_OP, func_union)
    # composed: nan_to_num(+inf) -> argmin, i.e. NumPy's nanargmin
    func_union.reduction_op = aclop_NanArgMin
    register_acl_ufunc("ascend_nanargmin", REDUCTION_OP, func_union)


# general ops
cdef extern from "../acl_general_ops.h" nogil:
    aclError aclop_Copy(const aclTensor* self,  aclTensor* out, aclrtStream stream)
    aclError aclop_Nonzero(const aclTensor* self,  aclTensor* out, aclrtStream stream)

    # 参数通道探针（只记录参数，不做计算；见 acl_general_ops.h）
    aclError aclop_DumpArgs(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    const char* aclop_GetLastDumpArgs()

    aclError aclop_Fill(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)

    aclError aclop_Arange(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Linspace(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)

    aclError aclop_Concat(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Stack(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Flip(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Permute(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Roll(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Cast(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Sort(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Argsort(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)

    aclError aclop_PutRaise(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Take(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Gather(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_IndexPutImpl(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)

    # Stage-A batch (CANN 9.0.1), see docs/ascend/DeveloperNotes.md
    aclError aclop_Slogdet(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_FillDiagonal(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Repeat(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_IndexSelect(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_MaskedFillScalar(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_MaskedFillTensor(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_IndexCopy(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_GatherNd(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_UniqueConsecutive(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_MaxN(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_MinN(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_MaxV2(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)

    # prefix scan: cumsum / cumprod (backing cupy.cumsum & the mask scan)
    aclError aclop_Cumsum(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Cumprod(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)

    # setitem / boolean indexing
    aclError aclop_ScatterUpdate(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_ScatterMax(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_ScatterMin(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_ScatterAdd(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_ScatterUpdateMask(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_ScatterAddMask(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_GetitemMask(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)

    # numpy.searchsorted / numpy.where
    aclError aclop_SearchSorted(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Bincount(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_VarCore(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Where(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)

    # set op: unique2 -> unique_all / unique_counts / unique_inverse / unique_values
    aclError aclop_Unique2(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)

    # linalg
    aclError aclop_Trace(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Tril(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Triu(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Qr(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Svd(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Inverse(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)

    # statistics / histogram
    aclError aclop_Aminmax(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Histc(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)

    # complex
    aclError aclop_Complex(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)

    # special math ops
    aclError aclop_Round(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_NanToNum(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Divmod(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Clamp(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_IsClose(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Einsum(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
    aclError aclop_Heaviside(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)


cdef extern from "../acl_random_ops.h" nogil:
    # BASE op, unconditional: aclnnMultinomial is part of the core CANN op
    # library (libopapi), NOT the ops-rand family. torch semantics — draws
    # numsamples category indices per row; numpy counts are derived by the
    # caller via flat bincount (cupy/random/_sample.py).
    aclError aclop_Multinomial(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
        const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)

IF CUPY_CANN_HAS_RAND:
    cdef extern from "../acl_random_ops.h" nogil:
        # cupy.random stateless fill ops (docs/ascend/DeveloperNotes.md
        # §ops-rand): destination is outs[0], `ins` is empty, and
        # (from/to/mean/std/seed/offset) arrive as scalars through the
        # unified args channel. Compiled in only when the aclnn_rand op
        # family is feature-detected in the SDK (CUPY_CANN_HAS_RAND, set by
        # AscendBackend.has_aclnn_rand in cupy_builder); the C++ header
        # additionally guards with __has_include.
        aclError aclop_RandomUniform(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
            const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
        aclError aclop_RandomNormal(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
            const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)
        aclError aclop_RandomInt(const vector[const aclTensor*]& ins, const vector[aclTensor*]& outs,
            const ArgsType& args, const KwargsType& kwargs, aclrtStream stream)


cdef void register_irregular_operators():
    cdef FuncPtrUnion func_union
    func_union.general_op = aclop_Concat
    register_acl_ufunc("ascend_concatenate", GENERAL_OP, func_union)
    func_union.general_op = aclop_Stack
    register_acl_ufunc("ascend_stack", GENERAL_OP, func_union)
    func_union.general_op = aclop_Flip
    register_acl_ufunc("ascend_flip", GENERAL_OP, func_union)
    func_union.general_op = aclop_Permute
    register_acl_ufunc("ascend_permute", GENERAL_OP, func_union)
    func_union.general_op = aclop_Roll
    register_acl_ufunc("ascend_roll", GENERAL_OP, func_union)
    func_union.general_op = aclop_Cast
    register_acl_ufunc("ascend_cast", GENERAL_OP, func_union)

    func_union.general_op = aclop_Sort
    register_acl_ufunc("ascend_sort", GENERAL_OP, func_union)
    func_union.general_op = aclop_Argsort
    register_acl_ufunc("ascend_argsort", GENERAL_OP, func_union)

    func_union.general_op = aclop_Take
    register_acl_ufunc("ascend_take", GENERAL_OP, func_union)
    register_acl_ufunc("ascend_take_scalar", GENERAL_OP, func_union)
    # `_put_raise_kernel` is an ElementwiseKernel named `cupy_put_raise`, and the
    # Ascend dispatcher maps kernel names by prefix (`cupy_` -> `ascend_`), so the
    # registration must use `ascend_put_raise` — the old `ascend_raise_put` never
    # matched any caller (review §3 T1).
    func_union.general_op = aclop_PutRaise
    register_acl_ufunc("ascend_put_raise", GENERAL_OP, func_union)

    # gather along a dim: the building block of `_take`'s axis branch
    # (aclnnTake is flatten-only, see aclop_Take)
    func_union.general_op = aclop_Gather
    register_acl_ufunc("ascend_gather", GENERAL_OP, func_union)

    # ndarray.put / numpy.put (aclnnIndexPutImpl); consumed by `_ndarray_put`'s
    # Ascend branch in cupy/_core/_routines_indexing.pyx
    func_union.general_op = aclop_IndexPutImpl
    register_acl_ufunc("ascend_index_put_impl", GENERAL_OP, func_union)

    # --- Stage-A batch (CANN 9.0.1), see docs/ascend/DeveloperNotes.md ---
    # linalg.slogdet (real float inputs; complex keeps the cpu_fallback path)
    func_union.general_op = aclop_Slogdet
    register_acl_ufunc("ascend_slogdet", GENERAL_OP, func_union)

    # fill_diagonal (scalar val only; array_like val keeps the Python path)
    func_union.general_op = aclop_FillDiagonal
    register_acl_ufunc("ascend_fill_diagonal", GENERAL_OP, func_union)

    # np.tile via aclnnRepeat (torch.repeat semantics); consumed by tiling.py
    func_union.general_op = aclop_Repeat
    register_acl_ufunc("ascend_repeat", GENERAL_OP, func_union)

    # np.take along a dim; consumed by _take's Ascend branch
    func_union.general_op = aclop_IndexSelect
    register_acl_ufunc("ascend_index_select", GENERAL_OP, func_union)

    # copyto(dst, src, where=mask): scalar vs tensor source
    func_union.general_op = aclop_MaskedFillScalar
    register_acl_ufunc("ascend_masked_fill_scalar", GENERAL_OP, func_union)
    func_union.general_op = aclop_MaskedFillTensor
    register_acl_ufunc("ascend_masked_fill_tensor", GENERAL_OP, func_union)

    # --- building blocks without a cupy API consumer yet ---
    # a[idx] = v along a dim (overlaps ascend_scatter_update)
    func_union.general_op = aclop_IndexCopy
    register_acl_ufunc("ascend_index_copy", GENERAL_OP, func_union)
    # multi-coordinate fancy indexing
    func_union.general_op = aclop_GatherNd
    register_acl_ufunc("ascend_gather_nd", GENERAL_OP, func_union)
    # torch.unique_consecutive (3 outputs)
    func_union.general_op = aclop_UniqueConsecutive
    register_acl_ufunc("ascend_unique_consecutive", GENERAL_OP, func_union)
    # NB: ascend_multinomial is registered in the random-ops block
    # (acl_random_ops.h / cupy.random WIP) -- no duplicate here.
    # elementwise max/min over N tensors (would back maximum.reduce)
    func_union.general_op = aclop_MaxN
    register_acl_ufunc("ascend_maxn", GENERAL_OP, func_union)
    func_union.general_op = aclop_MinN
    register_acl_ufunc("ascend_minn", GENERAL_OP, func_union)
    # multi-dim max reduction (redundant with ascend_max; no aclnn_min_v2)
    func_union.general_op = aclop_MaxV2
    register_acl_ufunc("ascend_max_v2", GENERAL_OP, func_union)

    # prefix scan (cupy.cumsum / cupy.cumprod / boolean-index mask scan)
    func_union.general_op = aclop_Cumsum
    register_acl_ufunc("ascend_cumsum", GENERAL_OP, func_union)
    func_union.general_op = aclop_Cumprod
    register_acl_ufunc("ascend_cumprod", GENERAL_OP, func_union)

    # setitem / boolean indexing: `_scatter_*_kernel` / `_getitem_mask_kernel`
    func_union.general_op = aclop_ScatterUpdate
    register_acl_ufunc("ascend_scatter_update", GENERAL_OP, func_union)
    func_union.general_op = aclop_ScatterAdd
    register_acl_ufunc("ascend_scatter_add", GENERAL_OP, func_union)
    # scatter_max/min：CANN 无原生 reduce=max/min，由 gather+Maximum/Minimum+
    # InplaceScatterUpdate 三段组合（acl_general_ops.h ScatterMaxMin）。
    # `cupy.maximum.at` / `cupy.minimum.at` / `cupyx.scatter_max/min` 的落点。
    func_union.general_op = aclop_ScatterMax
    register_acl_ufunc("ascend_scatter_max", GENERAL_OP, func_union)
    func_union.general_op = aclop_ScatterMin
    register_acl_ufunc("ascend_scatter_min", GENERAL_OP, func_union)
    func_union.general_op = aclop_ScatterUpdateMask
    register_acl_ufunc("ascend_scatter_update_mask", GENERAL_OP, func_union)
    func_union.general_op = aclop_ScatterAddMask
    register_acl_ufunc("ascend_scatter_add_mask", GENERAL_OP, func_union)
    func_union.general_op = aclop_GetitemMask
    register_acl_ufunc("ascend_getitem_mask", GENERAL_OP, func_union)

    # numpy.searchsorted / numpy.where(cond, x, y)
    func_union.general_op = aclop_SearchSorted
    register_acl_ufunc("ascend_searchsorted_kernel", GENERAL_OP, func_union)
    # cupy.bincount (histogram.py) is not a ufunc: it launches the
    # ElementwiseKernels cupy_bincount_kernel (unweighted) and
    # cupy_bincount_with_weight_kernel. Both route here; the C++ side picks
    # weights from ins[1] when present.
    func_union.general_op = aclop_Bincount
    register_acl_ufunc("ascend_bincount_kernel", GENERAL_OP, func_union)
    register_acl_ufunc("ascend_bincount_with_weight_kernel", GENERAL_OP, func_union)
    func_union.general_op = aclop_Where
    register_acl_ufunc("ascend_where", GENERAL_OP, func_union)
    # cupy_var_core_float*（ReductionKernel 3-in/1-out）的 Ascend 组合：
    # ascend_var_core 算 sum((x - mean)^2)，alpha 乘法由
    # py_launch_var_core（下）用 ascend_inplace_multiply 完成。
    func_union.general_op = aclop_VarCore
    register_acl_ufunc("ascend_var_core", GENERAL_OP, func_union)

    func_union.general_op = aclop_Arange
    register_acl_ufunc("ascend_arange", GENERAL_OP, func_union)
    func_union.general_op = aclop_Linspace
    register_acl_ufunc("ascend_linspace", GENERAL_OP, func_union)

    func_union.general_op = aclop_Round
    register_acl_ufunc("ascend_round", GENERAL_OP, func_union)
    func_union.general_op = aclop_NanToNum
    register_acl_ufunc("ascend_nan_to_num", GENERAL_OP, func_union)
    register_acl_ufunc("ascend_nan_to_num_", GENERAL_OP, func_union)
    func_union.general_op = aclop_Divmod
    register_acl_ufunc("ascend_divmod", GENERAL_OP, func_union)
    func_union.general_op = aclop_Clamp
    register_acl_ufunc("ascend_clip", GENERAL_OP, func_union)
    func_union.general_op = aclop_IsClose
    register_acl_ufunc("ascend_is_close", GENERAL_OP, func_union)
    # einsum：equation 走统一参数通道的 ARG_STRING（cupy/linalg/_einsum.py 的
    # Ascend 快速路径，默认关闭，CUPY_ASCEND_NATIVE_EINSUM=1 启用）
    func_union.general_op = aclop_Einsum
    register_acl_ufunc("ascend_einsum", GENERAL_OP, func_union)
    func_union.general_op = aclop_Heaviside
    register_acl_ufunc("ascend_heaviside", GENERAL_OP, func_union)

    # cupy.random stateless fill ops (aclnnInplaceUniform/Normal/Random):
    # launched from pure Python via py_launch_general("ascend_random_*",
    # [], [out], [from, to, seed, offset], {}) — see cupy/random/_generator.py.
    # Compiled in only when feature-detected (CUPY_CANN_HAS_RAND); when the
    # ops are absent cupy.random fails at runtime via the py_is_acl_ufunc_
    # registered check in _generator.py instead of here.
    # cupy.random.multinomial — BASE op (core libopapi), unconditional:
    # torch-style index sampling; counts are derived in
    # cupy/random/_sample.py via flat bincount
    func_union.general_op = aclop_Multinomial
    register_acl_ufunc("ascend_multinomial", GENERAL_OP, func_union)

    IF CUPY_CANN_HAS_RAND:
        func_union.general_op = aclop_RandomUniform
        register_acl_ufunc("ascend_random_uniform", GENERAL_OP, func_union)
        func_union.general_op = aclop_RandomNormal
        register_acl_ufunc("ascend_random_normal", GENERAL_OP, func_union)
        func_union.general_op = aclop_RandomInt
        register_acl_ufunc("ascend_random_int", GENERAL_OP, func_union)

    # set op: unique2 covers unique_all/counts/inverse/values in one kernel
    func_union.general_op = aclop_Unique2
    register_acl_ufunc("ascend_unique2", GENERAL_OP, func_union)

    # linalg (aclnn-backed; reached from cupy/_core/_ascend/_routines_linalg.pyx)
    func_union.general_op = aclop_Trace
    register_acl_ufunc("ascend_trace", GENERAL_OP, func_union)
    func_union.general_op = aclop_Tril
    register_acl_ufunc("ascend_tril", GENERAL_OP, func_union)
    func_union.general_op = aclop_Triu
    register_acl_ufunc("ascend_triu", GENERAL_OP, func_union)
    func_union.general_op = aclop_Qr
    register_acl_ufunc("ascend_qr", GENERAL_OP, func_union)
    func_union.general_op = aclop_Svd
    register_acl_ufunc("ascend_svd", GENERAL_OP, func_union)
    func_union.general_op = aclop_Inverse
    register_acl_ufunc("ascend_inverse", GENERAL_OP, func_union)

    # statistics / histogram
    func_union.general_op = aclop_Aminmax
    register_acl_ufunc("ascend_aminmax", GENERAL_OP, func_union)
    func_union.general_op = aclop_Histc
    register_acl_ufunc("ascend_histc", GENERAL_OP, func_union)

    # complex construction
    func_union.general_op = aclop_Complex
    register_acl_ufunc("ascend_complex", GENERAL_OP, func_union)

    func_union.unary_op = aclop_Copy
    register_acl_ufunc("ascend_copy", UNARY_OP, func_union)
    # numpy.positive(+x) is the identity for every non-bool dtype.
    register_acl_ufunc("ascend_positive", UNARY_OP, func_union)
    func_union.general_op  = aclop_Fill
    register_acl_ufunc("ascend_fill", GENERAL_OP, func_union)
    func_union.unary_op = aclop_Nonzero
    register_acl_ufunc("ascend_nonzero", UNARY_OP, func_union)

    # 参数通道探针：只记录收到的参数（scalar / int 序列 / str / None 的 tag 与值），
    # 不做任何计算，也没有对应的 numpy API。tests/ascend/test_unified_args.py 用它
    # 在无 NPU 环境验证「统一参数通道」端到端可达（py_dump_args / py_last_dump_args）。
    func_union.general_op = aclop_DumpArgs
    register_acl_ufunc("ascend_dump_args", GENERAL_OP, func_union)

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

cdef bint is_acl_ufunc_registered(str opname) except *:
    """Whether ``opname`` (already ``ascend_*``) has an implementation at all.

    The Ascend `ElementwiseKernel`/`ufunc` never compile their CUDA body; they
    dispatch purely by name (``cupy_xxx`` -> ``ascend_xxx``). So "is there an
    ``ascend_xxx``?" is exactly "will this kernel work on Ascend?". Used by
    ``cupy/_core/_ascend/_kernel.pyx`` to fail with an actionable message instead
    of a bare ``KeyError`` from the dispatcher.

    Any arity/inplace/scalar variant counts, and custom AscendC kernels
    (``py_register_custom_kernel``) count as well.
    """
    if opname in _custom_kernel_specs:
        return True
    cdef string cname = opname.encode("utf-8")
    return _registered_op_names.find(cname) != _registered_op_names.end()


def py_is_acl_ufunc_registered(str opname) -> bool:
    """Python-visible wrapper around :func:`is_acl_ufunc_registered` (for tests)."""
    return is_acl_ufunc_registered(opname)


def py_get_op_type(object ops, bint inplace, bint has_scalar=False,
                   bint scalar_is_lhs=False) -> int:
    """测试用：暴露 `get_op_type`，验证「操作数位置 -> OpType」的判定（无需 NPU）。"""
    return <int>get_op_type(ops, inplace, has_scalar, scalar_is_lhs)


def py_launch_general(str opname, tuple ins, tuple outs, tuple args,
                      dict kwargs, intptr_t stream_ptr=0):
    """Python 层的 general-op 启动器（einsum 等纯 python routine 的 Ascend 快速路径用）。

    与 pyx/C 调用方走同一张注册表和统一参数通道（args 按 tag 转换：str 需在
    `_STRING_ARG_OPS` 白名单内）；错误传播语义同 ``launch_general_func``
    （aclError != 0 -> RuntimeError，消息带 aclGetRecentErrMsg）。
    """
    return launch_general_func(opname, ins, outs, list(args), dict(kwargs),
                               stream_ptr)


def py_is_registered(str opname, int op_type) -> bool:
    """测试用：查询 ``(opname, OpType)`` 是否已注册（无需 NPU）。"""
    cdef OpInfo op_info
    op_info.op_name = opname.encode("utf-8")
    op_info.op_type = <OpType>op_type
    return _builtin_operators.find(op_info) != _builtin_operators.end()


def _no_ascend_impl_msg(str opname) -> str:
    """The explicit "this op is simply not implemented on Ascend" message.

    Used by `launch_acl_func`/`launch_reduction_op` when the registry has no
    entry at all.  Rationale (review P1/D3): on Ascend an
    `ElementwiseKernel`/`ufunc` never compiles its CUDA body -- it is dispatched
    purely by name (`cupy_xxx` -> `ascend_xxx`) -- so a missing registry entry
    used to surface as a bare `KeyError` from deep inside the dispatcher. With
    this message the ~47 op names that still lack an implementation fail loudly
    and point at the place to fix.
    """
    return (
        "Ascend backend: no implementation registered for %r. ElementwiseKernel/"
        "ufunc bodies are not compiled on Ascend; kernels are dispatched by name "
        "(cupy_xxx -> ascend_xxx), so this op needs an `ascend_*` registration in "
        "cupy/backends/ascend/api/acl_utils.pyx (register_*_operators) plus a "
        "wrapper in cupy/backends/ascend/acl_*.h. The list of op names still "
        "missing an implementation is tracked in "
        "docs/ascend/code_review_ascend_core.md." % (opname,))


def py_list_acl_ufuncs():
    """Return the list of registered aclnn op names as ``(opname, op_type)``.

    Introspection helper for tooling/tests: it exposes the contents of the
    internal ``_builtin_operators`` registry so that op coverage can be audited
    without a device. `op_type` is the integer value of the ``OpType`` enum
    (see ``acl_opinfo.h``).
    """
    cdef list result = []
    cdef cpp_map[OpInfo, FuncPtrUnion, OpInfo.Hash].iterator it = _builtin_operators.begin()
    cdef cpp_map[OpInfo, FuncPtrUnion, OpInfo.Hash].iterator end = _builtin_operators.end()
    cdef OpInfo op_info
    while it != end:
        op_info = deref(it).first
        result.append((op_info.op_name.decode('utf-8'), <int>op_info.op_type))
        inc(it)
    return result


cdef void init_builtin_operators():
    register_math_operators()
    register_reduction_operators()
    register_irregular_operators()

# only one function can be run during module init
init_builtin_operators()

