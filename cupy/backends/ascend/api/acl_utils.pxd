cimport cpython
import cython
from libc.stdint cimport intptr_t
from libcpp.string cimport string
from cupy._core.core cimport _ndarray_base
from cupy._core._scalar cimport CScalar as _cupy_scalar
include 'acl_types.pxi' # TODO: should not include in pxd file??

cdef str ASCEND_OP_PREFIX

cdef aclDataType numpy_dtype_to_acl_dtype(dtype,
    bint is_half_allowed=*, bint is_double_supported=*) except*
cdef aclTensor* cupy_ndarray_to_acl_tensor(_ndarray_base cupy_array) except*
cdef aclScalar* cupy_scalar_to_acl_scalar(_cupy_scalar s) except*

ctypedef fused sequence:
    list


# TODO: is size_t is the best type to pass C void* stream Pointer??
#
# 两层派发 API（见 docs/ascend/arg_passing_plan.md §2.5）：
#   * `launch_*`      —— 默认入口：返回 aclError，非 0 时抛 RuntimeError
#                        （带 aclGetRecentErrMsg()）；Python 路径用这个。
#   * `launch_*_raw`  —— 原语：返回 aclError 且**不抛**，把 Ascend 错误码交给
#                        caller（noexcept / C 侧入口方向）。
cdef aclError launch_general_func(str opname, sequence ins, sequence outs,
    list args, dict kargs, intptr_t stream_ptr) except *
cdef aclError launch_acl_func(str opname, sequence ins, sequence outs,
    list args, dict kargs, intptr_t stream_ptr) except *
cdef aclError launch_reduction_op(str opname, sequence ins, sequence outs,
    object axes, bint keepdims, dict kargs, intptr_t stream_ptr) except *
cdef aclError launch_general_func_raw(str opname, sequence ins, sequence outs,
    list args, dict kargs, intptr_t stream_ptr) except *
cdef aclError launch_acl_func_raw(str opname, sequence ins, sequence outs,
    list args, dict kargs, intptr_t stream_ptr) except *
cdef aclError launch_reduction_op_raw(str opname, sequence ins, sequence outs,
    object axes, bint keepdims, dict kargs, intptr_t stream_ptr) except *

# Whether `opname` (already `ascend_*`) has some implementation registered.
# Consumed by `cupy/_core/_ascend/_kernel.pyx` to fail loudly on kernels that
# have no Ascend implementation instead of raising a bare KeyError.
cdef bint is_acl_ufunc_registered(str opname) except *

# Live state of the enable_float64_to_float32 switch (A.1.1). Cimported by
# `cupy/_core/_ascend/_reduction.pyx` so the runtime setter
# (py_enable_float64_to_float32) applies to the reduction channel as well.
cdef bint ascend_float64_promote_enabled()