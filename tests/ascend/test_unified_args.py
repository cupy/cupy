"""M3 tests: unified (typed) argument channel + reverse-scalar dispatch.

Covers the two things added in this milestone (docs/ascend/arg_passing_plan.md):

1. **统一参数通道** — args/kwargs 现在按 tag 传递（scalar / int_array / string /
   tensor / none），`ascend_dump_args`（`acl_general_ops.h` 里的探针 op）会在
   C++ 侧记录真正收到的内容，所以「参数到底有没有按类型送达」可以在**没有 NPU**
   的环境里断言。

2. **reverse scalar binary**（`scalar <op> tensor`）—— `1 - x` / `2 / x` /
   `1 > x` 这类调用的标量在左操作数。以前 `2 / x` 会被算成 `x / 2`
   （静默错误结果），`x - 1` 之外的组合要么没注册、要么方向反。现在 dispatch 按
   操作数位置选 `REVERSE_SCALAR_BINARY_OP`，C++ 侧有对应的 aclop_R* 实现。
"""

from __future__ import annotations

import glob
import os
import re

import pytest

# OpType 值（acl_opinfo.h）
GENERAL_OP = 0
UNARY_OP = 1
BINARY_OP = 4
SCALAR_BINARY_OP = 6
REVERSE_SCALAR_BINARY_OP = 10


@pytest.fixture
def acl_utils():
    from cupy.backends.ascend.api import acl_utils
    return acl_utils


@pytest.fixture
def strict_env(monkeypatch):
    monkeypatch.delenv('CUPY_ASCEND_LENIENT_ARGS', raising=False)


# ---------------------------------------------------------------------------
# 1. 类型化参数：describe（pyx 侧）与 dump（C++ 侧）一致
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('value, kind', [
    (3, 'scalar'),
    (2.5, 'scalar'),
    (True, 'scalar'),
    ((0, 1), 'int_array'),
    ([1, 2, 3], 'int_array'),
    (None, 'none'),
])
def test_describe_arg_kinds(acl_utils, strict_env, value, kind):
    desc = acl_utils.py_describe_args('ascend_dump_args', (value,))
    assert desc[0][1] == kind


def test_dump_args_reaches_cpp_with_tags(acl_utils, strict_env):
    """端到端：参数按 tag 送达 C++（探针 op 不做任何 aclnn 计算，无需 NPU）。"""
    dump = acl_utils.py_dump_args((7, (0, 1), None), {'axis': [2, 3]})
    assert 'arg[0]=kind=scalar' in dump
    assert 'arg[1]=kind=int_array, values=[0, 1]' in dump
    assert 'arg[2]=kind=none' in dump
    assert 'kwarg[axis]=kind=int_array, values=[2, 3]' in dump


def test_dump_args_string_is_whitelisted(acl_utils, strict_env):
    """str -> ARG_STRING（自带所有权），只有 _STRING_ARG_OPS 里的算子能收到。"""
    dump = acl_utils.py_dump_args(('stable',), {'order': 'C'})
    assert 'arg[0]=kind=string, value="stable"' in dump
    assert 'kwarg[order]=kind=string, value="C"' in dump


def test_string_rejected_for_non_whitelisted_op(acl_utils, strict_env):
    with pytest.raises(NotImplementedError, match='host 侧解析'):
        acl_utils.py_describe_args('ascend_sort', ('stable',))


def test_int_array_rejects_non_integer_elements(acl_utils, strict_env):
    """序列里混入 float/str 时不能静默截断。"""
    with pytest.raises(NotImplementedError, match='不是整数'):
        acl_utils.py_describe_args('ascend_dump_args', ((1, 2.5),))


def test_none_is_not_unsupported(acl_utils, strict_env):
    """None 是合法参数（axis=None 表示「全部轴」），不再当作不可转换。"""
    assert acl_utils.py_describe_args('ascend_flip', (None,))[0][1] == 'none'


# ---------------------------------------------------------------------------
# 2. 操作数位置 -> OpType
# ---------------------------------------------------------------------------
def test_get_op_type_scalar_position(acl_utils):
    # x - 1  : ins = [tensor, scalar], outs = [out] -> SCALAR_BINARY_OP
    assert acl_utils.py_get_op_type([1, 2, 3], False, True, False) == SCALAR_BINARY_OP
    # 1 - x  : ins = [scalar, tensor]             -> REVERSE_SCALAR_BINARY_OP
    assert acl_utils.py_get_op_type([1, 2, 3], False, True, True) == \
        REVERSE_SCALAR_BINARY_OP
    # x - y  : 两个 tensor                        -> BINARY_OP
    assert acl_utils.py_get_op_type([1, 2, 3], False, False) == BINARY_OP


# ---------------------------------------------------------------------------
# 3. reverse 变体已注册（名字 + OpType 都要对得上，否则运行期静默不派发）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('opname', [
    'ascend_subtract',
    'ascend_true_divide',
    'ascend_floor_divide',
    'ascend_fmod',
    'ascend_power',
    'ascend_float_power',
    'ascend_remainder',
    'ascend_greater',
    'ascend_greater_equal',
    'ascend_less',
    'ascend_less_equal',
])
def test_reverse_scalar_registered(acl_utils, opname):
    assert acl_utils.py_is_registered(opname, REVERSE_SCALAR_BINARY_OP), (
        f'{opname} has no REVERSE_SCALAR_BINARY_OP implementation: '
        f'`scalar {opname[len("ascend_"):]} tensor` would either raise or '
        f'silently compute the operands in the wrong order')


@pytest.mark.parametrize('opname', [
    'ascend_subtract',      # `cupy_subtract` 的 ufunc 名（历史拼写是 ascend_sub）
    'ascend_true_divide',
    'ascend_floor_divide',
    'ascend_fmod',
    'ascend_power',
    'ascend_remainder',
])
def test_forward_scalar_registered(acl_utils, opname):
    """tensor <op> scalar 也要在（本次顺带补上 ascend_subtract / floor_divide）。"""
    assert acl_utils.py_is_registered(opname, SCALAR_BINARY_OP)


def test_commutative_ops_need_no_reverse(acl_utils):
    """交换律算子（add/multiply/maximum...）没有 reverse 实现也应可用：dispatch 回退。"""
    from cupy.backends.ascend.api import acl_utils
    assert 'add' in acl_utils._COMMUTATIVE_OPS
    assert not acl_utils.py_is_registered('ascend_add', REVERSE_SCALAR_BINARY_OP)
    assert acl_utils.py_is_registered('ascend_add', SCALAR_BINARY_OP)


# ---------------------------------------------------------------------------
# 4. 注册表里没有「名字对不上」的 scalar 变体（历史 bug：ascend_sub / 拼写错误）
# ---------------------------------------------------------------------------
def test_subtract_registered_under_real_ufunc_name():
    """`create_arithmetic('subtract', ...)` 生成的 ufunc 是 cupy_subtract。"""
    src, _ = _acl_utils_src()
    assert re.search(r'register_acl_ufunc\("ascend_subtract",\s*SCALAR_BINARY_OP', src)
    assert re.search(
        r'register_acl_ufunc\("ascend_subtract",\s*REVERSE_SCALAR_BINARY_OP', src)


# ---------------------------------------------------------------------------
# 5. 错误码契约（两层 API）
#
#   * `launch_*`      —— 默认入口：返回 aclError，非 0 时抛 RuntimeError。
#     「检查」是默认行为，所以**没有后缀**；Python 路径用它。
#   * `launch_*_raw`  —— 原语：返回 aclError 且**不抛**，把 Ascend 错误码交给
#     caller（C/C++ / noexcept 方向）。
# ---------------------------------------------------------------------------
def _acl_utils_src():
    from cupy.backends.ascend.api import acl_utils
    base = os.path.dirname(acl_utils.__file__)
    with open(os.path.join(base, 'acl_utils.pyx')) as f:
        pyx = f.read()
    with open(os.path.join(base, 'acl_utils.pxd')) as f:
        pxd = f.read()
    return pyx, pxd


@pytest.mark.parametrize('name', ['launch_general_func', 'launch_acl_func',
                                  'launch_reduction_op'])
def test_launch_api_returns_error_code(name):
    """两层入口都必须返回 `aclError`（不是 void）。"""
    _, pxd = _acl_utils_src()
    assert re.search(r'cdef aclError ' + name + r'\(', pxd), name
    assert re.search(r'cdef aclError ' + name + r'_raw\(', pxd), name


def test_only_default_launchers_raise():
    """`raise_acl_op_error` 只允许出现在三个默认（无后缀）入口里；_raw 只传错误码。"""
    pyx, _ = _acl_utils_src()
    first_default = pyx.index('cdef aclError launch_general_func(')
    # 只数调用点（定义是 `raise_acl_op_error(str opname, long ret)`）
    positions = [m.start() for m in re.finditer(r'raise_acl_op_error\(opname, ret\)', pyx)]
    assert len(positions) == 3, positions
    assert all(p > first_default for p in positions)


def test_core_modules_do_not_bypass_the_check():
    """其它模块不允许直接调用 *_raw：错误码会被静默丢弃（本次要修的一类 bug）。"""
    import cupy
    core = os.path.join(os.path.dirname(cupy.__file__), '_core')
    offenders = []
    for path in glob.glob(os.path.join(core, '**', '*.pyx'), recursive=True):
        src = open(path).read()
        for name in ('launch_general_func_raw', 'launch_acl_func_raw',
                     'launch_reduction_op_raw'):
            if re.search(r'\b' + name + r'\(', src):
                offenders.append((os.path.relpath(path), name))
    assert not offenders, (
        f'{offenders}: core 层请用默认入口（检查 + 抛），_raw 只给 C/noexcept 侧')
