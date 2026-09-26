"""Ascend scatter 算子的注册与组合实现（无 NPU 可跑的部分）。

背景（见 commit 说明 / docs/ascend/dtype_promotion.md 同套评审方法）：

* ``cupyx.scatter_add`` / ``cupyx.scatter_max`` / ``cupyx.scatter_min`` 落到
  ``ndarray._scatter_op`` -> ``_scatter_op_single`` -> ``cupy_scatter_*``
  ElementwiseKernel，按名字派发到 ``ascend_scatter_*``。
* ``ascend_scatter_update`` / ``ascend_scatter_add`` 走原生
  aclnnInplaceScatterUpdate / aclnnScatterAdd；
* CANN **没有** reduce=max/min 的 scatter（aclnnScatter.reduce 只有 add/mul/none，
  aclnnIndexPutImpl 只有 accumulate/replace），所以 ``ascend_scatter_max/min``
  是 gather + Maximum/Minimum + InplaceScatterUpdate 的三段组合实现。

dtype 门槛在 ``cupy/_core/_routines_indexing.pyx::_scatter_op_single``：
Ascend 下收窄为 int32/float16/float32（aclnnScatterAdd 的白名单），
需要设备才能端到端验证，这里只锁注册与组合实现的存在性。
"""

import os

import pytest

GENERAL_OP = 0


@pytest.fixture
def acl_utils():
    from cupy.backends.ascend.api import acl_utils
    return acl_utils


@pytest.mark.parametrize('opname', [
    'ascend_scatter_update',
    'ascend_scatter_add',
    'ascend_scatter_max',
    'ascend_scatter_min',
])
def test_scatter_ops_registered(acl_utils, opname):
    assert acl_utils.py_is_registered(opname, GENERAL_OP), opname


def test_scatter_max_min_are_composed(acl_utils):
    """max/min 无原生 aclnn，必须由 gather+Maximum/Minimum+ScatterUpdate 组合。"""
    base = os.path.dirname(os.path.dirname(acl_utils.__file__))
    src = open(os.path.join(base, 'acl_general_ops.h'), encoding='utf-8').read()
    assert 'ScatterMaxMin' in src
    assert 'aclnnGatherGetWorkspaceSize' in src
    assert 'aclnnMaximumGetWorkspaceSize' in src
    assert 'aclnnMinimumGetWorkspaceSize' in src
    assert 'aclnnInplaceScatterUpdateGetWorkspaceSize' in src


def test_add_at_dtype_gate_is_narrowed_on_ascend():
    """Ascend 下 host 侧 dtype 白名单收窄为 aclnnScatterAdd 支持的类型。"""
    from cupy._core import _routines_indexing as idx
    # 本环境按 Ascend 构建，runtime 判定应为 True（is_ascend 来自 backend）
    from cupy.backends.backend import is_ascend
    if not is_ascend:
        pytest.skip('non-Ascend backend build')
    assert idx.py_scatter_ascend_gate_active()
