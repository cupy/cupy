"""M1 tests: argument validation / strict mode
(docs/ascend/arg_passing_plan.md §2.1).

No NPU required: the checks run through ``py_describe_args``, which uses the
same conversion rules as the real dispatch path, plus static consistency
checks on the registration table.
"""

from __future__ import annotations

import os
import re

import numpy
import pytest


@pytest.fixture
def acl_utils():
    from cupy.backends.ascend.api import acl_utils
    return acl_utils


@pytest.fixture
def strict_env(monkeypatch):
    monkeypatch.delenv('CUPY_ASCEND_LENIENT_ARGS', raising=False)


@pytest.fixture
def lenient_env(monkeypatch):
    monkeypatch.setenv('CUPY_ASCEND_LENIENT_ARGS', '1')


# ---------------------------------------------------------------------------
# 值类型：能转换的
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('value', [1, 3.5, True, numpy.int32(7),
                                   numpy.float64(2.5), numpy.bool_(False)])
def test_supported_scalars(acl_utils, strict_env, value):
    desc = acl_utils.py_describe_args('ascend_sort', (value,))
    assert desc[0][1] == 'scalar'


def test_known_keys_accepted(acl_utils, strict_env):
    desc = acl_utils.py_describe_args(
        'ascend_sort', (), {'axis': 0, 'stable': 1, 'descending': 0, 'k': 2})
    assert {d[0] for d in desc} == {'axis', 'stable', 'descending', 'k'}
    assert all(d[1] == 'scalar' for d in desc)


# ---------------------------------------------------------------------------
# 值类型：不支持的（原来静默丢弃 -> 现在必须报错）
# ---------------------------------------------------------------------------
# NOTE: `[1, 2, 3]` 与 `None` 自 M3（统一参数通道）起是**合法**参数：
# 前者 -> ARG_INT_ARRAY，后者 -> ARG_NONE（详见 test_unified_args.py）。
@pytest.mark.parametrize('value', [{'a': 1}, object()])
def test_unsupported_values_raise(acl_utils, strict_env, value):
    with pytest.raises(NotImplementedError, match='无法转换为'):
        acl_utils.py_describe_args('ascend_sort', (value,))


def test_ndarray_value_raises(acl_utils, strict_env):
    # `where=mask` 注入的是 ndarray（_kernel.pyx:913）：必须响亮失败而不是丢弃
    with pytest.raises(NotImplementedError):
        acl_utils.py_describe_args('ascend_add', (), {'where': numpy.ones(4)})


# ---------------------------------------------------------------------------
# 字符串：host 侧解析优先，未声明 ARG_STRING 的算子直接报错
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('s', ['stable', 'quicksort'])
def test_string_args_rejected_by_default(acl_utils, strict_env, s):
    with pytest.raises(NotImplementedError, match='host 侧解析'):
        acl_utils.py_describe_args('ascend_sort', (s,))


# ---------------------------------------------------------------------------
# 未知 key
# ---------------------------------------------------------------------------
def test_unknown_kwarg_key_raises(acl_utils, strict_env):
    with pytest.raises(ValueError, match='未知参数 key'):
        acl_utils.py_describe_args('ascend_sort', (), {'no_such_param': 1})


def test_where_is_known_key(acl_utils, strict_env):
    # where 在白名单里，避免误报「未知 key」；标量值可转换（ndarray 值见上一条用例）
    desc = acl_utils.py_describe_args('ascend_add', (), {'where': True})
    assert desc[0][0] == 'where' and desc[0][1] == 'scalar'


# ---------------------------------------------------------------------------
# 宽松模式（迁移期开关）
# ---------------------------------------------------------------------------
def test_lenient_mode_keeps_old_behaviour(acl_utils, lenient_env):
    # 宽松模式：未知 key 不再抛错；不可转换的值标记 unsupported 后跳过。
    # NOTE: int 序列（[1, 2]）自 M3 起是合法参数（ARG_INT_ARRAY），不再是 unsupported。
    desc = acl_utils.py_describe_args('ascend_sort', ([1, 2], object()),
                                      {'bogus': [3, 4]})
    kinds = {d[0]: d[1] for d in desc}
    assert kinds['#0'] == 'int_array'
    assert kinds['#1'] == 'unsupported'
    assert kinds['bogus'] == 'int_array'


def test_lenient_mode_string(acl_utils, lenient_env):
    desc = acl_utils.py_describe_args('ascend_sort', ('stable',))
    assert desc[0][1] == 'unsupported'


# ---------------------------------------------------------------------------
# 静态一致性：reduction 注册表不允许再出现「op 名 ↔ aclop 名」错位
# ---------------------------------------------------------------------------
_REDUCTION_ALIASES = {
    'any': 'Any', 'all': 'All', 'max': 'Max', 'min': 'Min',
    'argmax': 'ArgMax', 'argmin': 'ArgMin', 'mean': 'Mean',
    'sum': 'Sum', 'prod': 'Prod',
}


def _parse_reduction_registrations():
    from cupy.backends.ascend.api import acl_utils
    path = os.path.join(os.path.dirname(acl_utils.__file__), 'acl_utils.pyx')
    with open(path) as f:
        src = f.read()
    pairs = re.findall(
        r'func_union\.reduction_op\s*=\s*(aclop_\w+)\s*\n\s*'
        r'register_acl_ufunc\("ascend_(\w+)",\s*REDUCTION_OP', src)
    assert pairs, 'no reduction registrations found (parse failure?)'
    return pairs


def test_reduction_registration_pairing():
    """argmax/argmin/mean 曾整体错位（review §2.1）——静态卡住这类回归。"""
    for aclop_name, ufunc_suffix in _parse_reduction_registrations():
        expected = _REDUCTION_ALIASES.get(ufunc_suffix)
        if expected is None:
            continue        # nan* 等别名不在本表的覆盖范围内
        assert aclop_name == f'aclop_{expected}', (
            f'ascend_{ufunc_suffix} registered to {aclop_name}, '
            f'expected aclop_{expected}')


def test_reduction_registration_findable_in_source():
    pairs = _parse_reduction_registrations()
    names = {u for _, u in pairs}
    for required in ('argmax', 'argmin', 'mean', 'sum', 'prod', 'max', 'min'):
        assert required in names
