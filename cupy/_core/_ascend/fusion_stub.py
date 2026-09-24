# placeholder to reduce core code modification to upstream cupy
"""Ascend stub for ``cupy._core.fusion`` (fusion is not ported).

共享代码（与后端无关的 ``cupy/`` / ``cupyx/`` 模块）通过
``from cupy._core import fusion`` 引用 fusion，并在真正融合前用
``fusion._is_fusing()`` 做短路判断。Ascend 下 ``cupy/_core/__init__.py``
把本模块绑定成 ``cupy._core.fusion``（以及 ``cupy._core.new_fusion``）。

因此本模块必须覆盖共享代码引用到的**全部属性**：即使 ``_is_fusing()`` 恒为
False、部分属性永远不会被真正调用，属性查找本身也可能发生（例如
``cupy.get_array_module`` 里 ``isinstance(arg, (..., _core.fusion._FusionVarArray,
_core.new_fusion._ArrayProxy))`` 的元组是在**每次调用**时构造的 —— 属性缺失会
直接 AttributeError）。检测引用点：``grep -rn "fusion\\._\\|new_fusion\\." cupy/``。

参考（GPU 侧同名对象）：``cupy/_core/_gpu/{fusion,_fusion_variable,_fusion_op}.pyx``、
``_gpu/_fusion_interface.py``。
"""


# --- 融合变量/代理类型 -------------------------------------------------------
# 仅作为 isinstance 元组里的占位类型；_is_fusing() 恒为 False，永远不会实例化。


class _FusionVarArray:
    """占位：GPU 侧 ``cupy._core.fusion._FusionVarArray``。"""


class _FusionVarScalar:
    """占位：GPU 侧 ``cupy._core.fusion._FusionVarScalar``。"""


class _ArrayProxy:
    """占位：GPU 侧 ``cupy._core.new_fusion._ArrayProxy``。"""


class _ScalarProxy:
    """占位：GPU 侧 ``cupy._core._fusion_interface._ScalarProxy``。"""


# --- 融合状态查询 -----------------------------------------------------------


def _is_fusing():
    return False


def is_fusing():
    return False


# --- 融合路径入口 -----------------------------------------------------------
# 共享代码在 `if fusion._is_fusing():` 之后才会调用，本 stub 下不可达；
# 显式报错而非静默返回错误结果。


def _call_ufunc(*args, **kwargs):
    raise NotImplementedError(
        'cupy fusion (ufunc tracing) is not ported to the Ascend backend')


def call_reduction(*args, **kwargs):
    raise NotImplementedError(
        'cupy fusion (reduction tracing) is not ported to the Ascend backend')


def fuse(*args, **kwargs):
    """No-op stand-in for ``cupy._core.fusion.fuse`` (fusion is not ported).

    Upstream uses it as a decorator factory (``@fuse()``); with fusion disabled
    the upstream decorator is transparent too, so returning the wrapped function
    unchanged keeps the decorated code working -- e.g. ``cupy.unique``'s
    ``_unique_update_mask_equal_nan``, the only user in the tree.
    """
    if len(args) == 1 and not kwargs and callable(args[0]):
        return args[0]

    def _decorator(func):
        return func
    return _decorator
