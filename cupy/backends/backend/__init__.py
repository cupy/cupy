# CANN 的 libop_common.so 引用了三个 ge:: 错误串辅助函数
# (GetViewErrorCodeStr / TypeUtils::FormatToSerialString /
#  TypeUtils::DataTypeToSerialString)，其提供者是 OPP 包里的 liboptiling.so，
# 但 libop_common 自身带 BIND_NOW(-z now)，而 CPython 默认以 RTLD_NOW dlopen
# 我们链接了 op_common 的扩展，于是 import 阶段就解析失败。
# 解决：先用 RTLD_GLOBAL 预加载 liboptiling.so，把符号放进全局名字空间，
# 之后 libop_common 即可正常解析。非 Ascend 环境没有该库，静默跳过。
# 注意：本模块 (`cupy.backends.backend`) 是 backend 扩展 (api.runtime 等)
# 的入口包，必须先于它们完成预加载，否则 `from cupy.backends.backend
# import is_ascend` / `import cupy` 会在 cupy/__init__ 早期 (line 15 的
# xpu_backend 别名处) 就触发 api.runtime 的动态库解析而失败。
import ctypes as _ctypes
import os as _os

try:
    _ctypes.CDLL('liboptiling.so', mode=_os.RTLD_LAZY | _os.RTLD_GLOBAL)
except (OSError, AttributeError):
    pass

from cupy.backends.backend.api.runtime import is_hip  # NOQA
from cupy.backends.backend.api.runtime import is_ascend  # NOQA
