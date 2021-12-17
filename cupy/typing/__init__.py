from numpy.typing import ArrayLike  # NOQA
from numpy.typing import DTypeLike  # NOQA
from numpy.typing import NBitBase  # NOQA

try:
    from cupy.typing._generic_alias import NDArray  # NOQA
except ImportError:
    pass
