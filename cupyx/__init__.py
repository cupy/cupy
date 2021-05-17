# "NOQA" to suppress flake8 warning
from cupyx._rsqrt import rsqrt  # NOQA
from cupyx._runtime import get_runtime_info  # NOQA
from cupyx._scatter import scatter_add  # NOQA
from cupyx._scatter import scatter_max  # NOQA
from cupyx._scatter import scatter_min  # NOQA

from cupyx import linalg  # NOQA
from cupyx import time  # NOQA
from cupyx import scipy  # NOQA
from cupyx import optimizing  # NOQA
from cupyx import lapack  # NOQA

from cupyx._ufunc_config import errstate  # NOQA
from cupyx._ufunc_config import geterr  # NOQA
from cupyx._ufunc_config import seterr  # NOQA
from cupy._core.syncdetect import allow_synchronize  # NOQA
from cupy._core.syncdetect import DeviceSynchronized  # NOQA

from cupyx._pinned_array import empty_pinned  # NOQA
from cupyx._pinned_array import empty_like_pinned  # NOQA
from cupyx._pinned_array import zeros_pinned  # NOQA
from cupyx._pinned_array import zeros_like_pinned  # NOQA
