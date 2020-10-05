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

from cupyx._ufunc_config import errstate  # NOQA
from cupyx._ufunc_config import geterr  # NOQA
from cupyx._ufunc_config import seterr  # NOQA
from cupy.core.syncdetect import allow_synchronize  # NOQA
from cupy.core.syncdetect import DeviceSynchronized  # NOQA
