# "NOQA" to suppress flake8 warning
from cupyx.rsqrt import rsqrt  # NOQA
from cupyx.runtime import get_runtime_info  # NOQA
from cupyx.scatter import scatter_add  # NOQA
from cupyx.scatter import scatter_max  # NOQA
from cupyx.scatter import scatter_min  # NOQA

from cupyx import linalg  # NOQA
from cupyx import scipy  # NOQA

from cupyx._ufunc_config import errstate  # NOQA
from cupyx._ufunc_config import geterr  # NOQA
from cupyx._ufunc_config import seterr  # NOQA
