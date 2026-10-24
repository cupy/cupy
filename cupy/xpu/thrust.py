# Forwarder kept for the fork sources that refer to `cupy.xpu.thrust`.
from cupy.cuda.thrust import *  # NOQA
from cupy.cuda import thrust as _thrust
available = _thrust.available
