import warnings

from cupy import _util
from cupy._core import core


_cuda_compute = False


cpdef _get_cuda_compute():
    global _cuda_compute

    if _cuda_compute is False:
        try:
            from cuda import compute
        except ImportError:
            _cuda_compute = None
        else:
            if hasattr(compute, 'OpKind'):
                _cuda_compute = compute
            else:
                warnings.warn(
                    'cuda.compute is installed but its CUDA bindings '
                    'could not be loaded, so the cuda_compute '
                    'accelerator will be skipped', RuntimeWarning)
                _cuda_compute = None
    return _cuda_compute


@_util.memoize(for_each_device=True)
def _make_raw_op(str src, str name):
    ltoir = core.compile_to_ltoir(src)
    return _get_cuda_compute().op.RawOp(ltoir=ltoir, name=name)
