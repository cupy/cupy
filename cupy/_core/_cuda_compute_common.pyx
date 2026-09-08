import threading
import warnings

import cupy
from cupy import _util
from cupy._core import core
from cupy.cuda import compiler
from cupy.cuda._compiler_cache import _hash_hexdigest
from cupy.cuda.device cimport get_compute_capability


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
def _make_raw_ops(str src, tuple names):
    """Compile ``src`` once and return ``{name: RawOp}`` for every entry
    point in ``names``.

    Ops linked into one kernel must come from one module, or nvJitLink sees
    the header device functions defined once per module.
    """
    ltoir = core.compile_to_ltoir(src, prepend_cupy_headers=True)
    compute = _get_cuda_compute()
    return {name: compute.op.RawOp(ltoir=ltoir, name=name) for name in names}


def _make_raw_op(str src, str name):
    return _make_raw_ops(src, (name,))[name]


cdef object _thread_local = threading.local()
_cache_key_prefix = None


cpdef str _environment_cache_key_prefix():
    global _cache_key_prefix
    if _cache_key_prefix is None:
        import cuda.cccl
        _cache_key_prefix = '|'.join((
            cuda.cccl.__version__,
            str(cupy.cuda.runtime.runtimeGetVersion()),
            compiler._get_cupy_cache_key()))
    return _cache_key_prefix


cpdef cached_algorithm(str family, tuple key, str key_src, build):
    """Per-thread, per-device memo of a built cuda.compute algorithm object,
    backed by the kernel cache on disk."""

    cache = getattr(_thread_local, family, None)
    if cache is None:
        cache = {}
        setattr(_thread_local, family, cache)
    full_key = (cupy.cuda.device.get_device_id(),) + key
    cached = cache.get(full_key)
    if cached is not None:
        return cached

    compute = _get_cuda_compute()
    prefix = _environment_cache_key_prefix()
    name = _hash_hexdigest(
        f'{prefix}|{get_compute_capability()}|{key_src}'.encode()
    ) + f'.cc_{family}'
    algo = None
    blob = compiler._kernel_cache_backend.load(name)
    if blob is not None:
        try:
            algo = compute.deserialize(blob)
        except Exception:
            algo = None
    if algo is None:
        algo = build()
        compiler._kernel_cache_backend.save(name, algo.serialize(), '')
    cache[full_key] = algo
    return algo
