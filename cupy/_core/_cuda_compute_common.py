from __future__ import annotations

import warnings


_cuda_compute = False


def _get_cuda_compute():
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


_cache_key_prefix = None


def _environment_cache_key_prefix():
    global _cache_key_prefix
    if _cache_key_prefix is None:
        import cuda.cccl

        import cupy
        from cupy.cuda import compiler
        _cache_key_prefix = '|'.join((
            cuda.cccl.__version__,
            str(cupy.cuda.runtime.runtimeGetVersion()),
            compiler._get_cupy_cache_key()))
    return _cache_key_prefix
