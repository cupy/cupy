from __future__ import annotations

import contextvars
import warnings

from cupy import _util
# expose cache handles via `config` object
from cupy.fft._cache import (get_plan_cache,
                             clear_plan_cache,
                             get_plan_cache_size,
                             set_plan_cache_size,
                             get_plan_cache_max_memsize,
                             set_plan_cache_max_memsize,
                             show_plan_cache_info)
# expose callback handles via `config` object
from cupy.fft._callback import (
    get_current_callback_manager, set_cufft_callbacks)


class _FFTConfig:
    get_plan_cache = get_plan_cache
    clear_plan_cache = clear_plan_cache
    get_plan_cache_size = get_plan_cache_size
    set_plan_cache_size = set_plan_cache_size
    get_plan_cache_max_memsize = get_plan_cache_max_memsize
    set_plan_cache_max_memsize = set_plan_cache_max_memsize
    show_plan_cache_info = show_plan_cache_info
    get_current_callback_manager = get_current_callback_manager
    set_cufft_callbacks = set_cufft_callbacks

    _enable_nd_planning = contextvars.ContextVar(
        'cupy.fft.config.enable_nd_planning', default=True)
    _use_multi_gpus = contextvars.ContextVar(
        'cupy.fft.config.use_multi_gpus', default=False)
    _devices = contextvars.ContextVar(
        'cupy.fft.config.devices', default=None)

    def set_cufft_gpus(self, gpus):
        '''Set the GPUs to be used in multi-GPU FFT.

        Args:
            gpus (int or list of int): The number of GPUs or a list of GPUs
                to be used. For the former case, the first ``gpus`` GPUs
                will be used.

        .. warning::
            This API is currently experimental and may be changed in the future
            version.

        .. seealso:: `Multiple GPU cuFFT Transforms`_

        .. _Multiple GPU cuFFT Transforms:
            https://docs.nvidia.com/cuda/cufft/index.html#multiple-GPU-cufft-transforms
        '''
        _util.experimental('cupy.fft.config.set_cufft_gpus')

        if isinstance(gpus, int):
            devs = [i for i in range(gpus)]
        elif isinstance(gpus, (list, tuple)):
            devs = gpus
        else:
            raise ValueError("gpus must be an int or a list or tuple of int.")
        if len(devs) <= 1:
            raise ValueError("Must use at least 2 GPUs.")

        # make it hashable
        self._devices.set(tuple(devs))

    @property
    def enable_nd_planning(self):
        warnings.warn(
            'enable_nd_planning is deprecated as of CuPy 15 and will always '
            'be True in the future.', DeprecationWarning, stacklevel=2)
        return self._enable_nd_planning.get()

    @enable_nd_planning.setter
    def enable_nd_planning(self, value):
        warnings.warn(
            'enable_nd_planning is deprecated as of CuPy 15 and will always '
            'be True in the future.', DeprecationWarning, stacklevel=2)
        self._enable_nd_planning.set(value)

    @property
    def use_multi_gpus(self):
        return self._use_multi_gpus.get()

    @use_multi_gpus.setter
    def use_multi_gpus(self, value):
        self._use_multi_gpus.set(bool(value))

    @property
    def devices(self):
        return self._devices.get() if self._use_multi_gpus.get() else None


config = _FFTConfig()
