from cupy.fft._config import config  # NOQA
from cupy.fft._fft import fft  # NOQA
from cupy.fft._fft import fft2  # NOQA
from cupy.fft._fft import fftfreq  # NOQA
from cupy.fft._fft import fftn  # NOQA
from cupy.fft._fft import fftshift  # NOQA
from cupy.fft._fft import hfft  # NOQA
from cupy.fft._fft import ifft  # NOQA
from cupy.fft._fft import ifft2  # NOQA
from cupy.fft._fft import ifftn  # NOQA
from cupy.fft._fft import ifftshift  # NOQA
from cupy.fft._fft import ihfft  # NOQA
from cupy.fft._fft import irfft  # NOQA
from cupy.fft._fft import irfft2  # NOQA
from cupy.fft._fft import irfftn  # NOQA
from cupy.fft._fft import rfft  # NOQA
from cupy.fft._fft import rfft2  # NOQA
from cupy.fft._fft import rfftfreq  # NOQA
from cupy.fft._fft import rfftn  # NOQA

__all__ = ["fft", "fft2", "fftfreq", "fftn", "fftshift", "hfft",
           "ifft", "ifft2", "ifftn", "ifftshift", "ihfft",
           "irfft", "irfft2", "irfftn",
           "rfft", "rfft2", "rfftfreq", "rfftn", "config"]

# Make config accessible as a submodule for backward compatibility.
# Allows `from cupy.fft.config import get_plan_cache` and even
# `import cupy.fft.config as fft_config` to work.
# (older mypy versions do not know about __spec__)
__import__("sys").modules[__spec__.name + ".config"] = config  # type: ignore
