"""Backend feature definitions for the CuPy build system.

Each accelerator backend (CUDA, ROCm/HIP, Ascend/CANN) is defined in its own
module so that version-specific logic and library lists stay isolated:

* :mod:`~cupy_builder.features._base`  - ``Feature`` base class and helpers
* :mod:`~cupy_builder.features.cuda`   - NVIDIA CUDA backend
* :mod:`~cupy_builder.features.rocm`   - AMD ROCm/HIP backend
* :mod:`~cupy_builder.features.ascend` - Huawei Ascend/CANN backend

``cupy_builder._features.get_features()`` is the single entry point and
dispatches to whichever backend is selected by the build context.
"""

from cupy_builder.features._base import Feature  # NOQA
from cupy_builder.features._base import from_dict  # NOQA

__all__ = ['Feature', 'from_dict']
