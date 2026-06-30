"""
Wheel build configurations.

Vendored subset of cupy-release-tools' ``dist_config.py``. These constants
drive ``ci/tools/prepare_wheel_build.py`` and the ``build-wheel`` workflow.
Keep in sync if cupy-release-tools changes (the long-term goal of #9974 is
to retire that repo in favor of these definitions).
"""
from __future__ import annotations

# Wheel package names per CTK major.
WHEEL_PACKAGE_NAMES: dict[str, str] = {
    "12": "cupy-cuda12x",
    "13": "cupy-cuda13x",
}

# Preload libraries to bundle metadata for, by host platform per CTK major.
# Matches what ``cupyx/tools/install_library.py`` can download.
PRELOAD_LIBRARIES: dict[str, dict[str, tuple[str, ...]]] = {
    "12": {
        "linux": ("cutensor", "nccl"),
        "win": ("cutensor",),
    },
    "13": {
        "linux": ("cutensor", "nccl"),
        "win": ("cutensor",),
    },
}

_LONG_DESCRIPTION_HEADER = """\
.. image:: https://raw.githubusercontent.com/cupy/cupy/main/docs/image/cupy_logo_1000px.png
   :width: 400

CuPy : NumPy & SciPy for GPU
============================

`CuPy <https://cupy.dev/>`_ is a NumPy/SciPy-compatible array library for GPU-accelerated computing with Python.

"""

# ``{version}`` and ``{wheel_suffix}`` are filled in by ``prepare_wheel_build.py``.
WHEEL_LONG_DESCRIPTION_CUDA: str = _LONG_DESCRIPTION_HEADER + """\
This is a CuPy wheel (precompiled binary) package for CUDA {version}.
You need to install `CUDA Toolkit {version} <https://developer.nvidia.com/cuda-toolkit-archive>`_ locally to use these packages.
Alternatively, you can install this package together with all needed CUDA components from PyPI by passing the ``[ctk]`` tag::

   $ pip install cupy-cuda{wheel_suffix}[ctk]

If you have another version of CUDA, or want to build from source, refer to the `Installation Guide <https://docs.cupy.dev/en/latest/install.html>`_ for instructions.
"""
