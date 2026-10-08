"""
Wheel build configurations.

Vendored subset of cupy-release-tools' ``dist_config.py``. These constants
drive ``ci/tools/prepare_wheel_build.py`` and the ``build-wheel`` workflow.
Keep in sync if cupy-release-tools changes (the long-term goal of #9974 is
to retire that repo in favor of these definitions).
"""
from __future__ import annotations

import argparse

# Wheel package names per CTK major.
WHEEL_PACKAGE_NAMES: dict[str, str] = {
    "12": "cupy-cuda12x",
    "13": "cupy-cuda13x",
}

# Wheel flavors. ``native`` is the wheel that is released to PyPI.
# ``cuda-python`` is a CI-only build of the same sdist with
# ``CUPY_USE_CUDA_PYTHON=1`` (CuPy built against and running on
# cuda.bindings / nvmath.bindings); it is never published.
WHEEL_FLAVORS: tuple[str, ...] = ("native", "cuda-python")

# Requirements of the ``cuda-python`` flavor, per CTK major. CuPy cimports
# ``cuda.bindings`` and ``nvmath.bindings``, so they are needed both to build
# the wheel and to import it. Keep in sync with the ``cuda-python`` and
# ``nvmath-python`` entries of the ``cuda-python`` variant lanes in
# ``.pfnci/matrix.yaml`` (``.pfnci/generate.py`` checks this).
BINDINGS_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "12": ("cuda-python==12.*", "nvmath-python==1.*"),
    "13": ("cuda-python==13.4.*", "nvmath-python==1.*"),
}


def wheel_artifact_name(
    cuda_major: str,
    flavor: str,
    python_version: str,
    host_platform: str,
    artifact_suffix: str,
) -> str:
    """Name of the GHA artifact holding a wheel built by build-wheel.yml.

    Both the producer (build-wheel.yml) and the consumer
    (.pfnci/linux/tests/actions/fetch-wheel.sh) get the name from here.

    The ``native`` name is what consumers such as docs.yml expect. The
    ``cuda-python`` name deliberately does NOT start with ``cupy-cuda12x-`` /
    ``cupy-cuda13x-``: ci-nightly.yml uploads every artifact matching those
    prefixes to the nightly wheel index, and the CI-only bindings wheels (same
    distribution name and version as the native ones) must never be uploaded.
    """
    if flavor == "native":
        prefix = f"cupy-cuda{cuda_major}x"
    elif flavor == "cuda-python":
        prefix = f"cupy-cuda-python-cuda{cuda_major}x"
    else:
        raise ValueError(f"unknown wheel flavor: {flavor!r}")
    return f"{prefix}-py{python_version}-{host_platform}-{artifact_suffix}"


# Preload libraries to bundle metadata for, by host platform per CTK major.
# Matches what ``cupyx/tools/install_library.py`` can download.
PRELOAD_LIBRARIES: dict[str, dict[str, tuple[str, ...]]] = {
    "12": {
        "linux-64": ("cutensor", "nccl"),
        "linux-aarch64": ("cutensor", "nccl"),
        "win-64": ("cutensor",),
    },
    "13": {
        "linux-64": ("cutensor", "nccl"),
        "linux-aarch64": ("cutensor", "nccl"),
        "win-64": ("cutensor",),
        # No cuTENSOR / NCCL redist for Windows ARM64 yet (as of CTK 13.4).
        # Revisit once NVIDIA ships those binaries; see cupy/cupy#10294.
        "win-arm64": (),
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

SDIST_LONG_DESCRIPTION: str = _LONG_DESCRIPTION_HEADER + """\
This package (``cupy``) is a source distribution.
For most users, use of pre-build wheel distributions are recommended:

- `cupy-cuda13x <https://pypi.org/project/cupy-cuda13x/>`_ (for NVIDIA CUDA 13.x)
- `cupy-cuda12x <https://pypi.org/project/cupy-cuda12x/>`_ (for NVIDIA CUDA 12.x)

- `cupy-rocm-7-0 <https://pypi.org/project/cupy-rocm-7-0/>`_ (for AMD ROCm 7.0)

Please see `Installation Guide <https://docs.cupy.dev/en/latest/install.html>`_ for the detailed instructions.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wheel naming helpers.")
    sub = parser.add_subparsers(dest="command", required=True)
    name = sub.add_parser(
        "artifact-name", help="Print the GHA artifact name of a built wheel.")
    name.add_argument("--cuda-major", required=True,
                      choices=sorted(WHEEL_PACKAGE_NAMES))
    name.add_argument("--flavor", required=True, choices=WHEEL_FLAVORS)
    name.add_argument("--python-version", required=True)
    name.add_argument("--host-platform", required=True)
    name.add_argument("--artifact-suffix", required=True)
    args = parser.parse_args(argv)
    print(wheel_artifact_name(
        args.cuda_major, args.flavor, args.python_version,
        args.host_platform, args.artifact_suffix,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
