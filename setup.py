#!/usr/bin/env python
from __future__ import annotations


import glob
import os
import sys

from setuptools import setup
from setuptools.dist import Distribution

source_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(source_root, 'install'))

import cupy_builder  # NOQA
from cupy_builder import cupy_setup_build  # NOQA

ctx = cupy_builder.Context(source_root)
cupy_builder.initialize(ctx)


# Base distribution name used for the Ascend build. `pyproject.toml` keeps
# `cupy` for the CUDA/HIP/CPU builds; the Ascend distribution adds the CANN
# release train as a suffix (`numpy-ascend-cann85`, `numpy-ascend-cann90`),
# mirroring upstream's `cupy-cuda11x` / `cupy-cuda12x` scheme so that
# `pip list` / `pip freeze` / `pip uninstall` can tell the trains apart. The
# import package is `cupy` on every backend.
ASCEND_DISTRIBUTION_NAME = 'numpy-ascend'


def _ascend_distribution_name() -> str:
    """Return e.g. ``'numpy-ascend-cann85'`` for the active CANN release train.

    Falls back to the bare ``'numpy-ascend'`` when the CANN version could not be
    determined (no platform tag to derive the suffix from).
    """
    tag = cupy_setup_build.get_wheel_platform_tag(ctx)
    if not tag:
        return ASCEND_DISTRIBUTION_NAME
    # 'cann8.5' -> 'cann85' (a dot is not a valid PEP 508 name character)
    return f'{ASCEND_DISTRIBUTION_NAME}-{tag.replace(".", "")}'


class _BackendAwareDistribution(Distribution):
    """Rename the distribution when building for Ascend.

    * Wheels and editable installs are named after the CANN release train,
      e.g. ``numpy-ascend-cann85``.
    * ``sdist`` is deliberately left as the bare ``numpy-ascend``: the source
      tarball holds Cython sources only (``MANIFEST.in`` excludes the generated
      ``.cpp``) and no CANN-specific artifact, so one tarball can build wheels
      for every CPython and every CANN release train.

    PEP 621 forbids a dynamic ``[project].name``, and a static ``[project]``
    value silently wins over ``setup(name=...)``, so the rename has to be done
    on the ``Distribution`` object itself. setuptools applies the ``[project]``
    table in ``parse_config_files()``, hence the override is placed *after* that
    call (doing it in ``__init__`` would be overwritten).
    """

    def parse_config_files(self, *args, **kwargs):
        super().parse_config_files(*args, **kwargs)
        if ctx.get_backend_name() != 'ascend':
            return
        if ctx.setup_command == 'sdist':
            # One tarball for every CPython/CANN combination: keep the bare
            # name instead of falling back to `[project].name` (which is
            # `cupy`, the CUDA/HIP/CPU name).
            self.metadata.name = ASCEND_DISTRIBUTION_NAME
            return
        self.metadata.name = _ascend_distribution_name()
# ASCEND: temp disable third-party submodule, by add dlpack.h into source 
# if not cupy_builder.preflight_check(ctx):
#     sys.exit(1)


# List of files that needs to be in the distribution (sdist/wheel).
# Notes:
# - Files only needed in sdist should be added to `MANIFEST.in`.
# - The following glob (`**`) ignores items starting with `.`.
# - libcudacxx's test files exceed the default path length limit on Windows, so
#   we have to exclude them so as to avoid asking users to touch the registry.
cupy_package_data = [
    'cupy/cuda/cupy_thrust.cu',
    'cupy/cuda/cupy_cub.cu',
    'cupy/cuda/cupy_cufftXt.cu',  # for cuFFT callback
    'cupy/cuda/cupy_cufftXt.h',  # for cuFFT callback
    'cupy/cuda/cupy_cufft.h',  # for cuFFT callback
    'cupy/cuda/cufft.pxd',  # for cuFFT callback
    'cupy/cuda/cufft.pyx',  # for cuFFT callback
    'cupy/random/cupy_distributions.cu',
    'cupy/random/cupy_distributions.cuh',
    'cupyx/scipy/ndimage/cuda/LICENSE',
    'cupyx/scipy/ndimage/cuda/pba_kernels_2d.h',
    'cupyx/scipy/ndimage/cuda/pba_kernels_3d.h',
] + [
    x for x in glob.glob('cupy/_core/include/cupy/**', recursive=True)
    if os.path.isfile(x)
]

package_data = {
    'cupy': [
        os.path.relpath(x, 'cupy') for x in cupy_package_data
    ],
}

package_data['cupy'] += cupy_setup_build.prepare_wheel_libs(ctx)


if ctx.setup_command in ('dist_info', 'egg_info'):
    # Extensions are unnecessary for dist_info generation as all sources files
    # can be enumerated via MANIFEST.in.
    print('Skipping extensions configuration')
    ext_modules = []
else:
    ext_modules = cupy_setup_build.get_ext_modules(True, ctx)


long_description = ''
if ctx.long_description_path is not None:
    with open(ctx.long_description_path) as f:
        long_description = f.read()


# ASCEND: append the backend's SDK tag (e.g. `cann8.5`) to the wheel platform
# tag so that wheels built against mutually-incompatible CANN releases can not
# be confused with one another, giving e.g.
#   numpy_ascend_cann85-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann8.5.whl
# (a CUDA/HIP build of the same tree still produces `cupy-14.0.0a1-...whl`, and
# the `sdist` command produces the CANN-agnostic `numpy_ascend-14.0.0a1.tar.gz`;
# see `_BackendAwareDistribution` above). Either way the importable packages are
# `cupy`/`cupyx`/`cupy_backends`.
_sdk_tag = cupy_setup_build.get_wheel_platform_tag(ctx)


def _make_cmdclass():
    cmds = {'build_ext': cupy_builder._command.custom_build_ext}
    if _sdk_tag:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

        class bdist_wheel(_bdist_wheel):
            def finalize_options(self):
                super().finalize_options()
                # The extensions are native, so never claim purity.
                self.root_is_pure = False

            def get_tag(self):
                python, abi, plat = super().get_tag()
                # Keep the python/abi tags untouched; only specialise the
                # platform tag, appending the SDK tag.
                if _sdk_tag not in plat:
                    plat = f'{plat}.{_sdk_tag}'
                return python, abi, plat

        cmds['bdist_wheel'] = bdist_wheel
    return cmds


setup(
    long_description=long_description,
    long_description_content_type='text/x-rst',
    package_data=package_data,
    zip_safe=False,
    ext_modules=ext_modules,
    cmdclass=_make_cmdclass(),
    distclass=_BackendAwareDistribution,
)
