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


# Distribution name used for every Ascend artifact (wheels AND sdist).
# `pyproject.toml` keeps `cupy` for the CUDA/HIP/CPU builds. A single PyPI
# project `numpy-ascend` hosts every CANN release train: the trains are told
# apart by the wheel's PEP 440 local version segment (`+cann8.5` / `+cann9.0`,
# see `_sdk_tag` below), which also keeps the wheel filenames distinct, so
# wheels of different trains can live side by side. The import package is
# `cupy` on every backend.
ASCEND_DISTRIBUTION_NAME = 'numpy-ascend'

# Pure-Python cupyx subpackages removed from the Ascend wheel. They are
# CUDA-specific (cusolver/cusparse/NCCL-backed or CUDA-only kernels) and
# essentially unusable on Ascend; accessing them raises ImportError instead
# of silently returning wrong results. The compiled cupyx extension modules
# (cusolver.pyx/cusparse.pyx/...) are already not built for Ascend (see
# cupy_builder/features/ascend.py) and degrade to _StubModule in cupyx.
#
# MUST KEEP (hard/lazy imports from `cupy` itself or `cupyx/__init__.py`):
#   cupyx.scipy.fft      - imported at module scope by cupy/_math/misc.py and
#                          cupy/lib/_routines_poly.py
#   cupyx.scipy.sparse   - imported at module scope by cupyx/scipy/__init__.py
#                          and cupy/testing
#   cupyx.scipy.special  - lazily imported by cupy/_core/core.pyx (ufuncs)
#   cupyx.linalg, cupyx.jit, cupyx.optimizing, cupyx.scipy (top level) -
#                          imported by cupyx/__init__.py / cupy._functional
_CUPYX_EXCLUDED_ON_ASCEND = [
    'cupyx.distributed',      # NCCL-based DistributedArray, CUDA-only
    'cupyx.scipy.fftpack',
    'cupyx.scipy.interpolate',
    'cupyx.scipy.ndimage',
    'cupyx.scipy.signal',
    'cupyx.scipy.spatial',
    'cupyx.scipy.stats',
]


class _BackendAwareDistribution(Distribution):
    """Rename the distribution when building for Ascend.

    * Every Ascend artifact (wheel, editable install, sdist) is named
      ``numpy-ascend``: one PyPI project. Wheels of the primary CANN train
      (``_PYPI_CANN_TRAIN``) keep the bare public version and are uploaded
      to PyPI; wheels of every other train carry a ``+cannX.Y`` local
      version segment and are distributed via GitHub Releases only, because
      PyPI rejects local version identifiers. The source tarball is
      CANN-agnostic by nature (``MANIFEST.in`` excludes the generated
      ``.cpp``).

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
        # Keep the bare name instead of falling back to `[project].name`
        # (which is `cupy`, the CUDA/HIP/CPU name).
        self.metadata.name = ASCEND_DISTRIBUTION_NAME
        # Prune CUDA-leaning pure-Python cupyx subpackages from the Ascend
        # wheel: they are unusable there (CUDA-specific algorithms, no aclnn
        # port) and only bloat the distribution. See _CUPYX_EXCLUDED_ON_ASCEND
        # for the list and _CUPYX_KEEP_ON_ASCEND for why the rest must stay.
        self.packages = [
            p for p in self.packages
            if not any(p == x or p.startswith(x + '.')
                       for x in _CUPYX_EXCLUDED_ON_ASCEND)
        ]
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


if ctx.setup_command in ('sdist', 'dist_info', 'egg_info'):
    # Extensions are unnecessary for these commands as all sources files
    # can be enumerated via MANIFEST.in (the sdist keeps the .pyx sources,
    # NOT the generated .cpp). Constructing them would probe the host
    # build environment (CUDA/CANN toolchain), which is unavailable on a
    # plain CI runner building the CANN-agnostic sdist.
    print('Skipping extensions configuration')
    ext_modules = []
else:
    ext_modules = cupy_setup_build.get_ext_modules(True, ctx)


long_description = ''
if ctx.long_description_path is not None:
    with open(ctx.long_description_path) as f:
        long_description = f.read()


# ASCEND: encode the backend's SDK tag (e.g. `cann8.5`) as a PEP 440
# *local version segment* of the wheel (PyTorch-style CANN train selection),
# giving e.g.
#   numpy_ascend-14.0.0a1+cann8.5-cp311-cp311-manylinux_2_28_x86_64.whl
# EXCEPT for the train published to PyPI (`_PYPI_CANN_TRAIN` below), whose
# wheels keep the bare public version - PyPI (Warehouse) rejects uploads
# whose version carries a local segment ("The use of local versions in
# '14.0.0a1+cann8.5' is not allowed"), which is also why PyTorch's
# `torch==2.5.1+cu121` wheels only ever live on download.pytorch.org.
# Every other train is distributed through the GitHub Release, where users
# pin the train explicitly:
#   pip install numpy-ascend==14.0.0a1+cann8.5 \
#       --find-links https://github.com/<owner>/<repo>/releases/expanded_assets/<tag>
# (`--find-links` accepts an HTML page such as GitHub's expanded_assets view;
# `--index-url` does NOT work there - GitHub Releases are not a PEP 503
# simple index.) The platform tag stays standard manylinux (produced by the
# auditwheel repair step in CI; PyPI rejects bare `linux_*` tags). A CUDA/HIP
# build of the same tree produces `cupy-14.0.0a1-...whl` (no local segment),
# and the `sdist` command produces the CANN-agnostic
# `numpy_ascend-14.0.0a1.tar.gz` (see `_BackendAwareDistribution` above).
# Either way the importable packages are `cupy`/`cupyx`/`cupy_backends`.
#
# NOTE on the dotted form (`cann8.5`, not the `cann85` style PyTorch uses for
# `+cu121`): dots are valid in PEP 440 local segments and pip handles them
# fine; they keep the CANN version readable. Be aware that a plain dotted
# `cannX.Y` compares the `cannX` part *lexically*, so `cann10.0` would sort
# BELOW `cann9.0` - if that ever matters, switch to fully dot-separated
# segments (`+cann.9.0` vs `+cann.10.0`), which compare the numbers
# numerically. Today only single-digit CANN majors/minors exist, and the
# primary train (bare version) is what pip installs by default anyway.
_sdk_tag = cupy_setup_build.get_wheel_platform_tag(ctx)

#: The CANN release train published to PyPI under the bare public version
#: (e.g. `14.0.0a1`). Wheels of every other train carry the `+cannX.Y` local
#: version segment and are distributed via GitHub Releases only. Bump this
#: when the primary supported CANN release changes; the workflow's publish
#: step selects the plain-versioned wheels accordingly.
_PYPI_CANN_TRAIN = 'cann9.0'


def _make_cmdclass():
    cmds = {'build_ext': cupy_builder._command.custom_build_ext}
    if _sdk_tag:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

        class bdist_wheel(_bdist_wheel):
            def finalize_options(self):
                # Mutate BEFORE super(): wheel's finalize_options pre-finalizes
                # `egg_info`, and `egg_info.finalize_options` snapshots the
                # version (egg_version) and resets metadata.version to it, so
                # the local segment must already be in place — otherwise only
                # the filename carries it while PKG-INFO/dist-info keep the
                # bare version (Warehouse rejects that mismatch).
                version = self.distribution.get_version()
                if (_sdk_tag and _sdk_tag != _PYPI_CANN_TRAIN
                        and f'+{_sdk_tag}' not in version):
                    self.distribution.metadata.version = f'{version}+{_sdk_tag}'
                super().finalize_options()
                # The extensions are native, so never claim purity.
                self.root_is_pure = False

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
