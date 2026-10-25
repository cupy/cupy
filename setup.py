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

# Public import name of the Ascend build (`import numpy_xpu as xp`). It is a
# thin shim over `cupy` (see `numpy_xpu/__init__.py`), shipped Ascend-only;
# `cupy`/`cupyx`/`cupy_backends` stay importable for compatibility.
ASCEND_ALIAS_PACKAGE = 'numpy_xpu'

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
#   cupyx.scipy.fftpack  - `cupyx/scipy/fft/__init__.py:13` imports
#                          `fftpack.get_fft_plan` at module scope, so dropping
#                          it makes `import cupy` fail (verified by installing
#                          the wheel); it also works on Ascend (pure Python on
#                          top of the ops-fft backend).
#   cupyx.scipy.sparse   - imported at module scope by cupyx/scipy/__init__.py
#                          and cupy/testing
#   cupyx.scipy.special  - lazily imported by cupy/_core/core.pyx (ufuncs)
#   cupyx.linalg, cupyx.jit, cupyx.optimizing, cupyx.scipy (top level) -
#                          imported by cupyx/__init__.py / cupy._functional
_CUPYX_EXCLUDED_ON_ASCEND = [
    # `cupyx.distributed` used to be dropped ("NCCL-based, CUDA-only"); it is
    # kept now that `cupy.cuda.nccl` is backed by HCCL
    # (cupy/backends/ascend/cupy_hccl.h + the optional `ascend_hccl` build
    # feature). Without HCCL at build time the package still imports fine and
    # `init_process_group` raises "NCCL is not available" -- the same
    # degradation as an upstream CuPy built without NCCL.
    'cupyx.scipy.interpolate',
    'cupyx.scipy.ndimage',
    'cupyx.scipy.signal',
    'cupyx.scipy.spatial',
    'cupyx.scipy.stats',
    # depends on `cupyx.scipy.signal` (excluded above); nothing imports it
    'cupyx.signal',
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
            # `numpy_xpu` is the Ascend-facing import name only; keep the
            # CUDA/HIP wheel identical to upstream (`import cupy`).
            self.packages = [
                p for p in self.packages if p != ASCEND_ALIAS_PACKAGE]
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
    'cupy_backends/cuda/_softlink.pxd',  # for cuFFT callback
    'cupy/random/cupy_distributions.cu',
    'cupy/random/cupy_distributions.cuh',
    'cupyx/scipy/ndimage/cuda/LICENSE',
    'cupyx/scipy/ndimage/cuda/pba_kernels_2d.h',
    'cupyx/scipy/ndimage/cuda/pba_kernels_3d.h',
] + [
    x for x in glob.glob('cupy/_core/include/cupy/**', recursive=True)
    if os.path.isfile(x)
]

# Compliance (docs/ascend/Compliance.md): never ship the proprietary CUDA
# Toolkit headers (they are no longer vendored; a locally fetched copy would
# otherwise be picked up by the glob above). On Ascend, additionally drop the
# open-source CUDA-only trees/sources (CCCL, jitify, *.cuh, PBA kernels) that
# only the CUDA NVRTC JIT needs.
_ascend_build = ctx.get_backend_name() == 'ascend'
cupy_package_data = cupy_setup_build.strip_cuda_headers(
    cupy_package_data, include_cccl=_ascend_build)

package_data = {
    'cupy': [
        os.path.relpath(x, 'cupy') for x in cupy_package_data
    ],
}

package_data['cupy'] += cupy_setup_build.prepare_wheel_libs(ctx)


# CUDA-less setup.py invocations: direct `sdist`, and PEP 517's
# get_requires_for_build_sdist, which setuptools backs with
# `setup.py egg_info`. Both must skip the CUDA probe + [ctk] so the
# sdist builder can run without a CUDA toolkit.
#
# Do NOT add `dist_info` here: that's the subcommand setuptools uses
# for `prepare_metadata_for_build_wheel` (and _for_build_editable).
# Pip resolves extras (`pip install .[ctk]`) from that prepared
# metadata, so skipping [ctk] in dist_info silently drops the toolkit
# wheels. bdist_wheel is a separate setup() invocation on the wheel
# builders where the probe + [ctk] both run normally.
_sdist_phase = ctx.setup_command in {'sdist', 'egg_info'}

ext_modules = [] if _sdist_phase else cupy_setup_build.get_ext_modules(
    True, ctx)


# The long description (rendered on the PyPI project page) is provided
# declaratively by pyproject.toml `[tool.setuptools.dynamic] readme`
# (README.md, text/markdown) -- it applies to both wheel and sdist builds,
# with no env var needed. NOTE: do NOT hardcode `text/x-rst` here: README.md
# is Markdown, and a mismatched content type makes PyPI render it as raw text.


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

    # `build_py` only ever *adds* files to `build/lib/`, so anything packaged
    # by an earlier build survives and silently leaks into the next wheel:
    #   * forbidden CUDA headers (after they were un-vendored / CCCL dropped),
    #   * packages that are excluded on Ascend (`_CUPYX_EXCLUDED_ON_ASCEND`).
    # Both were observed in practice (the first made the compliance filter look
    # broken, the second shipped `cupyx/signal` while `cupyx/scipy/signal` was
    # gone). Prune the build tree after copying so clean and incremental builds
    # produce identical artifacts.
    from setuptools.command.build_py import build_py as _build_py

    class build_py(_build_py):
        def run(self):
            super().run()
            self._prune_stale()

        def _prune_stale(self):
            import shutil

            build_lib = self.build_lib
            removed = 0
            for root, _dirs, files in os.walk(build_lib):
                for name in files:
                    path = os.path.join(root, name)
                    rel = os.path.relpath(path, build_lib)
                    if cupy_setup_build.is_cuda_only_file(
                            rel, include_cccl=_ascend_build):
                        os.unlink(path)
                        removed += 1
            if removed:
                print('INFO: pruned {} CUDA-only file(s) left in {} by an '
                      'earlier build'.format(removed, build_lib))

            if _ascend_build:
                for pkg in _CUPYX_EXCLUDED_ON_ASCEND:
                    path = os.path.join(build_lib, *pkg.split('.'))
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        print('INFO: pruned stale package {!r} from {}'.format(
                            pkg, build_lib))

    cmds['build_py'] = build_py
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

    # Compliance (docs/ascend/Compliance.md): an Ascend sdist must not carry
    # CUDA-only sources (*.cuh, CCCL, jitify, PBA kernels). MANIFEST.in still
    # lists them so the plain CUDA/HIP tarball stays complete; the filter is
    # applied to the file list handed to `make_release_tree`, i.e. only when
    # `CUPY_INSTALL_USE_ASCEND=1` / the Ascend backend is active. Proprietary
    # CUDA Toolkit headers are dropped from every artifact via `package_data`
    # and MANIFEST.in, independently of this command.
    from setuptools.command.sdist import sdist as _sdist

    class sdist(_sdist):
        def make_release_tree(self, base_dir, files):
            files = cupy_setup_build.strip_cuda_headers(
                files, include_cccl=_ascend_build)
            super().make_release_tree(base_dir, files)

    cmds['sdist'] = sdist
    return cmds


dependencies = [
    "numpy>=2.0,<2.7",  # see #4773
]
optional_dependencies = {
    "all": [
        "scipy>=1.14,<1.19",  # see #4773
        "Cython>=3,!=3.2.6",
        "optuna>=2.0",
    ],
    "test": [
        "packaging",
        "pytest>=7.2",
        "hypothesis>=6.37.2,<6.55.0",
        "mpmath",
    ],
}
# `cuda-pathfinder` / the `ctk` extra are CUDA-only: an Ascend build must not
# pull in CUDA wheels (and `features["cuda"]` does not exist there).
if not ctx.use_hip and not ctx.use_ascend:
    dependencies.append("cuda-pathfinder>=1.3.4,==1.*")
    # Same reasoning as the ext_modules guard: sdist/egg_info/dist_info
    # run on a CUDA-less builder, so skip the version-detecting [ctk]
    # computation. The wheel builder re-runs setup.py under bdist_wheel
    # with CUDA available and populates [ctk] then.
    if not _sdist_phase:
        cuda_major = ctx.features["cuda"].get_version() // 1000
        optional_dependencies["ctk"] = [
            f"cuda-toolkit[cudart,nvrtc,cublas,cufft,cusolver,cusparse,curand]=={cuda_major}.*"  # NOQA
        ]


setup(
    package_data=package_data,
    install_requires=dependencies,
    extras_require=optional_dependencies,
    zip_safe=False,
    ext_modules=ext_modules,
    cmdclass=_make_cmdclass(),
    distclass=_BackendAwareDistribution,
)
