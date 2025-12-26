#!/usr/bin/env python
from __future__ import annotations


import glob
import os
from setuptools import setup
import sys

source_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(source_root, 'install'))

import cupy_builder  # NOQA
from cupy_builder import cupy_setup_build  # NOQA

ctx = cupy_builder.Context(source_root)
cupy_builder.initialize(ctx)
if not cupy_builder.preflight_check(ctx):
    sys.exit(1)


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

package_data = {
    'cupy': [
        os.path.relpath(x, 'cupy') for x in cupy_package_data
    ],
}

package_data['cupy'] += cupy_setup_build.prepare_wheel_libs(ctx)


ext_modules = cupy_setup_build.get_ext_modules(True, ctx)


long_description = ''
if ctx.long_description_path is not None:
    with open(ctx.long_description_path) as f:
        long_description = f.read()


dependencies = [
    "numpy>=2.0,<2.6",  # see #4773
]
optional_dependencies = {
    "all": [
        "scipy>=1.10,<1.17",  # see #4773
        "Cython>=3",
        "optuna>=2.0",
    ],
    "test": [
        "packaging",
        "pytest>=7.2",
        "hypothesis>=6.37.2,<6.55.0",
        "mpmath",
    ],
}
if not ctx.use_hip:
    dependencies.append("cuda-pathfinder>=1.3.2,==1.*")
    if not ctx.use_stub:
        cuda_major = ctx.features["cuda"].get_version() // 1000
        optional_dependencies["ctk"] = [
            f"cuda-toolkit[cudart,nvrtc,cublas,cufft,cusolver,cusparse,curand]=={cuda_major}.*"  # NOQA
        ]


setup(
    long_description=long_description,
    long_description_content_type='text/x-rst',
    package_data=package_data,
    install_requires=dependencies,
    extras_require=optional_dependencies,
    zip_safe=False,
    ext_modules=ext_modules,
    cmdclass={'build_ext': cupy_builder._command.custom_build_ext},
)
