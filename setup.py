#!/usr/bin/env python

import glob
import os
from setuptools import setup, find_packages
import sys

source_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(source_root, 'install'))

import cupy_builder  # NOQA
from cupy_builder import cupy_setup_build  # NOQA

ctx = cupy_builder.Context(source_root)
cupy_builder.initialize(ctx)
if not cupy_builder.preflight_check(ctx):
    sys.exit(1)


# TODO(kmaehashi): migrate to pyproject.toml (see #4727, #4619)
setup_requires = [
    'Cython>=0.29.22,<3',
    'fastrlock>=0.5',
]
install_requires = [
    'numpy>=1.20,<1.26',  # see #4773
    'fastrlock>=0.5',
]
extras_require = {
    'all': [
        'scipy>=1.6,<1.12',  # see #4773
        'Cython>=0.29.22,<3',
        'optuna>=2.0',
    ],
    # TODO(kmaehashi): remove stylecheck and update the contribution guide
    'stylecheck': [
        'autopep8==1.5.5',
        'flake8==3.8.4',
        'pbr==5.5.1',
        'pycodestyle==2.6.0',

        'mypy==0.950',
        'types-setuptools==57.4.14',
    ],
    'test': [
        # 4.2 <= pytest < 6.2 is slow collecting tests and times out on CI.
        # pytest < 7.2 has some different behavior that makes our CI fail
        'pytest>=7.2',
        'hypothesis>=6.37.2,<6.55.0',
    ],
}
tests_require = extras_require['test']


# List of files that needs to be in the distribution (sdist/wheel).
# Notes:
# - Files only needed in sdist should be added to `MANIFEST.in`.
# - The following glob (`**`) ignores items starting with `.`.
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


if len(sys.argv) < 2 or sys.argv[1] == 'egg_info':
    # Extensions are unnecessary for egg_info generation as all sources files
    # can be enumerated via MANIFEST.in.
    ext_modules = []
else:
    ext_modules = cupy_setup_build.get_ext_modules(True, ctx)


# Get __version__ variable
with open(os.path.join(source_root, 'cupy', '_version.py')) as f:
    exec(f.read())

long_description = None
if ctx.long_description_path is not None:
    with open(ctx.long_description_path) as f:
        long_description = f.read()


CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3.11
Programming Language :: Python :: 3 :: Only
Programming Language :: Cython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: Microsoft :: Windows
"""


setup(
    name=ctx.package_name,
    version=__version__,  # NOQA
    description='CuPy: NumPy & SciPy for GPU',
    long_description=long_description,
    author='Seiya Tokui',
    author_email='tokui@preferred.jp',
    maintainer='CuPy Developers',
    url='https://cupy.dev/',
    license='MIT License',
    project_urls={
        "Bug Tracker": "https://github.com/cupy/cupy/issues",
        "Documentation": "https://docs.cupy.dev/",
        "Source Code": "https://github.com/cupy/cupy",
    },
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    packages=find_packages(exclude=['install', 'tests']),
    package_data=package_data,
    zip_safe=False,
    python_requires='>=3.7',
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    ext_modules=ext_modules,
    cmdclass={'build_ext': cupy_builder._command.custom_build_ext},
)
