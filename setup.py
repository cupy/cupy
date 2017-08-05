#!/usr/bin/env python

import os
from setuptools import setup
import sys

import cupy_setup_build


if sys.version_info[:3] == (3, 5, 0):
    if not int(os.getenv('CUPY_PYTHON_350_FORCE', '0')):
        msg = """
CuPy does not work with Python 3.5.0.

We strongly recommend to use another version of Python.
If you want to use CuPy with Python 3.5.0 at your own risk,
set 1 to CUPY_PYTHON_350_FORCE environment variable."""
        print(msg)
        sys.exit(1)


setup_requires = []
install_requires = [
    'nose',
    'numpy>=1.9.0',
    'six>=1.9.0',
]

ext_modules = cupy_setup_build.get_ext_modules()
build_ext = cupy_setup_build.custom_build_ext

setup(
    name='cupy',
    version='2.0.0b1',
    description='CuPy: NumPy-like API accelerated with CUDA',
    author='Seiya Tokui',
    author_email='tokui@preferred.jp',
    url='https://docs-cupy.chainer.org/',
    license='MIT License',
    packages=['cupy',
              'cupy.binary',
              'cupy.core',
              'cupy.creation',
              'cupy.cuda',
              'cupy.ext',
              'cupy.indexing',
              'cupy.io',
              'cupy.linalg',
              'cupy.logic',
              'cupy.manipulation',
              'cupy.math',
              'cupy.padding',
              'cupy.prof',
              'cupy.random',
              'cupy.sorting',
              'cupy.sparse',
              'cupy.statistics',
              'cupy.testing'],
    package_data={
        'cupy': [
            'core/cupy/complex/arithmetic.h',
            'core/cupy/complex/catrig.h',
            'core/cupy/complex/catrigf.h',
            'core/cupy/complex/ccosh.h',
            'core/cupy/complex/ccoshf.h',
            'core/cupy/complex/cexp.h',
            'core/cupy/complex/cexpf.h',
            'core/cupy/complex/clog.h',
            'core/cupy/complex/clogf.h',
            'core/cupy/complex/complex.h',
            'core/cupy/complex/complex_inl.h',
            'core/cupy/complex/cpow.h',
            'core/cupy/complex/cproj.h',
            'core/cupy/complex/csinh.h',
            'core/cupy/complex/csinhf.h',
            'core/cupy/complex/csqrt.h',
            'core/cupy/complex/csqrtf.h',
            'core/cupy/complex/ctanh.h',
            'core/cupy/complex/ctanhf.h',
            'core/cupy/complex/math_private.h',
            'core/cupy/carray.cuh',
            'core/cupy/complex.cuh',
            'cuda/cupy_thrust.cu',
        ],
    },
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=['mock',
                   'nose'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
