#!/usr/bin/env python

import imp
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


setup_requires = [
    'fastrlock>=0.3',
]
install_requires = [
    'numpy>=1.9.0',
    'six>=1.9.0',
    'fastrlock>=0.3',
]

ext_modules = cupy_setup_build.get_ext_modules()
build_ext = cupy_setup_build.custom_build_ext
sdist = cupy_setup_build.sdist_with_cython

here = os.path.abspath(os.path.dirname(__file__))
__version__ = imp.load_source(
    '_version', os.path.join(here, 'cupy', '_version.py')).__version__

setup(
    name='cupy',
    version=__version__,
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
              'cupy.cuda.memory_hooks',
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
            'core/include/cupy/complex/arithmetic.h',
            'core/include/cupy/complex/catrig.h',
            'core/include/cupy/complex/catrigf.h',
            'core/include/cupy/complex/ccosh.h',
            'core/include/cupy/complex/ccoshf.h',
            'core/include/cupy/complex/cexp.h',
            'core/include/cupy/complex/cexpf.h',
            'core/include/cupy/complex/clog.h',
            'core/include/cupy/complex/clogf.h',
            'core/include/cupy/complex/complex.h',
            'core/include/cupy/complex/complex_inl.h',
            'core/include/cupy/complex/cpow.h',
            'core/include/cupy/complex/cproj.h',
            'core/include/cupy/complex/csinh.h',
            'core/include/cupy/complex/csinhf.h',
            'core/include/cupy/complex/csqrt.h',
            'core/include/cupy/complex/csqrtf.h',
            'core/include/cupy/complex/ctanh.h',
            'core/include/cupy/complex/ctanhf.h',
            'core/include/cupy/complex/math_private.h',
            'core/include/cupy/carray.cuh',
            'core/include/cupy/complex.cuh',
            'cuda/cupy_thrust.cu',
        ],
    },
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=['mock',
                   'pytest'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext,
              'sdist': sdist},
)
