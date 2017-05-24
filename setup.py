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
    'filelock',
    'nose',
    'numpy>=1.9.0',
    'six>=1.9.0',
    'pynvrtc>=7.0',
]

ext_modules = cupy_setup_build.get_ext_modules()

setup(
    name='cupy',
    version='1.0.0b1',
    description='CuPy: NumPy-like API accelerated with CUDA',
    author='Seiya Tokui',
    author_email='tokui@preferred.jp',
    url='http://docs.cupy.chainer.org/',
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
              'cupy.statistics',
              'cupy.testing'],
    package_data={
        'cupy': ['core/carray.cuh'],
    },
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=['mock',
                   'nose'],
    ext_modules=ext_modules,
)
