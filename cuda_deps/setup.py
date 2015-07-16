#!/usr/bin/env python
from setuptools import setup

setup(
    name='chainer-cuda-deps',
    version='1.1.0.1',
    description='Install dependent packages to use Chainer on CUDA',
    author='Seiya Tokui',
    author_email='tokui@preferred.jp',
    url='http://chainer.org/',
    packages=[],
    install_requires=[
        'chainer',
        'pycuda>=2014.1',
        'scikit-cuda>=0.5.0',
        'Mako',
        'six>=1.9.0',
    ],
)
