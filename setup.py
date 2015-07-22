#!/usr/bin/env python
from setuptools import setup

setup(
    name='chainer',
    version='1.1.1',
    description='A flexible framework of neural networks',
    author='Seiya Tokui',
    author_email='tokui@preferred.jp',
    url='http://chainer.org/',
    packages=['chainer',
              'chainer.cudnn',
              'chainer.functions',
              'chainer.functions.caffe',
              'chainer.optimizers',
              'chainer.testing',
              'chainer.utils'],
    install_requires=['nose',
                      'numpy',
                      'protobuf',
                      'six>=1.9.0'],
    tests_require=['nose'],
)
