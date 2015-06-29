#!/usr/bin/env python
from setuptools import setup

setup(
    name='chainer',
    version='1.0.1',
    description='A flexible framework of neural networks',
    author='Seiya Tokui',
    author_email='tokui@preferred.jp',
    url='http://chainer.org/',
    packages=['chainer',
              'chainer.cudnn',
              'chainer.functions',
              'chainer.optimizers',
              'chainer.utils'],
    install_requires=['numpy',
                      'six>=1.9.0'],
    tests_require=['nose'],
)
