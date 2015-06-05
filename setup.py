#!/usr/bin/env python
from distutils.core import setup

setup(
    name='chainer',
    version='1.0.0',
    description='Neural network framework with on-the-fly graph construction',
    author='Seiya Tokui',
    author_email='tokui@preferred.jp',
    url='https://github.com/pfnet/chainer',
    packages=['chainer',
              'chainer.cudnn',
              'chainer.functions',
              'chainer.optimizers',
              'chainer.utils'],
    install_requires=['numpy'],
    scripts=['scripts/chainer-cuda-requirements'],
)
