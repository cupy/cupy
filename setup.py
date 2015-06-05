#!/usr/bin/env python
from distutils.core import setup

setup(
    name='chainer',
    version='1.0.0',
    description='A flexible framework of neural networks',
    author='Seiya Tokui',
    author_email='tokui@preferred.jp',
    url='http://chainer.org/',
    packages=['chainer',
              'chainer.cudnn',
              'chainer.functions',
              'chainer.optimizers',
              'chainer.requirements',
              'chainer.utils'],
    install_requires=['numpy'],
    scripts=['scripts/chainer-cuda-requirements'],
)
