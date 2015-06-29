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
              'chainer.requirements',
              'chainer.utils'],
    package_data={'chainer.requirements': ['cuda-requirements.txt']},
    install_requires=['numpy',
                      'six>=1.9.0'],
    scripts=['scripts/chainer-cuda-requirements'],
    tests_require=['nose'],
)
