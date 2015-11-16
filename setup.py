#!/usr/bin/env python
import os

from setuptools import setup


on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

install_requires = [
    'filelock',
    'nose',
    'numpy>=1.9.0',
    'protobuf',
    'six>=1.9.0']

if not on_rtd:
    install_requires.append('h5py>=2.5.0')

setup(
    name='chainer',
    version='1.4.1',
    description='A flexible framework of neural networks',
    author='Seiya Tokui',
    author_email='tokui@preferred.jp',
    url='http://chainer.org/',
    packages=['chainer',
              'chainer.functions',
              'chainer.functions.activation',
              'chainer.functions.array',
              'chainer.functions.caffe',
              'chainer.functions.connection',
              'chainer.functions.evaluation',
              'chainer.functions.loss',
              'chainer.functions.math',
              'chainer.functions.noise',
              'chainer.functions.normalization',
              'chainer.functions.pooling',
              'chainer.links',
              'chainer.links.activation',
              'chainer.links.connection',
              'chainer.links.loss',
              'chainer.links.model',
              'chainer.links.normalization',
              'chainer.optimizers',
              'chainer.serializers',
              'chainer.testing',
              'chainer.utils',
              'cupy',
              'cupy.binary',
              'cupy.creation',
              'cupy.cuda',
              'cupy.indexing',
              'cupy.io',
              'cupy.linalg',
              'cupy.logic',
              'cupy.manipulation',
              'cupy.math',
              'cupy.padding',
              'cupy.random',
              'cupy.sorting',
              'cupy.statistics',
              'cupy.testing'],
    package_data={
        'cupy': ['carray.cuh'],
    },
    install_requires=install_requires,
    # Cython is required to setup h5py, not for chainer itself.
    # This line is required for h5py-2.5.0, as `setup_requires` is missing in
    # its `setup.py` and you cannot install h5py directly.
    # In the msater branch of h5py, this problem is fixed.
    setup_requires=['Cython>=0.17'],
    tests_require=['mock',
                   'nose'],
)
