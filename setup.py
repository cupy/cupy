#!/usr/bin/env python

import sys

from setuptools import setup

import chainer_setup_build


setup_requires = []
install_requires = [
    'filelock',
    'nose',
    'numpy>=1.9.0',
    'protobuf',
    'six>=1.9.0',
]


# Hack for Read the Docs
on_rtd = chainer_setup_build.check_readthedocs_environment()
if on_rtd:
    print('Add develop command for Read the Docs')
    sys.argv.insert(1, 'develop')
    setup_requires = ['Cython>=0.23'] + setup_requires

chainer_setup_build.parse_args()

setup(
    name='chainer',
    version='1.8.2',
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
              'chainer.function_hooks',
              'chainer.links',
              'chainer.links.activation',
              'chainer.links.caffe',
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
              'cupy.core',
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
        'cupy': ['core/carray.cuh'],
    },
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=['mock',
                   'nose'],
    # To trick build into running build_ext
    ext_modules=[chainer_setup_build.dummy_extension],
    cmdclass={
        'build_ext': chainer_setup_build.chainer_build_ext,
    },
)
