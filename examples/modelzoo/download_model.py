#!/usr/bin/env python
from __future__ import print_function
import argparse
import zipfile

import six


parser = argparse.ArgumentParser(
    description='Download a Caffe reference model')
parser.add_argument('model_type',
                    choices=('alexnet', 'caffenet', 'googlenet', 'resnet'),
                    help='Model type (alexnet, caffenet, googlenet, resnet)')
args = parser.parse_args()

if args.model_type == 'alexnet':
    url = 'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel'
    name = 'bvlc_alexnet.caffemodel'
elif args.model_type == 'caffenet':
    url = 'http://dl.caffe.berkeleyvision.org/' \
          'bvlc_reference_caffenet.caffemodel'
    name = 'bvlc_reference_caffenet.caffemodel'
elif args.model_type == 'googlenet':
    url = 'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel'
    name = 'bvlc_googlenet.caffemodel'
elif args.model_type == 'resnet':
    url = 'http://research.microsoft.com/en-us/um/people/kahe/resnet/' \
          'models.zip'
    name = 'models.zip'
else:
    raise RuntimeError('Invalid model type. Choose from '
                       'alexnet, caffenet, googlenet and resnet.')

print('Downloading model file...')
six.moves.urllib.request.urlretrieve(url, name)

if args.model_type == 'resnet':
    with zipfile.ZipFile(name, 'r') as zf:
        zf.extractall('.')

print('Done')
