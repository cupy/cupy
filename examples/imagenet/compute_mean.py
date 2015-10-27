#!/usr/bin/env python
import argparse
import os
import sys

import numpy
from PIL import Image
import six.moves.cPickle as pickle


parser = argparse.ArgumentParser(description='Compute images mean array')
parser.add_argument('dataset', help='Path to training image-label list file')
parser.add_argument('--root', '-r', default='.',
                    help='Root directory path of image files')
parser.add_argument('--output', '-o', default='mean.npy',
                    help='path to output mean array')
args = parser.parse_args()

sum_image = None
count = 0
for line in open(args.dataset):
    filepath = os.path.join(args.root, line.strip().split()[0])
    image = numpy.asarray(Image.open(filepath)).transpose(2, 0, 1)
    if sum_image is None:
        sum_image = numpy.ndarray(image.shape, dtype=numpy.float32)
        sum_image[:] = image
    else:
        sum_image += image
    count += 1
    sys.stderr.write('\r{}'.format(count))
    sys.stderr.flush()

sys.stderr.write('\n')

mean = sum_image / count
pickle.dump(mean, open(args.output, 'wb'), -1)
