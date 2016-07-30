import gzip
import os

import numpy as np
import six
from six.moves.urllib import request

parent = 'http://yann.lecun.com/exdb/mnist'
train_images = 'train-images-idx3-ubyte.gz'
train_labels = 'train-labels-idx1-ubyte.gz'
test_images = 't10k-images-idx3-ubyte.gz'
test_labels = 't10k-labels-idx1-ubyte.gz'
num_train = 60000
num_test = 10000
dim = 784


def load_mnist(images, labels, num):
    data = np.zeros(num * dim, dtype=np.uint8).reshape((num, dim))
    target = np.zeros(num, dtype=np.uint8).reshape((num, ))

    with gzip.open(images, 'rb') as f_images,\
            gzip.open(labels, 'rb') as f_labels:
        f_images.read(16)
        f_labels.read(8)
        for i in six.moves.range(num):
            target[i] = ord(f_labels.read(1))
            for j in six.moves.range(dim):
                data[i, j] = ord(f_images.read(1))

    return data, target


def download_mnist_data():
    print('Downloading {:s}...'.format(train_images))
    request.urlretrieve('{:s}/{:s}'.format(parent, train_images), train_images)
    print('Done')
    print('Downloading {:s}...'.format(train_labels))
    request.urlretrieve('{:s}/{:s}'.format(parent, train_labels), train_labels)
    print('Done')
    print('Downloading {:s}...'.format(test_images))
    request.urlretrieve('{:s}/{:s}'.format(parent, test_images), test_images)
    print('Done')
    print('Downloading {:s}...'.format(test_labels))
    request.urlretrieve('{:s}/{:s}'.format(parent, test_labels), test_labels)
    print('Done')

    print('Converting training data...')
    data_train, target_train = load_mnist(train_images, train_labels,
                                          num_train)
    print('Done')
    print('Converting test data...')
    data_test, target_test = load_mnist(test_images, test_labels, num_test)
    mnist = {'data': np.append(data_train, data_test, axis=0),
             'target': np.append(target_train, target_test, axis=0)}
    print('Done')
    print('Save output...')
    with open('mnist.pkl', 'wb') as output:
        six.moves.cPickle.dump(mnist, output, -1)
    print('Done')
    print('Convert completed')


def load_mnist_data():
    if not os.path.exists('mnist.pkl'):
        download_mnist_data()
    with open('mnist.pkl', 'rb') as mnist_pickle:
        mnist = six.moves.cPickle.load(mnist_pickle)
    return mnist
