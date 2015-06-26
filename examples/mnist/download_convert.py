#!/usr/bin/env python

import gzip
import numpy as np
import six
import urllib

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

    with gzip.open(images, "rb") as f_images,\
            gzip.open(labels, "rb") as f_labels:
        f_images.read(16)
        f_labels.read(8)
        for i in six.moves.range(num):
            target[i] = ord(f_labels.read(1))
            for j in six.moves.range(dim):
                data[i, j] = ord(f_images.read(1))

    return data, target

print("Downloading %s..." % train_images)
urllib.urlretrieve(
    '%s/%s' % (parent, train_images), train_images)
print("Done\r\nDownloading %s..." % train_labels)
urllib.urlretrieve(
    '%s/%s' % (parent, train_labels), train_labels)
print("Done\r\nDownloading %s..." % test_images)
urllib.urlretrieve(
    '%s/%s' % (parent, test_images), test_images)
print("Done\r\nDownloading %s..." % test_labels)
urllib.urlretrieve(
    '%s/%s' % (parent, test_labels), test_labels)
print("Done")

mnist = {}
print("Converting training data...")
data_train, target_train = load_mnist(train_images, train_labels, num_train)
print("Done\r\nConverting test data...")
data_test, target_test = load_mnist(test_images, test_labels, num_test)
mnist['data'] = np.append(data_train, data_test, axis=0)
mnist['target'] = np.append(target_train, target_test, axis=0)

print("Done\r\nSave output...")
with open('mnist.pkl', 'wb') as output:
    pickle.dump(mnist, output, -1)
print("Done\r\nConvert completed")
