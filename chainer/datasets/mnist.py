import gzip
import os
import struct

import numpy
import six

from chainer.dataset import download
from chainer.datasets import tuple_dataset


def get_mnist(withlabel=True, ndim=1, scale=1., dtype=numpy.float32,
              label_dtype=numpy.int32):
    """Gets the MNIST dataset.

    `MNIST <http://yann.lecun.com/exdb/mnist/>`_ is a set of hand-written
    digits represented by grey-scale 28x28 images. In the original images, each
    pixel is represented by one-byte unsigned integer. This function
    scales the pixels to floating point values in the interval ``[0, scale]``.

    This function returns the training set and the test set of the official
    MNIST dataset. If ``withlabel`` is ``True``, each dataset consists of
    tuples of images and labels, otherwise it only consists of images.

    Args:
        withlabel (bool): If ``True``, it returns datasets with labels. In this
            case, each example is a tuple of an image and a label. Otherwise,
            the datasets only contain images.
        ndim (int): Number of dimensions of each image. The shape of each image
            is determined depending on ``ndim`` as follows:

            - ``ndim == 1``: the shape is ``(784,)``
            - ``ndim == 2``: the shape is ``(28, 28)``
            - ``ndim == 3``: the shape is ``(1, 28, 28)``

        scale (float): Pixel value scale. If it is 1 (default), pixels are
            scaled to the interval ``[0, 1]``.
        dtype: Data type of resulting image arrays.
        label_dtype: Data type of the labels.

    Returns:
        A tuple of two datasets. If ``withlabel`` is ``True``, both datasets
        are :class:`~chainer.datasets.TupleDataset` instances. Otherwise, both
        datasets are arrays of images.

    """
    train_raw = _retrieve_mnist_training()
    train = _preprocess_mnist(train_raw, withlabel, ndim, scale, dtype,
                              label_dtype)
    test_raw = _retrieve_mnist_test()
    test = _preprocess_mnist(test_raw, withlabel, ndim, scale, dtype,
                             label_dtype)
    return train, test


def _preprocess_mnist(raw, withlabel, ndim, scale, image_dtype, label_dtype):
    images = raw['x']
    if ndim == 2:
        images = images.reshape(-1, 28, 28)
    elif ndim == 3:
        images = images.reshape(-1, 1, 28, 28)
    elif ndim != 1:
        raise ValueError('invalid ndim for MNIST dataset')
    images = images.astype(image_dtype)
    images *= scale / 255.

    if withlabel:
        labels = raw['y'].astype(label_dtype)
        return tuple_dataset.TupleDataset(images, labels)
    else:
        return images


def _retrieve_mnist_training():
    urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz']
    return _retrieve_mnist('train.npz', urls)


def _retrieve_mnist_test():
    urls = ['http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
    return _retrieve_mnist('test.npz', urls)


def _retrieve_mnist(name, urls):
    root = download.get_dataset_directory('pfnet/chainer/mnist')
    path = os.path.join(root, name)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, urls), numpy.load)


def _make_npz(path, urls):
    x_url, y_url = urls
    x_path = download.cached_download(x_url)
    y_path = download.cached_download(y_url)

    with gzip.open(x_path, 'rb') as fx, gzip.open(y_path, 'rb') as fy:
        fx.read(4)
        fy.read(4)
        N, = struct.unpack('>i', fx.read(4))
        if N != struct.unpack('>i', fy.read(4))[0]:
            raise RuntimeError('wrong pair of MNIST images and labels')
        fx.read(8)

        x = numpy.empty((N, 784), dtype=numpy.uint8)
        y = numpy.empty(N, dtype=numpy.uint8)

        for i in six.moves.range(N):
            y[i] = ord(fy.read(1))
            for j in six.moves.range(784):
                x[i, j] = ord(fx.read(1))

    numpy.savez_compressed(path, x=x, y=y)
    return {'x': x, 'y': y}
