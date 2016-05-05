import gzip
import os
import struct

import numpy
import six
from six.moves.urllib import request

from chainer.dataset.datasets import tuple_dataset
from chainer.dataset import download


def get_mnist_training(withlabel=True, ndim=1, dtype=numpy.float32, scale=1.):
    """Gets the MNIST training set.

    `MNIST <http://yann.lecun.com/exdb/mnist/>`_ is a set of hand-written
    digits represented by grey-scale 28x28 images. Each pixel is scaled to
    values in the interval ``[0, scale]``.

    This function returns the training set of the official MNIST dataset.

    Args:
        withlabel (bool): If True, it returns a dataset with labels. In this
            case, each example is a tuple of an image and a label. Otherwise,
            the dataset only contains images.
        ndim (int): Number of dimensions of each image. The shape of each image
            is determined depending on ndim as follows:
                - ``ndim == 1``: the shape is ``(784,)``
                - ``ndim == 2``: the shape is ``(28, 28)``
                - ``ndim == 3``: the shape is ``(1, 28, 28)``
        dtype: Data type of images.
        scale (float): Pixel value scale. If it is 1 (default), pixels are
            scaled to the interval ``[0, 1]``.

    Returns:
        Dataset of tuples if ``withlabel`` is True, or an array of images
        otherwise. In latter case, each row corresponds to an image.

    .. seealso::
       Use :func:`get_mnist_test` to retrieve the MNIST test set.

    """
    raw = _retrieve_mnist_training()
    return _preprocess_mnist(raw, withlabel, ndim, dtype, scale)


def get_mnist_test(withlabel=True, ndim=1, dtype=numpy.float32, scale=1.):
    """Gets the MNIST test set.

    `MNIST <http://yann.lecun.com/exdb/mnist/>`_ is a set of hand-written
    digits represented by grey-scale 28x28 images. Each pixel is scaled to
    values in the interval ``[0, scale]``.

    This function returns the test set of the official MNIST dataset.

    Args:
        withlabel (bool): If True, it returns a dataset with labels. In this
            case, each example is a tuple of an image and a label. Othewrise,
            the dataset only contains images.
        ndim (int): Number of dimensions of each image. See
            :func:`get_mnist_training` for details.
        dtype: Data type of images.
        scale (float): Pixel value scale. If it is 1 (default), pixels are
            scaled to the interval ``[0, 1]``.

    Returns:
        Dataset of tuples if ``withlabel`` is True, or an array of images
        otherwise. In latter case, each row corresponds to an image.

    """
    raw = _retrieve_mnist_test()
    return _preprocess_mnist(raw, withlabel, ndim, dtype, scale)


def _preprocess_mnist(raw, withlabel, ndim, dtype, scale):
    images = raw['x']
    if ndim == 2:
        images = images.reshape(-1, 28, 28)
    elif ndim == 3:
        images = images.reshape(-1, 1, 28, 28)
    elif ndim != 1:
        raise ValueError('invalid ndim for MNIST dataset')
    images = images.astype(dtype)
    images *= scale / 255.

    if withlabel:
        labels = raw['y'].astype(numpy.int32)
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
    return download.cached_create_file(
        path, lambda path: _make_npz(path, urls), numpy.load)


def _make_npz(path, urls):
    x_url, y_url = urls
    x_path = download.cached_download(x_url)
    y_path = download.cached_download(y_url)

    with gzip.open(x_path, 'rb') as fx, gzip.open(y_path, 'rb') as fy:
        fx.read(4)
        fy.read(4)
        N = struct.unpack(fx.read(4), '>i')
        if N != struct.unpack(fy.read(4), '>i'):
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
