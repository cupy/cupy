from __future__ import print_function
from collections import OrderedDict
import os

import numpy
from PIL import Image

import chainer
import chainer.dataset
import chainer.functions as F
import chainer.initializers
import chainer.links as L
from chainer.links.caffe import CaffeFunction
from chainer import serializers
from chainer import Variable


class ImagePreprocessor(chainer.dataset.DatasetMixin):

    def __init__(self, images, size=None, mean=None, rgb_to_bgr=False):
        self.images = images
        self.size = size
        self.mean = mean
        self.rgb_to_bgr = rgb_to_bgr

    def __len__(self):
        return len(self.images)

    def get_example(self, i):
        img = self.images[i]
        if isinstance(img, numpy.ndarray):
            img = Image.fromarray(img)
        img = img.convert('RGB')
        if self.size is not None:
            img = img.resize(self.size)
        img = numpy.asarray(img, dtype=numpy.float32)
        if self.rgb_to_bgr:
            img = img[:, :, ::-1]
        if self.mean is not None:
            img -= self.mean
        img = img.transpose((2, 0, 1))
        return img


def _make_npz(path_npz, url, model):
    path_caffemodel = chainer.dataset.cached_download(url)
    print('Now loading caffemodel (usually it may take few and ten minutes)')
    caffemodel = CaffeFunction(path_caffemodel)
    serializers.save_npz(path_npz, caffemodel, compression=False)
    serializers.load_npz(path_npz, model)
    return model


def _retrieve(name, url, model):
    root = chainer.dataset.get_dataset_directory('pfnet/chainer/models/')
    path = os.path.join(root, name)
    return chainer.dataset.cache_or_load_file(
        path, lambda path: _make_npz(path, url, model),
        lambda path: serializers.load_npz(path, model))


class VGG16Layers(chainer.Chain):

    def __init__(self, pretrained_model='auto'):
        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            init = chainer.initializers.Zero()
            kwargs = {'initialW': init, 'initial_bias': init}
        else:
            # employ default initializers used in the original paper
            kwargs = {
                'initialW': chainer.initializers.Normal(0.01),
                'initial_bias': chainer.initializers.Zero(),
            }
        super(VGG16Layers, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, 1, 1, **kwargs),
            conv1_2=L.Convolution2D(64, 64, 3, 1, 1, **kwargs),
            conv2_1=L.Convolution2D(64, 128, 3, 1, 1, **kwargs),
            conv2_2=L.Convolution2D(128, 128, 3, 1, 1, **kwargs),
            conv3_1=L.Convolution2D(128, 256, 3, 1, 1, **kwargs),
            conv3_2=L.Convolution2D(256, 256, 3, 1, 1, **kwargs),
            conv3_3=L.Convolution2D(256, 256, 3, 1, 1, **kwargs),
            conv4_1=L.Convolution2D(256, 512, 3, 1, 1, **kwargs),
            conv4_2=L.Convolution2D(512, 512, 3, 1, 1, **kwargs),
            conv4_3=L.Convolution2D(512, 512, 3, 1, 1, **kwargs),
            conv5_1=L.Convolution2D(512, 512, 3, 1, 1, **kwargs),
            conv5_2=L.Convolution2D(512, 512, 3, 1, 1, **kwargs),
            conv5_3=L.Convolution2D(512, 512, 3, 1, 1, **kwargs),
            fc6=L.Linear(512 * 7 * 7, 4096, **kwargs),
            fc7=L.Linear(4096, 4096, **kwargs),
            fc8=L.Linear(4096, 1000, **kwargs),
        )
        if pretrained_model == 'auto':
            _retrieve(
                'VGG_ILSVRC_16_layers.npz',
                'http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/'
                'caffe/VGG_ILSVRC_16_layers.caffemodel',
                self)
        elif pretrained_model:
            serializers.load_npz(pretrained_model, self)

        self.functions = OrderedDict([
            ('conv1_1', [self.conv1_1, F.relu]),
            ('conv1_2', [self.conv1_2, F.relu]),
            ('pool1', [(F.max_pooling_2d, {'ksize': 2})]),
            ('conv2_1', [self.conv2_1, F.relu]),
            ('conv2_2', [self.conv2_2, F.relu]),
            ('pool2', [(F.max_pooling_2d, {'ksize': 2})]),
            ('conv3_1', [self.conv3_1, F.relu]),
            ('conv3_2', [self.conv3_2, F.relu]),
            ('conv3_3', [self.conv3_3, F.relu]),
            ('pool3', [(F.max_pooling_2d, {'ksize': 2})]),
            ('conv4_1', [self.conv4_1, F.relu]),
            ('conv4_2', [self.conv4_2, F.relu]),
            ('conv4_3', [self.conv4_3, F.relu]),
            ('pool4', [(F.max_pooling_2d, {'ksize': 2})]),
            ('conv5_1', [self.conv5_1, F.relu]),
            ('conv5_2', [self.conv5_2, F.relu]),
            ('conv5_3', [self.conv5_3, F.relu]),
            ('pool5', [(F.max_pooling_2d, {'ksize': 2})]),
            ('fc6', [self.fc6, F.relu]),
            ('fc7', [self.fc7, F.relu]),
            ('fc8', [self.fc8, F.relu]),
            ('prob', [F.softmax]),
        ])
        self.mean = numpy.asarray(
            [103.939, 116.779, 123.68], dtype=numpy.float32)

    @property
    def available_layers(self):
        return list(self.functions.iterkeys())

    def __call__(self, x, layers=[]):
        h = x
        activations = {}
        target_layers = set(layers)
        for key, funcs in self.functions.iteritems():
            for func in funcs:
                if isinstance(func, tuple):
                    func, kwargs = func
                    h = func(h, **kwargs)
                else:
                    h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)
            if len(target_layers) == 0:
                break
        return activations

    def preprocess(self, images, resize=True):
        kwargs = {'images': images, 'mean': self.mean, 'rgb_to_bgr': True}
        if resize:
            kwargs['size'] = (224, 224)
        processor = ImagePreprocessor(**kwargs)
        batch = chainer.iterators.SerialIterator(
            processor, 2**32, repeat=False, shuffle=False).next()
        return chainer.dataset.concat_examples(batch)

    def extract(self, images, layers=['fc7'], resize=True):
        x = self.preprocess(images, resize=resize)
        x = Variable(self.xp.asarray(x))
        return self(x, layers=layers)

    def predict(self, images):
        return self.extract(images, layers=['prob'], resize=True)['prob']
