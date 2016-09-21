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

    """A pre-trained CNN model with 16 layers provided by VGG team [1].

    During initialization, this chain model automatically downloads
    the pre-trained caffemodel from the internet, convert to another
    chainer model, store it on your local directory, and initialize
    all the parameters with it.
    This model would be useful when you want to extract a semantic
    feature vector from a given image, or fine-tune the model
    on a different dataset.

    Note that this pre-trained model is released under Creative Commons
    Attribution License.

    [1] ``Very Deep Convolutional Networks for Large-Scale Image
    Recognition <https://arxiv.org/abs/1409.1556>``

    Args:
        pretrained_model (str): the destination of the pre-trained
            caffemodel. If this argument is specified as ``auto``,
            it automatically downloads the caffemodel from the internet.
            Note that in this case the converted chainer model is stored
            on ``$CHAINER_DATASET_ROOT/pfnet/chainer/models`` directory,
            where ``$CHAINER_DATASET_ROOT`` is set as
            ``$HOME/.chainer/dataset`` unless you specify another value
            as a environment variable.

    Attributes:
        available_layers (list of str): The list of available layer names
            used by ``__call__`` and ``extract`` methods.

    """

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

    @property
    def available_layers(self):
        return list(self.functions.iterkeys())

    def __call__(self, x, layers=['prob']):
        """Computes all the feature maps specified by ``layers``.

        Args:
            x (~chainer.Variable): Input variable.
            layers (list of str): The list of layernames you want to extract.

        Returns:
            Dictionary of ~chainer.Variable: The directory in which
            the key contains the layer name and the value contains
            the corresponding variable.

        """

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

    def prepare(self, image, size=(224, 224)):
        """Converts the given image to the numpy array.

        Note that you have to call this method before ``__call__``
        because the pre-trained vgg model requires to resize the given image,
        covert the RGB to the BGR, subtract the mean,
        and permute the dimensions before calling.

        Args:
            image (PIL.Image or numpy.ndarray): Input image.
            size (pair of ints): Size of converted images.
                If ``None``, the given image is not resized.

        Returns:
            numpy.ndarray: The converted output array.

        """

        if isinstance(image, numpy.ndarray):
            image = Image.fromarray(image)
        image = image.convert('RGB')
        if size is not None:
            image = image.resize(size)
        image = numpy.asarray(image, dtype=numpy.float32)
        image = image[:, :, ::-1]
        image -= numpy.array(
            [103.939, 116.779, 123.68], dtype=numpy.float32)
        image = image.transpose((2, 0, 1))
        return image

    def extract(self, images, layers=['fc7'], size=(224, 224)):
        """Extracts all the feature maps of given images.

        The difference of directory executing ``__call__`` is that
        it directory accepts the list of images as an input, and
        automatically transforms them to a proper variable. That is,
        it is also interpreted as a shortcut method that implicitly call
        ``prepare`` and ``__call__`` methods.

        Args:
            image (list of PIL.Image or numpy.ndarray): Input images.
            layers (list of str): The list of layernames you want to extract.
            size (pair of ints): The resolution of resized images used as
                an input of CNN. All the given images are not resized
                if this argument is ``None``, but the resolutions of
                all the images should be the same.

        Returns:
            Dictionary of ~chainer.Variable: The directory in which
            the key contains the layer name and the value contains
            the corresponding variable.

        """

        x = chainer.dataset.concat_examples(
            [self.prepare(img, size=size) for img in images])
        x = Variable(self.xp.asarray(x))
        return self(x, layers=layers)

    def predict(self, images):
        """Computes all the probabilities of given images.

        Args:
            image (list of PIL.Image or numpy.ndarray): Input images.

        Returns:
            ~chainer.Variable: Output that contains the class probabilities
                of given images.

        """

        return self.extract(images, layers=['prob'])['prob']
