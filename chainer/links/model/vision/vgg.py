from __future__ import print_function
import collections
import os

import numpy
try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e

from chainer.dataset.convert import concat_examples
from chainer.dataset import download
from chainer import flag
from chainer.functions.activation.relu import relu
from chainer.functions.activation.softmax import softmax
from chainer.functions.array.reshape import reshape
from chainer.functions.math.sum import sum
from chainer.functions.noise.dropout import dropout
from chainer.functions.pooling.max_pooling_2d import max_pooling_2d
from chainer.initializers import constant
from chainer.initializers import normal
from chainer import link
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.links.connection.linear import Linear
from chainer.serializers import npz
from chainer.utils import imgproc
from chainer.variable import Variable


class VGG16Layers(link.Chain):

    """A pre-trained CNN model with 16 layers provided by VGG team [1].

    During initialization, this chain model automatically downloads
    the pre-trained caffemodel, convert to another chainer model,
    stores it on your local directory, and initializes all the parameters
    with it. This model would be useful when you want to extract a semantic
    feature vector from a given image, or fine-tune the model
    on a different dataset.
    Note that this pre-trained model is released under Creative Commons
    Attribution License.

    If you want to manually convert the pre-trained caffemodel to a chainer
    model that can be specified in the constructor,
    please use ``convert_caffemodel_to_npz`` classmethod instead.

    .. [1] K. Simonyan and A. Zisserman, `Very Deep Convolutional Networks
        for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`_

    Args:
        pretrained_model (str): the destination of the pre-trained
            chainer model serialized as a ``.npz`` file.
            If this argument is specified as ``auto``,
            it automatically downloads the caffemodel from the internet.
            Note that in this case the converted chainer model is stored
            on ``$CHAINER_DATASET_ROOT/pfnet/chainer/models`` directory,
            where ``$CHAINER_DATASET_ROOT`` is set as
            ``$HOME/.chainer/dataset`` unless you specify another value
            as a environment variable. The converted chainer model is
            automatically used from the second time.
            If the argument is specified as ``None``, all the parameters
            are not initialized by the pre-trained model, but the default
            initializer used in the original paper, i.e.,
            ``chainer.initializers.Normal(scale=0.01)``.

    Attributes:
        available_layers (list of str): The list of available layer names
            used by ``__call__`` and ``extract`` methods.

    """

    def __init__(self, pretrained_model='auto'):
        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            init = constant.Zero()
            kwargs = {'initialW': init, 'initial_bias': init}
        else:
            # employ default initializers used in the original paper
            kwargs = {
                'initialW': normal.Normal(0.01),
                'initial_bias': constant.Zero(),
            }
        super(VGG16Layers, self).__init__(
            conv1_1=Convolution2D(3, 64, 3, 1, 1, **kwargs),
            conv1_2=Convolution2D(64, 64, 3, 1, 1, **kwargs),
            conv2_1=Convolution2D(64, 128, 3, 1, 1, **kwargs),
            conv2_2=Convolution2D(128, 128, 3, 1, 1, **kwargs),
            conv3_1=Convolution2D(128, 256, 3, 1, 1, **kwargs),
            conv3_2=Convolution2D(256, 256, 3, 1, 1, **kwargs),
            conv3_3=Convolution2D(256, 256, 3, 1, 1, **kwargs),
            conv4_1=Convolution2D(256, 512, 3, 1, 1, **kwargs),
            conv4_2=Convolution2D(512, 512, 3, 1, 1, **kwargs),
            conv4_3=Convolution2D(512, 512, 3, 1, 1, **kwargs),
            conv5_1=Convolution2D(512, 512, 3, 1, 1, **kwargs),
            conv5_2=Convolution2D(512, 512, 3, 1, 1, **kwargs),
            conv5_3=Convolution2D(512, 512, 3, 1, 1, **kwargs),
            fc6=Linear(512 * 7 * 7, 4096, **kwargs),
            fc7=Linear(4096, 4096, **kwargs),
            fc8=Linear(4096, 1000, **kwargs),
        )
        if pretrained_model == 'auto':
            _retrieve(
                'VGG_ILSVRC_16_layers.npz',
                'http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/'
                'caffe/VGG_ILSVRC_16_layers.caffemodel',
                self)
        elif pretrained_model:
            npz.load_npz(pretrained_model, self)

        self.functions = collections.OrderedDict([
            ('conv1_1', [self.conv1_1, relu]),
            ('conv1_2', [self.conv1_2, relu]),
            ('pool1', [_max_pooling_2d]),
            ('conv2_1', [self.conv2_1, relu]),
            ('conv2_2', [self.conv2_2, relu]),
            ('pool2', [_max_pooling_2d]),
            ('conv3_1', [self.conv3_1, relu]),
            ('conv3_2', [self.conv3_2, relu]),
            ('conv3_3', [self.conv3_3, relu]),
            ('pool3', [_max_pooling_2d]),
            ('conv4_1', [self.conv4_1, relu]),
            ('conv4_2', [self.conv4_2, relu]),
            ('conv4_3', [self.conv4_3, relu]),
            ('pool4', [_max_pooling_2d]),
            ('conv5_1', [self.conv5_1, relu]),
            ('conv5_2', [self.conv5_2, relu]),
            ('conv5_3', [self.conv5_3, relu]),
            ('pool5', [_max_pooling_2d]),
            ('fc6', [self.fc6, relu, dropout]),
            ('fc7', [self.fc7, relu, dropout]),
            ('fc8', [self.fc8, relu]),
            ('prob', [softmax]),
        ])

    @property
    def available_layers(self):
        return list(self.functions.keys())

    @classmethod
    def convert_caffemodel_to_npz(cls, path_caffemodel, path_npz):
        """Converts a pre-trained caffemodel to a chainer model.

        Args:
            path_caffemodel (str): Path of the pre-trained caffemodel.
            path_npz (str): Path of the converted chainer model.
        """

        # As CaffeFunction uses shortcut symbols,
        # we import CaffeFunction here.
        from chainer.links.caffe.caffe_function import CaffeFunction
        caffemodel = CaffeFunction(path_caffemodel)
        npz.save_npz(path_npz, caffemodel, compression=False)

    def __call__(self, x, layers=['prob'], test=True):
        """Computes all the feature maps specified by ``layers``.

        Args:
            x (~chainer.Variable): Input variable.
            layers (list of str): The list of layer names you want to extract.
            test (bool): If ``True``, dropout runs in test mode.

        Returns:
            Dictionary of ~chainer.Variable: A directory in which
            the key contains the layer name and the value contains
            the corresponding feature map variable.

        """

        h = x
        activations = {}
        target_layers = set(layers)
        for key, funcs in self.functions.items():
            if len(target_layers) == 0:
                break
            for func in funcs:
                if func is dropout:
                    h = func(h, train=not test)
                else:
                    h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)
        return activations

    def extract(self, images, layers=['fc7'], size=(224, 224),
                test=True, volatile=flag.OFF):
        """Extracts all the feature maps of given images.

        The difference of directly executing ``__call__`` is that
        it directly accepts images as an input and automatically
        transforms them to a proper variable. That is,
        it is also interpreted as a shortcut method that implicitly calls
        ``prepare`` and ``__call__`` functions.

        Args:
            images (iterable of PIL.Image or numpy.ndarray): Input images.
            layers (list of str): The list of layer names you want to extract.
            size (pair of ints): The resolution of resized images used as
                an input of CNN. All the given images are not resized
                if this argument is ``None``, but the resolutions of
                all the images should be the same.
            test (bool): If ``True``, dropout runs in test mode.
            volatile (~chainer.Flag): Volatility flag used for input variables.

        Returns:
            Dictionary of ~chainer.Variable: A directory in which
            the key contains the layer name and the value contains
            the corresponding feature map variable.

        """

        x = concat_examples([prepare(img, size=size) for img in images])
        x = Variable(self.xp.asarray(x), volatile=volatile)
        return self(x, layers=layers, test=test)

    def predict(self, images, oversample=True):
        """Computes all the probabilities of given images.

        Args:
            images (iterable of PIL.Image or numpy.ndarray): Input images.
            oversample (bool): If ``True``, it averages results across
                center, corners, and mirrors. Otherwise, it uses only the
                center.

        Returns:
            ~chainer.Variable: Output that contains the class probabilities
            of given images.

        """

        x = concat_examples([prepare(img, size=(256, 256)) for img in images])
        if oversample:
            x = imgproc.oversample(x, crop_dims=(224, 224))
        else:
            x = x[:, :, 16:240, 16:240]
        # Set volatile option to ON to reduce memory consumption
        x = Variable(self.xp.asarray(x), volatile=flag.ON)
        y = self(x, layers=['prob'])['prob']
        if oversample:
            n = y.data.shape[0] // 10
            y_shape = y.data.shape[1:]
            y = reshape(y, (n, 10) + y_shape)
            y = sum(y, axis=1) / 10
        return y


def prepare(image, size=(224, 224)):
    """Converts the given image to the numpy array for VGG models.

    Note that you have to call this method before ``__call__``
    because the pre-trained vgg model requires to resize the given image,
    covert the RGB to the BGR, subtract the mean,
    and permute the dimensions before calling.

    Args:
        image (PIL.Image or numpy.ndarray): Input image.
            If an input is ``numpy.ndarray``, its shape must be
            ``(height, width)``, ``(height, width, channels)``,
            or ``(channels, height, width)``, and
            the order of the channels must be RGB.
        size (pair of ints): Size of converted images.
            If ``None``, the given image is not resized.

    Returns:
        numpy.ndarray: The converted output array.

    """

    if not available:
        raise ImportError('PIL cannot be loaded. Install Pillow!\n'
                          'The actual import error is as follows:\n' +
                          str(_import_error))
    if isinstance(image, numpy.ndarray):
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image[0, :, :]
            elif image.shape[0] == 3:
                image = image.transpose((1, 2, 0))
        image = Image.fromarray(image.astype(numpy.uint8))
    image = image.convert('RGB')
    if size:
        image = image.resize(size)
    image = numpy.asarray(image, dtype=numpy.float32)
    image = image[:, :, ::-1]
    image -= numpy.array(
        [103.939, 116.779, 123.68], dtype=numpy.float32)
    image = image.transpose((2, 0, 1))
    return image


def _max_pooling_2d(x):
    return max_pooling_2d(x, ksize=2)


def _make_npz(path_npz, url, model):
    path_caffemodel = download.cached_download(url)
    print('Now loading caffemodel (usually it may take few minutes)')
    VGG16Layers.convert_caffemodel_to_npz(path_caffemodel, path_npz)
    npz.load_npz(path_npz, model)
    return model


def _retrieve(name, url, model):
    root = download.get_dataset_directory('pfnet/chainer/models/')
    path = os.path.join(root, name)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, url, model),
        lambda path: npz.load_npz(path, model))
