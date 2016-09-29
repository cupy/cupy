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


class ResNet50Layers(chainer.Chain):

    """A pre-trained CNN model with 50 layers provided by MSRA [1].

    When you specify the path of the pre-trained chainer model serialized as
    a ``.npz`` file in the constructor, this chain model automatically
    initializes all the parameters with it.
    This model would be useful when you want to extract a semantic feature
    vector from given images, or fine-tune the model on a different dataset.
    Note that unlike ``VGG16Layers``, it does not automatically download a
    pre-trained caffemodel. This caffemodel can be downloaded at
    ``GitHub <https://github.com/KaimingHe/deep-residual-networks>``.

    If you want to manually convert the pre-trained caffemodel to a chainer
    model that can be spesified in the constractor,
    please use ``convert_caffemodel_to_npz`` classmethod instead.

    [1] ``Deep Residual Learning for Image Recognition
    <http://arxiv.org/abs/1512.03385>``

    Args:
        pretrained_model (str): the destination of the pre-trained
            chainer model serialized as a ``.npz`` file.
            If this argument is specified as ``auto``,
            it automatically loads and converts the caffemodel from
            ``$CHAINER_DATASET_ROOT/pfnet/chainer/models/ResNet-50-model.caffemodel``,
            where ``$CHAINER_DATASET_ROOT`` is set as
            ``$HOME/.chainer/dataset`` unless you specify another value
            as an environment variable. Note that in this case the converted
            chainer model is stored on the same directory and automatically
            used from the second time.
            If the argument is specfied as ``None``, all the parameters
            are not initialized by the pre-trained model, but the default
            initializer used in the original paper.

    Attributes:
        available_layers (list of str): The list of available layer names
            used by ``__call__`` and ``extract`` methods.

    """

    def __init__(self, pretrained_model='auto'):
        if pretrained_model is not None:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            kwargs = {'initialW': chainer.initializers.Zero()}
        else:
            # employ default initializers used in the original paper
            kwargs = {'initialW': None}
        super(ResNet50Layers, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, 2, 3, **kwargs),
            bn1=L.BatchNormalization(64),
            res2=BuildingBlock(3, 64, 64, 256, 1, **kwargs),
            res3=BuildingBlock(4, 256, 128, 512, 2, **kwargs),
            res4=BuildingBlock(6, 512, 256, 1024, 2, **kwargs),
            res5=BuildingBlock(3, 1024, 512, 2048, 2, **kwargs),
            fc=L.Linear(2048, 1000),
        )
        if pretrained_model == 'auto':
            _retrieve(
                'ResNet-50-model.npz', 'ResNet-50-model.caffemodel', self)
        elif pretrained_model is not None:
            serializers.load_npz(pretrained_model, self)
        max_pooling_2d = lambda x: F.max_pooling_2d(x, ksize=3, stride=2)
        self.functions = OrderedDict([
            ('conv1', [self.conv1, self.bn1, F.relu]),
            ('pool1', [max_pooling_2d]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('res4', [self.res4]),
            ('res5', [self.res5]),
            ('pool6', [_global_average_pooling_2d]),
            ('fc', [self.fc]),
            ('prob', [F.softmax]),
        ])

    @property
    def available_layers(self):
        return list(self.functions.iterkeys())

    @classmethod
    def convert_caffemodel_to_npz(cls, path_caffemodel, path_npz):
        """Converts a pre-trained caffemodel to a chainer model.

        Args:
            path_caffemodel (str): Path of the pre-trained caffemodel.
            path_npz (str): Path of the converted chainer model.
        """

        caffemodel = CaffeFunction(path_caffemodel)
        chainermodel = cls(pretrained_model=None)
        _transfer_resnet50(caffemodel, chainermodel)
        serializers.save_npz(path_npz, chainermodel, compression=False)

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
            if len(target_layers) == 0:
                break
            for func in funcs:
                h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)
        return activations

    def prepare(self, image, size=(224, 224)):
        """Converts the given image to the numpy array.

        Note that you have to call this method before ``__call__``
        because the pre-trained vgg model requires to resize the given image,
        covert the RGB to the BGR, subtract the mean,
        and permute the dimensions before calling.

        Args:
            image (PIL.Image or numpy.ndarray): Input image.
                If an input is ``numpy.ndarray``, its shape must be
                ``(height, width)`` or ``(height, width, channels)``.
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
        # NOTE: in the original paper they subtract a fixed mean image,
        #       however, in order to support arbitrary size we instead use the
        #       mean pixel (rather than mean image) as with VGG team. The mean
        #       value used in ResNet is slightly different from that of VGG16.
        image -= numpy.array(
            [103.063,  115.903,  123.152], dtype=numpy.float32)
        image = image.transpose((2, 0, 1))
        return image

    def extract(self, images, layers=['pool6'], size=(224, 224)):
        """Extracts all the feature maps of given images.

        The difference of directly executing ``__call__`` is that
        it directly accepts the list of images as an input, and
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


class BuildingBlock(chainer.Chain):

    """A building block that consists of several Bottleneck layers.

    Args:
        n_layer (int): Number of layers used in the building block.
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, n_layer, in_channels, mid_channels,
                 out_channels, stride, initialW=None):
        links = [
            ('a', BottleneckA(
                in_channels, mid_channels, out_channels, stride, initialW))
        ]
        for i in range(n_layer - 1):
            name = 'b{}'.format(i + 1)
            link = BottleneckB(out_channels, mid_channels, initialW)
            links.append((name, link))
        super(BuildingBlock, self).__init__(**dict(links))
        self.forward = links

    def __call__(self, x, test=True):
        for name, func in self.forward:
            x = func(x, test=test)
        return x


class BottleneckA(chainer.Chain):

    """A bottleneck layer that reduces the resolution of the feature map.

    Args:
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=2, initialW=None):
        super(BottleneckA, self).__init__(
            conv1=L.Convolution2D(
                in_channels, mid_channels, 1, stride, 0,
                initialW=initialW, nobias=True),
            bn1=L.BatchNormalization(mid_channels),
            conv2=L.Convolution2D(
                mid_channels, mid_channels, 3, 1, 1,
                initialW=initialW, nobias=True),
            bn2=L.BatchNormalization(mid_channels),
            conv3=L.Convolution2D(
                mid_channels, out_channels, 1, 1, 0,
                initialW=initialW, nobias=True),
            bn3=L.BatchNormalization(out_channels),
            conv4=L.Convolution2D(
                in_channels, out_channels, 1, stride, 0,
                initialW=initialW, nobias=True),
            bn4=L.BatchNormalization(out_channels),
        )

    def __call__(self, x, test=True):
        h1 = F.relu(self.bn1(self.conv1(x), test=test))
        h1 = F.relu(self.bn2(self.conv2(h1), test=test))
        h1 = self.bn3(self.conv3(h1), test=test)
        h2 = self.bn4(self.conv4(x), test=test)
        return F.relu(h1 + h2)


class BottleneckB(chainer.Chain):

    """A bottleneck layer that maintains the resolution of the feature map.

    Args:
        in_channels (int): Number of channels of input and output arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, in_channels, mid_channels, initialW=None):
        super(BottleneckB, self).__init__(
            conv1=L.Convolution2D(
                in_channels, mid_channels, 1, 1, 0,
                initialW=initialW, nobias=True),
            bn1=L.BatchNormalization(mid_channels),
            conv2=L.Convolution2D(
                mid_channels, mid_channels, 3, 1, 1,
                initialW=initialW, nobias=True),
            bn2=L.BatchNormalization(mid_channels),
            conv3=L.Convolution2D(
                mid_channels, in_channels, 1, 1, 0,
                initialW=initialW, nobias=True),
            bn3=L.BatchNormalization(in_channels),
        )

    def __call__(self, x, test=True):
        h = F.relu(self.bn1(self.conv1(x), test=test))
        h = F.relu(self.bn2(self.conv2(h), test=test))
        h = self.bn3(self.conv3(h), test=test)
        return F.relu(h + x)


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.data.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = F.reshape(h, (n, channel))
    return h


def _transfer_components(src, dst_conv, dst_bn, bname, cname):
    src_conv = getattr(src, 'res{}_branch{}'.format(bname, cname))
    src_bn = getattr(src, 'bn{}_branch{}'.format(bname, cname))
    src_scale = getattr(src, 'scale{}_branch{}'.format(bname, cname))
    dst_conv.W.data[:] = src_conv.W.data
    dst_bn.avg_mean[:] = src_bn.avg_mean
    dst_bn.avg_var[:] = src_bn.avg_var
    dst_bn.gamma.data[:] = src_scale.W.data
    dst_bn.beta.data[:] = src_scale.bias.b.data


def _transfer_bottleneckA(src, dst, name):
    _transfer_components(src, dst.conv1, dst.bn1, name, '2a')
    _transfer_components(src, dst.conv2, dst.bn2, name, '2b')
    _transfer_components(src, dst.conv3, dst.bn3, name, '2c')
    _transfer_components(src, dst.conv4, dst.bn4, name, '1')


def _transfer_bottleneckB(src, dst, name):
    _transfer_components(src, dst.conv1, dst.bn1, name, '2a')
    _transfer_components(src, dst.conv2, dst.bn2, name, '2b')
    _transfer_components(src, dst.conv3, dst.bn3, name, '2c')


def _transfer_block(src, dst, names):
    _transfer_bottleneckA(src, dst.a, names[0])
    for i, name in enumerate(names[1:]):
        dst_bottleneckB = getattr(dst, 'b{}'.format(i + 1))
        _transfer_bottleneckB(src, dst_bottleneckB, name)


def _transfer_resnet50(src, dst):
    dst.conv1.W.data[:] = src.conv1.W.data
    dst.conv1.b.data[:] = src.conv1.b.data
    dst.bn1.avg_mean[:] = src.bn_conv1.avg_mean
    dst.bn1.avg_var[:] = src.bn_conv1.avg_var
    dst.bn1.gamma.data[:] = src.scale_conv1.W.data
    dst.bn1.beta.data[:] = src.scale_conv1.bias.b.data

    _transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    _transfer_block(src, dst.res3, ['3a', '3b', '3c', '3d'])
    _transfer_block(src, dst.res4, ['4a', '4b', '4c', '4d', '4e', '4f'])
    _transfer_block(src, dst.res5, ['5a', '5b', '5c'])

    dst.fc.W.data[:] = src.fc1000.W.data
    dst.fc.b.data[:] = src.fc1000.b.data


def _make_npz(path_npz, path_caffemodel, model):
    print('Now loading caffemodel (usually it may take few minutes)')
    if not os.path.exists(path_caffemodel):
        raise IOError(
            'The pre-trained caffemodel does not exist. Please download it '
            'from \'https://github.com/KaimingHe/deep-residual-networks\', '
            'and place it on {}'.format(path_caffemodel))
    ResNet50Layers.convert_caffemodel_to_npz(path_caffemodel, path_npz)
    serializers.load_npz(path_npz, model)
    return model


def _retrieve(name_npz, name_caffemodel, model):
    root = chainer.dataset.get_dataset_directory('pfnet/chainer/models/')
    path = os.path.join(root, name_npz)
    path_caffemodel = os.path.join(root, name_caffemodel)
    return chainer.dataset.cache_or_load_file(
        path, lambda path: _make_npz(path, path_caffemodel, model),
        lambda path: serializers.load_npz(path, model))
