import unittest

import numpy

from chainer import cuda
from chainer.links.model.vision import resnet
from chainer.links.model.vision import vgg
from chainer import testing
from chainer.testing import attr
from chainer.variable import Variable


@unittest.skipUnless(resnet.available, 'Pillow is required')
@attr.slow
class TestResNet50Layers(unittest.TestCase):

    def setUp(self):
        self.link = resnet.ResNet50Layers(pretrained_model=None)

    def test_available_layers(self):
        result = self.link.available_layers
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 9)

    def check_call(self):
        xp = self.link.xp

        # Suppress warning that arises from zero division in BatchNormalization
        with numpy.errstate(divide='ignore'):
            x1 = Variable(xp.asarray(numpy.random.uniform(
                -1, 1, (1, 3, 224, 224)).astype(numpy.float32)))
            y1 = cuda.to_cpu(self.link(x1)['prob'].data)
            self.assertEqual(y1.shape, (1, 1000))

            x2 = Variable(xp.asarray(numpy.random.uniform(
                -1, 1, (1, 3, 128, 128)).astype(numpy.float32)))
            y2 = cuda.to_cpu(self.link(x2, layers=['pool5'])['pool5'].data)
            self.assertEqual(y2.shape, (1, 2048))

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()

    def test_prepare(self):
        x1 = numpy.random.uniform(0, 255, (320, 240, 3)).astype(numpy.uint8)
        x2 = numpy.random.uniform(0, 255, (320, 240)).astype(numpy.uint8)
        x3 = numpy.random.uniform(0, 255, (160, 120, 3)).astype(numpy.float32)
        x4 = numpy.random.uniform(0, 255, (1, 160, 120)).astype(numpy.float32)
        x5 = numpy.random.uniform(0, 255, (3, 160, 120)).astype(numpy.uint8)

        y1 = resnet.prepare(x1)
        self.assertEqual(y1.shape, (3, 224, 224))
        self.assertEqual(y1.dtype, numpy.float32)
        y2 = resnet.prepare(x2)
        self.assertEqual(y2.shape, (3, 224, 224))
        self.assertEqual(y2.dtype, numpy.float32)
        y3 = resnet.prepare(x3, size=None)
        self.assertEqual(y3.shape, (3, 160, 120))
        self.assertEqual(y3.dtype, numpy.float32)
        y4 = resnet.prepare(x4)
        self.assertEqual(y4.shape, (3, 224, 224))
        self.assertEqual(y4.dtype, numpy.float32)
        y5 = resnet.prepare(x5, size=None)
        self.assertEqual(y5.shape, (3, 160, 120))
        self.assertEqual(y5.dtype, numpy.float32)

    def check_extract(self):
        x1 = numpy.random.uniform(0, 255, (320, 240, 3)).astype(numpy.uint8)
        x2 = numpy.random.uniform(0, 255, (320, 240)).astype(numpy.uint8)

        with numpy.errstate(divide='ignore'):
            result = self.link.extract([x1, x2], layers=['res3', 'pool5'])
            self.assertEqual(len(result), 2)
            y1 = cuda.to_cpu(result['res3'].data)
            self.assertEqual(y1.shape, (2, 512, 28, 28))
            self.assertEqual(y1.dtype, numpy.float32)
            y2 = cuda.to_cpu(result['pool5'].data)
            self.assertEqual(y2.shape, (2, 2048))
            self.assertEqual(y2.dtype, numpy.float32)

            x3 = numpy.random.uniform(0, 255, (80, 60)).astype(numpy.uint8)
            result = self.link.extract([x3], layers=['res2'], size=None)
            self.assertEqual(len(result), 1)
            y3 = cuda.to_cpu(result['res2'].data)
            self.assertEqual(y3.shape, (1, 256, 20, 15))
            self.assertEqual(y3.dtype, numpy.float32)

    def test_extract_cpu(self):
        self.check_extract()

    @attr.gpu
    def test_extract_gpu(self):
        self.link.to_gpu()
        self.check_extract()

    def check_predict(self):
        x1 = numpy.random.uniform(0, 255, (320, 240, 3)).astype(numpy.uint8)
        x2 = numpy.random.uniform(0, 255, (320, 240)).astype(numpy.uint8)

        with numpy.errstate(divide='ignore'):
            result = self.link.predict([x1, x2], oversample=False)
            y = cuda.to_cpu(result.data)
            self.assertEqual(y.shape, (2, 1000))
            self.assertEqual(y.dtype, numpy.float32)
            result = self.link.predict([x1, x2], oversample=True)
            y = cuda.to_cpu(result.data)
            self.assertEqual(y.shape, (2, 1000))
            self.assertEqual(y.dtype, numpy.float32)

    def test_predict_cpu(self):
        self.check_predict()

    @attr.gpu
    def test_predict_gpu(self):
        self.link.to_gpu()
        self.check_predict()


@unittest.skipUnless(resnet.available, 'Pillow is required')
@attr.slow
class TestVGG16Layers(unittest.TestCase):

    def setUp(self):
        self.link = vgg.VGG16Layers(pretrained_model=None)

    def test_available_layers(self):
        result = self.link.available_layers
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 22)

    def check_call(self):
        xp = self.link.xp

        x1 = Variable(xp.asarray(numpy.random.uniform(
            -1, 1, (1, 3, 224, 224)).astype(numpy.float32)))
        y1 = cuda.to_cpu(self.link(x1)['prob'].data)
        self.assertEqual(y1.shape, (1, 1000))

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()

    def test_prepare(self):
        x1 = numpy.random.uniform(0, 255, (320, 240, 3)).astype(numpy.uint8)
        x2 = numpy.random.uniform(0, 255, (320, 240)).astype(numpy.uint8)
        x3 = numpy.random.uniform(0, 255, (160, 120, 3)).astype(numpy.float32)
        x4 = numpy.random.uniform(0, 255, (1, 160, 120)).astype(numpy.float32)
        x5 = numpy.random.uniform(0, 255, (3, 160, 120)).astype(numpy.uint8)

        y1 = vgg.prepare(x1)
        self.assertEqual(y1.shape, (3, 224, 224))
        self.assertEqual(y1.dtype, numpy.float32)
        y2 = vgg.prepare(x2)
        self.assertEqual(y2.shape, (3, 224, 224))
        self.assertEqual(y2.dtype, numpy.float32)
        y3 = vgg.prepare(x3, size=None)
        self.assertEqual(y3.shape, (3, 160, 120))
        self.assertEqual(y3.dtype, numpy.float32)
        y4 = vgg.prepare(x4)
        self.assertEqual(y4.shape, (3, 224, 224))
        self.assertEqual(y4.dtype, numpy.float32)
        y5 = vgg.prepare(x5, size=None)
        self.assertEqual(y5.shape, (3, 160, 120))
        self.assertEqual(y5.dtype, numpy.float32)

    def check_extract(self):
        x1 = numpy.random.uniform(0, 255, (320, 240, 3)).astype(numpy.uint8)
        x2 = numpy.random.uniform(0, 255, (320, 240)).astype(numpy.uint8)
        result = self.link.extract([x1, x2], layers=['pool3', 'fc7'])
        self.assertEqual(len(result), 2)
        y1 = cuda.to_cpu(result['pool3'].data)
        self.assertEqual(y1.shape, (2, 256, 28, 28))
        self.assertEqual(y1.dtype, numpy.float32)
        y2 = cuda.to_cpu(result['fc7'].data)
        self.assertEqual(y2.shape, (2, 4096))
        self.assertEqual(y2.dtype, numpy.float32)

        x3 = numpy.random.uniform(0, 255, (80, 60)).astype(numpy.uint8)
        result = self.link.extract([x3], layers=['pool1'], size=None)
        self.assertEqual(len(result), 1)
        y3 = cuda.to_cpu(result['pool1'].data)
        self.assertEqual(y3.shape, (1, 64, 40, 30))
        self.assertEqual(y3.dtype, numpy.float32)

    def test_extract_cpu(self):
        self.check_extract()

    @attr.gpu
    def test_extract_gpu(self):
        self.link.to_gpu()
        self.check_extract()

    def check_predict(self):
        x1 = numpy.random.uniform(0, 255, (320, 240, 3)).astype(numpy.uint8)
        x2 = numpy.random.uniform(0, 255, (320, 240)).astype(numpy.uint8)
        result = self.link.predict([x1, x2], oversample=False)
        y = cuda.to_cpu(result.data)
        self.assertEqual(y.shape, (2, 1000))
        self.assertEqual(y.dtype, numpy.float32)
        result = self.link.predict([x1, x2], oversample=True)
        y = cuda.to_cpu(result.data)
        self.assertEqual(y.shape, (2, 1000))
        self.assertEqual(y.dtype, numpy.float32)

    def test_predict_cpu(self):
        self.check_predict()

    @attr.gpu
    def test_predict_gpu(self):
        self.link.to_gpu()
        self.check_predict()

testing.run_module(__name__, __file__)
