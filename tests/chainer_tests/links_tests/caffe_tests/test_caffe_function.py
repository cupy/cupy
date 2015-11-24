import tempfile
import unittest

import mock
import numpy
import six

import chainer
from chainer import links
from chainer.links import caffe
if six.PY2:
    from chainer.links.caffe import caffe_pb2


def _iter_init(param, data):
    if isinstance(data, list):
        for d in data:
            if hasattr(param, 'append'):
                param.append(d)
            else:
                param.add()
                if isinstance(d, (list, dict)):
                    _iter_init(param[-1], d)
                else:
                    param[-1] = d

    elif isinstance(data, dict):
        for k, d in data.items():
            if isinstance(d, (list, dict)):
                _iter_init(getattr(param, k), d)
            else:
                setattr(param, k, d)

    else:
        setattr(param, data)


def _make_param(data):
    param = caffe_pb2.NetParameter()
    _iter_init(param, data)
    return param


@unittest.skipUnless(six.PY2, 'Only py2 supports caffe_function')
class TestCaffeFunctionBase(unittest.TestCase):

    def setUp(self):
        self.model_file = tempfile.NamedTemporaryFile()
        param = _make_param(self.data)
        self.model_file.write(param.SerializeToString())
        self.model_file.flush()

    def tearDown(self):
        self.model_file.close()

    def init_func(self):
        self.func = caffe.CaffeFunction(self.model_file.name)


class TestCaffeFunctionBaseMock(TestCaffeFunctionBase):

    def setUp(self):
        outs = []
        for shape in self.out_shapes:
            out_data = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
            outs.append(chainer.Variable(out_data))
        self.outputs = tuple(outs)

        ret_value = outs[0] if len(outs) == 1 else tuple(outs)
        m = mock.MagicMock(name=self.func_name, return_value=ret_value)
        self.patch = mock.patch(self.func_name, m)
        self.mock = self.patch.start()

        super(TestCaffeFunctionBaseMock, self).setUp()

    def tearDown(self):
        super(TestCaffeFunctionBaseMock, self).tearDown()
        self.patch.stop()

    def call(self, inputs, outputs):
        invars = []
        for shape in self.in_shapes:
            data = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
            invars.append(chainer.Variable(data))
        self.inputs = invars

        out = self.func(inputs=dict(zip(inputs, invars)),
                        outputs=outputs, train=False)
        self.assertEqual(len(out), len(self.outputs))
        for actual, expect in zip(out, self.outputs):
            self.assertIs(actual, expect)


class TestConcat(TestCaffeFunctionBaseMock):

    func_name = 'chainer.functions.concat'
    in_shapes = [(3, 2, 3), (3, 2, 3)]
    out_shapes = [(3, 2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Concat',
                'bottom': ['x', 'y'],
                'top': ['z'],
                'concat_param': {
                    'axis': 2
                }
            }
        ]
    }

    def test_concat(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x', 'y'], ['z'])
        self.mock.assert_called_once_with(
            (self.inputs[0], self.inputs[1]), axis=2)


class TestConvolution(TestCaffeFunctionBase):

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Convolution',
                'bottom': ['x'],
                'top': ['y'],
                'convolution_param': {
                    'kernel_size': 2,
                    'stride': 3,
                    'pad': 4,
                    'group': 3,
                    'bias_term': True,
                },
                'blobs': [
                    {
                        'num': 6,
                        'channels': 4,
                        'data': list(range(96))
                    },
                    {
                        'data': list(range(6))
                    }
                ]
            }
        ]
    }

    def test_convolution(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        f = self.func.forwards['l1']
        self.assertIsInstance(f, links.Convolution2D)
        for i in range(3):  # 3 == group
            in_slice = slice(i * 4, (i + 1) * 4)  # 4 == channels
            out_slice = slice(i * 2, (i + 1) * 2)  # 2 == num / group
            w = f.W.data[out_slice, in_slice]
            numpy.testing.assert_array_equal(
                w.flatten(), range(i * 32, (i + 1) * 32))

        numpy.testing.assert_array_equal(
            f.b.data, range(6))


class TestData(TestCaffeFunctionBase):

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Data',
            }
        ]
    }

    def test_data(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 0)


class TestDropout(TestCaffeFunctionBaseMock):

    func_name = 'chainer.functions.dropout'
    in_shapes = [(3, 2, 3)]
    out_shapes = [(3, 2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Dropout',
                'bottom': ['x'],
                'top': ['y'],
                'dropout_param': {
                    'dropout_ratio': 0.25
                }
            }
        ]
    }

    def test_dropout(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x'], ['y'])
        self.mock.assert_called_once_with(
            self.inputs[0], ratio=0.25, train=False)


class TestInnerProduct(TestCaffeFunctionBase):

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'InnerProduct',
                'bottom': ['x'],
                'top': ['y'],
                'inner_product_param': {
                    'bias_term': True,
                    'axis': 1
                },
                'blobs': [
                    # weight
                    {
                        'shape': {
                            'dim': [2, 3]
                        },
                        'data': list(range(6)),
                    },
                    # bias
                    {
                        'shape': {
                            'dim': [2]
                        },
                        'data': list(range(2)),
                    }
                ]
            }
        ]
    }

    def test_linear(self):
        self.init_func()
        f = self.func.forwards['l1']
        self.assertIsInstance(f, links.Linear)
        numpy.testing.assert_array_equal(
            f.W.data, numpy.array([[0, 1, 2], [3, 4, 5]], dtype=numpy.float32))
        numpy.testing.assert_array_equal(
            f.b.data, numpy.array([0, 1], dtype=numpy.float32))


class TestLRN(TestCaffeFunctionBaseMock):

    func_name = 'chainer.functions.local_response_normalization'
    in_shapes = [(3, 2, 3)]
    out_shapes = [(3, 2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'LRN',
                'bottom': ['x'],
                'top': ['y'],
                'lrn_param': {
                    'local_size': 4,
                    'alpha': 0.5,
                    'beta': 0.25,
                    'norm_region': 0,  # ACROSS_CHANNELS
                    'k': 0.5
                },
            }
        ]
    }

    def test_lrn(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x'], ['y'])
        self.mock.assert_called_once_with(
            self.inputs[0], n=4, k=0.5, alpha=0.5 / 4, beta=0.25)


class TestMaxPooling(TestCaffeFunctionBaseMock):

    func_name = 'chainer.functions.max_pooling_2d'
    in_shapes = [(3, 2, 3)]
    out_shapes = [(3, 2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Pooling',
                'bottom': ['x'],
                'top': ['y'],
                'pooling_param': {
                    'pool': 0,  # MAX
                    'kernel_h': 2,
                    'kernel_w': 3,
                    'stride_h': 4,
                    'stride_w': 5,
                    'pad_h': 6,
                    'pad_w': 7,
                }
            }
        ]
    }

    def test_max_pooling(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x'], ['y'])
        self.mock.assert_called_once_with(
            self.inputs[0], (2, 3), stride=(4, 5), pad=(6, 7))


class TestAveragePooling(TestCaffeFunctionBaseMock):

    func_name = 'chainer.functions.average_pooling_2d'
    in_shapes = [(3, 2, 3)]
    out_shapes = [(3, 2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Pooling',
                'bottom': ['x'],
                'top': ['y'],
                'pooling_param': {
                    'pool': 1,  # AVE
                    'kernel_size': 2,
                    'stride': 4,
                    'pad': 6,
                }
            }
        ]
    }

    def test_max_pooling(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x'], ['y'])
        self.mock.assert_called_once_with(
            self.inputs[0], 2, stride=4, pad=6)


class TestReLU(TestCaffeFunctionBaseMock):

    func_name = 'chainer.functions.relu'
    in_shapes = [(3, 2, 3)]
    out_shapes = [(3, 2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'ReLU',
                'bottom': ['x'],
                'top': ['y'],
                'relu_param': {
                    'negative_slope': 0
                }
            }
        ]
    }

    def test_lrn(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x'], ['y'])
        self.mock.assert_called_once_with(self.inputs[0])


class TestLeakyReLU(TestCaffeFunctionBaseMock):

    func_name = 'chainer.functions.leaky_relu'
    in_shapes = [(3, 2, 3)]
    out_shapes = [(3, 2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'ReLU',
                'bottom': ['x'],
                'top': ['y'],
                'relu_param': {
                    'negative_slope': 0.5
                }
            }
        ]
    }

    def test_lrn(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x'], ['y'])
        self.mock.assert_called_once_with(self.inputs[0], slope=0.5)


class TestSoftmaxWithLoss(TestCaffeFunctionBaseMock):

    func_name = 'chainer.functions.softmax_cross_entropy'
    in_shapes = [(3, 2, 3)]
    out_shapes = [()]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'SoftmaxWithLoss',
                'bottom': ['x'],
                'top': ['y'],
            }
        ]
    }

    def test_softmax_with_loss(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x'], ['y'])
        self.mock.assert_called_once_with(self.inputs[0])


class TestSplit(TestCaffeFunctionBase):

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Split',
                'bottom': ['x'],
                'top': ['y', 'z'],
            }
        ]
    }

    def test_split(self):
        self.init_func()
        self.assertEqual(self.func.split_map, {'y': 'x', 'z': 'x'})


class TestCaffeFunctionAvailable(unittest.TestCase):

    @unittest.skipUnless(six.PY2, 'CaffeFunction is available on Py2')
    def test_py2_available(self):
        self.assertTrue(links.caffe.caffe_function.available)

    @unittest.skipUnless(six.PY3, 'CaffeFunction is unavailable on Py3')
    def test_py3_unavailable(self):
        self.assertFalse(links.caffe.caffe_function.available)
