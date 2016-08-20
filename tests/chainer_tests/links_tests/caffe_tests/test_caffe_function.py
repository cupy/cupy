import os
import pkg_resources
import tempfile
import unittest

import mock
import numpy
import six

import chainer
from chainer import links
from chainer.links import caffe
from chainer import testing
if links.caffe.caffe_function.available:
    from chainer.links.caffe.caffe_function import caffe_pb


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
    param = caffe_pb.NetParameter()
    _iter_init(param, data)
    return param


@unittest.skipUnless(links.caffe.caffe_function.available,
                     'protobuf>=3.0.0 is required for py3')
class TestCaffeFunctionBase(unittest.TestCase):

    def setUp(self):
        param = _make_param(self.data)
        # The name can be used to open the file a second time,
        # while the named temporary file is still open on the Windows.
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.temp_file_path = f.name
            f.write(param.SerializeToString())

    def tearDown(self):
        os.remove(self.temp_file_path)

    def init_func(self):
        self.func = caffe.CaffeFunction(self.temp_file_path)


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


class TestConvolution(TestCaffeFunctionBaseMock):

    func_name = 'chainer.links.Convolution2D.__call__'
    in_shapes = [(2, 3)]
    out_shapes = [(2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Convolution',
                'bottom': ['x'],
                'top': ['y'],
                'convolution_param': {
                    'kernel_size': [2],
                    'stride': [3],
                    'pad': [4],
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
        f = self.func.l1
        self.assertIsInstance(f, links.Convolution2D)
        for i in range(3):  # 3 == group
            in_slice = slice(i * 4, (i + 1) * 4)  # 4 == channels
            out_slice = slice(i * 2, (i + 1) * 2)  # 2 == num / group
            w = f.W.data[out_slice, in_slice]
            numpy.testing.assert_array_equal(
                w.flatten(), range(i * 32, (i + 1) * 32))

        numpy.testing.assert_array_equal(
            f.b.data, range(6))

        self.call(['x'], ['y'])
        self.mock.assert_called_once_with(self.inputs[0])


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


class TestInnerProduct(TestCaffeFunctionBaseMock):

    func_name = 'chainer.links.Linear.__call__'
    in_shapes = [(2, 3)]
    out_shapes = [(2, 3)]

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
        f = self.func.l1
        self.assertIsInstance(f, links.Linear)
        numpy.testing.assert_array_equal(
            f.W.data, numpy.array([[0, 1, 2], [3, 4, 5]], dtype=numpy.float32))
        numpy.testing.assert_array_equal(
            f.b.data, numpy.array([0, 1], dtype=numpy.float32))

        self.call(['x'], ['y'])
        self.mock.assert_called_once_with(self.inputs[0])


class TestInnerProductDim4(TestCaffeFunctionBaseMock):

    func_name = 'chainer.links.Linear.__call__'
    in_shapes = [(2, 3)]
    out_shapes = [(2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'InnerProduct',
                'bottom': ['x'],
                'top': ['y'],
                'inner_product_param': {
                    'bias_term': False,
                    'axis': 1
                },
                'blobs': [
                    # weight
                    {
                        'shape': {
                            'dim': [4, 5, 2, 3]
                        },
                        # when `ndim` == 4, `data` stored shape[2] x shape[3]
                        # data
                        'data': list(range(6)),
                    }
                ]
            }
        ]
    }

    def test_linear(self):
        self.init_func()
        f = self.func.l1
        self.assertIsInstance(f, links.Linear)
        numpy.testing.assert_array_equal(
            f.W.data, numpy.array([[0, 1, 2], [3, 4, 5]], dtype=numpy.float32))
        self.assertIsNone(f.b)

        self.call(['x'], ['y'])
        self.mock.assert_called_once_with(self.inputs[0])


class TestInnerProductInvalidDim(TestCaffeFunctionBase):

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'InnerProduct',
                'blobs': [
                    {
                        'shape': {
                            'dim': [2, 3, 4, 5, 6]  # 5-dim is not supported
                        },
                    },
                ]
            }
        ]
    }

    def test_linear(self):
        with self.assertRaises(RuntimeError):
            self.init_func()


class TestInnerProductNonDefaultAxis(TestCaffeFunctionBase):

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'InnerProduct',
                'inner_product_param': {
                    'axis': 0  # non-default axis
                }
            }
        ]
    }

    def test_linear(self):
        with self.assertRaises(RuntimeError):
            self.init_func()


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


class TestLRNWithinChannel(TestCaffeFunctionBase):

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'LRN',
                'lrn_param': {
                    'norm_region': 1,  # WITHIN_CHANNELS is not supported
                },
            }
        ]
    }

    def test_lrn(self):
        with self.assertRaises(RuntimeError):
            self.init_func()


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


class TestStochasticPooling(TestCaffeFunctionBase):

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Pooling',
                'pooling_param': {
                    'pool': 2,  # STOCHASTIC is not supported
                }
            }
        ]
    }

    def test_stochastic_pooling(self):
        with self.assertRaises(RuntimeError):
            self.init_func()


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


class TestBatchNorm(TestCaffeFunctionBaseMock):

    func_name = 'chainer.links.BatchNormalization.__call__'
    in_shapes = [(2, 3)]
    out_shapes = [(2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'BatchNorm',
                'bottom': ['x'],
                'top': ['y'],
                'blobs': [
                    # For average mean.
                    {
                        'shape': {
                            'dim': [3],
                        },
                        'data': list(six.moves.range(3)),
                    },
                    # For average variance.
                    {
                        'shape': {
                            'dim': [3],
                        },
                        'data': list(six.moves.range(3)),
                    },
                ],
                'batch_norm_param': {
                    'use_global_stats': False,
                }
            }
        ]
    }

    def test_batchnorm(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x'], ['y'])
        self.mock.assert_called_once_with(self.inputs[0],
                                          test=False, finetune=False)


class TestBatchNormUsingGlobalStats(TestCaffeFunctionBaseMock):

    func_name = 'chainer.links.BatchNormalization.__call__'
    in_shapes = [(2, 3)]
    out_shapes = [(2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'BatchNorm',
                'bottom': ['x'],
                'top': ['y'],
                'blobs': [
                    # For average mean.
                    {
                        'shape': {
                            'dim': [3],
                        },
                        'data': list(six.moves.range(3)),
                    },
                    # For average variance.
                    {
                        'shape': {
                            'dim': [3],
                        },
                        'data': list(six.moves.range(3)),
                    },
                ],
                'batch_norm_param': {
                    'use_global_stats': True,
                }
            }
        ]
    }

    def test_batchnorm(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x'], ['y'])
        self.mock.assert_called_once_with(self.inputs[0],
                                          test=True, finetune=False)


class TestEltwiseProd(TestCaffeFunctionBaseMock):

    func_name = 'chainer.variable.Variable.__mul__'
    in_shapes = [(2, 3), (2, 3), (2, 3)]
    out_shapes = [(2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Eltwise',
                'bottom': ['x1', 'x2', 'x3'],
                'top': ['y'],
                'eltwise_param': {
                    'operation': 0,  # PROD
                },
            }
        ]
    }

    def test_eltwise_prod(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x1', 'x2', 'x3'], ['y'])
        self.mock.assert_has_calls([mock.call(self.inputs[1]),
                                    mock.call(self.inputs[2])])


class TestEltwiseSum(TestCaffeFunctionBaseMock):

    func_name = 'chainer.variable.Variable.__add__'
    in_shapes = [(2, 3), (2, 3), (2, 3)]
    out_shapes = [(2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Eltwise',
                'bottom': ['x1', 'x2', 'x3'],
                'top': ['y'],
                'eltwise_param': {
                    'operation': 1,  # SUM
                },
            }
        ]
    }

    def test_eltwise_sum(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x1', 'x2', 'x3'], ['y'])
        self.mock.assert_has_calls([mock.call(self.inputs[1]),
                                    mock.call(self.inputs[2])])


class TestEltwiseSumCoeff(TestCaffeFunctionBaseMock):

    func_name = 'chainer.variable.Variable.__add__'
    in_shapes = [(2, 3), (2, 3), (2, 3)]
    out_shapes = [(2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Eltwise',
                'bottom': ['x1', 'x2', 'x3'],
                'top': ['y'],
                'eltwise_param': {
                    'operation': 1,  # SUM
                    'coeff': list(six.moves.range(3)),
                },
            }
        ]
    }

    def test_eltwise_sum(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x1', 'x2', 'x3'], ['y'])
        self.assertEqual(self.mock.call_count, 2)


class TestEltwiseSumInvalidCoeff(TestCaffeFunctionBaseMock):

    func_name = 'chainer.variable.Variable.__add__'
    in_shapes = [(2, 3), (2, 3), (2, 3)]
    out_shapes = [(2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Eltwise',
                'bottom': ['x1', 'x2', 'x3'],
                'top': ['y'],
                'eltwise_param': {
                    'operation': 1,           # SUM
                    # not same as number of bottoms
                    'coeff': list(six.moves.range(2)),
                },
            }
        ]
    }

    def test_eltwise_sum(self):
        self.init_func()
        with self.assertRaises(AssertionError):
            self.call(['x1', 'x2', 'x3'], ['y'])


class TestEltwiseMax(TestCaffeFunctionBaseMock):

    func_name = 'chainer.functions.maximum'
    in_shapes = [(2, 3), (2, 3), (2, 3)]
    out_shapes = [(2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Eltwise',
                'bottom': ['x1', 'x2', 'x3'],
                'top': ['y'],
                'eltwise_param': {
                    'operation': 2,  # MAX
                },
            }
        ]
    }

    def test_eltwise_max(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x1', 'x2', 'x3'], ['y'])
        self.mock.assert_has_calls(
            [mock.call(self.inputs[0], self.inputs[1]),
             mock.call(self.outputs[0], self.inputs[2])])


class TestScale(TestCaffeFunctionBaseMock):

    func_name = 'chainer.links.Scale.__call__'
    in_shapes = [(2, 3), (2, 3)]
    out_shapes = [(2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Scale',
                'bottom': ['x', 'y'],
                'top': ['z'],
                'scale_param': {
                    'axis': 0,
                }
            }
        ]
    }

    def test_scale(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x', 'y'], ['z'])
        self.mock.assert_called_once_with(self.inputs[0], self.inputs[1])


class TestScaleOneBottom(TestCaffeFunctionBaseMock):

    func_name = 'chainer.links.Scale.__call__'
    in_shapes = [(2, 3)]
    out_shapes = [(2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Scale',
                'bottom': ['x'],
                'top': ['y'],
                'blobs': [
                    {
                        'shape': {
                            'dim': [2, 3],
                        },
                        'data': list(six.moves.range(6)),
                    }
                ],
                'scale_param': {
                    'axis': 0,
                }
            }
        ]
    }

    def test_scale_one_bottom(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x'], ['y'])
        self.mock.assert_called_once_with(self.inputs[0])


class TestScaleWithBias(TestCaffeFunctionBaseMock):

    func_name = 'chainer.links.Scale.__call__'
    in_shapes = [(2, 3), (2, 3)]
    out_shapes = [(2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Scale',
                'bottom': ['x', 'y'],
                'top': ['z'],
                'blobs': [
                    {
                        'shape': {
                            'dim': [2, 3],
                        },
                        'data': list(six.moves.range(6)),
                    }
                ],
                'scale_param': {
                    'axis': 0,
                    'bias_term': True,
                }
            }
        ]
    }

    def test_scale_with_bias(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.assertTrue(hasattr(self.func.l1, 'bias'))
        self.call(['x', 'y'], ['z'])
        self.mock.assert_called_once_with(self.inputs[0], self.inputs[1])


class TestScaleOneBottomWithBias(TestCaffeFunctionBaseMock):

    func_name = 'chainer.links.Scale.__call__'
    in_shapes = [(2, 3)]
    out_shapes = [(2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Scale',
                'bottom': ['x'],
                'top': ['y'],
                'blobs': [
                    # For W parameter.
                    {
                        'shape': {
                            'dim': [2, 3],
                        },
                        'data': list(six.moves.range(6)),
                    },
                    # For bias.
                    {
                        'shape': {
                            'dim': [2, 3],
                        },
                        'data': list(six.moves.range(6)),
                    }
                ],
                'scale_param': {
                    'axis': 0,
                    'bias_term': True,
                }
            }
        ]
    }

    def test_scale_one_bottom_with_bias(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.assertTrue(hasattr(self.func.l1, 'bias'))
        self.call(['x'], ['y'])
        self.mock.assert_called_once_with(self.inputs[0])


class TestSlice(TestCaffeFunctionBaseMock):

    func_name = 'chainer.functions.split_axis'
    in_shapes = [(3, 4, 3)]
    out_shapes = [(3, 2, 3), (3, 2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Slice',
                'bottom': ['x'],
                'top': ['y1', 'y2'],
                'slice_param': {
                    'axis': 1
                }
            }
        ]
    }

    def test_slice(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x'], ['y1', 'y2'])
        self.mock.assert_called_once_with(
            self.inputs[0],
            indices_or_sections=2,
            axis=1
        )


class TestSliceNoAxis(TestCaffeFunctionBaseMock):

    func_name = 'chainer.functions.split_axis'
    in_shapes = [(4, 6, 4)]
    out_shapes = [(2, 6, 4), (2, 6, 4)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Slice',
                'bottom': ['x'],
                'top': ['y1', 'y2'],
                'slice_param': {
                    'slice_dim': 0
                }
            }
        ]
    }

    def test_slice_no_axis(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x'], ['y1', 'y2'])
        self.mock.assert_called_once_with(
            self.inputs[0],
            indices_or_sections=2,
            axis=0
        )


class TestSliceNoAxisNoSliceDim(TestCaffeFunctionBaseMock):

    func_name = 'chainer.functions.split_axis'
    in_shapes = [(4, 6, 4)]
    out_shapes = [(4, 3, 4), (4, 3, 4)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Slice',
                'bottom': ['x'],
                'top': ['y1', 'y2'],
            }
        ]
    }

    def test_slice_no_axis_no_slice_dim(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x'], ['y1', 'y2'])
        self.mock.assert_called_once_with(
            self.inputs[0],
            indices_or_sections=2,
            axis=1
        )


class TestSliceSlicePoint(TestCaffeFunctionBaseMock):

    func_name = 'chainer.functions.split_axis'
    in_shapes = [(4, 8, 6)]
    out_shapes = [(4, 3, 6), (4, 2, 6), (4, 3, 6)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Slice',
                'bottom': ['x'],
                'top': ['y1', 'y2', 'y3'],
                'slice_param': {
                    'axis': 1,
                    'slice_point': [3, 5]
                }
            }
        ]
    }

    def test_slice_slice_point(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x'], ['y1', 'y2', 'y3'])
        self.mock.assert_called_once_with(
            self.inputs[0],
            indices_or_sections=[3, 5],
            axis=1
        )


class TestSoftmax(TestCaffeFunctionBaseMock):

    func_name = 'chainer.functions.softmax'
    in_shapes = [(2, 3)]
    out_shapes = [(2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Softmax',
                'bottom': ['x'],
                'top': ['y'],
            }
        ]
    }

    def test_softmax(self):
        self.init_func()
        self.assertEqual(len(self.func.layers), 1)
        self.call(['x'], ['y'])
        self.mock.assert_called_once_with(self.inputs[0])


class TestSoftmaxCaffeEngine(TestCaffeFunctionBaseMock):

    func_name = 'chainer.functions.softmax'
    in_shapes = [(2, 3)]
    out_shapes = [(2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Softmax',
                'softmax_param': {
                    'engine': 1,  # CAFFE
                },
                'bottom': ['x'],
                'top': ['y'],
            }
        ]
    }

    def test_softmax_caffe_engine(self):
        self.init_func()
        self.call(['x'], ['y'])
        self.mock.assert_called_once_with(self.inputs[0], use_cudnn=False)


class TestSoftmaxcuDnnEngine(TestCaffeFunctionBaseMock):

    func_name = 'chainer.functions.softmax'
    in_shapes = [(2, 3)]
    out_shapes = [(2, 3)]

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Softmax',
                'softmax_param': {
                    'engine': 2,  # CUDNN
                },
                'bottom': ['x'],
                'top': ['y'],
            }
        ]
    }

    def test_softmax_cuDNN_engine(self):
        self.init_func()
        self.call(['x'], ['y'])
        self.mock.assert_called_once_with(self.inputs[0], use_cudnn=True)


class TestSoftmaxInvalidAxis(TestCaffeFunctionBase):

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'Softmax',
                'softmax_param': {
                    'axis': 0,  # invalid axis
                }
            }
        ]
    }

    def test_softmax_invalid_axis(self):
        with self.assertRaises(RuntimeError):
            self.init_func()


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


class TestSoftmaxWithLossInvalidAxis(TestCaffeFunctionBase):

    data = {
        'layer': [
            {
                'name': 'l1',
                'type': 'SoftmaxWithLoss',
                'softmax_param': {
                    'axis': 0,  # invalid axis
                }
            }
        ]
    }

    def test_softmax_with_loss_invalid_axis(self):
        with self.assertRaises(RuntimeError):
            self.init_func()


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

    @unittest.skipUnless(six.PY2, 'Only for Py2')
    def test_py2_available(self):
        self.assertTrue(links.caffe.caffe_function.available)

    @unittest.skipUnless(six.PY3, 'Only for Py3')
    def test_py3_available(self):
        ws = pkg_resources.WorkingSet()
        try:
            ws.require('protobuf<3.0.0')
            ver = 2
        except pkg_resources.VersionConflict:
            ver = 3

        if ver >= 3:
            self.assertTrue(links.caffe.caffe_function.available)
        else:
            self.assertFalse(links.caffe.caffe_function.available)

            with self.assertRaises(RuntimeError):
                caffe.CaffeFunction('')


testing.run_module(__name__, __file__)
