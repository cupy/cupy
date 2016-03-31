import collections
import pkg_resources
import sys
import warnings

import numpy
import six

from chainer import functions
from chainer import link
from chainer import links


def _protobuf3():
    ws = pkg_resources.WorkingSet()
    try:
        ws.require('protobuf>=3.0.0a')
        return True
    except pkg_resources.VersionConflict:
        return False


if _protobuf3():
    from chainer.links.caffe import caffe_pb3 as caffe_pb
    available = True
elif sys.version_info < (3, 0, 0):
    # caffe_pb2 does not support Py3
    from chainer.links.caffe import caffe_pb2 as caffe_pb
    available = True
else:
    available = False

if available:
    _type_to_method = {}
    _oldname_to_method = {}

    def _layer(typ, oldname):
        def decorator(meth):
            global _type_to_method
            _type_to_method[typ] = meth
            typevalue = getattr(caffe_pb.V1LayerParameter, oldname)
            _oldname_to_method[typevalue] = meth
            return meth
        return decorator
else:
    def _layer(typ, oldname):  # fallback
        def decorator(meth):
            return meth
        return decorator


class CaffeFunction(link.Chain):

    """Caffe emulator based on the model file of Caffe.

    Given a protocol buffers file of a Caffe model, this class loads and
    emulates it on :class:`~chainer.Variable` objects. It supports the official
    reference models provided by BVLC.

    .. note::

       This class only supports Python 2.7, since the compiled module for
       protocol buffers only supports Python 2. The ``__init__`` function
       raises an exception in Python 3.

    .. note::

       CaffeFunction ignores the following layers:

       - Layers that CaffeFunction does not support (including data layers)
       - Layers that have no top blobs
       - Layers whose bottom blobs are incomplete (i.e., some or all of them
         are not given nor computed)

    .. warning::

       It does not support full compatibility against Caffe. Some layers and
       configurations are not implemented in Chainer yet, though the reference
       models provided by the BVLC team are supported except data layers.

    .. admonition:: Example

       Consider we want to extract the (unnormalized) log class probability
       of given images using BVLC reference CaffeNet. The model can be
       downloaded from:

       http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

       We want to compute the ``fc8`` blob from the ``data`` blob. It is simply
       written as follows::

          # Load the model
          func = CaffeFunction('path/to/bvlc_reference_caffenet.caffemodel')

          # Minibatch of size 10
          x_data = numpy.ndarray((10, 3, 227, 227), dtype=numpy.float32)
          ...  # (Fill the minibatch here)

          # Forward the pre-trained net
          x = Variable(x_data)
          y, = func(inputs={'data': x}, outputs=['fc8'])

       The result ``y`` contains the Variable corresponding to the ``fc8``
       blob. The computational graph is memorized as a usual forward
       computation in Chainer, so we can run backprop through this pre-trained
       net.

    Args:
        model_path (str): Path to the binary-proto model file of Caffe.

    Attributes:
        fs (FunctionSet): A set of functions corresponding to parameterized
            layers of Caffe. The names of its attributes are same as the layer
            names of the given network.
        forwards (dict): A mapping from layer names to corresponding functions.

    """
    def __init__(self, model_path):
        if not available:
            msg = ('CaffeFunction is only supported on protobuf>=3 in Python3')
            raise RuntimeError(msg)

        super(CaffeFunction, self).__init__()

        net = caffe_pb.NetParameter()
        with open(model_path, 'rb') as model_file:
            net.MergeFromString(model_file.read())

        self.forwards = {}
        self.split_map = {}
        self.layers = []

        if net.layer:
            for layer in net.layer:
                meth = _type_to_method.get(layer.type)
                if meth:
                    meth(self, layer)
                else:
                    warnings.warn(
                        'Skip the layer "%s", since CaffeFunction does not'
                        'support %s layer' % (layer.name, layer.type))
        else:  # v1 format
            for layer in net.layers:
                meth = _oldname_to_method.get(layer.type)
                if meth:
                    meth(self, layer)
                else:
                    warnings.warn(
                        'Skip the layer "%s", since CaffeFunction does not'
                        'support it' % layer.name)

    def __call__(self, inputs, outputs, disable=(), train=True):
        """Executes a sub-network of the network.

        This function acts as an interpreter of the network definition for
        Caffe. On execution, it interprets each layer one by one, and if the
        bottom blobs are already computed, then emulates the layer and stores
        output blobs as :class:`~chainer.Variable` objects.

        Args:
            inputs (dict): A dictionary whose key-value pairs indicate initial
                correspondences between blob names and
                :class:`~chainer.Variable` objects.
            outputs (Iterable): A list of blob names whose corresponding
                :class:`~chainer.Variable` objects are returned.
            disable (Iterable): A list of layer names that will be ignored
                during the forward computation.
            train (bool): If ``True``, this function emulates the TRAIN phase
                of the Caffe layers. Otherwise, it emulates the TEST phase.

        Returns:
            tuple: A tuple of output :class:`~chainer.Variable` objects
                corresponding to elements of the  `outputs` argument.

        """
        self.train = train
        variables = dict(inputs)
        for func_name, bottom, top in self.layers:
            if (func_name in disable or
               func_name not in self.forwards or
               any(blob not in variables for blob in bottom)):
                continue

            func = self.forwards[func_name]
            input_vars = tuple(variables[blob] for blob in bottom)
            output_vars = func(*input_vars)
            if not isinstance(output_vars, collections.Iterable):
                output_vars = output_vars,
            for var, name in zip(output_vars, top):
                variables[name] = var

        self.variables = variables
        return tuple(variables[blob] for blob in outputs)

    def _add_layer(self, layer):
        bottom = []
        for blob_name in layer.bottom:
            bottom.append(self.split_map.get(blob_name, blob_name))
        self.layers.append((layer.name, bottom, list(layer.top)))

    @_layer('Concat', 'CONCAT')
    def _setup_concat(self, layer):
        param = layer.concat_param
        axis = param.axis
        if axis == 1 and param.concat_dim != 1:
            axis = param.concat_dim

        self.forwards[layer.name] = _ListArgumentFcuntion(
            functions.concat, axis=axis)
        self._add_layer(layer)

    @_layer('Convolution', 'CONVOLUTION')
    def _setup_convolution(self, layer):
        blobs = layer.blobs
        param = layer.convolution_param
        ksize = _get_ksize(param)
        stride = _get_stride(param)
        pad = _get_pad(param)
        num = _get_num(blobs[0])
        channels = _get_channels(blobs[0])

        n_in = channels * param.group
        n_out = num
        func = links.Convolution2D(n_in, n_out, ksize, stride, pad,
                                   nobias=not param.bias_term)
        func.W.data[...] = 0

        part_size = len(blobs[0].data) // param.group
        for i in six.moves.range(param.group):
            in_slice = slice(i * n_in // param.group,
                             (i+1) * n_in // param.group)
            out_slice = slice(i * n_out // param.group,
                              (i+1) * n_out // param.group)
            w = func.W.data[out_slice, in_slice]

            data = numpy.array(blobs[0].data[i*part_size:(i+1)*part_size])
            w[:] = data.reshape(w.shape)

        if param.bias_term:
            func.b.data[:] = blobs[1].data

        self.add_link(layer.name, func)
        self.forwards[layer.name] = _CallChildLink(self, layer.name)
        self._add_layer(layer)

    @_layer('Data', 'DATA')
    def _setup_data(self, layer):
        # We silently skip the data layer.
        pass

    @_layer('Dropout', 'DROPOUT')
    def _setup_dropout(self, layer):
        param = layer.dropout_param

        self.forwards[layer.name] = _DropoutFunction(
            self, ratio=param.dropout_ratio)
        self._add_layer(layer)

    @_layer('InnerProduct', 'INNER_PRODUCT')
    def _setup_inner_product(self, layer):
        param = layer.inner_product_param
        bias_term = param.bias_term
        if param.axis != 1:
            raise RuntimeError(
                'Non-default axis in InnerProduct is not supported')

        blobs = layer.blobs
        width, height = _get_width(blobs[0]), _get_height(blobs[0])
        func = links.Linear(width, height, nobias=not bias_term)
        func.W.data.ravel()[:] = blobs[0].data
        if bias_term:
            func.b.data[:] = blobs[1].data

        self.add_link(layer.name, func)
        self.forwards[layer.name] = _CallChildLink(self, layer.name)
        self._add_layer(layer)

    @_layer('LRN', 'LRN')
    def _setup_lrn(self, layer):
        param = layer.lrn_param
        if param.norm_region != param.ACROSS_CHANNELS:
            raise RuntimeError('Within-channel LRN is not supported')

        fwd = _SingleArgumentFunction(
            functions.local_response_normalization,
            n=param.local_size, k=param.k,
            alpha=param.alpha / param.local_size, beta=param.beta)
        self.forwards[layer.name] = fwd
        self._add_layer(layer)

    @_layer('Pooling', 'POOLING')
    def _setup_pooling(self, layer):
        param = layer.pooling_param
        ksize = _get_ksize(param)
        stride = _get_stride(param)
        pad = _get_pad(param)

        if param.pool == param.MAX:
            func = functions.max_pooling_2d
        elif param.pool == param.AVE:
            func = functions.average_pooling_2d
        else:
            raise RuntimeError('Stochastic pooling is not supported')

        fw = _SingleArgumentFunction(func, ksize, stride=stride, pad=pad)
        self.forwards[layer.name] = fw
        self._add_layer(layer)

    @_layer('ReLU', 'RELU')
    def _setup_relu(self, layer):
        slope = layer.relu_param.negative_slope

        if slope != 0:
            fw = _SingleArgumentFunction(functions.leaky_relu, slope=slope)
        else:
            fw = functions.relu

        self.forwards[layer.name] = fw
        self._add_layer(layer)

    @_layer('SoftmaxWithLoss', 'SOFTMAX_LOSS')
    def _setup_softmax_with_loss(self, layer):
        if layer.softmax_param.axis != 1:
            raise RuntimeError(
                'Softmax along non-channel axis is not supported')

        self.forwards[layer.name] = functions.softmax_cross_entropy
        self._add_layer(layer)

    @_layer('Split', 'SPLIT')
    def _setup_split(self, layer):
        for top in layer.top:
            self.split_map[top] = layer.bottom[0]


# Internal functions

def _get_ksize(param):
    if param.kernel_h > 0:
        return param.kernel_h, param.kernel_w
    else:
        return param.kernel_size


def _get_stride(param):
    if param.stride_h > 0:
        return param.stride_h, param.stride_w
    else:
        return param.stride


def _get_pad(param):
    if param.pad_h > 0:
        return param.pad_h, param.pad_w
    else:
        return param.pad


def _get_num(blob):
    if blob.num > 0:
        return blob.num
    else:
        return blob.shape.dim[0]


def _get_channels(blob):
    if blob.channels > 0:
        return blob.channels
    else:
        return blob.shape.dim[1]


def _get_height(blob):
    if blob.height > 0:
        return blob.height
    elif len(blob.shape.dim) == 2:
        return blob.shape.dim[0]
    elif len(blob.shape.dim) == 4:
        return blob.shape.dim[2]
    else:
        raise RuntimeError(
            '{}-dimentional array is not supported'.format(
                len(blob.shape.dim)))


def _get_width(blob):
    if blob.width > 0:
        return blob.width
    elif len(blob.shape.dim) == 2:
        return blob.shape.dim[1]
    elif len(blob.shape.dim) == 4:
        return blob.shape.dim[3]
    else:
        raise RuntimeError(
            '{}-dimentional array is not supported'.format(
                len(blob.shape.dim)))


# Internal class

class _SingleArgumentFunction(object):
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return self.func(x, *self.args, **self.kwargs)


class _ListArgumentFcuntion(object):
    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def __call__(self, *xs):
        return self.func(xs, **self.kwargs)


class _DropoutFunction(object):
    def __init__(self, caffe_func, ratio):
        # `caffe_func.train` is determined when calling `__call__`
        self.caffe_func = caffe_func
        self.ratio = ratio

    def __call__(self, x):
        return functions.dropout(
            x, ratio=self.ratio, train=self.caffe_func.train)


class _CallChildLink(object):
    def __init__(self, caffe_func, name):
        self.name = name
        self.caffe_func = caffe_func

    def __call__(self, *xs):
        return self.caffe_func[self.name](*xs)
