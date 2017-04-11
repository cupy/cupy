import binascii
import itertools
import os
import time

import numpy
import six

from chainer import cuda
from chainer import function
from chainer.functions.activation import relu
from chainer.functions.activation import tanh
from chainer.functions.array import concat
from chainer.functions.array import reshape
from chainer.functions.array import split_axis
from chainer.functions.array import stack
from chainer.functions.connection import linear
from chainer.functions.noise import dropout
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()


def _stack_weight(ws):
    # TODO(unno): Input of the current LSTM implementaiton is shuffled
    w = stack.stack(ws, axis=1)
    shape = w.shape
    return reshape.reshape(w, (shape[0] * shape[1],) + shape[2:])


class PointerArray(object):

    def __init__(self, lst, back_pointer):
        self._value = numpy.array(lst, dtype=numpy.intp)
        # Store back_pointer to prevent the GC removes the original variable
        self._back_pointer = back_pointer

    @property
    def data(self):
        return self._value.ctypes.data


def _make_tensor_descriptor_array(xs):
    """Make an array of pointers denoting pointers of tensor descriptors.

    """
    descs = []
    for x in xs:
        if x.ndim < 3:
            shape = x.shape + (1,) * (3 - x.ndim)
            x = x.reshape(shape)
        desc = cudnn.create_tensor_nd_descriptor(x)
        descs.append(desc)
    return PointerArray([d.value for d in descs], descs)


def _make_ptr_array(xs):
    """Make an array of pointers denoting pointers of ndarrays.

    """
    return PointerArray([x.data.ptr for x in xs], xs)


class DropoutStates(object):

    def __init__(self, states, desc):
        self.states = states
        self.desc = desc

    def set_dropout_ratio(self, handle, dropout):
        cudnn.set_dropout_descriptor(self.desc, handle, dropout)

    @staticmethod
    def create(handle, dropout, seed):
        states = cudnn.create_dropout_states(handle)
        desc = cudnn.create_dropout_descriptor(
            handle, dropout, states.data.ptr, states.size, seed)
        return DropoutStates(states, desc)


class DropoutRandomStates(object):

    def __init__(self, seed):
        self._states = None

        if seed is None:
            try:
                seed_str = binascii.hexlify(os.urandom(8))
                seed = numpy.uint64(int(seed_str, 16))
            except NotImplementedError:
                seed = numpy.uint64(time.clock() * 1000000)
        else:
            seed = numpy.uint64(seed)

        self._seed = seed

    def create_dropout_states(self, dropout):
        handle = cudnn.get_handle()
        if self._states is None:
            self._states = DropoutStates.create(handle, dropout, self._seed)
        else:
            self._states.set_dropout_ratio(handle, dropout)

        return self._states


def _split(inputs, pos):
    return inputs[:pos], inputs[pos:]

_random_states = {}


def get_random_state():
    global _random_states
    dev = cuda.Device()
    rs = _random_states.get(dev.id, None)
    if rs is None:
        rs = DropoutRandomStates(os.getenv('CHAINER_SEED'))
        _random_states[dev.id] = rs
    return rs

if cuda.cudnn_enabled and _cudnn_version >= 5000:
    # Define RNN parameters using dict.
    _rnn_dirs = {
        'uni': libcudnn.CUDNN_UNIDIRECTIONAL,
        'bi':  libcudnn.CUDNN_BIDIRECTIONAL,
    }

    _rnn_modes = {
        'rnn_relu': libcudnn.CUDNN_RNN_RELU,
        'rnn_tanh': libcudnn.CUDNN_RNN_TANH,
        'gru': libcudnn.CUDNN_GRU,
        'lstm': libcudnn.CUDNN_LSTM,
    }

    _rnn_n_params = {
        libcudnn.CUDNN_RNN_RELU: 2,
        libcudnn.CUDNN_RNN_TANH: 2,
        libcudnn.CUDNN_GRU: 6,
        libcudnn.CUDNN_LSTM: 8,
    }

    _rnn_params_direction = {
        libcudnn.CUDNN_UNIDIRECTIONAL: 1,
        libcudnn.CUDNN_BIDIRECTIONAL: 2,
    }

    _rnn_params_use_cell = {
        libcudnn.CUDNN_RNN_RELU: False,
        libcudnn.CUDNN_RNN_TANH: False,
        libcudnn.CUDNN_GRU: False,
        libcudnn.CUDNN_LSTM: True,
    }


class BaseNStepRNN(function.Function):
    def __init__(self, n_layers, states, rnn_dir, rnn_mode, train=True):
        if rnn_dir not in _rnn_dirs:
            candidate_list = ','.join(_rnn_dirs.keys())
            raise ValueError('Invalid rnn_dir: "%s". Please select from [%s]'
                             % (rnn_dir, candidate_list))
        if rnn_mode not in _rnn_modes:
            candidate_list = ','.join(_rnn_modes.keys())
            raise ValueError('Invalid rnn_mode: "%s". Please select from [%s]'
                             % (rnn_mode, candidate_list))
        self.rnn_dir = _rnn_dirs[rnn_dir]
        self.rnn_mode = _rnn_modes[rnn_mode]
        self.rnn_direction = _rnn_params_direction[self.rnn_dir]
        self.n_layers = n_layers
        self.train = train
        self.states = states
        self.use_cell = _rnn_params_use_cell[self.rnn_mode]
        self.n_W = _rnn_n_params[self.rnn_mode]

    @property
    def _n_cell(self):
        if self.use_cell:
            return 2
        else:
            return 1

    @property
    def _n_params(self):
        return self.n_layers * self.rnn_direction * self.n_W

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > self._n_cell + self._n_params * 2)

        if self.use_cell:
            (h_type, c_type), in_types = _split(in_types, 2)
            h_size = self.n_layers * self.rnn_direction
            type_check.expect(
                h_type.dtype == numpy.float32,
                c_type.dtype == numpy.float32,

                h_type.ndim == 3,
                h_type.shape[0] == h_size,
                c_type.ndim == 3,
                c_type.shape[0] == h_size,

                # mini-batch size
                h_type.shape[1] == c_type.shape[1],

                # hidden size
                h_type.shape[2] == c_type.shape[2],
            )

        else:
            (h_type, ), in_types = _split(in_types, 1)
            h_size = self.n_layers * self.rnn_direction
            type_check.expect(
                h_type.dtype == numpy.float32,

                h_type.ndim == 3,
                h_type.shape[0] == h_size,
            )

        w_types, in_types = _split(in_types, self._n_params)
        b_types, in_types = _split(in_types, self._n_params)

        x_types = in_types
        for x_type in x_types:
            type_check.expect(
                x_type.dtype == numpy.float32,
                x_type.ndim == 2,
            )
        for x1_type, x2_type in six.moves.zip(x_types, x_types[1:]):
            type_check.expect(
                # Check if xs are sorted by descending lengths
                x1_type.shape[0] >= x2_type.shape[0],
                x1_type.shape[1] == x2_type.shape[1])

        in_size = x_types[0].shape[1]
        out_size = h_type.shape[2]

        for layer in six.moves.range(self.n_layers):
            for i in six.moves.range(self.n_W):
                for di in six.moves.range(self.rnn_direction):
                    ind = (layer * self.rnn_direction + di) * self.n_W + i
                    w_type = w_types[ind]
                    b_type = b_types[ind]
                    if self.rnn_direction == 1:
                        # Uni-direction
                        if layer == 0 and i < (self.n_W // 2):
                            w_in = in_size
                        else:
                            w_in = out_size
                    else:
                        # Bi-direction
                        if layer == 0 and i < (self.n_W // 2):
                            w_in = in_size
                        elif layer > 0 and i < (self.n_W // 2):
                            w_in = out_size * self.rnn_direction
                        else:
                            w_in = out_size

                    type_check.expect(
                        w_type.dtype == numpy.float32,
                        w_type.ndim == 2,
                        w_type.shape[0] == out_size,
                        w_type.shape[1] == w_in,

                        b_type.dtype == numpy.float32,
                        b_type.ndim == 1,
                        b_type.shape[0] == out_size,
                    )

    def forward(self, inputs):
        if self.use_cell:
            # LSTM
            (hx, cx), inputs = _split(inputs, self._n_cell)
            cx = cuda.cupy.ascontiguousarray(cx)
            cx_desc = cudnn.create_tensor_nd_descriptor(cx)

            cy = cuda.cupy.empty_like(cx)
            cy_desc = cudnn.create_tensor_nd_descriptor(cy)

            cx_data_ptr = cx.data.ptr
            cy_data_ptr = cy.data.ptr

            cx_desc_value = cx_desc.value
            cy_desc_value = cy_desc.value
        else:
            # RNN, GRU
            (hx, ), inputs = _split(inputs, self._n_cell)
            cx = cy = None
            cx_data_ptr = cy_data_ptr = 0
            cx_desc_value = cy_desc_value = 0

        ws, inputs = _split(inputs, self._n_params)
        bs, inputs = _split(inputs, self._n_params)
        x_list = inputs
        hx = cuda.cupy.ascontiguousarray(hx)
        x_desc = cudnn.create_tensor_nd_descriptor(x_list[0][..., None])

        length = len(x_list)
        n_units = hx.shape[2]

        xs = cuda.cupy.concatenate(x_list, axis=0)
        ys = cuda.cupy.empty((len(xs),
                              n_units * self.rnn_direction), dtype=xs.dtype)

        handle = cudnn.get_handle()
        self.handle = handle

        rnn_desc = cudnn.create_rnn_descriptor(
            n_units, self.n_layers, self.states.desc,
            libcudnn.CUDNN_LINEAR_INPUT, self.rnn_dir,
            self.rnn_mode, libcudnn.CUDNN_DATA_FLOAT)
        self.rnn_desc = rnn_desc

        c_x_descs = _make_tensor_descriptor_array(x_list)
        hx_desc = cudnn.create_tensor_nd_descriptor(hx)

        weights_size = libcudnn.getRNNParamsSize(
            handle, rnn_desc.value, x_desc.value, libcudnn.CUDNN_DATA_FLOAT)
        w = cuda.cupy.empty((weights_size // 4, 1, 1), dtype=numpy.float32)
        w_desc = cudnn.create_filter_descriptor(w)

        for layer in six.moves.range(self.n_layers):
            for di in six.moves.range(self.rnn_direction):
                # di = 0: forward, 1: backward
                for lin_layer_id in six.moves.range(self.n_W):
                    mat_index = layer * self.rnn_direction + di
                    mat = cudnn.get_rnn_lin_layer_matrix_params(
                        handle, rnn_desc, mat_index,
                        x_desc, w_desc, w, lin_layer_id)
                    W_index = mat_index * self.n_W + lin_layer_id
                    m = mat.reshape(mat.size)
                    m[...] = ws[W_index].ravel()
                    bias = cudnn.get_rnn_lin_layer_bias_params(
                        handle, rnn_desc, mat_index,
                        x_desc, w_desc, w, lin_layer_id)
                    b = bias.reshape(bias.size)
                    b[...] = bs[W_index]
        self.w = w
        self.w_desc = w_desc

        sections = numpy.cumsum([len(x) for x in x_list[:-1]])
        y_list = cuda.cupy.split(ys, sections)

        c_y_descs = _make_tensor_descriptor_array(y_list)
        hy = cuda.cupy.empty_like(hx)
        hy_desc = cudnn.create_tensor_nd_descriptor(hy)

        work_size = libcudnn.getRNNWorkspaceSize(
            handle, rnn_desc.value, length, c_x_descs.data)
        workspace = cuda.cupy.empty((work_size,), dtype='b')
        self.workspace = workspace

        if not self.train:
            libcudnn.RNNForwardInference(
                handle, rnn_desc.value, length,
                c_x_descs.data, xs.data.ptr, hx_desc.value, hx.data.ptr,
                cx_desc_value, cx_data_ptr, w_desc.value, w.data.ptr,
                c_y_descs.data, ys.data.ptr, hy_desc.value, hy.data.ptr,
                cy_desc_value, cy_data_ptr, workspace.data.ptr, work_size)

        else:
            reserve_size = libcudnn.getRNNTrainingReserveSize(
                handle, rnn_desc.value, length, c_x_descs.data)
            self.reserve_space = cuda.cupy.empty((reserve_size,), dtype='b')
            libcudnn.RNNForwardTraining(
                handle, rnn_desc.value, length,
                c_x_descs.data, xs.data.ptr, hx_desc.value, hx.data.ptr,
                cx_desc_value, cx_data_ptr, w_desc.value, w.data.ptr,
                c_y_descs.data, ys.data.ptr, hy_desc.value, hy.data.ptr,
                cy_desc_value, cy_data_ptr,
                workspace.data.ptr, work_size,
                self.reserve_space.data.ptr, reserve_size)

        self.c_y_descs = c_y_descs
        self.ys = ys
        self.c_x_descs = c_x_descs

        if self.use_cell:
            # LSTM
            return tuple([hy, cy] + y_list)
        else:
            # GRU, RNN
            return tuple([hy, ] + y_list)

    def backward(self, inputs, grads):
        if self.use_cell:
            # LSTM
            (hx, cx), inputs = _split(inputs, self._n_cell)
            dhy, dcy = grads[:self._n_cell]
            if dcy is None:
                dcy = cuda.cupy.zeros_like(cx)

            cx = cuda.cupy.ascontiguousarray(cx)
            dcx = cuda.cupy.empty_like(cx)

            cx_desc = cudnn.create_tensor_nd_descriptor(cx)
            dcx_desc = cudnn.create_tensor_nd_descriptor(dcx)
            dcy_desc = cudnn.create_tensor_nd_descriptor(dcy)

            cx_data_ptr = cx.data.ptr
            dcy_data_ptr = dcy.data.ptr
            dcx_data_ptr = dcx.data.ptr
            cx_desc_value = cx_desc.value
            dcx_desc_value = dcx_desc.value
            dcy_desc_value = dcy_desc.value
        else:
            # GRU, RNN
            (hx, ), inputs = _split(inputs, self._n_cell)
            dhy, = grads[:self._n_cell]
            dcy = cx = dcx = None
            cx_data_ptr = dcy_data_ptr = dcx_data_ptr = 0
            cx_desc_value = dcx_desc_value = dcy_desc_value = 0

        ws_size = self.n_layers * self.rnn_direction * self.n_W
        ws, inputs = _split(inputs, ws_size)
        bs, inputs = _split(inputs, ws_size)
        x_list = inputs
        hx = cuda.cupy.ascontiguousarray(hx)

        if dhy is None:
            dhy = cuda.cupy.zeros_like(hx)

        dy_list = list(grads[self._n_cell:])
        for i in six.moves.range(len(dy_list)):
            if dy_list[i] is None:
                dy_list[i] = cuda.cupy.zeros_like(x_list[i])

        xs = cuda.cupy.concatenate(x_list, axis=0)
        length = len(x_list)

        dhx = cuda.cupy.empty_like(hx)

        hx_desc = cudnn.create_tensor_nd_descriptor(hx)
        dhy_desc = cudnn.create_tensor_nd_descriptor(dhy)

        c_dy_descs = _make_tensor_descriptor_array(dy_list)
        dys = cuda.cupy.concatenate(dy_list, axis=0)

        rnn_desc = self.rnn_desc
        handle = self.handle
        work_size = libcudnn.getRNNWorkspaceSize(
            handle, rnn_desc.value, length, self.c_x_descs.data)
        workspace = cuda.cupy.empty((work_size,), dtype='b')

        dhx_desc = cudnn.create_tensor_nd_descriptor(dhx)

        dxs = cuda.cupy.empty_like(xs)
        sections = numpy.cumsum([len(x) for x in x_list[:-1]])
        dx_list = cuda.cupy.split(dxs, sections, 0)
        c_dx_descs = _make_tensor_descriptor_array(dx_list)

        libcudnn.RNNBackwardData(
            handle, rnn_desc.value, length,
            self.c_y_descs.data, self.ys.data.ptr,
            c_dy_descs.data, dys.data.ptr, dhy_desc.value, dhy.data.ptr,
            dcy_desc_value, dcy_data_ptr, self.w_desc.value, self.w.data.ptr,
            hx_desc.value, hx.data.ptr, cx_desc_value, cx_data_ptr,
            c_dx_descs.data, dxs.data.ptr, dhx_desc.value, dhx.data.ptr,
            dcx_desc_value, dcx_data_ptr, workspace.data.ptr, work_size,
            self.reserve_space.data.ptr, self.reserve_space.size)

        dw = cuda.cupy.zeros_like(self.w)
        dw_desc = cudnn.create_filter_descriptor(dw)
        libcudnn.RNNBackwardWeights(
            handle, rnn_desc.value, length,
            self.c_x_descs.data, xs.data.ptr,
            hx_desc.value, hx.data.ptr, self.c_y_descs.data, self.ys.data.ptr,
            workspace.data.ptr, work_size, dw_desc.value, dw.data.ptr,
            self.reserve_space.data.ptr, self.reserve_space.size)

        dx = dx_list[0]
        dx = dx.reshape(dx.shape + (1,))
        dx_desc = cudnn.create_tensor_nd_descriptor(dx)
        dws = []
        dbs = []
        for layer in six.moves.range(self.n_layers):
            for di in six.moves.range(self.rnn_direction):
                for lin_layer_id in six.moves.range(self.n_W):
                    mat_index = layer * self.rnn_direction + di
                    mat = cudnn.get_rnn_lin_layer_matrix_params(
                        handle, rnn_desc, mat_index,
                        dx_desc, dw_desc, dw, lin_layer_id)
                    W_index = mat_index * self.n_W + lin_layer_id
                    dws.append(mat.reshape(ws[W_index].shape))
                    bias = cudnn.get_rnn_lin_layer_bias_params(
                        handle, rnn_desc, mat_index,
                        dx_desc, dw_desc, dw, lin_layer_id)
                    dbs.append(bias.reshape(bs[W_index].shape))

        if self.use_cell:
            # LSTM
            return tuple([dhx, dcx] + dws + dbs + dx_list)
        else:
            # GRU, RNN
            return tuple([dhx, ] + dws + dbs + dx_list)


class NStepRNNTanh(BaseNStepRNN):
    def __init__(self, n_layers, states, train=True):
        BaseNStepRNN.__init__(self, n_layers, states, rnn_dir='uni',
                              rnn_mode='rnn_tanh', train=train)


class NStepRNNReLU(BaseNStepRNN):
    def __init__(self, n_layers, states, train=True):
        BaseNStepRNN.__init__(self, n_layers, states, rnn_dir='uni',
                              rnn_mode='rnn_relu', train=train)


class NStepBiRNNTanh(BaseNStepRNN):
    def __init__(self, n_layers, states, train=True):
        BaseNStepRNN.__init__(self, n_layers, states, rnn_dir='bi',
                              rnn_mode='rnn_tanh', train=train)


class NStepBiRNNReLU(BaseNStepRNN):
    def __init__(self, n_layers, states, train=True):
        BaseNStepRNN.__init__(self, n_layers, states, rnn_dir='bi',
                              rnn_mode='rnn_relu', train=train)


def n_step_rnn(n_layers, dropout_ratio, hx, ws, bs, xs, train=True,
               use_cudnn=True, activation='tanh'):
    """Stacked Uni-directional RNN function for sequence inputs.

    This function calculates stacked RNN with sequences. This function gets
    an initial hidden state :math:`h_0`, an initial cell state :math:`c_0`,
    an input sequence :math:`x`, weight matrices :math:`W`, and bias vectors
    :math:`b`.
    This function calculates hidden states :math:`h_t` and :math:`c_t` for each
    time :math:`t` from input :math:`x_t`.

    .. math::
       h_t = f(W_0 x_t + W_1 h_{t-1} + b_0 + b_1)

    where :math:`f` is an activation function.

    Weight matrices :math:`W` contains two matrices :math:`W_0` and
    :math:`W_1`. :math:`W_0` is a parameter for an input sequence.
    :math:`W_1` is a parameter for a hidden state.
    Bias matrices :math:`b` contains two matrices :math:`b_0` and :math:`b_1`.
    :math:`b_0` is a parameter for an input sequence.
    :math:`b_1` is a parameter for a hidden state.


    As the function accepts a sequence, it calculates :math:`h_t` for all
    :math:`t` with one call. Two weight matrices and two bias vectors are
    required for each layer. So, when :math:`S` layers exist, you need to
    prepare :math:`2S` weigth matrices and :math:`2S` bias vectors.

    If the number of layers ``n_layers`` is greather than :math:`1`, input
    of ``k``-th layer is hidden state ``h_t`` of ``k-1``-th layer.
    Note that all input variables except first layer may have different shape
    from the first layer.

    Args:
        n_layers(int): Number of layers.
        dropout_ratio(float): Dropout ratio.
        hx (chainer.Variable): Variable holding stacked hidden states.
            Its shape is ``(S, B, N)`` where ``S`` is number of layers and is
            equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is
            dimention of hidden units.
        ws (list of list of chainer.Variable): Weight matrices. ``ws[i]``
            represents weights for i-th layer.
            Each ``ws[i]`` is a list containing two matrices.
            ``ws[i][j]`` is corresponding with ``W_j`` in the equation.
            Only ``ws[0][j]`` where ``0 <= j < 1`` is ``(I, N)`` shape as they
            are multiplied with input variables. All other matrices has
            ``(N, N)`` shape.
        bs (list of list of chainer.Variable): Bias vectors. ``bs[i]``
            represnents biases for i-th layer.
            Each ``bs[i]`` is a list containing two vectors.
            ``bs[i][j]`` is corresponding with ``b_j`` in the equation.
            Shape of each matrix is ``(N,)`` where ``N`` is dimention of
            hidden units.
        xs (list of chainer.Variable): A list of :class:`~chainer.Variable`
            holding input values. Each element ``xs[t]`` holds input value
            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is
            mini-batch size for time ``t``, and ``I`` is size of input units.
            Note that this functions supports variable length sequences.
            When sequneces has different lengths, sort sequences in descending
            order by length, and transpose the sorted sequence.
            :func:`~chainer.functions.transpose_sequence` transpose a list
            of :func:`~chainer.Variable` holding sequence.
            So ``xs`` needs to satisfy
            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.
        train (bool): If ``True``, this function executes dropout.
        use_cudnn (bool): If ``True``, this function uses cuDNN if available.
        activation (str): Activation function name.
            Please select ``tanh`` or ``relu``.

    Returns:
        tuple: This functions returns a tuple concaining three elements,
            ``hy`` and ``ys``.

            - ``hy`` is an updated hidden states whose shape is same as ``hx``.
            - ``ys`` is a list of :class:`~chainer.Variable` . Each element
              ``ys[t]`` holds hidden states of the last layer corresponding
              to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is
              mini-batch size for time ``t``, and ``N`` is size of hidden
              units. Note that ``B_t`` is the same value as ``xs[t]``.

    """
    return n_step_rnn_base(n_layers, dropout_ratio, hx, ws, bs, xs, train,
                           use_cudnn, activation, use_bi_direction=False)


def n_step_birnn(n_layers, dropout_ratio, hx, ws, bs, xs, train=True,
                 use_cudnn=True, activation='tanh'):
    """Stacked Bi-directional RNN function for sequence inputs.

    This function calculates stacked RNN with sequences. This function gets
    an initial hidden state :math:`h_0`, an initial cell state :math:`c_0`,
    an input sequence :math:`x`, weight matrices :math:`W`, and bias vectors
    :math:`b`.
    This function calculates hidden states :math:`h_t` and :math:`c_t` for each
    time :math:`t` from input :math:`x_t`.

    .. math::
        hf_t &=& f(W^{f}_0 x_t + W^{f}_1 hf_{t-1} + b^{f}_0 + b^{f}_1), \\\\
        hb_t &=& f(W^{b}_0 x_t + W^{b}_1 hb_{t-1} + b^{b}_0 + b^{b}_1), \\\\
        h_t  &=& [hf_t; hb_t], \\\\

    where :math:`f` is an activation function.

    Weight matrices :math:`W` contains two matrices :math:`W^{f}` and
    :math:`W^{b}`. :math:`W^{f}` is weight matrices for forward directional
    RNN. :math:`W^{b}` is weight matrices for backward directional RNN.

    :math:`W^{f}` contains :math:`W^{f}_0` for an input sequence and
    :math:`W^{f}_1` for a hidden state.
    :math:`W^{b}` contains :math:`W^{b}_0` for an input sequence and
    :math:`W^{b}_1` for a hidden state.

    Bias matrices :math:`b` contains two matrices :math:`b^{f}` and
    :math:`b^{f}`. :math:`b^{f}` contains :math:`b^{f}_0` for an input sequence
    and :math:`b^{f}_1` for a hidden state.
    :math:`b^{b}` contains :math:`b^{b}_0` for an input sequence and
    :math:`b^{b}_1` for a hidden state.

    As the function accepts a sequence, it calculates :math:`h_t` for all
    :math:`t` with one call. Two weight matrices and two bias vectors are
    required for each layer. So, when :math:`S` layers exist, you need to
    prepare :math:`2S` weigth matrices and :math:`2S` bias vectors.

    If the number of layers ``n_layers`` is greather than :math:`1`, input
    of ``k``-th layer is hidden state ``h_t`` of ``k-1``-th layer.
    Note that all input variables except first layer may have different shape
    from the first layer.

    Args:
        n_layers(int): Number of layers.
        dropout_ratio(float): Dropout ratio.
        hx (chainer.Variable): Variable holding stacked hidden states.
            Its shape is ``(S, B, N)`` where ``S`` is number of layers and is
            equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is
            dimention of hidden units.
        ws (list of list of chainer.Variable): Weight matrices. ``ws[i + di]``
            represents weights for i-th layer.
            Note that ``di = 0`` for forward-RNN and ``di = 1`` for
            backward-RNN.
            Each ``ws[i + di]`` is a list containing two matrices.
            ``ws[i + di][j]`` is corresponding with ``W^{f}_j`` if ``di = 0``
            and corresponding with ``W^{b}_j`` if ``di = 1`` in the equation.
            Only ``ws[0][j]`` and ``ws[1][j]`` where ``0 <= j < 1`` are
            ``(I, N)`` shape as they are multiplied with input variables.
            All other matrices has ``(N, N)`` shape.
        bs (list of list of chainer.Variable): Bias vectors. ``bs[i + di]``
            represnents biases for i-th layer.
            Note that ``di = 0`` for forward-RNN and ``di = 1`` for
            backward-RNN.
            Each ``bs[i + di]`` is a list containing two vectors.
            ``bs[i + di][j]`` is corresponding with ``b^{f}_j`` if ``di = 0``
            and corresponding with ``b^{b}_j`` if ``di = 1`` in the equation.
            Shape of each matrix is ``(N,)`` where ``N`` is dimention of
            hidden units.
        xs (list of chainer.Variable): A list of :class:`~chainer.Variable`
            holding input values. Each element ``xs[t]`` holds input value
            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is
            mini-batch size for time ``t``, and ``I`` is size of input units.
            Note that this functions supports variable length sequences.
            When sequneces has different lengths, sort sequences in descending
            order by length, and transpose the sorted sequence.
            :func:`~chainer.functions.transpose_sequence` transpose a list
            of :func:`~chainer.Variable` holding sequence.
            So ``xs`` needs to satisfy
            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.
        train (bool): If ``True``, this function executes dropout.
        use_cudnn (bool): If ``True``, this function uses cuDNN if available.
        activation (str): Activation function name.
            Please select ``tanh`` or ``relu``.

    Returns:
        tuple: This functions returns a tuple concaining three elements,
            ``hy`` and ``ys``.

            - ``hy`` is an updated hidden states whose shape is same as ``hx``.
            - ``ys`` is a list of :class:`~chainer.Variable` . Each element
              ``ys[t]`` holds hidden states of the last layer corresponding
              to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t``
              is mini-batch size for time ``t``, and ``N`` is size of hidden
              units. Note that ``B_t`` is the same value as ``xs[t]``.

    """
    return n_step_rnn_base(n_layers, dropout_ratio, hx, ws, bs, xs, train,
                           use_cudnn, activation, use_bi_direction=True)


def n_step_rnn_base(n_layers, dropout_ratio, hx, ws, bs, xs, train,
                    use_cudnn, activation, use_bi_direction):
    """Stacked RNN Base function for sequence inputs.

    This function calculates stacked RNN with sequences. This function gets
    an initial hidden state :math:`h_0`, an initial cell state :math:`c_0`,
    an input sequence :math:`x`, weight matrices :math:`W_0` for an input
    sequence, weight matrices :math:`W_1` for a hidden state, and bias vectors
    :math:`b`.
    This function calculates hidden states :math:`h_t` and :math:`c_t` for each
    time :math:`t` from input :math:`x_t`.

    .. math::
       h_t = f(W_0 x_t + W_1 h_{t-1} + b_0 + b_1)

    where :math:`f` is an activation function.

    As the function accepts a sequence, it calculates :math:`h_t` for all
    :math:`t` with one call. Eight weight matrices and eight bias vectors are
    required for each layer. So, when :math:`S` layers exist, you need to
    prepare :math:`2S` weigth matrices and :math:`2S` bias vectors.

    If the number of layers ``n_layers`` is greather than :math:`1`, input
    of ``k``-th layer is hidden state ``h_t`` of ``k-1``-th layer.
    Note that all input variables except first layer may have different shape
    from the first layer.

    Args:
        n_layers(int): Number of layers.
        dropout_ratio(float): Dropout ratio.
        hx (chainer.Variable): Variable holding stacked hidden states.
            Its shape is ``(S, B, N)`` where ``S`` is number of layers and is
            equal to ``n_layers``, ``B`` is mini-batch size, and ``N`` is
            dimention of hidden units.
        ws (list of list of chainer.Variable): Weight matrices. ``ws[i]``
            represents weights for i-th layer.
            Each ``ws[i]`` is a list containing eight matrices.
            ``ws[i][j]`` is corresponding with ``W_j`` in the equation.
            Only ``ws[0][j]`` where ``0 <= j < 1`` is ``(I, N)`` shape as they
            are multiplied with input variables. All other matrices has
            ``(N, N)`` shape.
        bs (list of list of chainer.Variable): Bias vectors. ``bs[i]``
            represnents biases for i-th layer.
            Each ``bs[i]`` is a list containing eight vectors.
            ``bs[i][j]`` is corresponding with ``b_j`` in the equation.
            Shape of each matrix is ``(N,)`` where ``N`` is dimention of
            hidden units.
        xs (list of chainer.Variable): A list of :class:`~chainer.Variable`
            holding input values. Each element ``xs[t]`` holds input value
            for time ``t``. Its shape is ``(B_t, I)``, where ``B_t`` is
            mini-batch size for time ``t``, and ``I`` is size of input units.
            Note that this functions supports variable length sequences.
            When sequneces has different lengths, sort sequences in descending
            order by length, and transpose the sorted sequence.
            :func:`~chainer.functions.transpose_sequence` transpose a list
            of :func:`~chainer.Variable` holding sequence.
            So ``xs`` needs to satisfy
            ``xs[t].shape[0] >= xs[t + 1].shape[0]``.
        train (bool): If ``True``, this function executes dropout.
        use_cudnn (bool): If ``True``, this function uses cuDNN if available.
        activation (str): Activation function name.
            Please select ``tanh`` or ``relu``.
        use_bi_direction (bool): If ``True``, this function uses
            Bi-direction RNN.

    Returns:
        tuple: This functions returns a tuple concaining three elements,
            ``hy`` and ``ys``.
            - ``hy`` is an updated hidden states whose shape is same as ``hx``.
            - ``ys`` is a list of :class:`~chainer.Variable` . Each element
              ``ys[t]`` holds hidden states of the last layer corresponding
              to an input ``xs[t]``. Its shape is ``(B_t, N)`` where ``B_t`` is
              mini-batch size for time ``t``, and ``N`` is size of hidden
              units. Note that ``B_t`` is the same value as ``xs[t]``.

    """
    activation_list = ['tanh', 'relu']
    if activation not in activation_list:
        candidate = ','.join(activation_list)
        raise ValueError('Invalid activation: "%s". Please select from [%s]'
                         % (activation, candidate))

    xp = cuda.get_array_module(hx)

    if use_cudnn and xp is not numpy and cuda.cudnn_enabled and \
       _cudnn_version >= 5000:
        states = get_random_state().create_dropout_states(dropout_ratio)
        # flatten all input variables
        inputs = tuple(itertools.chain(
            (hx, ),
            itertools.chain.from_iterable(ws),
            itertools.chain.from_iterable(bs),
            xs))
        if use_bi_direction:
            # Bi-directional RNN
            if activation == 'tanh':
                rnn = NStepBiRNNTanh(n_layers, states, train=train)
            elif activation == 'relu':
                rnn = NStepBiRNNReLU(n_layers, states, train=train)
        else:
            # Uni-directional RNN
            if activation == 'tanh':
                rnn = NStepRNNTanh(n_layers, states, train=train)
            elif activation == 'relu':
                rnn = NStepRNNReLU(n_layers, states, train=train)

        ret = rnn(*inputs)
        hy, = ret[:1]
        ys = ret[1:]
        return hy, ys

    else:

        direction = 2 if use_bi_direction else 1
        hx = split_axis.split_axis(hx, n_layers * direction, axis=0,
                                   force_tuple=True)
        hx = [reshape.reshape(h, h.shape[1:]) for h in hx]

        xws = [_stack_weight([w[0]]) for w in ws]
        hws = [_stack_weight([w[1]]) for w in ws]
        xbs = [_stack_weight([b[0]]) for b in bs]
        hbs = [_stack_weight([b[1]]) for b in bs]

        xs_next = xs
        hy = []
        for layer in six.moves.range(n_layers):

            def _one_directional_loop(di):
                # di=0, forward RNN
                # di=1, backward RNN
                xs_list = xs_next if di == 0 else reversed(xs_next)
                layer_idx = direction * layer + di
                h = hx[layer_idx]
                h_list = []
                for x in xs_list:
                    batch = x.shape[0]
                    if h.shape[0] > batch:
                        h, h_rest = split_axis.split_axis(h, [batch], axis=0)
                    else:
                        h_rest = None

                    if layer > 0:
                        x = dropout.dropout(x, ratio=dropout_ratio,
                                            train=train)

                    rnn_in = (linear.linear(x, xws[layer_idx],
                                            xbs[layer_idx]) +
                              linear.linear(h, hws[layer_idx], hbs[layer_idx]))
                    if activation == 'tanh':
                        h_bar = tanh.tanh(rnn_in)
                    elif activation == 'relu':
                        h_bar = relu.relu(rnn_in)

                    if h_rest is not None:
                        h = concat.concat([h_bar, h_rest], axis=0)
                    else:
                        h = h_bar
                    h_list.append(h_bar)
                return h, h_list

            # Forward RNN
            h, h_forward = _one_directional_loop(di=0)
            hy.append(h)

            if use_bi_direction:
                # Backward RNN
                h, h_backward = _one_directional_loop(di=1)
                h_backward.reverse()
                # Concat
                xs_next = [concat.concat([hfi, hbi], axis=1) for (hfi, hbi) in
                           six.moves.zip(h_forward, h_backward)]
                hy.append(h)
            else:
                # Uni-directional RNN
                xs_next = h_forward

        ys = xs_next
        hy = stack.stack(hy)
        return hy, tuple(ys)
