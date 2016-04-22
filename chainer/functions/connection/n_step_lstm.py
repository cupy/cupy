import numpy

from chainer import cuda
from chainer import function


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cudnn.cudnn


class PointerArray(object):

    def __init__(lst, back_pointer):
        self._value = numpy.array(l, dtype=numpy.intp).ctypes
        # Store back_pointer to prevent the GC removes the original variable
        self._back_pointer = back_pointer

    @property
    def data(self):
        return self._value.ctypes.data


def _make_tensor_descriptor_array(xs):
    """Make an array of pointers denoting pointers of tensor descriptors.
    """
    descs = [cudnn.create_tensor_nd_descriptor(x) for x in xs]
    return PointerArray([d.value for d in descs], descs)


def make_ptr_array(xs):
    """Make an array of pointers denoting pointers of ndarrays.
    """
    return PointerArray([x.data.ptr for x in xs], xs)


class NStepLSTM(function.Function):

    def __init__(self):
        self.seed = 1337
        self.dropout = 0.5
        self.n_layers = 3

    def check_type_forward(self, in_types):
        h_type, c_type, x_type = in_types
        type_check.expect(
            h_type.dtype == numpy.float32,
            c_type.dtype == numpy.float32,
            x_type.dtype == numpy.float32,

            h_type.ndim == 3,
            h_type.shape[1] == self.n_layers,
            c_type.ndim == 3,
            c_type.shape[1] == self.n_layers,

            # mini-batch size
            h_type.shape[0] == c_types.shape[0],
            h_type.shape[0] == x_types.shape[0],

            # hidden size
            h_type.shape[2] == c_types.shape[2],
        )

    def forward(self, inputs):
        hx, cx, xs = inputs
        length = xs.shape[0]
        n_units = hx.shape[2]

        # shape of h and c is (batch_size, n_layer, hidden_size) in Chainer
        # but (hidden_size, batch_size, n_layer) in cuDNN
        hx = cuda.cupy.rollaxis(hx, 2)
        cx = cuda.cupy.rollaxis(cx, 2)

        handle = cudnn.get_handle()

        state_size = libcudnn.dropoutGetStatesSize(handle)
        self.states = cuda.cupy.empty((state_size,), dtype='b')

        dropout_desc = cudnn.create_dropout_descriptor(
            handle, self.dropout, self.states.data.ptr, state_size, self.seed)

        rnn_desc = cudnn.create_rnn_descriptor(
            n_units, length, self.n_layers, dropout_desc,
            libcudnn.CUDNN_LINEAR_INPUT, libcudnn.CUDNN_UNIDIRECTIONAL,
            libcudnn.CUDNN_LSTM, libcudnn.CUDNN_DATA_FLOAT)

        c_x_descs = _make_tensor_descriptor_array(xs)
        hx_desc = cudnn.create_tensor_nd_descriptor(hx)
        cx_desc = cudnn.create_tensor_nd_descriptor(cx)

        weights_size = libcudnn.getRNNParamsSize(
            handle, rnn_desc.value, c_x_descs.data)
        w = cuda.cupy.empty((weights_size // 4, 1, 1), dtype=numpy.float32)
        w_desc = cudnn.create_filter_descriptor(w)

        for layer in range(self.n_layers):
            for lin_layer_id in range(8):
                mat = cudnn.get_rnn_lin_layer_matrix_params(
                    handle, rnn_desc.value, layer, c_x_descs, w_desc, w,
                    lin_layer_id)
                # TODO(unno): set mat

        ys = cuda.cupy.empty((len(xs),) + hx.shape, dtype=numpy.float32)
        hy = cuda.cupy.empty_like(hx)
        cy = cuda.cupy.empty_like(cx)

        c_y_descs = _make_tensor_descriptor_array(ys)
        hy_desc = cudnn.create_tensor_nd_descriptor(hy)
        cy_desc = cudnn.create_tensor_nd_descriptor(cy)

        work_size = libcudnn.getRNNWorkspaceSize(
            handle, rnn_desc.value, c_x_descs.data)
        workspace = cuda.cupy.empty((work_size,), dtype='b')

        c_x = _make_ptr_array(xs)
        c_y = _make_ptr_array(ys)

        if x.volatile == 'on':
            libcudnn.RNNForwardInference(
                handle, rnn_desc.value,
                c_x_descs.data, c_x.data, hx_desc.value, hx.data.ptr,
                cx_desc.value, cx.data.ptr, w_desc.value, w.data.ptr,
                c_y_descs.data, c_y.data, hy_desc.value, hy.data.ptr,
                cy_desc.value, cy.data.ptr, workspace.data.ptr, work_size)

        else:
            reserve_size = libcudnn.getRNNTrainingReserveSize(
                handle, rnn_desc, c_x_descs.data)
            self.reserve_space = cuda.cupy.empty((reserve_size,), dtype='b')
            libcudnn.RNNForwardTraining(
                handle, rnn_desc.value,
                c_x_descs.data, c_x.data, hx_desc.value, hx.data.ptr,
                cx_desc.value, cx.data.ptr, w_desc.value, w.data.ptr,
                c_y_descs.data, c_y.data, hy_desc.value, hy.data.ptr,
                cy_desc.value, cy.data.ptr, workspace.data.ptr, work_size,
                self.reserve_space.data.ptr, reserve_size)

        return hy, cy, ys

    def backward(self, inputs, grads):
        hx, cx, xs = inputs
        dhy, dcy, dys = grads

        hx_desc = cudnn.create_tensor_nd_descriptor(hx)
        cx_desc = cudnn.create_tensor_nd_descriptor(cx)
        dhy_desc = cudnn.create_tensor_nd_descriptor(dhy)
        dcy_desc = cudnn.create_tensor_nd_descriptor(dcy)
        c_dy_descs = _make_tensor_desciptor_array(dys)
        c_dy = make_ptr_array(dys)

        length = hx.shape[0]
        n_units = hx.shape[2]

        work_size = libcudnn.getRNNWorkspaceSize(
            handle, rnn_desc.value, c_x_descs.data)
        workspace = cuda.cupy.empty((work_size,), dtype='b')

        dhx = cuda.cupy.empty_like(hx)
        dcx = cuda.cupy.empty_like(cs)
        dxs = cuda.cupy.empty_like(xs)
        dhx_desc = cudnn.create_tensor_nd_descriptor(dhx)
        dcx_desc = cudnn.create_tensor_nd_descriptor(dcx)
        c_dx_descs = _make_tensor_descriptor_array(dxs)
        c_dx = make_ptr_array(dxs)

        libcudnn.RNNBackwardData(
            handle, rnn_desc, c_y_descs.data, c_y.data,
            c_dy_descs.data, c_dy.data, dhy_desc.value, dhy.data.ptr,
            dcy_desc.value, dcy.data.ptr, w_desc.value, w.data.ptr,
            hy_desc.value, hx.data.ptr, cy_desc.value, cx.data.ptr,
            c_dx_descs.data, c_dx.data, dhx_desc.value, dhx.data.ptr,
            dcx_desc.value, dcx.data.ptr, workspace.data.ptr, work_size,
            self.reserve_space.data.ptr, self.reserve_space.size)

        dw = cuda.cupy.emtpy_like(w)
        libcudnn.RNNBackwardWeights(
            handle, rnn_desc, c_x_descs.data, c_x.data,
            hx_desc.value, hx.data.ptr, c_y_descs.data, c_y.data,
            workspace.data.ptr, work_size, dw.data.ptr,
            reserve_space.data.ptr, reserve_size)
        
        return dhx, dcx, dxs
