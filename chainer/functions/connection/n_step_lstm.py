import numpy

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cudnn.cudnn


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

    @staticmethod
    def create(handle, dropout, seed):
        states = cudnn.create_dropout_states(handle)
        desc = cudnn.create_dropout_descriptor(
            handle, dropout, states.data.ptr, states.size, seed)
        return DropoutStates(states, desc)

    @staticmethod
    def from_states(handle, states, dropout):
        desc = cudnn.create_dropout_descriptor(handle, dropout, 0, 0, 0)
        return DropoutStates(states, desc)


def _make_indices(batches):
    pos = 0
    inds = []
    for b in batches[:-1]:
        pos += b
        inds.append(pos)
    return inds


def _fetch_input(inputs, pos, n):
    return inputs[pos:pos+n], pos + n


class NStepLSTM(function.Function):

    def __init__(self, n_layers, states, train=True):
        self.n_layers = n_layers
        self.train = train
        self.states = states

    def check_type_forward(self, in_types):
        h_type, c_type = in_types[:2]
        type_check.expect(
            h_type.dtype == numpy.float32,
            c_type.dtype == numpy.float32,

            h_type.ndim == 3,
            h_type.shape[0] == self.n_layers,
            c_type.ndim == 3,
            c_type.shape[0] == self.n_layers,

            # mini-batch size
            h_type.shape[1] == c_type.shape[1],

            # hidden size
            h_type.shape[2] == c_type.shape[2],
        )

    def forward(self, inputs):
        (hx, cx), pos = _fetch_input(inputs, 0, 2)
        ws, pos = _fetch_input(inputs, pos, self.n_layers * 8)
        bs, pos = _fetch_input(inputs, pos, self.n_layers * 8)
        x_list = inputs[pos:]
        x = x_list[0]
        x = x.reshape(x.shape + (1,))
        x_desc = cudnn.create_tensor_nd_descriptor(x)

        length = len(x_list)
        n_units = hx.shape[2]

        xs = cuda.cupy.concatenate(x_list, axis=0)
        ys = cuda.cupy.empty((len(xs), n_units), dtype=xs.dtype)

        # shape of h and c is (batch_size, n_layer, hidden_size) in Chainer
        # but (hidden_size, batch_size, n_layer) in cuDNN
        handle = cudnn.get_handle()
        self.handle = handle

        rnn_desc = cudnn.create_rnn_descriptor(
            n_units, self.n_layers, self.states.desc,
            libcudnn.CUDNN_LINEAR_INPUT, libcudnn.CUDNN_UNIDIRECTIONAL,
            libcudnn.CUDNN_LSTM, libcudnn.CUDNN_DATA_FLOAT)
        self.rnn_desc = rnn_desc

        c_x_descs = _make_tensor_descriptor_array(x_list)
        hx_desc = cudnn.create_tensor_nd_descriptor(hx)
        cx_desc = cudnn.create_tensor_nd_descriptor(cx)

        weights_size = libcudnn.getRNNParamsSize(
            handle, rnn_desc.value, x_desc.value, libcudnn.CUDNN_DATA_FLOAT)
        w = cuda.cupy.empty((weights_size // 4, 1, 1), dtype=numpy.float32)
        w_desc = cudnn.create_filter_descriptor(w)

        for layer in range(self.n_layers):
            for lin_layer_id in range(8):
                mat = cudnn.get_rnn_lin_layer_matrix_params(
                    handle, rnn_desc, layer, x_desc, w_desc, w,
                    lin_layer_id)
                m = mat.reshape(mat.size)
                m[...] = ws[layer * 8 + lin_layer_id].ravel()
                bias = cudnn.get_rnn_lin_layer_bias_params(
                    handle, rnn_desc, layer, x_desc, w_desc, w,
                    lin_layer_id)
                b = bias.reshape(bias.size)
                b[...] = bs[layer * 8 + lin_layer_id]
        self.w = w
        self.w_desc = w_desc

        lengths = [len(x) for x in x_list]
        indexes = _make_indices(lengths)
        y_list = cuda.cupy.split(ys, indexes)

        c_y_descs = _make_tensor_descriptor_array(y_list)
        hy = cuda.cupy.empty_like(hx)
        cy = cuda.cupy.empty_like(cx)
        hy_desc = cudnn.create_tensor_nd_descriptor(hy)
        cy_desc = cudnn.create_tensor_nd_descriptor(cy)

        work_size = libcudnn.getRNNWorkspaceSize(
            handle, rnn_desc.value, length, c_x_descs.data)
        workspace = cuda.cupy.empty((work_size,), dtype='b')
        self.workspace = workspace

        if not self.train:
            libcudnn.RNNForwardInference(
                handle, rnn_desc.value, length,
                c_x_descs.data, xs.data.ptr, hx_desc.value, hx.data.ptr,
                cx_desc.value, cx.data.ptr, w_desc.value, w.data.ptr,
                c_y_descs.data, ys.data.ptr, hy_desc.value, hy.data.ptr,
                cy_desc.value, cy.data.ptr, workspace.data.ptr, work_size)

        else:
            reserve_size = libcudnn.getRNNTrainingReserveSize(
                handle, rnn_desc.value, length, c_x_descs.data)
            self.reserve_space = cuda.cupy.empty((reserve_size,), dtype='b')
            libcudnn.RNNForwardTraining(
                handle, rnn_desc.value, length,
                c_x_descs.data, xs.data.ptr, hx_desc.value, hx.data.ptr,
                cx_desc.value, cx.data.ptr, w_desc.value, w.data.ptr,
                c_y_descs.data, ys.data.ptr, hy_desc.value, hy.data.ptr,
                cy_desc.value, cy.data.ptr,
                workspace.data.ptr, work_size,
                self.reserve_space.data.ptr, reserve_size)

        self.c_y_descs = c_y_descs
        self.ys = ys
        self.c_x_descs = c_x_descs

        ret = [hy, cy]
        ret.extend(y_list)
        return tuple(ret)

    def backward(self, inputs, grads):
        (hx, cx), pos = _fetch_input(inputs, 0, 2)
        ws, pos = _fetch_input(inputs, pos, self.n_layers * 8)
        bs, pos = _fetch_input(inputs, pos, self.n_layers * 8)
        x_list = inputs[pos:]
        lengths = [len(x) for x in x_list]
        indexes = _make_indices(lengths)

        dhy, dcy = grads[:2]
        dy_list = grads[2:]
        if dhy is None:
            dhy = cuda.cupy.zeros_like(hx)
        if dcy is None:
            dcy = cuda.cupy.zeros_like(cx)
        for i in range(len(dy_list)):
            if dy_list[i] is None:
                dy_list[i] = cuda.cupy.zeros_like(x_list[i])

        xs = cuda.cupy.concatenate(x_list, axis=0)
        length = len(x_list)

        dhx = cuda.cupy.empty_like(hx)
        dcx = cuda.cupy.empty_like(cx)

        hx_desc = cudnn.create_tensor_nd_descriptor(hx)
        cx_desc = cudnn.create_tensor_nd_descriptor(cx)
        dhy_desc = cudnn.create_tensor_nd_descriptor(dhy)
        dcy_desc = cudnn.create_tensor_nd_descriptor(dcy)

        c_dy_descs = _make_tensor_descriptor_array(dy_list)
        dys = cuda.cupy.concatenate(dys, axis=0)

        rnn_desc = self.rnn_desc
        handle = self.handle
        work_size = libcudnn.getRNNWorkspaceSize(
            handle, rnn_desc.value, length, self.c_x_descs.data)
        workspace = cuda.cupy.empty((work_size,), dtype='b')

        dhx_desc = cudnn.create_tensor_nd_descriptor(dhx)
        dcx_desc = cudnn.create_tensor_nd_descriptor(dcx)

        dxs = cuda.cupy.empty_like(xs)
        dx_list = cuda.cupy.split(dxs, indexes, 0)
        c_dx_descs = _make_tensor_descriptor_array(dx_list)

        libcudnn.RNNBackwardData(
            handle, rnn_desc.value, length,
            self.c_y_descs.data, self.ys.data.ptr,
            c_dy_descs.data, dys.data.ptr, dhy_desc.value, dhy.data.ptr,
            dcy_desc.value, dcy.data.ptr, self.w_desc.value, self.w.data.ptr,
            hx_desc.value, hx.data.ptr, cx_desc.value, cx.data.ptr,
            c_dx_descs.data, dxs.data.ptr, dhx_desc.value, dhx.data.ptr,
            dcx_desc.value, dcx.data.ptr, workspace.data.ptr, work_size,
            self.reserve_space.data.ptr, self.reserve_space.size)

        dw = cuda.cupy.zeros_like(self.w)
        dw_desc = cudnn.create_tensor_nd_descriptor(dw)
        libcudnn.RNNBackwardWeights(
            handle, rnn_desc.value, length,
            self.c_x_descs.data, xs.data.ptr,
            hx_desc.value, hx.data.ptr, self.c_y_descs.data, self.ys.data.ptr,
            workspace.data.ptr, work_size, dw_desc.value, dw.data.ptr,
            self.reserve_space.data.ptr, self.reserve_space.size)

        dx = dx_list[0]
        dx = dx.reshape(dx.shape + (1,))
        dx_desc = cudnn.create_tensor_nd_descriptor(dx)
        dws = [cuda.cupy.empty_like(w) for w in ws]
        dbs = [cuda.cupy.empty_like(b) for b in bs]
        for layer in range(self.n_layers):
            for lin_layer_id in range(8):
                mat = cudnn.get_rnn_lin_layer_matrix_params(
                    handle, rnn_desc, layer, dx_desc, dw_desc, dw,
                    lin_layer_id)
                v = dws[layer * 8 + lin_layer_id]
                v = v.reshape(v.size)
                v[:] = mat.ravel()
                bias = cudnn.get_rnn_lin_layer_bias_params(
                    handle, rnn_desc, layer, dx_desc, dw_desc, dw,
                    lin_layer_id)
                v = dbs[layer * 8 + lin_layer_id]
                v = v.reshape(v.size)
                v[:] = bias.ravel()

        return tuple([dhx, dcx] + dws + dbs + dx_list)

