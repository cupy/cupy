import multiprocessing
import warnings

import six

import chainer
from chainer import cuda
from chainer.dataset import convert
from chainer.training.updater import StandardUpdater
from chainer import variable

try:
    from cupy.cuda import nccl
    _available = True
except ImportError:
    _available = False

import numpy


class _Worker(multiprocessing.Process):

    def __init__(self, proc_id, pipe, master):
        super(_Worker, self).__init__()
        self.proc_id = proc_id
        self.pipe = pipe
        self.converter = master.converter
        self.model = master._master
        self.device = master._devices[proc_id]
        self.iterator = master._mpu_iterators[proc_id]
        self.n_devices = len(master._devices)

    def setup(self):
        _, comm_id = self.pipe.recv()
        self.comm = nccl.NcclCommunicator(self.n_devices, comm_id,
                                          self.proc_id)

        self.model.to_gpu(self.device)
        self.reporter = chainer.reporter.Reporter()
        self.reporter.add_observer('main', self.model)

    def run(self):
        dev = cuda.Device(self.device)
        dev.use()
        self.setup()
        gp = None
        while True:
            job, data = self.pipe.recv()
            if job == 'finalize':
                dev.synchronize()
                break
            if job == 'update':
                # For reducing memory
                self.model.cleargrads()

                batch = self.converter(self.iterator.next(), self.device)
                observation = {}
                with self.reporter.scope(observation):
                    loss = _calc_loss(self.model, batch)

                self.model.cleargrads()
                loss.backward()

                del loss

                gg = gather_grads(self.model)
                null_stream = cuda.Stream.null
                self.comm.reduce(gg.data.ptr, gg.data.ptr, gg.size,
                                 nccl.NCCL_FLOAT, nccl.NCCL_SUM, 0,
                                 null_stream.ptr)
                del gg
                self.model.cleargrads()
                gp = gather_params(self.model)
                self.comm.bcast(gp.data.ptr, gp.size, nccl.NCCL_FLOAT, 0,
                                null_stream.ptr)
                scatter_params(self.model, gp)
                gp = None


class MultiprocessParallelUpdater(StandardUpdater):

    """Implementation of a multiprocess parallel GPU Updater.

    This is an implementation of :class:`Updater` that uses multiple GPUs
    with multi-process data parallelism. It uses Nvidia NCCL for communication
    between multiple GPUs.

    It behaves similarly to :class:`~chainer.training.StandardUpdater`. The
    update routine is modified to support data-parallel computation on multiple
    GPUs in one machine. It is based on synchronous parallel SGD: it
    parallelizes the gradient computation over a mini-batch, and updates the
    parameters only in the main device.

    It does not transfer the values collected by :class:`Reporter` in the sub
    devices to the main device. So you can only see the reported values in
    the main device.

    Args:
        iterators: List of dataset iterator for the training dataset. The
            number of the iterators must be same to the number of GPUs you use.
        optimizer: Optimizer to update parameters. The model should be attached
            to the optimizer.
        converter: Converter function to build input arrays. Each batch
            extracted by the iterator is split equally between the devices and
            then passed with corresponding ``device`` option to this function.
            :func:`~chainer.dataset.concat_examples` is used by default.
        devices: Dictionary or list of devices to which the training data is
            sent. The master device will be the first one in the list or the
            value attached to the key ``'main'``.

    """

    def __init__(self, iterators, optimizer, converter=convert.concat_examples,
                 devices=None):
        if not MultiprocessParallelUpdater.available():
            raise Exception(
                'NCCL is not enabled. MultiprocessParallelUpdater '
                'requires NCCL.\n'
                'Please reinstall chainer after you install NCCL.\n'
                '(see https://github.com/pfnet/chainer#installation).')

        assert len(iterators) == len(devices)
        for iterator in iterators[1:]:
            assert len(iterator.dataset) == len(iterators[0].dataset)

        # Correct optimizer parameters for new minibatch size
        optim = optimizer.__class__.__name__
        if optim in ('Adam', 'AdaGrad', 'RMSprop'):
            optimizer.eps *= len(devices)
            warnings.warn('optimizer.eps is changed to {} '
                          'by MultiprocessParallelUpdater for new batch size.'.
                          format(optimizer.eps))
        elif optim in ('RMSpropGraves', 'AdaDelta'):
            optimizer.eps *= len(devices) ** 2  # not quite right for AdaDelta
            warnings.warn('optimizer.eps is changed to {} '
                          'by MultiprocessParallelUpdater for new batch size.'.
                          format(optimizer.eps))
        elif hasattr(optimizer, 'lr'):
            optimizer.lr /= len(devices)
            warnings.warn('optimizer.lr is changed to {} '
                          'by MultiprocessParallelUpdater for new batch size.'.
                          format(optimizer.lr))

        super(MultiprocessParallelUpdater, self).__init__(
            iterator=iterators[0],
            optimizer=optimizer,
            converter=converter
        )

        if isinstance(devices, dict):
            main = devices.pop('main')
            devices = list(six.itervalues(devices))
            devices = [main] + devices
        if devices is None or any(device is None for device in devices):
            raise ValueError('must specify GPU devices')

        self._master = optimizer.target
        self._devices = devices
        self._mpu_iterators = iterators
        self._initialized = False

        self._pipes = []
        self._workers = []
        self.comm = None

    @staticmethod
    def available():
        return _available

    def _send_message(self, message):
        for pipe in self._pipes:
            pipe.send(message)

    def setup_workers(self):
        if self._initialized:
            return
        self._initialized = True

        self._master.cleargrads()
        for i in six.moves.range(1, len(self._devices)):
            pipe, worker_end = multiprocessing.Pipe()
            worker = _Worker(i, worker_end, self)
            worker.start()
            self._workers.append(worker)
            self._pipes.append(pipe)

        with cuda.Device(self._devices[0]):
            self._master.to_gpu(self._devices[0])
            if len(self._devices) > 1:
                comm_id = nccl.get_unique_id()
                self._send_message(("set comm_id", comm_id))
                self.comm = nccl.NcclCommunicator(len(self._devices),
                                                  comm_id, 0)

    def update_core(self):
        self.setup_workers()

        self._send_message(('update', None))
        with cuda.Device(self._devices[0]):
            # For reducing memory
            self._master.cleargrads()

            optimizer = self.get_optimizer('main')
            batch = self.get_iterator('main').next()
            batch = self.converter(batch, self._devices[0])

            loss = _calc_loss(self._master, batch)

            self._master.cleargrads()
            loss.backward()

            # NCCL: reduce grads
            null_stream = cuda.Stream.null
            if self.comm is not None:
                gg = gather_grads(self._master)
                self.comm.reduce(gg.data.ptr, gg.data.ptr, gg.size,
                                 nccl.NCCL_FLOAT, nccl.NCCL_SUM,
                                 0, null_stream.ptr)
                scatter_grads(self._master, gg)
                del gg
            optimizer.update()
            if self.comm is not None:
                gp = gather_params(self._master)
                self.comm.bcast(gp.data.ptr, gp.size, nccl.NCCL_FLOAT,
                                0, null_stream.ptr)

    def finalize(self):
        self._send_message(('finalize', None))

        for worker in self._workers:
            worker.join()


def _calc_loss(model, in_arrays):
    if isinstance(in_arrays, tuple):
        in_vars = tuple(variable.Variable(x) for x in in_arrays)
        return model(*in_vars)
    elif isinstance(in_arrays, dict):
        in_vars = {key: variable.Variable(x)
                   for key, x in six.iteritems(in_arrays)}
        return model(**in_vars)
    else:
        in_vars = variable.Variable(in_arrays)
        return model(in_vars)


def size_num_grads(link):
    """Count total size of all gradient arrays of a given link

    Args:
        link (chainer.link.Link): Target link object.
    """
    size = 0
    num = 0
    for param in link.params():
        if param.size == 0:
            continue
        size += param.size
        num += 1
    return size, num


def _batch_memcpy():
    return cuda.cupy.ElementwiseKernel(
        'raw T ptrs, raw X info',
        'raw float32 dst',
        '''
            int id_min = id_pre;
            int id_max = num_src;
            while (id_max - id_min > 1) {
                int id = (id_max + id_min) / 2;
                if (i < info[id]) id_max = id;
                else              id_min = id;
            }
            int id = id_min;
            float *src = (float *)(ptrs[id]);
            int i_dst = i;
            int i_src = i;
            if (id > 0) i_src -= info[id];
            dst[i_dst] = 0;
            if (src != NULL) {
                dst[i_dst] = src[i_src];
            }
            id_pre = id;
        ''',
        'batch_memcpy',
        loop_prep='''
                int num_src = info[0];
                int id_pre = 0;
            ''')


def gather_grads(link):
    """Put together all gradient arrays and make a single array

    Args:
        link (chainer.link.Link): Target link object.
    Return:
        cupy.ndarray
    """
    if link.xp is numpy:
        raise RuntimeError('gather_grads works only on GPU.')

    size, num = size_num_grads(link)

    ptrs = numpy.empty(num, dtype=numpy.uint64)
    info = numpy.empty(num + 1, dtype=numpy.int32)
    info[0] = 0
    i = 0
    for param in link.params():
        if param.size == 0:
            continue
        ptrs[i] = 0  # NULL pointer
        if param.grad is not None:
            ptrs[i] = param.grad.data.ptr
        info[i + 1] = info[i] + param.size
        i += 1
    info[0] = num

    ptrs = cuda.to_gpu(ptrs, stream=cuda.Stream.null)
    info = cuda.to_gpu(info, stream=cuda.Stream.null)

    return _batch_memcpy()(ptrs, info, size=size)


def gather_params(link):
    """Put together all gradient arrays and make a single array

    Args:
        link (chainer.link.Link): Target link object.
    Return:
        cupy.ndarray
    """
    if link.xp is numpy:
        raise RuntimeError('Link.gather_params works only on GPU.')

    size, num = size_num_grads(link)

    ptrs = numpy.empty(num, dtype=numpy.uint64)
    info = numpy.empty(num + 1, dtype=numpy.int32)
    info[0] = 0
    i = 0
    for param in link.params():
        if param.size == 0:
            continue
        ptrs[i] = 0  # NULL pointer
        if param.data is not None:
            ptrs[i] = param.data.data.ptr
        info[i + 1] = info[i] + param.size
        i += 1
    info[0] = num

    ptrs = cuda.to_gpu(ptrs, stream=cuda.Stream.null)
    info = cuda.to_gpu(info, stream=cuda.Stream.null)

    return _batch_memcpy()(ptrs, info, size=size)


def scatter_grads(link, array):
    """Put back contents of the specified array to the related gradient arrays

    Args:
        link (chainer.link.Link): Target link object.
        array (cupy.ndarray): gathered array created by gather_grads()
    """
    offset = 0
    for param in link.params():
        next_offset = offset + param.size
        param.grad = array[offset:next_offset].reshape(param.data.shape)
        offset = next_offset


def scatter_params(link, array):
    """Put back contents of the specified array to the related gradient arrays

    Args:
        link (chainer.link.Link): Target link object.
        array (cupy.ndarray): gathered array created by gather_params()
    """
    offset = 0
    for param in link.params():
        next_offset = offset + param.size
        param.data = array[offset:next_offset].reshape(param.data.shape)
        offset = next_offset
