import copy
import multiprocessing

import six

from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import optimizer as optimizer_module
from chainer import variable
from chainer import cuda
import chainer
from cupy.cuda import nccl
from chainer.training.updater import StandardUpdater, _calc_loss

import os


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
        print("setup %d" % self.device)
        _, comm_id = self.pipe.recv()
        self.comm = nccl.NcclCommunicator(self.n_devices, comm_id,
                                          self.proc_id)

        self.model.to_gpu(self.device)
        self.reporter = chainer.reporter.Reporter()
        self.reporter.add_observer('main', self.model)
        print("%d setup done" % self.device)

    def run(self):
        dev = cuda.Device(self.device)
        dev.use()
        self.setup()
        pp = None
        while True:
            job, data = self.pipe.recv()
            if job == 'finalize':
                dev.synchronize()
                # chainer.serializers.save_npz('dump_model_{}'.format(self.device), self.model, compression=False)
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

                gg = self.model.gather_grads()
                null_stream = cuda.Stream.null
                self.comm.reduce(gg.data.ptr, gg.data.ptr, gg.size,
                                 nccl.NCCL_FLOAT, nccl.NCCL_SUM, 0, null_stream.ptr)
                if pp is None:
                    pp = self.model.gather_params()
                self.comm.bcast(pp.data.ptr, pp.size, nccl.NCCL_FLOAT, 0, null_stream.ptr)
                self.model.scatter_params(pp)

                # Sending observation via pipe is too slow.
                # self.pipe.send(observation)


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

    Args:
        iterators: List of dataset iterator for the training dataset. The number
            of the iterators must be same to the number of GPUs you use.
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

        assert len(iterators) == len(devices)
        for iterator in iterators[1:]:
            assert len(iterator.dataset) == len(iterators[0].dataset)

        # Correct optimizer parameters for new minibatch size
        optim = optimizer.__class__.__name__
        if optim in ('Adam', 'AdaGrad', 'RMSprop'):
            optimizer.eps *= len(devices)
        elif optim in ('RMSpropGraves', 'AdaDelta'):
            optimizer.eps *= len(devices) ** 2  # not quite right for AdaDelta
        else:
            optimizer.lr /= len(devices)

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
            comm_id = nccl.get_unique_id()
            self._send_message(("set comm_di", comm_id))
            self.comm = nccl.NcclCommunicator(len(self._devices), comm_id, 0)

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
            gg = self._master.gather_grads()
            null_stream = cuda.Stream.null
            self.comm.reduce(gg.data.ptr, gg.data.ptr, gg.size,
                             nccl.NCCL_FLOAT, nccl.NCCL_SUM,
                             0, null_stream.ptr)
            self._master.scatter_grads(gg)
            optimizer.update()
            pp = self._master.gather_params()
            self.comm.bcast(pp.data.ptr, pp.size, nccl.NCCL_FLOAT,
                            0, null_stream.ptr)

            # Sending observation via pipe is too slow.
            # for pipe in self._pipes:
            #     chainer.reporter.report(pipe.recv())

    def finalize(self):
        self._send_message(('finalize', None))

        for worker in self._workers:
            print("join", worker)
            worker.join()

            # chainer.serializers.save_npz('dump_model_root', self._master, compression=False)
