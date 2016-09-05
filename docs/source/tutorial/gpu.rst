Using GPU(s) in Chainer
-----------------------

.. currentmodule:: chainer

In this section, you will learn about the following things:

* Relationship between Chainer and CuPy
* Basics of CuPy
* Single-GPU usage of Chainer
* Multi-GPU usage of model-parallel computing
* Multi-GPU usage of data-parallel computing

After reading this section, you will be able to:

* Use Chainer on a CUDA-enabled GPU
* Write model-parallel computing in Chainer
* Write data-parallel computing in Chainer


Relationship between Chainer and CuPy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   As of the release of v1.3.0, Chainer changes its GPU backend from `PyCUDA <http://mathema.tician.de/software/pycuda/>`_ to CuPy.
   CuPy covers all features of PyCUDA used by Chainer, though their interfaces are not compatible.

Chainer uses :ref:`CuPy <cupy_reference>` as its backend for GPU computation.
In particular, the :class:`cupy.ndarray` class is the GPU array implementation for Chainer.
CuPy supports a subset of features of NumPy with a compatible interface.
It enables us to write a common code for CPU and GPU.
It also supports PyCUDA-like user-defined kernel generation, which enables us to write fast implementations dedicated to GPU.

.. note::

   The :mod:`chainer.cuda` module imports many important symbols from CuPy.
   For example, the cupy namespace is referred as ``cuda.cupy`` in the Chainer code.
   Note that the :mod:`chainer.cuda` module can be imported even if CUDA is not installed.

Chainer uses a memory pool for GPU memory allocation.
As shown in the previous sections, Chainer constructs and destructs many arrays during learning and evaluating iterations.
It is not well suited for CUDA architecture, since memory allocation and release in CUDA (i.e. ``cudaMalloc`` and ``cudaFree`` functions) synchronize CPU and GPU computations, which hurts performance.
In order to avoid memory allocation and deallocation during the computation, Chainer uses CuPy's memory pool as the standard memory allocator.
Chainer changes the default allocator of CuPy to the memory pool, so user can use functions of CuPy directly without dealing with the memory allocator.


Basics of :class:`cupy.ndarray`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   CuPy does not require explicit initialization, so ``cuda.init()`` function is removed as of v1.3.0.

CuPy is a GPU array backend that implements a subset of NumPy interface.
The :class:`cupy.ndarray` class is in its core, which is a compatible GPU alternative of :class:`numpy.ndarray`.
CuPy implements many functions on :class:`cupy.ndarray` objects.
:ref:`See the reference for the supported subset of NumPy API <cupy_reference>`.
Understanding NumPy might help utilizing most features of CuPy.
`See the NumPy documentation for learning it <http://docs.scipy.org/doc/numpy/index.html>`_.

The main difference of :class:`cupy.ndarray` from :class:`numpy.ndarray` is that the content is allocated on the device memory.
The allocation takes place on the current device by default.
The current device can be changed by :class:`cupy.cuda.Device` object as follows:

.. testcode::

   with cupy.cuda.Device(1):
       x_on_gpu1 = cupy.array([1, 2, 3, 4, 5])

Most operations of CuPy is done on the current device.
Be careful that it causes an error to process an array on a non-current device.

Chainer provides some convenient functions to automatically switch and choose the device.
For example, the :func:`chainer.cuda.to_gpu` function copies a :class:`numpy.ndarray` object to a specified device:

.. testcode::

   x_cpu = np.ones((5, 4, 3), dtype=np.float32)
   x_gpu = cuda.to_gpu(x_cpu, device=1)

It is equivalent to the following code using CuPy:

.. testcode::

   x_cpu = np.ones((5, 4, 3), dtype=np.float32)
   with cupy.cuda.Device(1):
       x_gpu = cupy.array(x_cpu)

Moving a device array to the host can be done by :func:`chainer.cuda.to_cpu` as follows:

.. testcode::

   x_cpu = cuda.to_cpu(x_gpu)

It is equivalent to the following code using CuPy:

.. testcode::

   with x_gpu.device:
       x_cpu = x_gpu.get()

.. note::

   The *with* statements in these codes are required to select the appropriate CUDA device.
   If user uses only one device, these device switching is not needed.
   :func:`chainer.cuda.to_cpu` and :func:`chainer.cuda.to_gpu` functions automatically switch the current device correctly.

Chainer also provides a convenient function :func:`chainer.cuda.get_device` to select a device.
It accepts an integer, CuPy array, NumPy array, or None (indicating the current device), and returns an appropriate device object.
If the argument is a NumPy array, then *a dummy device object* is returned.
The dummy device object supports *with* statements like above which does nothing.
Here are some examples:

.. testcode::

   cuda.get_device(1).use()
   x_gpu1 = cupy.empty((4, 3), dtype='f')  # 'f' indicates float32

   with cuda.get_device(1):
       x_gpu1 = cuda.empty((4, 3), dtype='f')

   with cuda.get_device(x_gpu1):
       y_gpu1 = x_gpu + 1

Since it accepts NumPy arrays, we can write a function that accepts both NumPy and CuPy arrays with correct device switching:

.. testcode::

   def add1(x):
       with cuda.get_device(x):
           return x + 1

The compatibility of CuPy with NumPy enables us to write CPU/GPU generic code.
It can be made easy by the :func:`chainer.cuda.get_array_module` function.
This function returns the :mod:`numpy` or :mod:`cupy` module based on arguments.
A CPU/GPU generic function is defined using it like follows:

.. testcode::

   # Stable implementation of log(1 + exp(x))
   def softplus(x):
       xp = cuda.get_array_module(x)
       return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))


Run Neural Networks on a Single GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Single-GPU usage is very simple.
What you have to do is transferring :class:`Link` and input arrays to the GPU beforehand.
In this subsection, the code is based on :ref:`our first MNIST example in this tutorial <mnist_mlp_example>`.

A :class:`Link` object can be transferred to the specified GPU using the :meth:`~Link.to_gpu` method.

.. testcode::
   :hide:

   class MLP(Chain):
       def __init__(self, n_units, n_out):
           super(MLP, self).__init__(
               l1=L.Linear(None, n_units),
               l2=L.Linear(None, n_units),
               l3=L.Linear(None, n_out),
           )

       def __call__(self, x):
           h1 = F.relu(self.l1(x))
           h2 = F.relu(self.l2(h1))
           y = self.l3(h2)
           return y

   model = L.Classifier(MLP(1000, 10)).to_gpu()  # to_gpu returns itself
   optimizer = optimizers.SGD()
   optimizer.setup(model)

This time, we make the number of input, hidden, and output units configurable.
The :meth:`~Link.to_gpu` method also accepts a device ID like ``model.to_gpu(0)``.
In this case, the link object is transferred to the appropriate GPU device.
The current device is used by default.

If we use :class:`chainer.training.Trainer`, what we have to do is just let the updater know the device ID to send each mini-batch.

.. testcode::
   :hide:

   data = np.random.rand(70000, 784).astype(np.float32)
   target = np.random.randint(10, size=70000).astype(np.int32)
   train = datasets.TupleDataset(data[:60000], target[:60000])
   test = datasets.TupleDataset(data[60000:], target[60000:])
   train_iter = iterators.SerialIterator(train, batch_size=100)
   test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

.. testcode::

   updater = training.StandardUpdater(train_iter, optimizer, device=0)
   trainer = training.Trainer(updater, (20, 'epoch'), out='result')

We also have to specify the device ID for an evaluator extension as well.

.. testcode::

   trainer.extend(extensions.Evaluator(test_iter, model, device=0))

When we write down the training loop by hand, we have to transfer each mini-batch to the GPU manually:

.. testcode::
   :hide:

   x_train = np.random.rand(600, 784).astype(np.float32)
   y_train = np.random.randint(10, size=600).astype(np.int32)

.. testcode::

   model.to_gpu()
   batchsize = 100
   datasize = len(x_train)
   for epoch in range(20):
       print('epoch %d' % epoch)
       indexes = np.random.permutation(datasize)
       for i in range(0, datasize, batchsize):
           x = Variable(cuda.to_gpu(x_train[indexes[i : i + batchsize]]))
           t = Variable(cuda.to_gpu(y_train[indexes[i : i + batchsize]]))
           optimizer.update(model, x, t)

.. testoutput::
   :hide:

   epoch 0
   ...


Model-parallel Computation on Multiple GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parallelization of machine learning is roughly classified into two types called "model-parallel" and "data-parallel".
Model-parallel means parallelizations of the computations inside the model.
In contrast, data-parallel means parallelizations using data sharding.
In this subsection, we show how to use the model-parallel approach on multiple GPUs in Chainer.

:ref:`Recall the MNIST example <mnist_mlp_example>`.
Now suppose that we want to modify this example by expanding the network to 6 layers with 2000 units each using two GPUs.
In order to make multi-GPU computation efficient, we only make the two GPUs communicate at the third and sixth layer.
The overall architecture looks like the following diagram::

  (GPU0) input --+--> l1 --> l2 --> l3 --+--> l4 --> l5 --> l6 --+--> output
                 |                       |                       |
  (GPU1)         +--> l1 --> l2 --> l3 --+--> l4 --> l5 --> l6 --+

We can use the above MLP chain as following diagram::

  (GPU0) input --+--> mlp1 --+--> mlp2 --+--> output
                 |           |           |
  (GPU1)         +--> mlp1 --+--> mlp2 --+

Let's write a link for the whole network.

.. testcode::

   class ParallelMLP(Chain):
       def __init__(self):
           super(ParallelMLP, self).__init__(
               # the input size, 784, is inferred
               mlp1_gpu0=MLP(1000, 2000).to_gpu(0),
               mlp1_gpu1=MLP(1000, 2000).to_gpu(1),

               # the input size, 2000, is inferred
               mlp2_gpu0=MLP(1000, 10).to_gpu(0),
               mlp2_gpu1=MLP(1000, 10).to_gpu(1),
           )

       def __call__(self, x):
           # assume x is on GPU 0
           z0 = self.mlp1_gpu0(x)
           z1 = self.mlp1_gpu1(F.copy(x, 1))

           # sync
           h0 = F.relu(z0 + F.copy(z1, 0))
           h1 = F.relu(z1 + F.copy(z0, 1))

           y0 = self.mlp2_gpu0(h0)
           y1 = self.mlp2_gpu1(h1)

           # sync
           y = y0 + F.copy(y1, 0)
           return y  # output is on GPU0

Recall that the :meth:`Link.to_gpu` method returns the link itself.
The :func:`~chainer.functions.copy` function copies an input variable to specified GPU device and returns a new variable on the device.
The copy supports backprop, which just reversely transfers an output gradient to the input device.

.. note::

   Above code is not parallelized on CPU, but is parallelized on GPU.
   This is because all the functions in the above code run asynchronously to the host CPU.

An almost identical example code can be found at `examples/mnist/train_mnist_model_parallel.py <https://github.com/pfnet/chainer/blob/master/examples/mnist/train_mnist_model_parallel.py>`_.


Data-parallel Computation on Multiple GPUs with Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data-parallel computation is another strategy to parallelize online processing.
In the context of neural networks, it means that a different device does computation on a different subset of the input data.
In this subsection, we review the way to achieve data-parallel learning on two GPUs.

Suppose again our task is :ref:`the MNIST example <mnist_mlp_example>`.
This time we want to directly parallelize the three-layer network.
The most simple form of data-parallelization is parallelizing the gradient computation for a distinct set of data.
First, define a model and optimizer instances:

.. doctest::

   model = L.Classifier(MLP(1000, 10))  # the input size, 784, is inferred
   optimizer = optimizers.SGD()
   optimizer.setup(model)

Recall that the ``MLP`` link implements the multi-layer perceptron, and the :class:`~chainer.links.Classifier` link wraps it to provide a classifier interface.
We used :class:`~training.StandardUpdater` in the previous example.
In order to enable data-parallel computation with multiple GPUs, we only have to replace it with :class:`~training.ParallelUpdater`.

.. doctest::

   updater = training.ParallelUpdater(train_iter, optimizer,
                                      devices={'main': 0, 'second': 1})

The ``devices`` option specifies which devices to use in data-parallel learning.
The device with name ``'main'`` is used as the main device.
The original model is sent to this device, so the optimization runs on the main device.
In the above example, the model is also cloned and sent to GPU 1.
Half of each mini-batch is fed to this cloned model.
After every backward computation, the gradient is accumulated into the main device, the parameter update runs on it, and then the updated parameters are sent to GPU 1 again.

See also the example code in `examples/mnist/train_mnist_data_parallel.py <https://github.com/pfnet/chainer/blob/master/examples/mnist/train_mnist_data_parallel.py>`_.


Data-parallel Computation on Multiple GPUs without Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We here introduce a way to write data-parallel computation without the help of :class:`~training.Trainer`.
Most users can skip this section.
If you are interested in how to write a data-parallel computation by yourself, this section should be informative.
It is also helpful to, e.g., customize the :class:`~training.ParallelUpdater` class.

We again start from the MNIST example.
At this time, we use a suffix like ``_0`` and ``_1`` to distinguish objects on each device.
First, we define a model.

.. testcode::

   model_0 = L.Classifier(MLP(1000, 10))  # the input size, 784, is inferred

We want to make two copies of this instance on different GPUs.
The :meth:`Link.to_gpu` method runs in place, so we cannot use it to make a copy.
In order to make a copy, we can use :meth:`Link.copy` method.

.. testcode::

   import copy
   model_1 = copy.deepcopy(model_0)
   model_0.to_gpu(0)
   model_1.to_gpu(1)

The :meth:`Link.copy` method copies the link into another instance.
*It just copies the link hierarchy*, and does not copy the arrays it holds.

Then, set up an optimizer:

.. testcode::

   optimizer = optimizers.SGD()
   optimizer.setup(model_0)

Here we use the first copy of the model as *the master model*.
Before its update, gradients of ``model_1`` must be aggregated to those of ``model_0``.

Then, we can write a data-parallel learning loop as follows:

.. testcode::

   batchsize = 100
   datasize = len(x_train)
   for epoch in range(20):
       print('epoch %d' % epoch)
       indexes = np.random.permutation(datasize)
       for i in range(0, datasize, batchsize):
           x_batch = x_train[indexes[i : i + batchsize]]
           y_batch = y_train[indexes[i : i + batchsize]]

           x0 = Variable(cuda.to_gpu(x_batch[:batchsize//2], 0))
           t0 = Variable(cuda.to_gpu(y_batch[:batchsize//2], 0))
           x1 = Variable(cuda.to_gpu(x_batch[batchsize//2:], 1))
           t1 = Variable(cuda.to_gpu(y_batch[batchsize//2:], 1))

           loss_0 = model_0(x0, t0)
           loss_1 = model_1(x1, t1)

           model_0.cleargrads()
           model_1.cleargrads()

           loss_0.backward()
           loss_1.backward()

           model_0.addgrads(model_1)
           optimizer.update()

           model_1.copyparams(model_0)

.. testoutput::
   :hide:

   epoch 0
   ...

Do not forget to clear the gradients of both model copies!
One half of the mini-batch is forwarded to GPU 0, the other half to GPU 1.
Then the gradients are accumulated by the :meth:`Link.addgrads` method.
This method adds the gradients of a given link to those of the self.
After the gradients are prepared, we can update the optimizer in usual way.
Note that the update only modifies the parameters of ``model_0``.
So we must manually copy them to ``model_1`` using :meth:`Link.copyparams` method.

.. note::

   If the batchsize used in one model remain the same, the scale of the gradient
   is roughly proportional to the number of models, when we aggregate
   gradients from all models by :func:`chainer.Link.addgrads`. So you need to adjust the batchsize
   and/or learning rate of the optimizer accordingly.

--------

Now you can use Chainer with GPUs.
All examples in the ``examples`` directory support GPU computation, so please refer to them if you want to know more practices on using GPUs.
In the next section, we will show how to define a differentiable (i.e. *backpropable*) function on Variable objects.
We will also show there how to write a simple (elementwise) CUDA kernel using Chainer's CUDA utilities.
