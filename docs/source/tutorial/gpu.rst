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


Basics of cupy.ndarray
~~~~~~~~~~~~~~~~~~~~~~

.. note::

   CuPy does not require explicit initialization, so ``cuda.init()`` function is removed as of v1.3.0.

CuPy is a GPU array backend that implements a subset of NumPy interface.
The :class:`cupy.ndarray` class is in its core, which is a compatible GPU alternative of :class:`numpy.ndarray`.
CuPy implements many functions on cupy.ndarray objects.
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
What you have to do is transferring :class:`FunctionSet` and input arrays to the GPU beforehand.
In this subsection, the code is based on :ref:`our first MNIST example in this tutorial <mnist_mlp_example>`.

A :class:`FunctionSet` object can be transferred to the specified GPU using the :meth:`~FunctionSet.to_gpu` method.
Make sure to give parameters and gradients of the GPU version to the optimizer. :

.. testcode::

   model = FunctionSet(
       l1 = F.Linear(784, 100),
       l2 = F.Linear(100, 100),
       l3 = F.Linear(100,  10),
   ).to_gpu()

   optimizer = optimizers.SGD()
   optimizer.setup(model)

Note that this method returns the :class:`FunctionSet` itself.
The device specifier can be omitted, in which case it uses the current device.

Then, all we have to do is transferring each minibatch to the GPU:

.. testcode::
   :hide:

   x_train = np.random.rand(600, 784).astype(np.float32)
   y_train = np.random.randint(10, size=600).astype(np.int32)

   def forward(x_data, y_data):
      x = Variable(x_data)
      t = Variable(y_data)
      y = model.l3(model.l1(x))
      return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

.. testcode::

   batchsize = 100
   datasize = len(x_train)
   for epoch in range(20):
       print('epoch %d' % epoch)
       indexes = np.random.permutation(datasize)
       for i in range(0, datasize, batchsize):
           x_batch = cuda.to_gpu(x_train[indexes[i : i + batchsize]])
           y_batch = cuda.to_gpu(y_train[indexes[i : i + batchsize]])

           optimizer.zero_grads()
           loss, accuracy = forward(x_batch, y_batch)
           loss.backward()
           optimizer.update()

.. testoutput::
   :hide:

   epoch 0
   ...

This is almost identical to the code of the original example,
we just inserted a call to the :func:`cuda.to_gpu` function to the minibatch arrays.


Model-parallel Computation on Multiple GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parallelization of machine learning is roughly classified into two types called "model-parallel" and "data-parallel".
Model-parallel means parallelizations of the computations inside the model.
In contrast, data-parallel means parallelizations using data sharding.
In this subsection, we show how to use the model-parallel approach on multiple GPUs in Chainer.

`Recall the MNIST example <mnist_mlp_example>`_.
Now suppose that we want to modify this example by expanding the network to 6 layers with 2000 units each using two GPUs.
In order to make multi-GPU computation efficient, we only make the two GPUs communicate at the third and sixth layer.
The overall architecture looks like the following diagram::

  (GPU0) input --+--> l1 --> l2 --> l3 --+--> l4 --> l5 --> l6 --+--> output
                 |                       |                       |
  (GPU1)         +--> l1 --> l2 --> l3 --+--> l4 --> l5 --> l6 --+

We first have to define a :class:`FunctionSet`.
Be careful that parameters that will be used on a device must reside on that device.
Here is a simple example of the model definition:

.. testcode::

   model = FunctionSet(
       gpu0 = FunctionSet(
           l1=F.Linear( 784, 1000),
           l2=F.Linear(1000, 1000),
           l3=F.Linear(1000, 2000),
           l4=F.Linear(2000, 1000),
           l5=F.Linear(1000, 1000),
           l6=F.Linear(1000,   10)
       ).to_gpu(0),
       gpu1 = FunctionSet(
           l1=F.Linear( 784, 1000),
           l2=F.Linear(1000, 1000),
           l3=F.Linear(1000, 2000),
           l4=F.Linear(2000, 1000),
           l5=F.Linear(1000, 1000),
           l6=F.Linear(1000,   10)
       ).to_gpu(1)
   )

Recall that :meth:`FunctionSet.to_gpu` returns the FunctionSet object itself.
Note that FunctionSet can be nested as above.

Now we can define the network architecture that we have shown in the diagram:

.. testcode::

   def forward(x_data, y_data):
       x_0 = Variable(cuda.to_gpu(x_data, 0))
       x_1 = Variable(cuda.to_gpu(x_data, 1))
       t   = Variable(cuda.to_gpu(y_data, 0))

       h1_0 = F.relu(model.gpu0.l1(x_0))
       h1_1 = F.relu(model.gpu1.l1(x_1))

       h2_0 = F.relu(model.gpu0.l2(h1_0))
       h2_1 = F.relu(model.gpu1.l2(h1_1))

       h3_0 = F.relu(model.gpu0.l3(h2_0))
       h3_1 = F.relu(model.gpu1.l3(h2_1))

       # Synchronize
       h3_0 += F.copy(h3_1, 0)
       h3_1  = F.copy(h3_0, 1)

       h4_0 = F.relu(model.gpu0.l4(h3_0))
       h4_1 = F.relu(model.gpu1.l4(h3_1))

       h5_0 = F.relu(model.gpu0.l5(h4_0))
       h5_1 = F.relu(model.gpu1.l5(h4_1))

       h6_0 = F.relu(model.gpu0.l6(h5_0))
       h6_1 = F.relu(model.gpu1.l6(h5_1))

       # Synchronize
       y = h6_0 + F.copy(h6_1, 0)
       return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

First, recall that :func:`cuda.to_gpu` accepts an optional argument to specify the device identifier.
We use this to transfer the input minibatch to both the 0th and the 1st devices.
Then, we can write this model-parallel example employing the :func:`functions.copy` function.
This function transfers an input array to another device.
Since it is a function on :class:`Variable`, the operation supports backprop, which reversely transfers an output gradient to the input device.

.. note::

   Above code is not parallelized on CPU, but is parallelized on GPU.
   This is because most of the GPU computation is asynchronous to the host CPU.

An almost identical example code can be found at ``examples/mnist/train_mnist_model_parallel.py``.


Data-parallel Computation on Multiple GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data-parallel computation is another strategy to parallelize online processing.
In the context of neural networks, it means that a different device does computation on a different subset of the input data.
In this subsection, we review the way to achieve data-parallel learning on two GPUs.

Suppose again our task is `the MNIST example <mnist_mlp_example>`_.
This time we want to directly parallelize the three-layer network.
The most simple form of data-parallelization is parallelizing the gradient computation for a distinct set of data.
First, define the model:

.. testcode::

   model = FunctionSet(
       l1 = F.Linear(784, 100),
       l2 = F.Linear(100, 100),
       l3 = F.Linear(100,  10),
   )

We have to copy this model into two different devices.
This is done by using :func:`copy.deepcopy` and :meth:`FunctionSet.to_gpu` method:

.. testcode::

   import copy
   model_0 = copy.deepcopy(model).to_gpu(0)
   model_1 = model.to_gpu(1)

Then, set up optimizer as:

.. testcode::

   optimizer = optimizers.SGD()
   optimizer.setup(model_0)

Here we use the first copy of the model as *the master model*.
Before its update, gradients of ``model_1`` must be aggregated to those of ``model_0``.

Forward function is almost same as the original example:

.. testcode::

   def forward(x_data, y_data, model):
       x = Variable(x_data)
       t = Variable(y_data)
       h1 = F.relu(model.l1(x))
       h2 = F.relu(model.l2(h1))
       y = model.l3(h2)
       return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

The only difference is that ``forward`` accepts ``model`` as an argument.
We can feed it with a model and arrays on an appropriate device.
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

           optimizer.zero_grads()

           loss_0, accuracy_0 = forward(
               cuda.to_gpu(x_batch[:batchsize//2], 0),
               cuda.to_gpu(y_batch[:batchsize//2], 0),
               model_0)
           loss_0.backward()

           loss_1, accuracy_1 = forward(
               cuda.to_gpu(x_batch[batchsize//2:], 1),
               cuda.to_gpu(y_batch[batchsize//2:], 1),
               model_1)
           loss_1.backward()

           optimizer.accumulate_grads(model_1.gradients)
           optimizer.update()

           model_1.copy_parameters_from(model_0.parameters)

.. testoutput::
   :hide:

   epoch 0
   ...

One half of the minibatch is forwarded to GPU 0, the other half to GPU 1.
Then the gradients are accumulated by the :meth:`Optimizer.accumulate_grads` method.
After the gradients are prepared, we can update the optimizer in usual way.
Note that the update only modifies the parameters of ``model_0``.
So we must manually copy them to ``model_1`` using :meth:`FunctionSet.copy_parameters_from` method.

--------

Now you can use Chainer with GPUs.
All examples in the ``examples`` directory support GPU computation, so please refer to them if you want to know more practices on using GPUs.
In the next section, we will show how to define a differentiable (i.e. *backpropable*) function on Variable objects.
We will also show there how to write a simple (elementwise) CUDA kernel using Chainer's CUDA utilities.
