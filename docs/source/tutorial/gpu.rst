Using GPU(s) in Chainer
-----------------------

.. currentmodule:: chainer

In this section, you will learn about the following things:

* Relationship between Chainer and PyCUDA
* Basics of GPUArray
* Single-GPU usage of Chainer
* Multi-GPU usage of model-parallel computing
* Multi-GPU usage of data-parallel computing

After reading this section, you will be able to:

* Use Chainer on a CUDA-enabled GPU
* Write model-parallel computing in Chainer
* Write data-parallel computing in Chainer


Relationship between Chainer and PyCUDA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chainer uses `PyCUDA <http://mathema.tician.de/software/pycuda/>`_ as its backend for GPU computation and the :class:`pycuda.gpuarray.GPUArray` class as the GPU array implementation.
GPUArray has far less features compared to :class:`numpy.ndarray`, though it is still enough to implement the required features for Chainer.

.. note::

   :mod:`chainer.cuda` module imports many important symbols from PyCUDA.
   For example, the GPUArray class is referred as ``cuda.GPUArray`` in the Chainer code.

Chainer provides wrappers of many PyCUDA functions and classes, mainly in order to support customized default allocation mechanism.
As shown in the previous sections, Chainer constructs and destructs many arrays during learning and evaluating iterations.
It is not well suited for CUDA architecture, since memory allocation and release in CUDA (i.e. ``cuMemAlloc`` and ``cuMemFree`` functions) synchronize CPU and GPU computations, which hurts performance.
In order to avoid memory allocation and deallocation during the computation, Chainer uses PyCUDA's memory pool utilities as the standard memory allocator.
Since memory pool is not the default allocator in PyCUDA, Chainer provides many wrapper functions and classes to use memory pools in a simple way.
At the same time, Chainer's wrapper functions and classes make it easy to handle multiple GPUs.

.. note::

   Chainer also uses `scikit-cuda <http://scikit-cuda.readthedocs.org/en/latest/>`_ for a wrapper of CUBLAS, and some functions use `CuDNN v2 <https://developer.nvidia.com/cuDNN>`_ if available.
   We omit their usage in this tutorial.

.. note::

   We also do not touch the detail of PyCUDA.
   See `PyCUDA's documentation <http://documen.tician.de/pycuda/>`_ instead.


Basics of GPUArray in Chainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to use GPU in Chainer, we must initialize :mod:`chainer.cuda` module before any GPU-related operations::

  cuda.init()

The :func:`cuda.init` function initializes global state and PyCUDA.
This function accepts an optional argument ``device``, which indicates the GPU device ID to select initially.

.. warning::

   If you are using :mod:`multiprocessing`, the initialization must take place for each process *after* the fork.
   The main process is no exception, i.e., :func:`cuda.init` should not be called before all the children that use GPU have been forked.

Then we can create a GPUArray object using functions of the :mod:`~chainer.cuda` module.
Chainer provides many constructor functions resembling the ones of NumPy: :func:`~cuda.empty`, :func:`~cuda.empty_like`, :func:`~cuda.full`, :func:`~cuda.full_like`, :func:`~cuda.zeros`, :func:`~cuda.zeros_like`, :func:`~cuda.ones`, :func:`~cuda.ones_like`.

Another useful function to create a GPUArray object is :func:`~cuda.to_gpu`.
This function copies a :class:`numpy.ndarray` object to a newly allocated GPUArray object.
For example, the following code ::

  x_cpu = np.ones((5, 4, 3), dtype=np.float32)
  x_gpu = cuda.to_gpu(x_cpu)

generates the same ``x_gpu`` as the following code::

  x_gpu = cuda.ones((5, 4, 3))

.. note::

   Allocation functions of the :mod:`~chainer.cuda` module use :class:`numpy.float32` as the default element type.

The :mod:`~chainer.cuda` module also has :func:`~cuda.to_cpu` function to copy a GPUArray object to an ndarray object::

  x_cpu = cuda.to_cpu(x_gpu)

All GPUArray constructors allocate memory on the current device.
In order to allocate memory on a different device, we can use device switching utilities.
:func:`cuda.use_device` function changes the current device::

  cuda.use_device(1)
  x_gpu1 = cuda.empty((4, 3))

There are many situations in which we want to temporarily switch the device, where the :func:`cuda.using_device` function is useful.
It returns an resource object that can be combinated with the ``with`` statement::

  with cuda.using_device(1):
      x_gpu1 = cuda.empty((4, 3))

These device switching utilities also accepts a GPUArray object as a device specifier.
In this case, Chainer switches the current device to one that the array is allocated on::

  with cuda.using_device(x_gpu1):
      y_gpu1 = x_gpu1 + 1

.. warning::

   An array that is not allocated by Chainer's allocator cannot be used as a device specifier.

A GPUArray object allocated by Chainer can be copied between GPUs by :func:`cuda.copy` function::

  cuda.use_device(0)
  x0 = cuda.ones((4, 3))
  x1 = cuda.copy(x0, out_device=1)


Run Neural Networks on a Single GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Single-GPU usage is very simple.
What you have to do is transferring :class:`FunctionSet` and input arrays to the GPU beforehand.
In this subsection, the code is based on :ref:`our first MNIST example in this tutorial <mnist_mlp_example>`.

A :class:`FunctionSet` object can be transferred to the specified GPU using the :meth:`~FunctionSet.to_gpu` method.
Make sure to give parameters and gradients of the GPU version to the optimizer. ::

  model = FunctionSet(
      l1 = F.Linear(784, 100),
      l2 = F.Linear(100, 100),
      l3 = F.Linear(100,  10),
  ).to_gpu()

  optimizer = optimizers.SGD()
  optimizer.setup(model)

Note that this method returns the function set itself.
The device specifier can be omitted, in which case it uses the current device.

Then, all we have to do is transferring each minibatch to the GPU::

  batchsize = 100
  for epoch in xrange(20):
      print 'epoch', epoch
      indexes = np.random.permutation(60000)
      for i in xrange(0, 60000, batchsize):
          x_batch = cuda.to_gpu(x_train[indexes[i : i + batchsize]])
          y_batch = cuda.to_gpu(y_train[indexes[i : i + batchsize]])

          optimizer.zero_grads()
          loss, accuracy = forward(x_batch, y_batch)
          loss.backward()
          optimizer.update()

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
Here is a simple example of the model definition::

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

Now we can define the network architecture that we have shown in the diagram::

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
First, define the model::

  model = FunctionSet(
      l1 = F.Linear(784, 100),
      l2 = F.Linear(100, 100),
      l3 = F.Linear(100,  10),
  )

We have to copy this model into two different devices.
This is done by using :func:`copy.deepcopy` and :meth:`FunctionSet.to_gpu` method::

  import copy
  model_0 = copy.deepcopy(model).to_gpu(0)
  model_1 = model.to_gpu(1)

Then, set up optimizer as::

  optimizer = optimizers.SGD()
  optimizer.setup(model_0)

Here we use the first copy of the model as *the master model*.
Before its update, gradients of ``model_1`` must be aggregated to those of ``model_0``.

Forward function is almost same as the original example::

  def forward(x_data, y_data, model):
      x = Variable(x_data)
      t = Variable(y_data)
      h1 = F.relu(model.l1(x))
      h2 = F.relu(model.l2(h1))
      y = model.l3(h2)
      return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

The only difference is that ``forward`` accepts ``model`` as an argument.
We can feed it with a model and arrays on an appropriate device.
Then, we can write a data-parallel learning loop as follows::

  batchsize = 100
  for epoch in xrange(20):
      print 'epoch', epoch
      indexes = np.random.permutation(60000)
      for i in xrange(0, 60000, batchsize):
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

          optimizer.acumulate_grads(model_1.gradients)
          optimizer.update()

          model_1.copy_parameters_from(model_0.parameters)

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
