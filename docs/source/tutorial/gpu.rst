Using GPU(s) in Chainer
-----------------------

.. currentmodule:: chainer

In this section, you will learn following things.

* Relationship between Chainer and PyCUDA
* Basics of GPUArray
* Single GPU usage of Chainer
* Multi GPU usage of model-parallel computing
* Multi GPU usage of data-parallel computing
* Combination of model-parallel and data-parallel

By reading this section, you will come to be able to

* Use Chainer on CUDA-enabled GPU
* Write model-parallel computing in Chainer
* Write data-parallel computing in Chainer


Relationship between Chainer and PyCUDA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chainer uses `PyCUDA <http://mathema.tician.de/software/pycuda/>`_ as its backend of GPU computation, and :class:`pycuda.gpuarray.GPUArray` class as the array implementation on GPU.
GPUArray has far less features compared to :class:`numpy.ndarray`, though it is still enough to implement features of Chainer.

.. note::

   :mod:`chainer.cuda` module imports many important symbols from PyCUDA.
   For example, the GPUArray class is referred as ``cuda.GPUArray`` in the code of Chainer.

Chainer provides wrappers of many PyCUDA functions and classes, mainly in order to support customized default allocation mechanism.
As we have shown in the previous sections, Chainer constructs and destructs many arrays during learning and evaluating iterations.
It is not suited for CUDA architecture, since memory allocation and release in CUDA (i.e. ``cuMemAlloc`` and ``cuMemFree`` functions) synchronize CPU and GPU computations, which much hurts its speed.
In order to avoid memory allocation and deallocation during the computation, Chainer uses PyCUDA's memory pool utilities as the standard memory allocator.
Since memory pool is not the default allocator of PyCUDA, Chainer provides many wrapper functions and classes to simply use memory pools.
At the same time, Chainer's wrapper functions and classes make it easy to handle multiple GPUs.

.. note::

   Chainer also uses `scikits.cuda <http://scikit-cuda.readthedocs.org/en/latest/>`_ for a wrapper of CUBLAS, and some functions use `CuDNN v2 <https://developer.nvidia.com/cuDNN>`_ if available.
   We omit their usage in this tutorial.

.. note::

   We also do not touch the detail of PyCUDA.
   See `PyCUDA's documentation <http://documen.tician.de/pycuda/>`_ instead.


Basics of GPUArray in Chainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to use GPU in Chainer, we must initialize :mod:`chainer.cuda` module before any GPU-related operations::

  >>> cuda.init()

:func:`cuda.init` function initializes its global state and PyCUDA.
This function accepts optional argument ``device``, which indicates the GPU device ID to initially select.

.. warning::

   If you are using :mod:`multiprocessing`, the initialization must take place for each process after the fork.
   The main process is not the exception, i.e. :func:`cuda.init` should not be called before all the forks that use GPU.

Then we can create GPUArray using functions of :mod:`cuda` module.
Chainer provides many constructor functions resembling ones of NumPy: :func:`~cuda.empty`, :func:`~cuda.empty_like`, :func:`~cuda.full`, :func:`~cuda.full_like`, :func:`~cuda.zeros`, :func:`~cuda.zeros_like`, :func:`~cuda.ones`, :func:`~cuda.ones_like`.

Another useful function to create GPUArray is :func:`~cuda.to_gpu`.
This function copies :class:`numpy.ndarray` to a newly allocated GPUArray.
For example, the following code ::

  >>> x_cpu = np.ones((5, 4, 3), dtype=np.float32)
  >>> x_gpu = cuda.to_gpu(x_cpu)

generates same ``x_gpu`` as the following code::

  >>> x_gpu = cuda.ones((5, 4, 3))

.. note::

   Allocation functions of :mod:`cuda` module uses :class:`numpy.float32` as the default element type.

:mod:`cuda` also has :func:`~cuda.to_cpu` function to copy GPUArray to ndarray::

  >>> x_cpu = cuda.to_cpu(x_gpu)

All GPUArray constructors allocate memory on the current device.
In order to allocate GPUArray on a different device, we can use device switching utilities.
:func:`cuda.use_device` function simply changes the current device::

  >>> cuda.use_device(1)
  >>> x_gpu1 = cuda.empty((4, 3))

There are many situations that we want to temporarily switch the device, where :func:`cuda.using_device` function is useful, which returns resource object that can be combinated with ``with`` statement::

  >>> with cuda.using_device(1):
  ...     x_gpu1 = cuda.empty((4, 3))

These device switching utilities also accepts GPUArray as a device specifier.
In this case, Chainer switches the current device to one on which the array is allocated::

  >>> with cuda.using_device(x_gpu1):
  ...     y_gpu1 = x_gpu1 + 1

.. warning::

   An array that is not allocated by Chainer's allocator cannot be used as a device specifier.

Chainer's GPUArray can be copied between GPUs by :func:`cuda.copy` function::

  >>> cuda.use_device(0)
  >>> x0 = cuda.ones((4, 3))
  >>> x1 = cuda.copy(x0, out_device=1)


Run neural networks on single GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Single GPU usage is very simple.
What you have to do is transferring :class:`FunctionSet` and input arrays to GPU beforehand.
In this subsection, the code is based on :ref:`our first MNIST example on this tutorial <mnist_mlp_example>`.

:class:`FunctionSet` object can be transferred to the specified GPU by :meth:`~FunctionSet.to_gpu` method.
Be careful that you must give parameters and gradients on GPU to the optimizer. ::

  >>> model = FunctionSet(
  ...     l1 = F.Linear(784, 100),
  ...     l2 = F.Linear(100, 100),
  ...     l3 = F.Linear(100,  10),
  ... ).to_gpu()
  ...
  >>> optimizer = optimizers.SGD()
  >>> optimizer.setup(model.collect_parameters())

Note that this method returns the function set itself.
The device specifier can be omitted, where it uses the current device.

Then, what we only have to do is transferring each minibatch to GPU::

  >>> batchsize = 100
  >>> for epoch in xrange(20):
  ...     print 'epoch', epoch
  ...     indexes = np.random.permutation(60000)
  ...     for i in xrange(0, 60000, batchsize):
  ...         x_batch = cuda.to_gpu(x_train[indexes[i : i + batchsize]])
  ...         y_batch = cuda.to_gpu(y_train[indexes[i : i + batchsize]])
  ...         
  ...         optimizer.zero_grads()
  ...         loss, accuracy = forward(x_batch, y_batch)
  ...         loss.backward()
  ...         optimizer.update()

It is almost same as that of the original example.
We just insert :func:`cuda.to_gpu` function to the minibatch arrays.


Model-parallel computation on multiple GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parallelization of machine learning is roughly classified into two types called "model-parallel" and "data-parallel".
Model-parallel indicates parallelizations of the computation inside of the model.
On the other hand, data-parallel indicates parallelizations by data sharding.
In this subsection, we show how to do model-parallel on multiple GPUs in Chainer.

`Recall the MNIST example <mnist_mlp_example>`_.
Here suppose that we want to modify this example by expanding the network to 6 layers with 2000 units for each using two GPUs.
In order to make multi-GPU computation efficient, we only make two GPUs communicate at the third and sixth layer.
The overall architecture looks like the following diagram::

  (GPU0) input --+--> l1 --> l2 --> l3 --+--> l4 --> l5 --> l6 --+--> output
                 |                     |                     |
  (GPU1)         ---> l1 --> l2 --> l3 --+--> l4 --> l5 --> l6 --+

We first have to define :class:`FunctionSet`.
Be careful that parameters that will be used on a device must reside on the devie.
Here is a simple example of the model definition::

  >>> model = FunctionSet(
  ...     gpu0 = FunctionSet(
  ...         l1=F.Linear( 784, 1000),
  ...         l2=F.Linear(1000, 1000),
  ...         l3=F.Linear(1000, 2000),
  ...         l4=F.Linear(2000, 1000),
  ...         l5=F.Linear(1000, 1000),
  ...         l6=F.Linear(1000,   10)
  ...     ).to_gpu(0),
  ...     gpu1 = FunctionSet(
  ...         l1=F.Linear( 784, 1000),
  ...         l2=F.Linear(1000, 1000),
  ...         l3=F.Linear(1000, 2000),
  ...         l4=F.Linear(2000, 1000),
  ...         l5=F.Linear(1000, 1000),
  ...         l6=F.Linear(1000,   10)
  ...     ).to_gpu(1)
  ... )

Recall that :meth:`FunctionSet.to_gpu` returns the FunctionSet object itself.
Note that FunctionSet can be nested as above.

Now we can define the network architecture that we have shown in the diagram::

  >>> def forward(x_data, y_data):
  ...     x_0 = Variable(cuda.to_gpu(x_data, 0))
  ...     x_1 = Variable(cuda.to_gpu(x_data, 1))
  ...     t   = Variable(cuda.to_gpu(y_data, 0))
  ... 
  ...     h1_0 = F.relu(model.gpu0.l1(x_0))
  ...     h1_1 = F.relu(model.gpu1.l1(x_1))
  ... 
  ...     h2_0 = F.relu(model.gpu0.l2(h1_0))
  ...     h2_1 = F.relu(model.gpu1.l2(h1_1))
  ...
  ...     h3_0 = F.relu(model.gpu0.l3(h2_0))
  ...     h3_1 = F.relu(model.gpu1.l3(h2_1))
  ...
  ...     # Synchronize
  ...     h3_0 += F.copy(h3_1, 0)
  ...     h3_1  = F.copy(h3_0, 1)
  ...
  ...     h4_0 = F.relu(model.gpu0.l4(h3_0))
  ...     h4_1 = F.relu(model.gpu1.l4(h3_1))
  ...
  ...     h5_0 = F.relu(model.gpu0.l5(h4_0))
  ...     h5_1 = F.relu(model.gpu1.l5(h4_1))
  ...
  ...     h6_0 = F.relu(model.gpu0.l6(h5_0))
  ...     h6_1 = F.relu(model.gpu1.l6(h5_1))
  ...
  ...     # Synchronize
  ...     y = h6_0 + F.copy(h6_1, 0)
  ...     return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

First, recall that :func:`cuda.to_gpu` accepts an optional argument to specify the device identifier.
We use this to transfer the input minibatch to both the 0th and the 1st devices.
Then, we can write this model-parallel example just by using :func:`functions.copy` function.
This function transfers an input array to another device.
Since it is a function on :class:`Variable`, the operation supports backprop, which reversely transfers an output gradient to the input device.

.. note::

   Above code is not parallelized on CPU, but is parallelized on GPU.
   This is because most of the GPU computation is asynchronous to the host CPU.

Almost same example code is available at ``examples/mnist/train_mnist_model_parallel.py``.


Data-parallel computation on multiple GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Combination of model and data parallelisms on multiple GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
