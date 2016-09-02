Introduction to Chainer
-----------------------

.. currentmodule:: chainer

This is the first section of the Chainer Tutorial.
In this section, you will learn about the following things:

* Pros and cons of existing frameworks and why we are developing Chainer
* Simple example of forward and backward computation
* Usage of links and their gradient computation
* Construction of chains (a.k.a. "model" in most frameworks)
* Parameter optimization
* Serialization of links and optimizers

After reading this section, you will be able to:

* Compute gradients of some arithmetics
* Write a multi-layer perceptron with Chainer


Core Concept
~~~~~~~~~~~~

As mentioned on the front page, Chainer is a flexible framework for neural networks.
One major goal is flexibility, so it must enable us to write complex architectures simply and intuitively.

Most existing deep learning frameworks are based on the **"Define-and-Run"** scheme.
That is, first a network is defined and fixed, and then the user periodically feeds it with mini-batches.
Since the network is statically defined before any forward/backward computation, all the logic must be embedded into the network architecture as *data*.
Consequently, defining a network architecture in such systems (e.g. Caffe) follows a declarative approach.
Note that one can still produce such a static network definition using imperative languages (e.g. torch.nn, Theano-based frameworks, and TensorFlow).

In contrast, Chainer adopts a **"Define-by-Run"** scheme, i.e., the network is defined on-the-fly via the actual forward computation.
More precisely, Chainer stores the history of computation instead of programming logic.
This strategy enables to fully leverage the power of programming logic in Python.
For example, Chainer does not need any magic to introduce conditionals and loops into the network definitions.
The Define-by-Run scheme is the core concept of Chainer.
We will show in this tutorial how to define networks dynamically.

This strategy also makes it easy to write multi-GPU parallelization, since logic comes closer to network manipulation.
We will review such amenities in later sections of this tutorial.


.. note::

   In example codes of this tutorial, we assume for simplicity that the following symbols are already imported::

     import numpy as np
     import chainer
     from chainer import cuda, Function, gradient_check, report, training, utils, Variable
     from chainer import datasets, iterators, optimizers, serializers
     from chainer import Link, Chain, ChainList
     import chainer.functions as F
     import chainer.links as L
     from chainer.training import extensions

   These imports appear widely in Chainer's codes and examples. For simplicity, we omit these imports in this tutorial.


Forward/Backward Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As described above, Chainer uses "Define-by-Run" scheme, so forward computation itself *defines* the network.
In order to start forward computation, we have to set the input array to :class:`Variable` object.
Here we start with simple :class:`~numpy.ndarray` with only one element:

.. doctest::

   >>> x_data = np.array([5], dtype=np.float32)
   >>> x = Variable(x_data)

A Variable object has basic arithmetic operators.
In order to compute :math:`y = x^2 - 2x + 1`, just write:

.. doctest::

   >>> y = x**2 - 2 * x + 1

The resulting ``y`` is also a Variable object, whose value can be extracted by accessing the :attr:`~Variable.data` attribute:

.. doctest::

   >>> y.data
   array([ 16.], dtype=float32)

What ``y`` holds is not only the result value.
It also holds the history of computation (or computational graph), which enables us to compute its differentiation.
This is done by calling its :meth:`~Variable.backward` method:

.. doctest::

   >>> y.backward()

This runs *error backpropagation* (a.k.a. *backprop* or *reverse-mode automatic differentiation*).
Then, the gradient is computed and stored in the :attr:`~Variable.grad` attribute of the input variable ``x``:

.. doctest::

   >>> x.grad
   array([ 8.], dtype=float32)

Also we can compute gradients of intermediate variables.
Note that Chainer, by default, releases the gradient arrays of intermediate variables for memory efficiency.
In order to preserve gradient information, pass the ``retain_grad`` argument to the backward method:

.. doctest::

   >>> z = 2*x
   >>> y = x**2 - z + 1
   >>> y.backward(retain_grad=True)
   >>> z.grad
   array([-1.], dtype=float32)

All these computations are easily generalized to multi-element array input.
Note that if we want to start backward computation from a variable holding a multi-element array, we must set the *initial error* manually.
This is simply done by setting the :attr:`~Variable.grad` attribute of the output variable:

.. doctest::

   >>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
   >>> y = x**2 - 2*x + 1
   >>> y.grad = np.ones((2, 3), dtype=np.float32)
   >>> y.backward()
   >>> x.grad
   array([[  0.,   2.,   4.],
          [  6.,   8.,  10.]], dtype=float32)

.. note::

   Many functions taking :class:`Variable` object(s) are defined in the :mod:`~chainer.functions` module.
   You can combine them to realize complicated functions with automatic backward computation.


Links
~~~~~

In order to write neural networks, we have to combine functions with *parameters* and optimize the parameters.
You can use **links** to do this.
Link is an object that holds parameters (i.e. optimization targets).

The most fundamental ones are links that behave like regular functions while replacing some arguments by their parameters.
We will introduce higher level links, but here think links just like functions with parameters.

One of the most frequently-used links is the :class:`~functions.Linear` link (a.k.a. *fully-connected layer* or *affine transformation*).
It represents a mathematical function :math:`f(x) = Wx + b`, where the matrix :math:`W` and the vector :math:`b` are parameters.
This link is corresponding to its pure counterpart :func:`~functions.linear`, which accepts :math:`x, W, b` as arguments.
A linear link from three-dimensional space to two-dimensional space is defined by the following line:

.. doctest::

   >>> f = L.Linear(3, 2)

.. note::
   Most functions and links only accept mini-batch input, where the first dimension of input arrays is considered as the *batch dimension*.
   In the above Linear link case, input must have shape of (N, 3), where N is the mini-batch size.

The parameters of a link are stored as attributes.
Each parameter is an instance of :class:`~chainer.Variable`.
In the case of Linear link, two parameters, ``W`` and ``b``, are stored.
By default, the matrix ``W`` is initialized randomly, while the vector ``b`` is initialized with zeros.

.. doctest::

   >>> f.W.data
   array([[ 1.01847613,  0.23103087,  0.56507462],
          [ 1.29378033,  1.07823515, -0.56423163]], dtype=float32)
   >>> f.b.data
   array([ 0.,  0.], dtype=float32)

An instance of the Linear link acts like a usual function:

.. doctest::

   >>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
   >>> y = f(x)
   >>> y.data
   array([[ 3.1757617 ,  1.75755572],
          [ 8.61950684,  7.18090773]], dtype=float32)

Gradients of parameters are computed by :meth:`~Variable.backward` method.
Note that gradients are **accumulated** by the method rather than overwritten.
So first you must initialize gradients to zero to renew the computation.
It can be done by calling the :meth:`~Link.zerograds` method.

.. doctest::

   >>> f.zerograds()

Now we can compute the gradients of parameters by simply calling backward method.

.. doctest::

   >>> y.grad = np.ones((2, 2), dtype=np.float32)
   >>> y.backward()
   >>> f.W.grad
   array([[ 5.,  7.,  9.],
          [ 5.,  7.,  9.]], dtype=float32)
   >>> f.b.grad
   array([ 2.,  2.], dtype=float32)


Write a model as a chain
~~~~~~~~~~~~~~~~~~~~~~~~

Most neural network architectures contain multiple links.
For example, a multi-layer perceptron consists of multiple linear layers.
We can write complex procedures with parameters by combining multiple links like this:

.. doctest::

   >>> l1 = L.Linear(4, 3)
   >>> l2 = L.Linear(3, 2)
   >>> def my_forward(x):
   ...     h = l1(x)
   ...     return l2(h)

Here the ``L`` indicates the :mod:`~chainer.links` module.
A procedure with parameters defined in this way is hard to reuse.
More Pythonic way is combining the links and procedures into a class:

.. doctest::

   >>> class MyProc(object):
   ...     def __init__(self):
   ...         self.l1 = L.Linear(4, 3)
   ...         self.l2 = L.Linear(3, 2)
   ...         
   ...     def forward(self, x):
   ...         h = self.l1(x)
   ...         return self.l2(h)

In order to make it more reusable, we want to support parameter management, CPU/GPU migration support, robust and flexible save/load features, etc.
These features are all supported by the :class:`Chain` class in Chainer.
Then, what we have to do here is just defining the above class as a subclass of Chain:

.. doctest::

   >>> class MyChain(Chain):
   ...     def __init__(self):
   ...         super(MyChain, self).__init__(
   ...             l1=L.Linear(4, 3),
   ...             l2=L.Linear(3, 2),
   ...         )
   ...        
   ...     def __call__(self, x):
   ...         h = self.l1(x)
   ...         return self.l2(h)

.. note::
   We often define a single forward method of a link by ``__call__`` operator.
   Such links and chains are callable and behave like regular functions of Variables.

It shows how a complex chain is constructed by simpler links.
Links like ``l1`` and ``l2`` are called *child links* of MyChain.
**Note that Chain itself inherits Link**.
It means we can define more complex chains that hold MyChain objects as their child links.

Another way to define a chain is using the :class:`ChainList` class, which behaves like a list of links:

.. doctest::

   >>> class MyChain2(ChainList):
   ...     def __init__(self):
   ...         super(MyChain2, self).__init__(
   ...             L.Linear(4, 3),
   ...             L.Linear(3, 2),
   ...         )
   ...         
   ...     def __call__(self, x):
   ...         h = self[0](x)
   ...         return self[1](h)

ChainList is convenient to use an arbitrary number of links.
If the number of links is fixed like the above case, the Chain class is recommended as a base class.


Optimizer
~~~~~~~~~

In order to get good values for parameters, we have to optimize them by the :class:`Optimizer` class.
It runs a numerical optimization algorithm given a link.
Many algorithms are implemented in :mod:`~chainer.optimizers` module.
Here we use the simplest one, called Stochastic Gradient Descent (SGD):

.. doctest::

   >>> model = MyChain()
   >>> optimizer = optimizers.SGD()
   >>> optimizer.setup(model)

The method :meth:`~Optimizer.setup` prepares for the optimization given a link.

Some parameter/gradient manipulations, e.g. weight decay and gradient clipping, can be done by setting *hook functions* to the optimizer.
Hook functions are called after the gradient computation and right before the actual update of parameters.
For example, we can set weight decay regularization by running the next line beforehand:

.. doctest::

   >>> optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

Of course, you can write your own hook functions.
It should be a function or a callable object, taking the optimizer as the argument.

There are two ways to use the optimizer.
One is using it via :class:`~chainer.training.Trainer`, which we will see in the following sections.
The other way is using it directly.
We here review the latter case.
*If you are interested in getting able to use the optimizer in a simple way, skip this section and go to the next one.*

There are further two ways to use the optimizer directly.
One is manually computing gradients and then call the :meth:`~Optimizer.update` method with no arguments.
Do not forget resetting gradients beforehand!

.. doctest::

   >>> model.zerograds()
   >>> # compute gradient here...
   >>> optimizer.update()

The other way is just passing a loss function to the :meth:`~Optimizer.update` method.
In this case, :meth:`~Link.zerograds` is automatically called by the update method, so user do not have to call it manually.

   >>> def lossfun(args...):
   ...     ...
   ...     return loss
   >>> optimizer.update(lossfun, args...)

See :meth:`Optimizer.update` for the full specification.


Trainer
~~~~~~~

When we want to train neural networks, we have to run *training loops* that update parameters many times.
A typical training loop consists of following procedures:

1. Iterations over training datasets
2. Preprocessing of extracted minibatches
3. Forward/backward computations of the neural networks
4. Parameter updates
5. Evaluations of the current parameters on validation datasets
6. Logging and printing of the intermediate results

Chainer provides a simple yet powerful way to make it easy to write such training processes.
The training loop abstraction mainly consists of two components:

- **Dataset abstraction**.
  It implements 1 and 2 in the above list.
  The core components are defined in the :mod:`dataset` module.
  There are also many implementations of datasets and iterators in :mod:`~chainer.datasets` and :mod:`~chainer.iterators` modules, respectively.
- **Trainer**.
  It implements 3, 4, 5, and 6 in the above list.
  The whole procedure is implemented by :class:`~training.Trainer`.
  The way to update parameters (3 and 4) is defined by :class:`~training.Updater`, which can be freely customized.
  The 5 and 6 are implemented by instances of :class:`~training.Extension`, which appends an extra procedure to the training loop.
  Users can freely customize the training procedure by adding extensions. Users can also implement their own extensions.

We will see how to use Trainer in the example section below.


Serializer
~~~~~~~~~~

Before proceeding to the first example, we introduce Serializer, which is the last core feature described in this page.
Serializer is a simple interface to serialize or deserialize an object.
:class:`Link`, :class:`Optimizer`, and :class:`~training.Trainer` supports serialization.

Concrete serializers are defined in the :mod:`serializers` module.
It supports NumPy NPZ and HDF5 formats.

For example, we can serialize a link object into NPZ file by the :func:`serializers.save_npz` function:

.. doctest::

   >>> serializers.save_npz('my.model', model)

It saves the parameters of ``model`` into the file ``'my.model'`` in NPZ format.
The saved model can be read by the :func:`serializers.load_npz` function:

.. doctest::

   >>> serializers.load_npz('my.model', model)

.. note::
   Note that only the parameters and the *persistent values* are serialized by these serialization code.
   Other attributes are not saved automatically.
   You can register arrays, scalars, or any serializable objects as persistent values by the :meth:`Link.add_persistent` method.
   The registered values can be accessed by attributes of the name passed to the add_persistent method.

The state of an optimizer can also be saved by the same functions:

.. doctest::

   >>> serializers.save_npz('my.state', optimizer)
   >>> serializers.load_npz('my.state', optimizer)

.. note::
   Note that serialization of optimizer only saves its internal states including number of iterations, momentum vectors of MomentumSGD, etc.
   It does not save the parameters and persistent values of the target link.
   We have to explicitly save the target link with the optimizer to resume the optimization from saved states.

Support of the HDF5 format is enabled if the h5py package is installed.
Serialization and deserialization with the HDF5 format are almost identical to those with the NPZ format;
just replace :func:`~serializers.save_npz` and :func:`~serializers.load_npz` by :func:`~serializers.save_hdf5` and :func:`~serializers.load_hdf5`, respectively.

.. _mnist_mlp_example:

Example: Multi-layer Perceptron on MNIST
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now you can solve a multiclass classification task using a multi-layer perceptron.
We use hand-written digits dataset called `MNIST <http://yann.lecun.com/exdb/mnist/>`_, which is one of the long-standing de facto "hello world" of machine learning.
This MNIST example is also found in the `examples/mnist <https://github.com/pfnet/chainer/tree/master/examples/mnist>`_ directory of the official repository.
We show how to use :class:`~training.Trainer` to construct and run the training loop in this section.

We first have to prepare the MNIST dataset.
The MNIST dataset consists of 70,000 grayscale images of size 28x28 (i.e. 784 pixels) and corresponding digit labels.
The dataset is divided into 60,000 training images and 10,000 test images by default.
We can obtain the vectorized version (i.e., a set of 784 dimensional vectors) by :func:`datasets.get_mnist`.

   >>> train, test = datasets.get_mnist()

.. testcode::
   :hide:

   data = np.random.rand(70000, 784).astype(np.float32)
   target = np.random.randint(10, size=70000).astype(np.int32)
   train = datasets.TupleDataset(data[:60000], target[:60000])
   test = datasets.TupleDataset(data[60000:], target[60000:])

This code automatically downloads the MNIST dataset and saves the NumPy arrays to the ``$(HOME)/.chainer`` directory.
The returned ``train`` and ``test`` can be seen as lists of image-label pairs (strictly speaking, they are instances of :class:`~datasets.TupleDataset`).

We also have to define how to iterate over these datasets.
We want to shuffle the training dataset for every *epoch*, i.e. at the beginning of every sweep over the dataset.
In this case, we can use :class:`iterators.SerialIterator`.

.. doctest::

   >>> train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)

On the other hand, we do not have to shuffle the test dataset.
In this case, we can pass ``shuffle=False`` argument to disable the shuffling.
It makes the iteration faster when the underlying dataset supports fast slicing.

.. doctest::

   >>> test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

We also pass ``repeat=False``, which means we stop iteration when all examples are visited.
This option is usually required for the test/validation datasets; without this option, the iteration enters an infinite loop.

Next, we define the architecture.
We use a simple three-layer rectifier network with 100 units per layer as an example.

.. doctest::

   >>> class MLP(Chain):
   ...     def __init__(self, n_units, n_out):
   ...         super(MLP, self).__init__(
   ...             # the size of the inputs to each layer will be inferred
   ...             l1=L.Linear(None, n_units),  # n_in -> n_units
   ...             l2=L.Linear(None, n_units),  # n_units -> n_units
   ...             l3=L.Linear(None, n_out),    # n_units -> n_out
   ...         )
   ...         
   ...     def __call__(self, x):
   ...         h1 = F.relu(self.l1(x))
   ...         h2 = F.relu(self.l2(h1))
   ...         y = self.l3(h2)
   ...         return y

This link uses :func:`~functions.relu` as an activation function.
Note that the ``'l3'`` link is the final linear layer whose output corresponds to scores for the ten digits.

In order to compute loss values or evaluate the accuracy of the predictions, we define a classifier chain on top of the above MLP chain:

.. doctest::

   >>> class Classifier(Chain):
   ...     def __init__(self, predictor):
   ...         super(Classifier, self).__init__(predictor=predictor)
   ...         
   ...     def __call__(self, x, t):
   ...         y = self.predictor(x)
   ...         loss = F.softmax_cross_entropy(y, t)
   ...         accuracy = F.accuracy(y, t)
   ...         report({'loss': loss, 'accuracy': accuracy}, self)
   ...         return loss

This Classifier class computes accuracy and loss, and returns the loss value.
The pair of arguments ``x`` and ``t`` corresponds to each example in the datasets (a tuple of an image and a label).
:func:`~functions.softmax_cross_entropy` computes the loss value given prediction and ground truth labels.
:func:`~functions.accuracy` computes the prediction accuracy.
We can set an arbitrary predictor link to an instance of the classifier.

The :func:`~chainer.report` function reports the loss and accuracy values to the trainer.
For the detailed mechanism of collecting training statistics, see :ref:`reporter`.
You can also collect other types of observations like activation statistics in a similar ways.

Note that a class similar to the Classifier above is defined as :class:`chainer.links.Classifier`.
So instead of using the above example, we will use this predefined Classifier chain.

.. doctest::

   >>> model = L.Classifier(MLP(100, 10))  # the input size, 784, is inferred
   >>> optimizer = optimizers.SGD()
   >>> optimizer.setup(model)

Now we can build a trainer object.

.. doctest::

   >>> updater = training.StandardUpdater(train_iter, optimizer)
   >>> trainer = training.Trainer(updater, (20, 'epoch'), out='result')

The second argument ``(20, 'epoch')`` represents the duration of training.
We can use either ``epoch`` or ``iteration`` as the unit.
In this case, we train the multi-layer perceptron by iterating over the training set 20 times.

In order to invoke the training loop, we just call the :meth:`~training.Trainer.run` method.

.. doctest::

   >>> trainer.run()

.. testcode::
   :hide:

   trainer = training.Trainer(updater, (20, 'epoch'), out='result')

This method executes the whole training sequence.

The above code just optimizes the parameters.
In most cases, we want to see how the training proceeds, where we can use extensions inserted before calling the ``run`` method.

.. doctest::

   >>> trainer.extend(extensions.Evaluator(test_iter, model))
   >>> trainer.extend(extensions.LogReport())
   >>> trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
   >>> trainer.extend(extensions.ProgressBar())
   >>> trainer.run()  # doctest: +SKIP

These extensions perform the following tasks:

:class:`~training.extensions.Evaluator`
   Evaluates the current model on the test dataset at the end of every epoch.
:class:`~training.extensions.LogReport`
   Accumulates the reported values and emits them to the log file in the output directory.
:class:`~training.extensions.PrintReport`
   Prints the selected items in the LogReport.
:class:`~training.extensions.ProgressBar`
   Shows the progress bar.

There are many extensions implemented in the :mod:`chainer.training.extensions` module.
The most important one that is not included above is :func:`~training.extensions.snapshot`, which saves the snapshot of the training procedure (i.e., the Trainer object) to a file in the output directory.

The `example code <https://github
.com/pfnet/chainer/blob/master/examples/mnist/train_mnist.py>`_ in the
`examples/mnist` directory additionally contains GPU support, though the
essential part is same as the code in this tutorial. We will review in later
 sections how to use GPU(s).
