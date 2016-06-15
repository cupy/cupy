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
Note that one can still produce such a static network definition using imperative languages (e.g. Torch7 and Theano-based frameworks).

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
     from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
     from chainer import Link, Chain, ChainList
     import chainer.functions as F
     import chainer.links as L

   These imports appear widely in Chainer's codes and examples. For simplicity, we omit this idiom in this tutorial.


Forward/Backward Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As described above, Chainer uses "Define-by-Run" scheme, so forward computation itself *defines* the network.
In order to start forward computation, we have to set the input array to :class:`Variable` object.
Here we start with simple :class:`~numpy.ndarray` with only one element:

.. doctest::

   >>> x_data = np.array([5], dtype=np.float32)
   >>> x = Variable(x_data)

.. warning::

   Chainer currently only supports 32-bit float for most computations.

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

   Many functions taking :class:`Variable` object(s) are defined in the :mod:`functions` module.
   You can combine them to realize complicated functions with automatic backward computation.


Links
~~~~~

In order to write neural networks, we have to combine functions with *parameters* and optimize the parameters.
You can use **links** to do this.
Link is an object that holds parameters (i.e. optimization targets).

The most fundamental ones are links that behave like regular functions while replacing some arguments by their parameters.
We will introduce higher level links, but here think links just like functions with parameters.

.. note::
   Actually, these are corresponding to "parameterized functions" in versions up to v1.4.

One of the most frequently-used links is the :class:`~functions.Linear` link (a.k.a. *fully-connected layer* or *affine transformation*).
It represents a mathematical function :math:`f(x) = Wx + b`, where the matrix :math:`W` and the vector :math:`b` are parameters.
This link is corresponding to its pure counterpart :func:`~functions.linear`, which accepts :math:`x, W, b` as arguments.
A linear link from three-dimensional space to two-dimensional space is defined by:

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
We can write complex procedures with parameters by combining multiple links like:

.. doctest::

   >>> l1 = L.Linear(4, 3)
   >>> l2 = L.Linear(3, 2)
   >>> def my_forward(x):
   ...     h = l1(x)
   ...     return l2(h)

Here the ``L`` indicates the :mod:`chainer.links` module.
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
If the number of links is fixed like above case, the Chain class is recommended as a base class.


Optimizer
~~~~~~~~~

In order to get good values for parameters, we have to optimize them by the :class:`Optimizer` class.
It runs a numerical optimization algorithm given a link.
Many algorithms are implemented in :mod:`optimizers` module.
Here we use the simplest one, called Stochastic Gradient Descent:

.. doctest::

   >>> model = MyChain()
   >>> optimizer = optimizers.SGD()
   >>> optimizer.setup(model)

The method :meth:`~Optimizer.setup` prepares for the optimization given a link.

There are two ways to run optimization.
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

Some parameter/gradient manipulations, e.g. weight decay and gradient clipping, can be done by setting *hook functions* to the optimizer.
Hook functions are called by the :meth:`~Optimizer.update` method in advance of the actual update.
For example, we can set weight decay regularization by running the next line beforehand:

.. doctest::

   >>> optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

Of course, you can write your own hook functions.
It should be a function or a callable object, taking the optimizer as the argument.


Serializer
~~~~~~~~~~

The last core feature described in this page is serializer.
Serializer is a simple interface to serialize or deserialize an object.
:class:`Link` and :class:`Optimizer` supports serialization by serializers.

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
Here we use hand-written digits dataset called `MNIST <http://yann.lecun.com/exdb/mnist/>`_, which is one of the long-standing defacto "hello world" of machine learning.
This MNIST example is also found in ``examples/mnist`` directory of the official repository.

In order to use MNIST, we prepared ``load_mnist_data`` function at ``examples/mnist/data.py``::

   >>> import data
   >>> mnist = data.load_mnist_data()

.. testcode::
   :hide:

   mnist = {'data': np.random.randint(255, size=(70000, 784)).astype(np.uint8),
            'target': np.random.randint(10, size=70000).astype(np.uint8)}

The mnist dataset consists of 70,000 grayscale images of size 28x28 (i.e. 784 pixels) and corresponding digit labels.
First, we scale pixels to [0, 1] values, and divide the dataset into 60,000 training samples and 10,000 test samples.

.. doctest::

   >>> x_all = mnist['data'].astype(np.float32) / 255
   >>> y_all = mnist['target'].astype(np.int32)
   >>> x_train, x_test = np.split(x_all, [60000])
   >>> y_train, y_test = np.split(y_all, [60000])

Next, we want to define the architecture.
We use a simple three-layer rectifier network with 100 units per layer as an example.

.. doctest::

   >>> class MLP(Chain):
   ...     def __init__(self):
   ...         super(MLP, self).__init__(
   ...             l1=L.Linear(784, 100),
   ...             l2=L.Linear(100, 100),
   ...             l3=L.Linear(100, 10),
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
   ...         self.loss = F.softmax_cross_entropy(y, t)
   ...         self.accuracy = F.accuracy(y, t)
   ...         return self.loss

This Classifier class computes accuracy and loss, and returns the loss value.
:func:`~functions.softmax_cross_entropy` computes the loss value given prediction and ground truth labels.
:func:`~functions.accuracy` computes the prediction accuracy.
We can set an arbitrary predictor link to an instance of the classifier.

Note that a similar class is defined as :class:`chainer.links.Classifier`.
So instead of using the above example, we will use this predefined Classifier chain instead.

.. doctest::

   >>> model = L.Classifier(MLP())
   >>> optimizer = optimizers.SGD()
   >>> optimizer.setup(model)

Finally, we can write a learning loop as following:

.. testcode::
   :hide:

   datasize = 600

.. doctest::

   >>> batchsize = 100
   >>> datasize = 60000  #doctest: +SKIP
   >>> for epoch in range(20):
   ...     print('epoch %d' % epoch)
   ...     indexes = np.random.permutation(datasize)
   ...     for i in range(0, datasize, batchsize):
   ...         x = Variable(x_train[indexes[i : i + batchsize]])
   ...         t = Variable(y_train[indexes[i : i + batchsize]])
   ...         optimizer.update(model, x, t)
   epoch 0...

Only the last three lines are the code related to Chainer, which are already described above.
Note that, in the last line, we pass ``model`` as a loss function.

These three lines can also be rewritten as follows, with explicit gradient computation:

.. doctest::

   >>> batchsize = 100
   >>> datasize = 60000  #doctest: +SKIP
   >>> for epoch in range(20):
   ...     print('epoch %d' % epoch)
   ...     indexes = np.random.permutation(datasize)
   ...     for i in range(0, datasize, batchsize):
   ...         x = Variable(x_train[indexes[i : i + batchsize]])
   ...         t = Variable(y_train[indexes[i : i + batchsize]])
   ...         model.zerograds()
   ...         loss = model(x, t)
   ...         loss.backward()
   ...         optimizer.update()
   epoch 0...

You may find that, at each iteration, the network is defined by forward computation, used for backprop, and then disposed.
By leveraging this "Define-by-Run" scheme, you can imagine that recurrent nets with variable length input are simply handled by just using loop over different length input for each iteration.

After or during optimization, we want to evaluate the model on the test set.
It can be achieved simply by calling forward function:

.. doctest::

   >>> sum_loss, sum_accuracy = 0, 0
   >>> for i in range(0, 10000, batchsize):
   ...     x = Variable(x_test[i : i + batchsize])
   ...     t = Variable(y_test[i : i + batchsize])
   ...     loss = model(x, t)
   ...     sum_loss += loss.data * batchsize
   ...     sum_accuracy += model.accuracy.data * batchsize
   ...     
   >>> mean_loss = sum_loss / 10000
   >>> mean_accuracy = sum_accuracy / 10000

The example code in the `examples/mnist` directory contains GPU support, though the essential part is same as the code in this tutorial.
We will review in later sections how to use GPU(s).
