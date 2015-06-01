Basics of Chainer
-----------------

.. currentmodule:: chainer

This is the first section of the Chainer Tutorial.
In this section, you will learn following things:

* Pros and cons of existing frameworks and why we are developing Chainer
* Simple example of forward and backward computation
* Usage of parameterized functions and its gradient computation
* Management of a set of parameterized functions (a.k.a. "model" in most frameworks)
* Optimization of parameters

After reading this section, you will be able to

* Compute gradients of some arithmetics
* Write multi-layer perceptron with Chainer


Core concept
~~~~~~~~~~~~

Chainer is, as shown at the front page, a flexible framework of neural networks.
We aim at its flexibility, so it must enable us to write complex architectures simply and intuitively.

Most existing deep learning frameworks are based on **"Define-and-Run"** scheme.
That is, a network is first defined and fixed, and then the user periodically feeds minibatches to it.
Since the network is statically defined before any forward/backward computation, all the logics must be embedded into the network architecture as *data*.
Consequently, it is declarative to define a network architecture in such systems (e.g. Caffe).
Note that one can still produce such static network definition using imperative languages (e.g. Torch7 and Theano-based frameworks).

Instead, Chainer adopts **"Define-by-Run"** scheme, i.e. the network is defined on-the-fly by the actual forward computation.
More precisely, Chainer stores the history of computation instead of programming logic.
This strategy enables to fully leverage the power of programming logics in Python.
For example, Chainer does not need any magic to introduce conditionals and loops into the network definitions.
The Define-by-Run scheme is the core concept of Chainer.
We will show in this tutorial how to define networks dynamically.

This strategy also makes it easy to write multi-GPU parallelization, since logic comes closer to network manipulation.
We will review such amenities in later sections of this tutorial.


.. note::

   In example codes of this tutorial, we suppose following symbols are implicitly imported::

     import numpy as np
     from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
     import chainer.functions as F

   These imports appear widely in Chainer's codes and examples. For simplicity, we omit this idiom in this tutorial.


Forward/Backward computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As described above, Chainer uses "Define-by-Run" scheme, so forward computation itself is the definition of networks.
In order to start forward computation, we have to set the input array to :class:`Variable` object.
Here we start with simple :class:`~numpy.ndarray` with only one element::

  >>> x_data = np.array([5], dtype=np.float32)
  >>> x = Variable(x_data)

.. warning::

   Chainer currently only supports 32-bit float for most computations.

Variable object has basic arithmetic operators.
In order to compute :math:`y = x^2 - 2x + 1`, just write ::

  >>> y = x**2 - 2 * x + 1

The resulting ``y`` is also Variable object, whose value can be extracted by :attr:`~Variable.data` attribute::

  >>> y.data
  array([ 16.], dtype=float32)

What ``y`` holds is not only the result value.
It also holds history of computation (or computation graph), which enables us to compute its differentiation.
This is done by calling its :meth:`~Variable.backward` method::

  >>> y.backward()

It runs what is called *error backpropagation* (a.k.a. *backprop* or *reverse-mode auto differentiation*).
Then, the gradient is computed into the :attr:`~Variable.grad` attribute of the input variable ``x``::

  >>> x.grad
  array([ 8.], dtype=float32)

Also we can compute gradients of intermediate variables.
Note that Chainer, by default, releases the gradient arrays of intermediate variables for memory efficiency.
In order to preserve gradient information, pass ``retain_grad`` argument to the backward method::

  >>> z = 2*x
  >>> y = x**2 - z + 1
  >>> y.backward(retain_grad=True)
  >>> z.grad
  array([-1.], dtype=float32)

All these computations are easily generalized to multi-element array input.
Note that if we want to start backward computation from variable with multi-element array, we must set *initial error* manually.
This is simply done by setting :attr:`~Variable.grad` attribute of the output variable::

  >>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
  >>> y = x**2 - 2*x + 1
  >>> y.grad = np.ones((2, 3), dtype=np.float32)
  >>> y.backward()
  >>> x.grad
  array([[  0.,   2.,   4.],
         [  6.,   8.,  10.]], dtype=float32)

.. note::

   Many functions taking :class:`Variable` object(s) are defined in :mod:`functions` module.
   You can combine them to realize complicated function with automatic backward computation.


Parameterized functions
~~~~~~~~~~~~~~~~~~~~~~~

In order to write neural networks, we have to use some *parameterized functions* and optimize their parameters.
As noted above, functions are predefined in :mod:`functions` module, which also includes parameterized functions.

One of the most fundamental parameterized function is :class:`~functions.Linear` function (a.k.a. *fully-connected layer* or *affine transformation*).
It represents a mathematical function :math:`f(x) = Wx + b`, where the matrix :math:`W` and the vector :math:`b` are parameters.
Linear function from three dimensional space to two dimensional space is defined by::
  
  >>> f = F.Linear(3, 2)

.. note::

   Most functions only accept minibatch input, where the first dimension of input arrays is considered as *batch dimension*.
   In the above Linear function case, input must has shape of (N, 3), where N is the minibatch size.

The parameters of Linear function are stored in :attr:`~functions.Linear.W` and :attr:`~functions.Linear.b` attributes.
By default, the matrix W is randomly initialized, while the vector b initialized by zeros.

  >>> f.W
  array([[ 1.33545339, -0.01839679,  0.7662735 ],
         [-1.21562171, -0.44784674, -0.07128379]], dtype=float32)
  >>> f.b
  array([ 0.,  0.], dtype=float32)

Instance of parameterized function class acts as a usual function::

  >>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
  >>> y = f(x)
  >>> y.data
  array([[ 3.5974803 , -2.3251667 ],
         [ 9.84747124, -7.52942371]], dtype=float32)

Gradients of parameters are computed by :meth:`~Variable.backward` method.
Note that gradients are **accumulated** by the method rather than overwritten.
So first you must initialize gradients to zero to renew the computation.
Gradients of Linear function are stored in :attr:`~functions.Linear.gW` and :attr:`~functions.Linear.gb` attributes::

  >>> f.gW.fill(0)
  >>> f.gb.fill(0)

.. note::

   This procedure is simplified by FunctionSet and Optimizer, which we will see in the next seciton.

Now we can compute the gradients of parameters by simply calling backward method::

  >>> y.grad = np.ones((2, 2), dtype=np.float32)
  >>> y.backward()
  >>>
  >>> f.gW
  array([[ 5.,  7.,  9.],
         [ 5.,  7.,  9.]], dtype=float32)
  >>> f.gb
  array([ 2.,  2.], dtype=float32)


FunctionSet
~~~~~~~~~~~

Most neural network architectures contain multiple parameterized functions.
:class:`FunctionSet` makes it easy to manage them.
This class acts like a simple object, with attributes initialized by keyword arguments of initializer::

  >>> model = FunctionSet(
  ...     l1 = F.Linear(4, 3),
  ...     l2 = F.Linear(3, 2),
  ... )
  >>> model.l1
  <chainer.functions.linear.Linear object at 0x7f7f03e4f350>
  >>> model.l2
  <chainer.functions.linear.Linear object at 0x7f7f03e4f590>

You can also set additional functions later by setting attributes::

  >>> model.l3 = F.Linear(2, 2)

Since :class:`FunctionSet` is just an object with functions stored as its attributes, we can use these functions in forward computation::

  >>> x = Varaible(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32))
  >>> h1 = model.l1(x)
  >>> h2 = model.l2(h1)
  >>> h3 = model.l3(h2)

One of the feature of FunctionSet is collecting parameters and gradients.
A tuple of all parameters and a tuple of all gradients are extracted by :attr:`FunctionSet.parameters` and :attr:`FunctionSet.gradients` properties, respectively.


Optimizer
~~~~~~~~~

:class:`Optimizer` is the final core feature of Chainer described in this section.
It runs a numerical optimization algorithm given tuples of parameters and gradients.
Many algorithms are implemented in :mod:`optimizers` module.
Here we use the most simple optimizer called Stochastic Gradient Descent::

  >>> optimizer = optimizers.SGD()
  >>> optimizer.setup(model.collect_parameters())

The method :meth:`~Optimizer.setup` prepares for the optimization given parameters and gradients.
The interface is aligned with :meth:`FunctionSet.collect_parameters` method.

.. note::

   Since Optimizer does not know functions who actually own the parameters and gradients,
   once parameters and gradients are given to Optimizer,
   functions must use same parameter and gradient array objects throughout all forward/backward computations.

In order to run optimization, you first have to compute gradients.
Zeroing the initial gradient arrays are simply done by calling :meth:`~Optimizer.zero_grads` method::

  >>> optimizer.zero_grads()

This is equivalent to filling gradients of functions with zero in the previous section.

Then, after computing gradient of each parameter, :meth:`~Optimizer.update` method runs one iteration of optimization::

  >>> (compute gradient)
  >>> optimizer.update()

Optimizer also contains some features related to parameter and gradient manipulation, e.g. weight decay and gradient clipping.


.. _mnist_mlp_example:

Example: multi-layer perceptron on MNIST
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now you can solve multiclass classification task using multi-layer perceptron.
Here we use hand-written digits dataset called `MNIST <http://yann.lecun.com/exdb/mnist/>`_, which is the long-standing de-facto "hello world" of machine learning.
This MNIST example is also found in ``examples/mnist`` directory of the official repository.

In order to use MNIST, :func:`sklearn.datasets.fetch_mldata` function of `scikit-learn <http://scikit-learn.org/>`_ is useful::

  >>> from sklearn.datasets import fetch_mldata
  >>> mnist = fetch_mldata('MNIST original')

Note that mnist dataset consists of 70,000 grayscale images of size 28x28 (i.e. 784 pixels) and corresponding digit labels.
First, we scale pixels to [0, 1] values, and divide the dataset into 60,000 training samples and 10,000 test samples.

  >>> x_all = mnist.data.astype(np.float32) / 255
  >>> y_all = mnist.target.astype(np.int32)
  >>> x_train, x_test = np.split(x_all, [60000])
  >>> y_train, y_test = np.split(y_all, [60000])

Next, we want to define the architecture.
We use a simple three-layer rectifier network with 100 units per layer as an example.
Before defining forward routine, we have to prepare parameterized functions::

  >>> model = FunctionSet(
  ...     l1 = F.Linear(784, 100),
  ...     l2 = F.Linear(100, 100),
  ...     l3 = F.Linear(100,  10),
  ... )
  >>> optimizer = optimizers.SGD()
  >>> optimizer.setup(model.collect_parameters())

Note that ``model.l3`` is the final linear layer whose output corresponding to the ten digits.
We also set up optimizer here.

Now we can define the forward routine using these Linear functions.
Typically it is defined as a simple python function given input arrays::

  >>> def forward(x_data, y_data):
  ...     x = Variable(x_data)
  ...     t = Variable(y_data)
  ...     h1 = F.relu(model.l1(x))
  ...     h2 = F.relu(model.l2(h1))
  ...     y = model.l3(h2)
  ...     return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

This function uses :func:`functions.relu` as an activation function.
Since ReLU does not have parameters to optimize, it need not to be included in `model`.
:func:`functions.softmax_cross_entropy` computes the loss function of softmax regression.
:func:`functions.accuracy` computes the classification accuracy of this minibatch.

Finally, we can write a learning loop as following::

  >>> batchsize = 100
  >>> for epoch in xrange(20):
  ...     print 'epoch', epoch
  ...     indexes = np.random.permutation(60000)
  ...     for i in xrange(0, 60000, batchsize):
  ...         x_batch = x_train[indexes[i : i + batchsize]]
  ...         y_batch = y_train[indexes[i : i + batchsize]]
  ...         
  ...         optimizer.zero_grads()
  ...         loss, accuracy = forward(x_batch, y_batch)
  ...         loss.backward()
  ...         optimizer.update()

Only the last four lines are the code related to Chainer, which are already described above.

Here you find that, at each iteration, the network is defined by forward computation, used for backprop, and then disposed.
By leveraging this "Define-by-Run" scheme, you can imagine that recurrent nets with variable length input are simply handled by just using loop over different length input for each iteration.

After or during optimization, we want to evaluate the model on the test set.
It can be achieved simply by calling forward function::

  >>> sum_loss, sum_accuracy = 0, 0
  >>> for i in xrange(0, 10000, batchsize):
  ...     x_batch = x_test[i : i + batchsize]
  ...     y_batch = y_test[i : i + batchsize]
  ...     loss, accuracy = forward(x_batch, y_batch)
  ...     sum_loss      += loss.data * batchsize
  ...     sum_accuracy  += accuracy.data * batchsize
  ...
  >>> mean_loss     = sum_loss / 10000
  >>> mean_accuracy = sum_accuracy / 10000

The example code contains GPU support, though the essential part is same as the code in this tutorial.
We will review in later sections how to use GPU(s).
