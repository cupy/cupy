Define your own function
------------------------

.. currentmodule:: chainer

In this section, you will learn about the following things:

* How to define a function on variables
* Useful tools to write a function using a GPU
* How to test the function definition

After reading this section, you will be able to:

* Write your own functions
* Define simple kernels in the function definition


Differentiable Functions
~~~~~~~~~~~~~~~~~~~~~~~~

Chainer provides a collection of functions in the :mod:`~chainer.functions` module.
It covers typical use cases in deep learning, so many existing works can be implemented with them.
On the other hand, deep learning is evolving rapidly and we cannot cover all possible functions to define unseen architectures.
So it is important to learn how to define your own functions.

First, suppose we want to define an elementwise function :math:`f(x, y, z) = x * y + z`.
While it is possible to implement this equation using a combination of the ``*`` and ``+`` functions,
defining it as a single function may reduce memory consumption, so it is not *only* a toy example.
Here we call this function *MulAdd*.

Let's start with defining MulAdd working on the CPU.
Any function must inherit the :class:`Function` class.
The skeleton of a function looks like::

   class MulAdd(Function):
       def forward_cpu(self, inputs):
           # do forward computation on CPU
           return some_tuple

       def backward_cpu(self, inputs, grad_outputs):
           # do backward computation on CPU
           return some_tuple

We must implement :meth:`~Function.forward_cpu` and :meth:`~Function.backward_cpu` methods.
The non-self arguments of these functions are tuples of array(s), and these functions must return a tuple of array(s).

.. warning::

   Be careful to return a tuple of arrays even if you have just one array to return.

MulAdd is simple and implemented as follows

.. testcode::

   class MulAdd(Function):
       def forward_cpu(self, inputs):
           x, y, z = inputs
           w = x * y + z
           return w,

       def backward_cpu(self, inputs, grad_outputs):
           x, y, z = inputs
           gw, = grad_outputs

           gx = y * gw
           gy = x * gw
           gz = gw
           return gx, gy, gz

.. testcode::
   :hide:

   x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   z = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   w = MulAdd()(x, y, z)
   w.grad = np.random.uniform(-1, 1, (3, 2)).astype(np.float32)
   w.backward()

As per the warning above, the ``forward_cpu`` method returns a tuple of single element.
Note that all arrays appearing in CPU functions are :class:`numpy.ndarray`.
The forward function is straightforward:
It unpacks the input tuple, computes the output, and packs it into a tuple.
The backward function is a bit more complicated.
Recall the rule of differentiation of multiplication.
This example just implements the rule.
Look at the return values, the function just packs the gradient of each input in same order and returns them.

By just defining the core computation of forward and backward,
Function class provides a chaining logic on it (i.e. storing the
history of computation, etc.).

.. note::
   Assuming we implement a (forward) function :math:`y=f(x)` which takes as input the
   vector :math:`x \in \mathbb{R}^n` and produces as output a vector
   :math:`y \in \mathbb{R}^m`. Then the ``backward`` method has to compute

   .. math::
      \lambda_i = \sum_{j=1}^m \frac{\partial y_j}{\partial x_i} \,
      \gamma_j \,\, \text{for}\, i = 1 \dots n

   where :math:`\gamma` is the ``grad_outputs``. Note, that the
   resulting vector :math:`\lambda` must have the same shape as the arguments of the ``forward`` method.



Now let's define the corresponding GPU methods.
You can easily predict that the methods we have to write are named :meth:`~Function.forward_gpu` and :meth:`~Function.backward_gpu`::

  class MulAdd(Function):
      def forward_cpu(self, inputs):
          ...

      def backward_cpu(self, inputs, grad_outputs):
          ...

      def forward_gpu(self, inputs):
          x, y, z = inputs
          w = x * y + z
          return w,

      def backward_gpu(self, inputs, grad_outputs):
          x, y, z = inputs
          gw, = grad_outputs

          gx = y * gw
          gy = x * gw
          gz = gw
          return gx, gy, gz

In GPU methods, arrays are of type :class:`cupy.ndarray`.
We use arithmetic operators defined for this class.
These operators implement the basic elementwise arithmetics.

You may find that the definitions of GPU methods are exactly same as those of CPU methods.
In that case, we can reduce them to :meth:`~Function.forward` and :meth:`~Function.backward` methods

.. testcode::

   class MulAdd(Function):
       def forward(self, inputs):
           x, y, z = inputs
           w = x * y + z
           return w,

       def backward(self, inputs, grad_outputs):
           x, y, z = inputs
           gw, = grad_outputs

           gx = y * gw
           gy = x * gw
           gz = gw
           return gx, gy, gz

.. testcode::
   :hide:

   x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   z = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   w = MulAdd()(x, y, z)
   w.grad = np.random.uniform(-1, 1, (3, 2)).astype(np.float32)
   w.backward()


Since the :class:`cupy.ndarray` class implements many methods of :class:`numpy.ndarray`, we can write these unified methods in most cases.

The MulAdd function is used as follows:

.. testcode::

   x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   z = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   w = MulAdd()(x, y, z)

It looks a bit ugly: we have to explicitly instantiate MulAdd before applying it to variables.
We also have to be careful that one instance of MulAdd must not be used multiple times, since it acts as a node in the computational graph.
In Chainer, we often define a thin wrapper Python function that hide the instantiation:

.. testcode::

   def muladd(x, y, z):
       return MulAdd()(x, y, z)

   w = muladd(x, y, z)


Unified forward/backward methods with NumPy/CuPy functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuPy also implements many functions that are compatible to those of NumPy.
We can write unified forward/backward methods with them.
Consider that we want to write a backprop-able function :math:`f(x, y) = \exp(x) + \exp(y)`.
We name it *ExpAdd* here.
It can be written straight-forward as follows

.. testcode::

   class ExpAdd(Function):
       def forward_cpu(self, inputs):
           x, y = inputs
           z = np.exp(x) + np.exp(y)
           return z,

       def backward_cpu(self, inputs, grad_outputs):
           x, y = inputs
           gz, = grad_outputs

           gx = gz * np.exp(x)
           gy = gz * np.exp(y)
           return gx, gy

       def forward_gpu(self, inputs):
           cupy = cuda.cupy
           x, y = inputs
           z = cupy.exp(x) + cupy.exp(y)
           return z,

       def backward_gpu(self, inputs, grad_outputs):
           cupy = cuda.cupy
           x, y = inputs
           gz, = grad_outputs

           gx = gz * cupy.exp(x)
           gy = gz * cupy.exp(y)
           return gx, gy

   def expadd(x, y):
       return ExpAdd()(x, y)


.. testcode::
   :hide:

   x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   z = ExpAdd()(x, y)
   z.grad = np.random.uniform(-1, 1, (3, 2)).astype(np.float32)
   z.backward()


.. note::
   Here we used ``cuda.cupy`` instead of directly accessing :mod:`cupy`.
   This is because the ``cupy`` module cannot be imported if the CUDA is not installed.
   In order to keep the implementation valid in non-CUDA environment, we have to defer the access to the ``cupy`` module.
   Note that the :mod:`chainer.cuda` module can be imported even if the CUDA is not installed.
   Of course, the module in such environment is almost useless, but if the interpreter does not run through the code accessing CUDA-dedicated functions, the code is still valid.

The CPU and GPU implementations are almost same, except that :mod:`numpy` is replaced by :mod:`cupy` in GPU methods.
We can unify these functions using the :func:`cuda.get_array_module` function.
This function accepts arbitrary number of arrays, and returns an appropriate module for them.
See the following code

.. testcode::

   class ExpAdd(Function):
       def forward(self, inputs):
           xp = cuda.get_array_module(*inputs)
           x, y = inputs
           z = xp.exp(x) + xp.exp(y)
           return z,

       def backward(self, inputs, grad_outputs):
           xp = cuda.get_array_module(*inputs)
           x, y = inputs
           gz, = grad_outputs

           gx = gz * xp.exp(x)
           gy = gz * xp.exp(y)
           return gx, gy

   def expadd(x, y):
       return ExpAdd()(x, y)


.. testcode::
   :hide:

   x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   z = ExpAdd()(x, y)
   z.grad = np.random.uniform(-1, 1, (3, 2)).astype(np.float32)
   z.backward()


Note that this code works correctly even if CUDA is not installed in the environment.
If CUDA is not found, get_array_module function always returns :mod:`numpy`.
We often use the name ``xp`` for the variadic module name, which is analogous to the abbreviation ``np`` for NumPy and ``cp`` for CuPy.


Write an Elementwise Kernel Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's turn back to the MulAdd example.

The GPU implementation of MulAdd as shown above is already fast and parallelized on GPU cores.
However, it invokes two kernels during each of forward and backward computations.
It might hurt performance, since the intermediate temporary arrays are read and written by possibly different GPU cores, which consumes much bandwidth.
We can reduce the number of invocations by defining our own kernel.
It also reduce the memory consumption.

Most functions only require elementwise operations like MulAdd.
CuPy provides a useful tool to define elementwise kernels, the :class:`cupy.elementwise.ElementwiseKernel` class, and Chainer wraps it by :func:`cuda.elementwise` function.
Our MulAdd implementation can be improved as follows::

  class MulAdd(Function):
      def forward_cpu(self, inputs):
          ...

      def backward_cpu(self, inputs, grad_outputs):
          ...

      def forward_gpu(self, inputs):
          cupy = cuda.cupy
          x, y, z = inputs
          w = cuda.elementwise(
              'float32 x, float32 y, float32 z',
              'float32 w',
              'w = x * y + z',
              'muladd_fwd')(x, y, z)
          return w,

      def backward_gpu(self, inputs, grad_outputs):
          x, y, z = inputs
          gw, = grad_outputs

          gx, gy = cuda.elementwise(
              'float32 x, float32 y, float32 gw',
              'float32 gx, float32 gy',
              '''
                 gx = y * gw;
                 gy = x * gw;
              ''',
              'muladd_bwd')(x, y, gw)

          gz = gw
          return gx, gy, gz

:func:`cuda.elementwise` function accepts the essential implementation of the kernel function, and returns a kernel invocation function (actually, it returns :class:`~cupy.elementwise.ElementwiseKernel` object, which is callable).
In typical usage, we pass four arguments to this function as follows:

1. Input argument list. This is a comma-separated string each entry of which consists of a type specification and an argument name.
2. Output argument list in the same format as the input argument list.
3. Body of *parallel loop*. We can use the input/output argument names as an element of these arrays.
4. Name of the kernel function, which is shown in debuggers and profilers.

Above code is not compiled on every forward/backward computation thanks to two caching mechanisms provided by :func:`cuda.elementwise`.

The first one is *binary caching*:
:func:`cuda.elementwise` function caches the compiled binary in the ``$(HOME)/.cupy/kernel_cache`` directory with a hash value of the CUDA code, and reuses it if the given code matches the hash value.
This caching mechanism is actually implemented in CuPy.

The second one is *upload caching*:
Given a compiled binary code, we have to upload it to the current GPU in order to execute it.
:func:`cuda.elementwise` function memoizes the arguments and the current device, and if it is called with the same arguments for the same device, it reuses the previously uploaded kernel code.

The above MulAdd code only works for float32 arrays.
The :class:`~cupy.elementwise.ElementwiseKernel` also supports the type-variadic kernel definition.
In order to define variadic kernel functions, you can use *type placeholder* by placing a single character as type specifier::

  class MulAdd(Function):
      def forward_cpu(self, inputs):
          ...

      def backward_cpu(self, inputs, grad_outputs):
          ...

      def forward_gpu(self, inputs):
          cupy = cuda.cupy
          x, y, z = inputs
          w = cuda.elementwise(
              'T x, T y, T z',
              'T w',
              'w = x * y + z',
              'muladd_fwd')(x, y, z)
          return w,

      def backward_gpu(self, inputs, grad_outputs):
          x, y, z = inputs
          gw, = grad_outputs

          gx, gy = cuda.elementwise(
              'T x, T y, T gw',
              'T gx, T gy',
              '''
                 gx = y * gw;
                 gy = x * gw;
              ''',
              'muladd_bwd')(x, y, gw)

          gz = gw
          return gx, gy, gz

The type placeholder ``T`` indicates an arbitrary data type that CuPy supports.

There are more functionalities on user-defined kernels in CuPy.
:ref:`See the CuPy documentation on user-defined kernels for more details. <udkernel>`


Links that wrap functions
~~~~~~~~~~~~~~~~~~~~~~~~~

Some functions are meant to be combined with parameters.
In such case, it is useful to write a small **link** that wraps the function.
We have already seen how to define a chain that wraps other links (by inheriting :class:`Chain` class).
Here we study how to define a link that does not hold any other links.

As the first example, suppose that we want to implement elementwise product function between the input array and the parameter array.
It can be defined as follows:

.. testcode::

   class EltwiseParamProduct(Link):
       def __init__(self, shape):
           # By passing a shape of the parameter, the initializer allocates a
           # parameter variable of the shape.
           super(EltwiseParamProduct, self).__init__(W=shape)
           self.W.data[...] = np.random.randn(*shape)

       def __call__(self, x):
           return self.W * x

.. testcode::
   :hide:

   x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
   f = EltwiseParamProduct((3, 2))
   y = f(x)
   y.grad = np.random.uniform(-1, 1, (3, 2)).astype(np.float32)
   y.backward()

We can also initialize the parameter after the initialization by the :meth:`Link.add_param` method.

.. testcode::

   class EltwiseParamProduct(Link):
       def __init__(self, shape):
           super(EltwiseParamProduct, self).__init__()
           self.add_param('W', shape)
           self.W.data[...] = np.random.randn(*shape)

       def __call__(self, x):
           return self.W * x

Note that the initializer and the :meth:`~Link.add_param` method does not initialize elements of the parameter array.
We have to manually initialize the elements by random values, zeros, etc.

For another example, assume we want to define a simple linear layer.
It is already defined as :class:`~chainer.links.Linear`, so this is an educational example.
The linear layer is divided into two parts: a function and its wrapper link.
First, we have to define a function on variables:

.. testcode::

   class LinearFunction(Function):
       def forward(self, inputs):
           x, W, b = inputs
           return x.dot(W.T) + b,

       def backward(self, inputs, grad_outputs):
           x, W, b = inputs
           gy, = grad_outputs

           gx = gy.dot(W)
           gW = gy.T.dot(x)
           gb = gy.sum(axis=0)
           return gx, gW, gb

   def linear(x, W, b):
       return LinearFunction()(x, W, b)

This function takes three arguments: input, weight, and bias.
It can be used as a part of model definition, though is inconvenient since the user have to manage the weight and bias parameters directly.
In order to make a convenient module, let's wrap it into a link:

.. testcode::

   class Linear(Link):
       def __init__(self, in_size, out_size):
           super(Linear, self).__init__(W=(out_size, in_size), b=out_size)
           self.W.data[...] = np.random.randn(out_size, in_size) / math.sqrt(in_size)
           self.b.data.fill(0)

       def __call__(self, x):
           return linear(x, self.W, self.b)

This link hides the parameters of the linear layer.

.. note::

   An advanced tip to implement functions: if you want to preserve some information between forward and backward computations (e.g. to cache some arrays), you can store it as attributes.
   Be careful that it might increase the memory consumption during the whole forward-backward computation.
   If you want to train very large networks on a GPU with limited memory, it is not recommended to cache arrays between forward and backward.
   There is one exception for this: caching the output arrays does not change the memory consumption, because they are also held by the output Variable objects.

   .. warning::

      You should not assume a one-to-one match of calls of forward and backward.
      Some users may call backward more than once after one forward call.


Testing Function
~~~~~~~~~~~~~~~~

In order to isolate the cause of learning failure from implementation bugs, it is important to test function implementations.
Chainer provides simple utilities to help writing unit tests.
They are defined in the :mod:`~chainer.gradient_check` module.

The most important test utility is the :func:`~gradient_check.numerical_grad` function.
This function computes the numerical gradient of given function using finite differences.
It can be used as follows

.. testcode::

   x  = np.random.randn(4, 3).astype(np.float32)
   gy = np.ones((4, 3), dtype=np.float32)
   f  = lambda: (x * x,)
   gx = gradient_check.numerical_grad(f, (x,), (gy,))

``f`` is a closure that returns a tuple of array(s) computed from input arrays.
The second and third arguments of :func:`~gradient_check.numerical_grad` are tuples of input arrays and output gradient arrays, respectively.
The code above computes the numerical gradients of ``sum(f(x))``, where ``sum`` indicates the summation over all elements.
The summation can be weighted by changing ``gy``.
:func:`~gradient_check.numerical_grad` function also accepts additional ``eps`` argument, which indicates the quantization width of finite differences.

.. note::

   :func:`~gradient_check.numerical_grad` function accepts both CPU and GPU arrays.
   Note that we cannot mix CPU and GPU arrays.

Another utility is :func:`~gradient_check.assert_allclose` function.
This is similar to :func:`numpy.testing.assert_allclose` function.
The difference is that Chainer's version accepts CPU and GPU arrays as inputs.
We can mix them in one invocation of :func:`~gradient_check.assert_allclose`.
The default values of optional arguments are also different.

Here is a typical usage of gradient checking utilities.
This is a test example of :func:`functions.relu` function

.. testcode::

   import unittest

   class TestReLU(unittest.TestCase):
       def test_backward_cpu(self):
           x = Variable(np.random.randn(3, 2).astype(np.float32))
           y = F.relu(x)
           y.grad = np.random.randn(3, 2).astype(np.float32)
           y.backward()

           f = lambda: (F.relu(x).data,)
           gx, = gradient_check.numerical_grad(f, (x.data,), (y.grad,))

           gradient_check.assert_allclose(gx, x.grad)


.. testcode::
   :hide:

   suite = unittest.TestLoader().loadTestsFromTestCase(TestReLU)
   unittest.TextTestRunner().run(suite)


The first four lines of the test code are simple forward and backward computation of ReLU function.
The next two lines compute numerical gradient using the same forward function without backward routine.
And at last, we compare these two results elementwise.
Note that the above test code can be easily modified to test GPU version just by replacing CPU arrays to GPU arrays.

You can find many examples of function tests under ``tests/chainer_tests/function_tests`` directory.
