Define your own function
------------------------

.. currentmodule:: chainer

In this section, you will learn following things.

* How to define a non-parameterized function
* Useful tools to write a function with GPU
* How to define a parameterized function
* How to test the function definition

By reading this section, you will come to be able to

* Write your own non-parameterized function
* Define simple kernels in the function definition
* Write your own parameterized function


Non-parameterized functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chainer provides a collection of functions in :mod:`functions` module.
It covers typicall usages in deep learning, so many existing works can be implemented using them.
On the other hand, deep learning is evolving rapidly and we cannot cover all possible functions to define unseen architectures.
So it is important to learn how to define your own functions.

Since non-parameterized functions are simpler, we first review how to define them.
First, suppose we want to define elementwise function :math:`f(x, y, z) = x * y + z`.
We can implement this equation just by using ``*`` and ``+`` functions.
Though, defining a unified function may reduce memory consumption, so it is a bit meaningful.
Here we call this function as *MulAdd*.

Let's start with defining MulAdd working on CPU.
Any function must inherits :class:`Function` class.
The skeleton of a non-parameterized function looks like::

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

   Be careful that you must return a tuple of an array even if it returns just one array.

MulAdd is so simple, and implemented as follows::

  class MulAdd(Function):
      def forward_cpu(self, inputs):
          x, y, z = inputs
          w = x * y + z
          return w,

      def backward_cpu(self, inputs, grad_outputs):
          x, y, z = inputs
          gw      = grad_outputs[0]

          gx = y * gw
          gy = x * gw
          gz = gw
          return gx, gy, gz

As we warn you above, forward_cpu function returns a tuple of single element.
The forward function is straightforward.
Note that all arrays appeared in CPU functions are :class:`numpy.ndarray`.
It unpacks the input tuple, computes the output, and packs it into a tuple.
The backward function is a bit complicated.
Recall the rule of differentiation of multiplication.
This example just implement the rule.
And look at the return values: it just packs the gradient of each input in same order and returns it.

By just defining the core computation of forward and backward, Function class provides a chaining logic on it (i.e. storing the history of computation, etc.).

Then, let's define GPU methods.
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
          gw      = grad_outputs[0]

          gx = y * gw
          gy = x * gw
          gz = gw
          return gx, gy, gz

In GPU methods, arrays are of type :class:`pycuda.gpuarray.GPUArray`
We use arithmetic operators defined for GPUArray.
These operators implement the basic elementwise arithmetics.

You maybe find that the definitions of GPU methods are exactly same as those of CPU methods.
In such case, we can reduce them to :meth:`~Function.forward` and :meth:`~Function.backward` methods::

  class MulAdd(Function):
      def forward(self, inputs):
          x, y, z = inputs
          w = x * y + z
          return w,

      def backward(self, inputs, grad_outputs):
          x, y, z = inputs
          gw      = grad_outputs[0]

          gx = y * gw
          gy = x * gw
          gz = gw
          return gx, gy, gz

Note that this is a very rare case, since GPUArray does not imlement most features of :class:`numpy.ndarray`.


Write an elementwise kernel function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Above GPU implementation of MulAdd is already fast and parallelized on cores of GPU.
However, it invokes two kernels during each of forward and backward computations, which may hurt the performance.
We can reduce the number of invokations by defining our own kernel.

Most functions only require elementwise operations like MulAdd.
PyCUDA provides useful tool to define elementwise kernel, and Chainer wraps it by :func:`cuda.elementwise` function.
Our MulAdd implementation can be improved as follows::

  class MulAdd(Function):
      def forward_cpu(self, inputs):
          ...

      def backward_cpu(self, inputs, grad_outputs):
          ...

      def forward_gpu(self, inputs):
          x, y, z = inputs
          w = cuda.empty_like(x)
          cuda.elementwise(
              'float* w, const float* x, const float* y, const float* z',
              'w[i] = x[i] * y[i] + z[i]',
              'muladd_fwd')(w, x, y, z)
          return w,

      def backward_gpu(self, inputs, grad_outputs):
          x, y, z = inputs
          gw      = grad_outputs[0]

          gx = cuda.empty_like(x)
          gy = cuda.empty_like(y)
          cuda.elementwise(
              '''
                 float* gx, float* gy,
                 const float* x, const float* y, const float* gw
              ''', '''
                 gx[i] = gy[i] * gw[i];
                 gy[i] = gx[i] * gw[i];
              ''', 'muladd_bwd')(gx, gy, x, y, gw)

          gz = gw  # no copy
          return gx, gy, gz

:func:`cuda.elementwise` function accepts the essential implentation of the kernel function, and returns a kernel invokation function.
In typical usage, we pass three arguments to this function.
The first is an argument list of the kernel function.
The second is a body of *parallel loop*, where the variable ``i`` indicates the index in the loop.
Note that ``i`` runs through all indexes of the first array argument.
The third is the name of the kernel function, which is shown in debugger and profilers.

Above code does not run compilation on every forward/backward computation, since :func:`cuda.elementwise` function has two caching mechanisms.

First is *binary caching*.
:func:`cuda.elementwise` function caches the compiled binary under ``/tmp`` directory with a hash value of CUDA code, and reuse them if given code matches to the hash value.

Second is *upload caching*.
Given a compiled binary code, we have to upload it to the current GPU in order to execute it.
:func:`cuda.elementwise` function memoizes the arguments and the curent context, and if it is called with the same arguments and the same context, it reuses the previously uploaded kernel code.


Parameterized functions
~~~~~~~~~~~~~~~~~~~~~~~

Next, we show how to define a parameterized function.
At this time, suppose that we want to implement elementwise product function between the input array and the parameter array.

.. note::

   Note that the elementwise product between a variable and parameters can be simply implemented by :class:`functions.Parameter` function::

     p = F.Parameter(np.random.rand((4, 3), dtype=np.float32))
     x = Variable(...)
     y = p() * x

   The Parameter function takes no arguments and just returns a variable holding the parameter array.
   The example in this subsection may be slightly efficient on memory consumption, though.

There are two differences between parameterized functions and non-parameterized functions.

* Parameterized functions have parameter arrays and corresponding gradient arrays.
  They are typically stored as attributes of the function class, where the function should provide :attr:`~Function.parameter_names` and :attr:`~Function.gradient_names` attributes (or properties).
  Otherwise, the function must provide :attr:`~Function.parameters` and :attr:`~Function.gradients` properties directly.
* Parameterized functions must accumulate gradients on backward.

Note that gradient arrays are automatically zeroed by an optimizer, so function implementation only need to initialize their shapes.
Then, the implementation of elementwise product may be as following::

  class EltwiseParamProduct(Function):
      parameter_names = 'w',
      gradient_names  = 'gw',

      def __init__(self, shape):
          self.w  = np.random.randn(shape).astype(np.float32)
          self.gw = np.empty_like(self.w)

      def forward(self, inputs):
          x = inputs[0]
          y = self.w * x
          return y,

      def backward(self, inputs, grad_outputs):
          x  = inputs[0]
          gy = grad_outputs[0]

          self.gw += gy * x
          gx       = gy * self.w

          return gx,

.. note::

   An advanced tip to implement functions: if you want to preserve some information between forward and backward computations (e.g. to cache some arrays), you can store it as attributes.
   It does not make any trouble even if the function object is used more than once in the same network, since :meth:`Function.__call__` operator copies itself before the forward computation.

   .. warning::

      You should not assume that calls of forward and backward are one-to-one.
      You should think about users that call backward more than once after only one forward.


Testing function
~~~~~~~~~~~~~~~~

In order to isolate the cause of learning failure from implementation bugs, it is important to test function implementations.
Chainer provides simple utilities to help writing unittests.
They are defined in :mod:`gradient_check` module.

The most important test utility is :func:`~gradient_check.numerical_grad` function.
This function computes numerical gradient of given function using finite differences.
It can be used as follows::

  x  = np.random.randn(4, 3).astype(np.float32)
  gy = np.ones((4, 3), dtype=np.float32)
  f  = lambda: (x * x,)
  gx = gradient_check.numerical_grad(f, (x,), (gy,))

``f`` is a closure that returns a tuple of array(s) computed from input arrays.
The second and third arguments of :func:`~gradient_check.numerical_grad` are tuples of input arrays and output gradient arrays, respectively.
Above code computes numerical gradients of ``sum(f(x))``, where ``sum`` indicates the summation through all elements.
The summation can be weighted by changing ``gy``.
:func:`~gradient_check.numerical_grad` function also accepts additional ``eps`` argument, which indicates the quantization width of finite differences.

.. note::

   :func:`~gradient_check.numerical_grad` function accepts both CPU and GPU arrays.
   Note that we cannot mix CPU and GPU arrays.

Another utility is :func:`~gradient_check.assert_allclose` function.
This is similar to :func:`numpy.testing.assert_allclose` function.
The difference is that Chainer's version accepts CPU and GPU arrays as inputs.
We can mix them in one invokation of assert_allclose.
The default values of optional arguments are also different.

Here is a typical usage of gradient checking utilities.
This is a test example of :func:`functions.relu` function::

  class TestReLU(TestCase):
      def test_backward_cpu(self):
          x = Variable(np.random.randn(3, 2).astype(np.float32))
          y = F.relu(x)
          y.grad = np.random.randn(3, 2).astype(np.float32)
          y.backward()

          func = y.creator
          f = lambda: func.forward((x.data,))
          gx, = gradient_check.numerical_grad(f, (x.data,), (y.grad,))

          gradient_check.assert_allclose(gx, x.grad)

We used :attr:`Variable.creator` to extract creator function object of a variable.
The first four lines of the test code are simple forward and backward computation of ReLU function.
The next three lines compute numerical gradient using the same forward function without backward routine.
And at last, we compare these two results elementwise.
Note that above test code can be easily modified to test GPU version just by replacing CPU arrays to GPU arrays.

You can find many examples of function tests under ``tests/function_tests`` directory.
