Type check
----------

.. currentmodule:: chainer

In this section, you will learn about the following things:

* Basic usage of type check
* Detail of type information
* Internal mechanism of type check
* More complicated cases
* Call functions
* Typical type check example

After reading this section, you will be able to:

* Write a code to check types of input arguments of your own functions


Basic usage of type check
~~~~~~~~~~~~~~~~~~~~~~~~~

When you call a function with an invalid type of array, you sometimes receive no error, but get an unexpected result by broadcasting.
When you use CUDA with an illegal type of array, it causes memory corruption, and you get a serious error.
These bugs are hard to fix.
Chainer can check preconditions of each function, and helps to prevent such problems.
These conditions may help a user to understand specification of functions.

Each implementation of :class:`Function` has a method for type check, :meth:`check_type_forward`.
This function is called just before the :meth:`forward` method of the :class:`Function` class.
You can override this method to check the condition on types and shapes of arguments.

:meth:`check_type_forward` gets an argument ``in_types``::

   def check_type_forward(self, in_types):
     ...

``in_types`` is an instance of :class:`~utils.type_check.TypeInfoTuple`, which is a sub-class of :class:`tuple`.
To get type information about the first argument, use ``in_types[0]``.
If the function gets multiple arguments, we recommend to use new variables for readability:

.. testcode::
   :hide:

   data = (np.empty((3, 2, 2), dtype=np.float32), np.empty((1, 2, 2), dtype=np.float32))
   in_types = utils.type_check.get_types(data, 'in_types', False)


.. testcode::

   x_type, y_type = in_types

In this case, ``x_type`` represents the type of the first argument, and ``y_type`` represents the second one.

We describe usage of ``in_types`` with an example.
When you want to check if the number of dimension of ``x_type`` equals to ``2``, write this code:

.. testcode::

   utils.type_check.expect(x_type.ndim == 2)

When this condition is true, nothing happens.
Otherwise this code throws an exception, and the user gets a message like this:

.. testoutput::
   :options: -IGNORE_EXCEPTION_DETAIL

   Traceback (most recent call last):
   ...
   InvalidType: Expect: in_types[0].ndim == 2
   Actual: 3 != 2

This error message means that "``ndim`` of the first argument expected to be ``2``, but actually it is ``3``".


Detail of type information
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can access three information of ``x_type``.

* ``.shape`` is a tuple of ints. Each value is size of each dimension.
* ``.ndim`` is :class:`int` value representing the number of dimensions. Note that ``ndim == len(shape)``
* ``.dtype`` is ``numpy.dtype`` representing data type of the value.

You can check all members.
For example, the size of the first dimension must be positive, you can write like this:

.. testcode::

   utils.type_check.expect(x_type.shape[0] > 0)

You can also check data types with ``.dtype``:

.. testcode::

   utils.type_check.expect(x_type.dtype == np.float64)

And an error is like this:

.. testoutput::
   :options: -IGNORE_EXCEPTION_DETAIL +NORMALIZE_WHITESPACE

   Traceback (most recent call last):
   ...
   InvalidType: Expect: in_types[0].dtype == <type 'numpy.float64'>
   Actual: float32 != <type 'numpy.float64'>

You can also check ``kind`` of ``dtype``.
This code checks if the type is floating point

.. testcode::

   utils.type_check.expect(x_type.dtype.kind == 'f')

You can compare between variables.
For example, the following code checks if the first argument and the second argument have the same length:

.. testcode::

   utils.type_check.expect(x_type.shape[1] == y_type.shape[1])


Internal mechanism of type check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

How does it show an error message like ``"in_types[0].ndim == 2"``?
If ``x_type`` is an object containing ``ndim`` member variable, we cannot show such an error message because this equation is evaluated as a boolean value by Python interpreter.

Actually ``x_type`` is a :class:`~utils.type_check.Expr` objects, and doesn't have a ``ndim`` member variable itself.
:class:`~utils.type_check.Expr` represents a syntax tree.
``x_type.ndim`` makes a :class:`~utils.type_check.Expr` object representing ``(getattr, x_type, 'ndim')``.
``x_type.ndim == 2`` makes an object like ``(eq, (getattr, x_type, 'ndim'), 2)``.
:meth:`type_check.expect` gets a :class:`~utils.type_check.Expr` object and evaluates it.
When it is ``True``, it causes no error and shows nothing.
Otherwise, this method shows a readable error message.

If you want to evaluate a :class:`~utils.type_check.Expr` object, call :meth:`eval` method:

.. testcode::

   actual_type = x_type.eval()

``actual_type`` is an instance of :class:`TypeInfo`, while ``x_type`` is an instance of :class:`~utils.type_check.Expr`.
In the same way, ``x_type.shape[0].eval()`` returns an int value.


More powerful methods
~~~~~~~~~~~~~~~~~~~~~

:class:`~utils.type_check.Expr` class is more powerful.
It supports all mathematical operators such as ``+`` and ``*``.
You can write a condition that the first dimension of ``x_type`` is the first dimension of ``y_type`` times four:

.. testcode::

   utils.type_check.expect(x_type.shape[0] == y_type.shape[0] * 4)

When ``x_type.shape[0] == 3`` and ``y_type.shape[0] == 1``, users can get the error message below:

.. testoutput::
   :options: -IGNORE_EXCEPTION_DETAIL

   Traceback (most recent call last):
   ...
   InvalidType: Expect: in_types[0].shape[0] == in_types[1].shape[0] * 4
   Actual: 3 != 4


To compare a member variable of your function, wrap a value with :class:`~utils.type_check.Variable` to show readable error message:

.. testcode::
   :hide:

   class Object(object):
       in_size = 2
   self = Object()

.. testcode::

   x_type.shape[0] == utils.type_check.Variable(self.in_size, "in_size")

This code can check the equivalent condition below:

.. testcode::

   x_type.shape[0] == self.in_size

However, the latter condition doesn't know the meaning of this value.
When this condition is not satisfied, the latter code shows unreadable error message::

  InvalidType: Expect: in_types[0].shape[0] == 4  # what does '4' mean?
  Actual: 3 != 4

Note that the second argument of :class:`utils.type_check.Variable` is only for readability.

The former shows this message::

  InvalidType: Expect: in_types[0].shape[0] == in_size  # OK, `in_size` is a value that is given to the constructor
  Actual: 3 != 4  # You can also check actual value here


Call functions
~~~~~~~~~~~~~~

How to check summation of all values of shape?
:class:`~utils.type_check.Expr` also supports function call:

.. testcode::

    sum = utils.type_check.Variable(np.sum, 'sum')
    utils.type_check.expect(sum(x_type.shape) == 10)

Why do we need to wrap the function ``numpy.sum`` with :class:`utils.type_check.Variable`?
``x_type.shape`` is not a tuple but an object of :class:`~utils.type_check.Expr` as we have seen before.
Therefore, ``numpy.sum(x_type.shape)`` fails.
We need to evaluate this function lazily.

The above example produces an error message like this:

.. testoutput::
   :options: -IGNORE_EXCEPTION_DETAIL

   Traceback (most recent call last):
   ...
   InvalidType: Expect: sum(in_types[0].shape) == 10
   Actual: 7 != 10


More complicated cases
~~~~~~~~~~~~~~~~~~~~~~

How to write a more complicated condition that can't be written with these operators?
You can evaluate :class:`~utils.type_check.Expr` and get its result value with :meth:`eval` method.
Then check the condition and show warning message by hand:

.. testcode::
   :hide:

   def more_complicated_condition(x):
       return False

.. testcode::

  x_shape = x_type.shape.eval()  # get actual shape (int tuple)
  if not more_complicated_condition(x_shape):
      expect_msg = 'Shape is expected to be ...'
      actual_msg = 'Shape is ...'
      raise utils.type_check.InvalidType(expect_msg, actual_msg)

Please write a readable error message.
This code generates the following error message:

.. testoutput::
   :options: -IGNORE_EXCEPTION_DETAIL

   Traceback (most recent call last):
   ...
   InvalidType: Expect: Shape is expected to be ...
   Actual: Shape is ...



Typical type check example
~~~~~~~~~~~~~~~~~~~~~~~~~~

We show a typical type check for a function.

First check the number of arguments:

.. testcode::

   utils.type_check.expect(in_types.size() == 2)

``in_types.size()`` returns a :class:`~utils.type_check.Expr` object representing the number of arguments.
You can check it in the same way.

And then, get each type:

.. testcode::

   x_type, y_type = in_types

Don't get each value before checking ``in_types.size()``.
When the number of argument is illegal, ``type_check.expect`` might output unuseful error messages.
For example, this code doesn't work when the size of ``in_types`` is 0:

.. testcode::

   utils.type_check.expect(
     in_types.size() == 2,
     in_types[0].ndim == 3,
   )

After that, check each type:

.. testcode::

   utils.type_check.expect(
     x_type.dtype == np.float32,
     x_type.ndim == 3,
     x_type.shape[1] == 2,
   )

The above example works correctly even when ``x_type.ndim == 0`` as all conditions are evaluated lazily.
