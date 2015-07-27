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

Each implementations of :class:`Function` has a method for type check, :meth:`check_type_forward`.
This function is called just before :meth:`forward` method of the :class:`Function`.
You can override this method to check the condition about types of arguments.

:meth:`check_type_forward` gets an argument ``in_types``::

   def check_type_forward(self, in_types):
     ...

``in_types`` is an instance of :class:`utils.type_check.TypeInfoTuple`, that is a sub-class of ``tuple``.
To get type information about the first argument, use ``in_types[0]``.
If the function gets multiple arguments, we recommend to use new variables for readability::

  x_type, y_type = in_types

``x_type`` represents the type of the first argument, and ``y_type`` represents the second one in this case.

We describe usage of ``in_types`` with an example.
When you want to check if the number of dimention of ``x_type`` equals to ``2``, write this code::

  utils.type_check.expect(x_type.ndim == 2)

When this condition is true, nothing happens.
Otherwise this code throws an exception, and a user gets a message like this::

  Expect: in_types[0].ndim == 2
  Actual: 3 != 2

This error message means that "``ndim`` of the first argument expected to be ``2``, but actually it is ``3``".


Detail of type information
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can access three information of ``x_type``.

s- ``.shape`` is a ``tuple`` of ``int``. Each value is size of each dimension.
- ``.ndim`` is ``int`` value representing the number of dimensions. Note that ``ndim == len(shape)``
- ``.dtype`` is ``numpy.dtype`` representing data type of the value.

You can check all members.
For exaple, size of first dimension must to be positive, you can write like this::

  utils.type_check.expect(x_type.shape[0] > 0)

You can also check data types with ``.dtype``::

  utils.type_check.expect(x_type.dtype == numpy.float32)

And an error is like this::

  Expect: in_types[0].dtype == numpy.float32
  Actual: numpy.float64 != numpy.float32

You can also check ``kind`` of ``dtype``.
This code checks if the type is floating point::

  utils.type_check.expect(x_type.dtype.kind == 'f')

You can compare between variables.
For exaple, first argument and second arguments need to have the same length::

  utils.type_check.expect(x_type.shape[0] == y_type.shape[0])


Internal mechanism of type check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

How does it show an error message like ``"in_types[0].ndim == 2"``?
If ``x_type`` is an object containtnig ``ndim`` member variable, we cannot show such an error message because this equation is evaluated as an boolean value by Python interpreter.

Actually ``x_type`` is an :class:`utils.type_check.Expr` objects, and doesn't have a ``ndim`` member variable itself.
:class:`utils.type_check.Expr` represents a syntax tree.
``x_type.ndim`` makes a :class:`utils.type_check.Expr` object representing ``(getattr, x_type, 'ndim')``.
``x_type.ndim == 2`` makes an object like ``(eq, (getattr, x_type, 'ndim'), 2)``.
:meth:`tyep_check.expect` gets a :class:`utils.type_check.Expr` object and evaluate it.
When it ``True``, it causes nothing.
And, when it returns ``False``, this method shows a readable error message.

If you want to evaluate a :class:`utils.type_check.Expr` object, call :meth:`eval` method::

  actual_type = x_type.eval()

``actual_type`` is an instance of :class:`TypeInfo` though ``x_type`` is an instance of :class:`utils.type_check.Expr`.
In the same way, ``x_type.shape[0].eval()`` returns an int value.


More powerfull methods
~~~~~~~~~~~~~~~~~~~~~~

:class:`utils.type_check.Expr` class is more powerfull.
It supports all mathmatical operators such as ``+`` and ``*``.
You can write a condition that the first dimension of ``x_type`` is the first dimension of ``y_type`` times four::

  x_type.shape[0] == y_type.shape[0] * 4

When ``x_type.shape[0] == 3`` and ``y_type.shape[0] == 1``, users can get the error message below::

  Expect: in_types[0].shape[0] == in_types[1].shape[0] * 4
  Actual: 3 != 4


To compare a member variable of your function, wrap a value with :class:`utils.type_check.Variable` to show readable error message::

  x_type.shape[0] == utils.type_check.Variable(self.in_size, "in_size")

This code can check the equivalent condition below::

  x_type.shape[0] == self.in_size

However, the later condition doesn't know meanig of this value.
When this condition is not satisfied, the later shows unreadable error message::

  Expect: in_types[0].shape[0] == 4  # what does '4' mean?
  Actual: 3 != 4

Note that the second argument of :class:`utils.type_check.Variable` is only for readability.

The former shows this message::

  Expect: in_types[0].shape[0] == in_size  # OK, `in_size` is a value that is given to the constructor
  Actual: 3 != 4  # You can also check actual value here


Call functions
~~~~~~~~~~~~~~

How to check summation of all values of shape?
:class:`utils.type_check.Expr` also supports function call.
::

   sum = utils.type_check.Variable('sum', numpy.sum)
   utils.type_check.expect(sum(x_type.shape) == 10)

Why do we need to wrap the function ``numpy.sum`` with :class:`utils.type_check.Variable`?
``x_type.shape`` is not a tuple but an object of :class:`utils.type_check.Expr` as I wrote before.
Therefore, ``numpy.sum(x_type.shape)`` fails.
We need to evaluate this function lazily.

The above example makes an error message like this::

   Expect: sum(in_types[0].shape) == 10
   Actual: 7 != 10


More complicated cases
~~~~~~~~~~~~~~~~~~~~~~

How to write a more complicated condition that can't be written these operators?
You can evaluate :class:`utils.type_check.Expr` and get its result value with :meth:`eval` method.
And, check the condition and show warning message by your hand::

  x_shape = x_type.shape.eval()  # get actual shape (int tuple)
  if not more_complicated_condition(x_shape):
      expect_msg = 'Shape is expected to be ...'
      actual_msg = 'Shape is ...'
      raise utils.type_check.InvalidType(expect_msg, actual_msg)

Please make a readable error message.
This code generates an error below::

  Expect: Shape is expected to be ...
  Actual: Shape is ...



Typical type check example
~~~~~~~~~~~~~~~~~~~~~~~~~~

We show a typical type check for a function.

First check the number of arguments::

  utils.type_check.expect(in_types.size() == 2)

``in_types.size()`` returns a :class:`Expr` object representing a number of arguments.
You can check it in the same way.

And then, get each type::

  x_type, y_type = in_types

Don't get each value before check ``in_types.size()``.
When the number of argument is illegal, this process may fail.
For example, this code doesn't work when the size of ``in_types`` is zero::

  utils.type_check.expect(
    in_types.size() == 1,
    in_types[0].ndim == 1,
  )

After that, check each type::

  utils.type_check.expect(
    x_type.dtype == numpy.float32,
    x_type.ndim == 2,
    x_type.shape[1] == 4,
  )

The above example works correctly even when ``x_type.ndim == 0`` as all conditions are evaluated lazily.
