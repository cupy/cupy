Recurrent Nets and their Computational Graph
--------------------------------------------

.. currentmodule:: chainer

In this section, you will learn how to write

* recurrent nets with full backprop,
* recurrent nets with truncated backprop,
* evaluation of networks with few memory.

After reading this section, you will be able to:

* Handle input sequences of variable length
* Truncate upper stream of the network during forward computation
* Use volatile variables to prevent network construction


Recurrent Nets
~~~~~~~~~~~~~~

Recurrent nets are neural networks with loops.
They are often used to learn from sequential input/output.
Given an input stream :math:`x_1, x_2, \dots, x_t, \dots` and the initial state :math:`h_0`, a recurrent net iteratively updates its state by :math:`h_t = f(x_t, h_{t-1})`, and at some or every point in time :math:`t`, it outputs :math:`y_t = g(h_t)`.
If we expand the procedure along the time axis, it looks like a regular feed-forward network except that same parameters are periodically used within the network.

Here we learn how to write a simple one-layer recurrent net.
The task is language modeling: given a finite sequence of words, we want to predict the next word at each position without peeking the successive words.
Suppose that there are 1,000 different word types, and that we use 100 dimensional real vectors to represent each word (a.k.a. word embedding).

Before writing the forward computation, we have to define parameterized functions::

  model = FunctionSet(
      embed  = F.EmbedID(1000, 100),
      x_to_h = F.Linear(100,   50),
      h_to_h = F.Linear( 50,   50),
      h_to_y = F.Linear( 50, 1000),
  )
  optimizer = optimizers.SGD()
  optimizer.setup(model.collect_parameters())

Here :class:`~functions.EmbedID` is a parameterized function class for word embedding.
It converts input integers into corresponding fixed-dimensional embedding vectors.
Other Linear layers represent the transformation as their names indicate.
Here we use 50 hidden units.

Then, we can write down the forward computation.
Suppose that the input word sequence is given as a list of integer arrays.
The forward computation is simply written with a for loop::

  def forward_one_step(h, cur_word, next_word, volatile=False):
      word = Variable(cur_word, volatile=volatile)
      t    = Variable(next_word, volatile=volatile)
      x    = F.tanh(model.embed(word))
      h    = F.tanh(model.x_to_h(x) + model.h_to_h(h))
      y    = model.h_to_y(h)
      loss = F.softmax_cross_entropy(y, t)
      return h, loss

  def forward(x_list, volatile=False):
      h = Variable(np.zeros((1, 50), dtype=np.float32), volatile=volatile)
      loss = 0
      for cur_word, next_word in zip(x_list, x_list[1:]):
          h, new_loss = forward_one_step(h, cur_word, next_word, volatile=volatile)
          loss += new_loss
      return loss

Note that the first dimension of ``h`` and ``x_list`` is always the mini-batch size.
The mini-batch size is assumed to be ``1`` here.
We implemented the one-step-forward computation as a separate function, which is a best practice of writing recurrent nets for higher extensibility.
Ignore the argument ``volatile`` for now, we will review it in the next subsection.
The ``forward`` function is very simple and no special care needs to be taken with respect to the length of the input sequence.
This code actually handles variable length input sequences without any tricks.

Of course, the accumulated loss is a Variable object with the full history of computation.
So we can just call its :meth:`~Variable.backward` method to compute gradients of the total loss according to the model parameters::

  optimizer.zero_grads()
  loss = forward(x_list)
  loss.backward()
  optimizer.update()

Do not forget to call :meth:`Optimizer.zero_grads` before the backward computation!

Truncate the Graph by Unchaining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Learning from very long sequences is also a typical use case of recurrent nets.
Suppose that the input and state sequence is too long to fit into memory.
In such cases, we often truncate the backpropagation into a short time range.
This technique is called *truncated backprop*.
It is heuristic, and it makes the gradients biased.
However, this technique works well in practice if the time range is long enough.

How to implement truncated backprop in Chainer?
Chainer has a smart mechanism to achieve truncation, called **backward unchaining**.
It is implemented in the :meth:`Variable.unchain_backward` method.
Backward unchaining starts from the Variable object, and it chops the computation history backwards from the variable.
The chopped variables are disposed automatically (if they are not referenced explicitly from any other user object).
As a result, they are no longer a part of computation history, and are not involved in backprop anymore.

Let's write an example of truncated backprop.
Here we use the same network as the one used in the previous subsection.
Suppose that we are given a very long sequence, and we want to run backprop truncated at every 30 time steps.
We can write truncated backprop using the ``forward_one_step`` function that we wrote above. ::

  h = Variable(np.zeros((1, 50), dtype=np.float32))
  loss   = 0
  count  = 0
  seqlen = len(x_list[1:])
  
  for cur_word, next_word in zip(x_list, x_list[1:]):
      h, new_loss = forward_one_step(h, cur_word, next_word)
      loss  += new_loss
      count += 1
      if count % 30 == 0 or count == seqlen:
          optimizer.zero_grads()
          loss.backward()
          loss.unchain_backward()
          optimizer.update()

State is updated at ``foward_one_step``, and the losses are accumulated to ``loss`` variable.
At each 30 steps, backprop takes place at the accumulated loss.
Then, the :meth:`~Variable.unchain_backward` method is called, which deletes the computation history backward from the accumulated loss.
Note that the latest state ``h`` itself is not lost, since above code holds a reference to it.

The implementation of truncated backprop is simple, and since there is no complicated trick on it, we can generalize this method to different situations.
For example, we can easily extend the above code to use different schedules between backprop timing and truncation length.

Network Evaluation without Storing the Computation History
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On evaluation of recurrent nets, there is typically no need to store the computation history.
While unchaining enables us to walk through unlimited length of sequences with limited memory, it is a bit of a work-around.

As an alternative, Chainer provides an evaluation mode of forward computation which does not store the computation history.
This is enabled by just passing ``volatile`` flag to all input variables.
Such variables are called *volatile variables*.

.. warning::

   It is not allowed to mix volatile and non-volatile variables as arguments to same function.

Remember that our ``forward`` function accepts ``volatile`` argument.
So we can enable volatile forward computation by just passing ``volatile=True`` to this function::

  loss = forward(x_list, volatile=True)

Volatile variables are also useful to evaluate feed-forward networks.

Variable's volatility can be changed directly by setting the :attr:`Variable.volatile` attribute.
This enables us to combine a fixed feature extractor network and a trainable predictor network.
For example, suppose that we want to train a feed-forward network ``predictor_func``, which is located on top of another fixed pretrained network ``fixed_func``.
We want to train ``predictor_func`` without storing the computation history for ``fixed_func``.
This is simply done by following code snippets (suppose ``x_data`` and ``y_data`` indicate input data and label, respectively)::

  x = Variable(x_data, volatile=True)
  feat = fixed_func(x)
  feat.volatile = False
  y = predictor_func(feat)
  y.backward()

At first, the input variable ``x`` is volatile, so ``fixed_func`` is executed in volatile mode, i.e. without memorizing the computation history.
Then the intermediate variable ``feat`` is manually set to non-volatile, so ``predictor_func`` is executed in non-volatile mode, i.e., with memorizing the history of computation.
Since the history of computation is only memorized between variables ``feat`` and ``y``, the backward computation stops at the ``feat`` variable.

---------

In this section we have demonstrated how to write recurrent nets in Chainer and some fundamental techniques to manage the history of computation (a.k.a. computational graph).
The example in the ``examples/ptb`` directory implements truncated backprop learning of a LSTM language model from the Penn Treebank corpus.
In the next section, we will review how to use GPU(s) in Chainer.
