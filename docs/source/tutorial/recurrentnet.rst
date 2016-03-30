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
Suppose there are 1,000 different word types, and that we use 100 dimensional real vectors to represent each word (a.k.a. word embedding).

Let's start from defining the recurrent neural net language model (RNNLM) as a chain.
We can use the :class:`chainer.links.LSTM` link that implements a fully-connected stateful LSTM layer.
This link looks like an ordinary fully-connected layer.
On construction, you pass the input and output size to the constructor:

.. doctest::

   >>> l = L.LSTM(100, 50)

Then, call on this instance ``l(x)`` executes *one step of LSTM layer*:

.. doctest::

   >>> l.reset_state()
   >>> x = Variable(np.random.randn(10, 100).astype(np.float32))
   >>> y = l(x)

Do not forget to reset the internal state of the LSTM layer before the forward computation!
Every recurrent layer holds its internal state (i.e. the output of the previous call).
At the first application of the recurrent layer, you must reset the internal state.
Then, the next input can be directly fed to the LSTM instance:

.. doctest::

   >>> x2 = Variable(np.random.randn(10, 100).astype(np.float32))
   >>> y2 = l(x2)

Based on this LSTM link, let's write our recurrent network as a new chain:

.. testcode::

   class RNN(Chain):
       def __init__(self):
           super(RNN, self).__init__(
               embed=L.EmbedID(1000, 100),  # word embedding
               mid=L.LSTM(100, 50),  # the first LSTM layer
               out=L.Linear(50, 1000),  # the feed-forward output layer
           )

       def reset_state(self):
           self.mid.reset_state()

       def __call__(self, cur_word):
           # Given the current word ID, predict the next word.
           x = self.embed(cur_word)
           h = self.mid(x)
           y = self.out(h)
           return y

   rnn = RNN()
   model = L.Classifier(rnn)
   optimizer = optimizers.SGD()
   optimizer.setup(model)

Here :class:`~chainer.links.EmbedID` is a link for word embedding.
It converts input integers into corresponding fixed-dimensional embedding vectors.
The last linear link ``out`` represents the feed-forward output layer.

The ``RNN`` chain implements a *one-step-forward computation*.
It does not handle sequences by itself, but we can use it to process sequences by just feeding items in a sequence straight to the chain.

Suppose we have a list of word variables ``x_list``.
Then, we can compute loss values for the word sequence by simple ``for`` loop.

.. testcode::

   def compute_loss(x_list):
       loss = 0
       for cur_word, next_word in zip(x_list, x_list[1:]):
           loss += model(cur_word, next_word)
       return loss

Of course, the accumulated loss is a Variable object with the full history of computation.
So we can just call its :meth:`~Variable.backward` method to compute gradients of the total loss according to the model parameters:

.. testcode::
   :hide:

   x_list = [Variable(np.random.randint(255, size=(1,)).astype(np.int32))
             for _ in range(100)]

.. testcode::

   # Suppose we have a list of word variables x_list.
   rnn.reset_state()
   model.zerograds()
   loss = compute_loss(x_list)
   loss.backward()
   optimizer.update()

Or equivalently we can use the ``compute_loss`` as a loss function:

.. testcode::

   rnn.reset_state()
   optimizer.update(compute_loss, x_list)


Truncate the Graph by Unchaining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Learning from very long sequences is also a typical use case of recurrent nets.
Suppose the input and state sequence is too long to fit into memory.
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
Suppose we are given a very long sequence, and we want to run backprop truncated at every 30 time steps.
We can write truncated backprop using the model defined above:

.. testcode::

   loss = 0
   count = 0
   seqlen = len(x_list[1:])

   rnn.reset_state()
   for cur_word, next_word in zip(x_list, x_list[1:]):
       loss += model(cur_word, next_word)
       count += 1
       if count % 30 == 0 or count == seqlen:
           model.zerograds()
           loss.backward()
           loss.unchain_backward()
           optimizer.update()

State is updated at ``model()``, and the losses are accumulated to ``loss`` variable.
At each 30 steps, backprop takes place at the accumulated loss.
Then, the :meth:`~Variable.unchain_backward` method is called, which deletes the computation history backward from the accumulated loss.
Note that the last state of ``model`` is not lost, since the RNN instance holds a reference to it.

The implementation of truncated backprop is simple, and since there is no complicated trick on it, we can generalize this method to different situations.
For example, we can easily extend the above code to use different schedules between backprop timing and truncation length.


Network Evaluation without Storing the Computation History
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On evaluation of recurrent nets, there is typically no need to store the computation history.
While unchaining enables us to walk through unlimited length of sequences with limited memory, it is a bit of a work-around.

As an alternative, Chainer provides an evaluation mode of forward computation which does not store the computation history.
This is enabled by just passing ``volatile`` flag to all input variables.
Such variables are called *volatile variables*.

Volatile variable is created by passing ``volatile='on'`` at the construction::

   x_list = [Variable(..., volatile='on') for _ in range(100)]  # list of 100 words
   loss = compute_loss(x_list)

Note that we cannot call ``loss.backward()`` to compute the gradient here, since the volatile variable does not remember the computation history.

Volatile variables are also useful to evaluate feed-forward networks to reduce the memory footprint.

Variable's volatility can be changed directly by setting the :attr:`Variable.volatile` attribute.
This enables us to combine a fixed feature extractor network and a trainable predictor network.
For example, suppose we want to train a feed-forward network ``predictor_func``, which is located on top of another fixed pre-trained network ``fixed_func``.
We want to train ``predictor_func`` without storing the computation history for ``fixed_func``.
This is simply done by following code snippets (suppose ``x_data`` and ``y_data`` indicate input data and label, respectively)::

   x = Variable(x_data, volatile='on')
   feat = fixed_func(x)
   feat.volatile = 'off'
   y = predictor_func(feat)
   y.backward()

At first, the input variable ``x`` is volatile, so ``fixed_func`` is executed in volatile mode, i.e. without memorizing the computation history.
Then the intermediate variable ``feat`` is manually set to non-volatile, so ``predictor_func`` is executed in non-volatile mode, i.e., with memorizing the history of computation.
Since the history of computation is only memorized between variables ``feat`` and ``y``, the backward computation stops at the ``feat`` variable.

.. warning::

   It is not allowed to mix volatile and non-volatile variables as arguments to same function.
   If you want to create a variable that behaves like a non-volatile variable while can be mixed with volatile ones, use ``'auto'`` flag instead of ``'off'`` flag.

---------

In this section we have demonstrated how to write recurrent nets in Chainer and some fundamental techniques to manage the history of computation (a.k.a. computational graph).
The example in the ``examples/ptb`` directory implements truncated backprop learning of a LSTM language model from the Penn Treebank corpus.
In the next section, we will review how to use GPU(s) in Chainer.
