import collections

import numpy
from chainer import Function
from six import iteritems

from six.moves import zip
from six.moves.queue import PriorityQueue


class TreeParser(object):

    def __init__(self):
        self.next_id = 0

    def size(self):
        return self.next_id

    def get_paths(self):
        return self.paths

    def get_codes(self):
        return self.codes

    def parse(self, tree):
        self.next_id = 0
        self.path = []
        self.code = []
        self.paths = {}
        self.codes = {}
        self._parse(tree)

        assert(len(self.path) == 0)
        assert(len(self.code) == 0)
        assert(len(self.paths) == len(self.codes))

    def _parse(self, node):
        if isinstance(node, tuple):
            # internal node
            if len(node) != 2:
                raise ValueError(
                    'All internal nodes must have two child nodes')
            left, right = node
            self.path.append(self.next_id)
            self.next_id += 1
            self.code.append(1.0)
            self._parse(left)

            self.code[-1] = -1.0
            self._parse(right)

            self.path.pop()
            self.code.pop()

        else:
            # leaf node
            self.paths[node] = numpy.array(self.path).astype(numpy.int32)
            self.codes[node] = numpy.array(self.code).astype(numpy.float32)


class BinaryHierarchicalSoftmax(Function):

    """Implementation of hierarchical softmax (HSM).

    In natural language applications, vocabulary size is too large to use
    softmax loss.
    Instead, the hierarchical softmax uses product of sigmoid functions.
    It costs only :math:`O(\log(n))` time where :math:`n` is the vocabulary
    size in average.

    At first a user need to prepare a binary tree whose each leaf is
    corresponding to a word in a vocabulary.
    When a word :math:`x` is given, exactly one path from the root of the tree
    to the leaf of the word exists.
    Let :math:`\mbox{path}(x) = ((e_1, b_1), \dots, (e_m, b_m))` be the path of
    :math:`x`, where :math:`e_i` is an index of :math:`i`-th internal node, and
    :math:`b_i \in \{-1, 1\}` indicates direction to move at :math:`i`-th
    internal node (-1 is left, and 1 is right).
    Then, the probability of :math:`x` is given as below:

    .. math::

       P(x) &= \prod_{(e_i, b_i) \in \mbox{path}(x)}P(b_i | e_i)  \\\\
            &= \prod_{(e_i, b_i) \in \mbox{path}(x)}\sigma(b_i x^\\top w_{e_i}),

    where :math:`\sigma(\\cdot)` is a sigmoid function, and :math:`w` is a
    weight matrix.

    This function costs :math:`O(\log(n))` time as an average length of paths is
    :math:`O(\log(n))`, and :math:`O(n)` memory as the number of internal nodes
    equals :math:`n - 1`.

    Args:
        in_size (int): Dimension of input vectors.
        tree: A binary tree made with tuples like `((1, 2), 3)`.

    See: Hierarchical Probabilistic Neural Network Language Model [Morin+, AISTAT2005].

    """

    parameter_names = ('W',)
    gradient_names = ('gW',)

    def __init__(self, in_size, tree):
        parser = TreeParser()
        parser.parse(tree)
        self.paths = parser.get_paths()
        self.codes = parser.get_codes()

        self.W = numpy.random.uniform(-1, 1,
                                      (parser.size(), in_size)).astype(numpy.float32)
        self.gW = numpy.zeros(self.W.shape, numpy.float32)

    def forward_cpu(self, args):
        x, t = args
        assert x.ndim == 2 and x.dtype.kind == 'f'
        assert t.ndim == 1 and t.dtype.kind == 'i'
        assert len(x) == len(t)

        loss = 0.0
        for ix, it in zip(x, t):
            loss += self._forward_cpu_one(ix, it)
        return numpy.array([loss]),

    def _forward_cpu_one(self, x, t):
        assert t in self.paths

        w = self.W[self.paths[t]]
        wxy = w.dot(x) * self.codes[t]
        loss = numpy.logaddexp(0, -wxy)  # == log(1 + exp(-wxy))
        return numpy.sum(loss)

    def backward_cpu(self, args, loss):
        x, t = args
        gloss, = loss
        gx = numpy.empty_like(x)
        for i, (ix, it) in enumerate(zip(x, t)):
            gx[i] = self._backward_cpu_one(ix, it, gloss[0])
        return gx, None

    def _backward_cpu_one(self, x, t, gloss):
        path = self.paths[t]
        w = self.W[path]
        wxy = w.dot(x) * self.codes[t]
        g = -gloss * self.codes[t] / (1.0 + numpy.exp(wxy))
        gx = g.dot(w)
        gw = g.reshape((g.shape[0], 1)).dot(x.reshape(1, x.shape[0]))
        self.gW[path] += gw
        return gx


def create_huffman_tree(word_counts):
    """Make a huffman tree from a dictionary containing word counts.

    This method creates a binary huffman tree, that is required for
    :class:`BinaryHierarchicalSoftmax`.
    For example, ``{0: 8, 1: 5, 2: 6, 3: 4}`` is converted to
    ``((3, 1), (2, 0))``.

    Args:
        word_counts (``dict`` of ``int`` key and ``int`` or ``float`` values.):
            Dictionary representing counts of words.

    Returns:
        Binary huffman tree with tuples and keys of ``word_coutns``.

    """
    if len(word_counts) == 0:
        raise ValueError('Empty vocabulary')

    q = PriorityQueue()
    for w, c in iteritems(word_counts):
        q.put((c, w))

    while q.qsize() >= 2:
        (count1, word1) = q.get()
        (count2, word2) = q.get()
        count = count1 + count2
        tree = (word1, word2)
        q.put((count, tree))

    return q.get()[1]
