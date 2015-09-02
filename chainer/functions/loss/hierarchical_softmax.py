import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


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


class BinaryHierarchicalSoftmax(function.Function):

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
            &= \prod_{(e_i, b_i) \in \mbox{path}(x)}\sigma(b_i x^\\top
               w_{e_i}),

    where :math:`\sigma(\\cdot)` is a sigmoid function, and :math:`w` is a
    weight matrix.

    This function costs :math:`O(\log(n))` time as an average length of paths
    is :math:`O(\log(n))`, and :math:`O(n)` memory as the number of internal
    nodes equals :math:`n - 1`.

    Args:
        in_size (int): Dimension of input vectors.
        tree: A binary tree made with tuples like `((1, 2), 3)`.

    See: Hierarchical Probabilistic Neural Network Language Model [Morin+,
    AISTAT2005].

    """

    parameter_names = ('W',)
    gradient_names = ('gW',)

    def __init__(self, in_size, tree):
        parser = TreeParser()
        parser.parse(tree)
        paths = parser.get_paths()
        codes = parser.get_codes()
        n_vocab = max(paths.keys()) + 1

        self.paths = numpy.concatenate(
            [paths[i] for i in range(n_vocab) if i in paths])
        self.codes = numpy.concatenate(
            [codes[i] for i in range(n_vocab) if i in codes])
        begins = numpy.empty((n_vocab + 1,), dtype=numpy.int32)
        begins[0] = 0
        for i in range(0, n_vocab):
            length = len(paths[i]) if i in paths else 0
            begins[i + 1] = begins[i] + length
        self.begins = begins

        self.W = numpy.random.uniform(
            -1, 1, (parser.size(), in_size)).astype(numpy.float32)
        self.gW = numpy.full_like(self.W, numpy.nan)

    @staticmethod
    def create_huffman_tree(word_counts):
        """Make a huffman tree from a dictionary containing word counts.

        This method creates a binary huffman tree, that is required for
        :class:`BinaryHierarchicalSoftmax`.
        For example, ``{0: 8, 1: 5, 2: 6, 3: 4}`` is converted to
        ``((3, 1), (2, 0))``.

        Args:
            word_counts (dict of int key and int or float values):
                Dictionary representing counts of words.

        Returns:
            Binary huffman tree with tuples and keys of ``word_coutns``.

        """
        if len(word_counts) == 0:
            raise ValueError('Empty vocabulary')

        q = six.moves.queue.PriorityQueue()
        # Add unique id to each entry so that we can compare two entries with
        # same counts.
        # Note that itreitems randomly order the entries.
        for uid, (w, c) in enumerate(six.iteritems(word_counts)):
            q.put((c, uid, w))

        while q.qsize() >= 2:
            (count1, id1, word1) = q.get()
            (count2, id2, word2) = q.get()
            count = count1 + count2
            tree = (word1, word2)
            q.put((count, min(id1, id2), tree))

        return q.get()[2]

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 2,
            t_type.dtype == numpy.int32,
            t_type.ndim == 1,
            x_type.shape[0] == t_type.shape[0]
        )

    def forward_cpu(self, args):
        x, t = args

        loss = numpy.float32(0.0)
        for ix, it in six.moves.zip(x, t):
            loss += self._forward_cpu_one(ix, it)
        return numpy.array(loss),

    def _forward_cpu_one(self, x, t):
        begin = self.begins[t]
        end = self.begins[t + 1]

        w = self.W[self.paths[begin:end]]
        wxy = w.dot(x) * self.codes[begin:end]
        loss = numpy.logaddexp(0.0, -wxy)  # == log(1 + exp(-wxy))
        return numpy.sum(loss)

    def backward_cpu(self, args, loss):
        x, t = args
        gloss, = loss
        gx = numpy.empty_like(x)
        for i, (ix, it) in enumerate(six.moves.zip(x, t)):
            gx[i] = self._backward_cpu_one(ix, it, gloss)
        return gx, None

    def _backward_cpu_one(self, x, t, gloss):
        begin = self.begins[t]
        end = self.begins[t + 1]

        path = self.paths[begin:end]
        w = self.W[path]
        wxy = w.dot(x) * self.codes[begin:end]
        g = -gloss * self.codes[begin:end] / (1.0 + numpy.exp(wxy))
        gx = g.dot(w)
        gw = g.reshape((g.shape[0], 1)).dot(x.reshape(1, x.shape[0]))
        self.gW[path] += gw
        return gx

    def to_gpu(self, device=None):
        function.Function.to_gpu(self, device)

        self.paths = cuda.to_gpu(self.paths, device)
        self.codes = cuda.to_gpu(self.codes, device)
        self.begins = cuda.to_gpu(self.begins, device)

    def to_cpu(self):
        function.Function.to_cpu(self)

        self.paths = cuda.to_cpu(self.paths)
        self.codes = cuda.to_cpu(self.codes)
        self.begins = cuda.to_cpu(self.begins)

    def forward_gpu(self, inputs):
        x, t = inputs
        max_length = cuda.reduce(
            'T t, raw T begins', 'T out', 'begins[t + 1] - begins[t]',
            'max(a, b)', 'out = a', '0',
            'binary_hierarchical_softmax_max_length')(t, self.begins)
        max_length = cuda.to_cpu(max_length)[()]

        length = max_length * x.shape[0]
        ls = cuda.empty((length,), dtype=numpy.float32)
        n_in = x.shape[1]
        wxy = cuda.empty((length,), dtype=numpy.float32)
        cuda.elementwise(
            '''raw T x, raw T w, raw int32 ts, raw int32 paths,
            raw T codes, raw int32 begins, int32 c, int32 max_length''',
            'T ls, T wxy',
            '''
            int ind = i / max_length;
            int offset = i - ind * max_length;
            int t = ts[ind];

            int begin = begins[t];
            int length = begins[t + 1] - begins[t];

            if (offset < length) {
              int p = begin + offset;
              int node = paths[p];

              T wx = 0;
              for (int j = 0; j < c; ++j) {
                int w_ind[] = {node, j};
                int x_ind[] = {ind, j};
                wx += w[w_ind] * x[x_ind];
              }
              wxy = wx * codes[p];
              ls = log(1 + exp(-wxy));
            } else {
              ls = 0;
            }
            ''',
            'binary_hierarchical_softmax_forward'
        )(x, self.W, t, self.paths, self.codes, self.begins, n_in, max_length,
          ls, wxy)
        self.max_length = max_length
        self.wxy = wxy
        return ls.sum(),

    def backward_gpu(self, inputs, loss):
        x, t = inputs
        gloss, = loss

        n_in = x.shape[1]
        gx = cuda.zeros_like(x)
        cuda.elementwise(
            '''T wxy, raw T x, raw T w, raw int32 ts, raw int32 paths,
            raw T codes, raw int32 begins, raw T gloss,
            int32 c, int32 max_length''',
            'raw T gx, raw T gw',
            '''
            int ind = i / max_length;
            int offset = i - ind * max_length;
            int t = ts[ind];

            int begin = begins[t];
            int length = begins[t + 1] - begins[t];

            if (offset < length) {
              int p = begin + offset;
              int node = paths[p];
              T code = codes[p];

              T g = -gloss[0] * code / (1.0 + exp(wxy));
              for (int j = 0; j < c; ++j) {
                int w_ind[] = {node, j};
                int x_ind[] = {ind, j};
                atomicAdd(&gx[x_ind], g * w[w_ind]);
                atomicAdd(&gw[w_ind], g * x[x_ind]);
              }
            }
            ''',
            'binary_hierarchical_softmax_bwd'
        )(self.wxy, x, self.W, t, self.paths, self.codes,
          self.begins, gloss, n_in, self.max_length, gx, self.gW)
        return gx, None
