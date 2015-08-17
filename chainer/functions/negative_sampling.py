import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check
from chainer.utils import walker_alias


class NegativeSampling(function.Function):
    """Implementation of negative sampling.

    In natural language processing, especially language modeling, the number of
    vocabulary is very large.
    Therefore, you need to spend a lot of time to calculate the gradient of the
    embedding matrix.

    Instead, in negative sampling trick, you only need to calculate the
    gradient for a few sampled negative examples.

    The objective function is below:

    .. math::

       f(x, p) = \log\sigma(x^\\top w_p) + \\
       k E_{i \sim P(i)}[\log\sigma(- x^\\top w_i)],

    where :math:`\sigma(\cdot)` is a sigmoid function, :math:`w_i` is the
    weight vector for the word :math:`i`, and :math:`p` is a positive example.
    It is approximeted with :math:`k` examples :math:`N` sampled from
    probability :math:`P(i)`, like this:

    .. math::

       f(x, p) \\approx \log\sigma(x^\\top w_p) + \\
       \sum_{n \in N} \log\sigma(-x^\\top w_n).

    Each sample of :math:`N` is drawn from the word distribution :math:`P(w)`.
    This is calculated as :math:`P(w) = \\frac{1}{Z} c(w)^\\alpha`, where
    :math:`c(w)` is the unigram count of the word :math:`w`, :math:`\\alpha` is
    a hyper-parameter, and :math:`Z` is the normalization constant.

    Args:
        in_size (int): Dimension of input vectors.
        counts (int list): Number of each identifiers.
        sample_size (int): Number of negative samples.
        power (float): Power factor :math:`\\alpha`.

    See: `Distributed Representations of Words and Phrases and their\
         Compositionality <http://arxiv.org/abs/1310.4546>`_
    """

    parameter_names = ('W',)
    gradient_names = ('gW',)

    def __init__(self, in_size, counts, sample_size, power=0.75):
        self.sample_size = sample_size
        p = numpy.array(counts, numpy.float32)
        p = numpy.power(p, power)
        self.sampler = walker_alias.WalkerAlias(p)

        vocab_size = len(counts)
        self.W = numpy.zeros((vocab_size, in_size)).astype(numpy.float32)
        self.gW = numpy.zeros_like(self.W)

    def _make_samples(self, t):
        if hasattr(self, 'samples'):
            return self.samples

        size = int(t.shape[0])
        # first one is the positive, and others are sampled negatives
        samples = self.sampler.sample((size, self.sample_size + 1))
        if isinstance(samples, numpy.ndarray):
            samples.T[0] = t
        else:
            cuda.elementwise(
                'const int* t, int* s, int m',
                ''' s[i * m] = t[i]; ''',
                'negative_sampling_assign'
            )(t, samples, self.sample_size + 1)

        self.samples = samples

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

    def to_gpu(self, device=None):
        function.Function.to_gpu(self, device)
        self.sampler.to_gpu()

    def to_cpu(self):
        function.Function.to_cpu(self)
        self.sampler.to_cpu()

    def forward_cpu(self, inputs):
        x, t = inputs
        self._make_samples(t)

        loss = numpy.float32(0.0)
        for i, (ix, k) in enumerate(six.moves.zip(x, self.samples)):
            w = self.W[k]
            f = w.dot(ix)
            f[0] *= -1  # positive sample
            loss += numpy.sum(numpy.logaddexp(f, 0))
        return numpy.array(loss, numpy.float32),

    def forward_gpu(self, inputs):
        x, t = inputs
        n_in = x.shape[1]
        self._make_samples(t)

        wx = cuda.empty((x.shape[0], self.sample_size + 1))
        cuda.elementwise(
            '''float* wx, const float* W, const float* x, const int* k, int c,
            int m''',
            '''
            x = &x[(i / m) * c];
            W = &W[k[i] * c];
            float f = 0;
            for (int j = 0; j < c; ++j) {
              f += x[j] * W[j];
            }
            wx[i] = f;
            ''',
            'negative_sampling_wx'
        )(wx, self.W, x, self.samples, n_in, self.sample_size + 1)
        self.wx = wx

        y = cuda.zeros_like(wx)
        cuda.elementwise(
            'float* y, const float* wx, int c, int m',
            '''
            float f = wx[i];
            if (i % m == 0) {
              f = -f;
            }
            float loss;
            if (f < 0) {
              loss = __logf(1 + __expf(f));
            } else {
              loss = f + __logf(1 + __expf(-f));
            }
            y[i] = loss;
            ''',
            'negative_sampling_forward'
        )(y, wx, n_in, self.sample_size + 1)
        loss = cuda.gpuarray.sum(y)
        return loss,

    def backward_cpu(self, inputs, grads):
        x, t = inputs
        gloss, = grads

        gx = numpy.zeros_like(x)

        for i, (ix, k) in enumerate(six.moves.zip(x, self.samples)):
            w = self.W[k]
            f = w.dot(ix)

            # g == -y * gloss / (1 + exp(yf))
            f[0] *= -1
            g = gloss / (1 + numpy.exp(-f))
            g[0] *= -1

            gx[i] = g.dot(w)
            for ik, ig in six.moves.zip(k, g):
                self.gW[ik] += ig * ix
        return gx, None

    def backward_gpu(self, inputs, grads):
        x, t = inputs
        gloss, = grads

        n_in = x.shape[1]
        g = cuda.empty_like(self.wx)
        cuda.elementwise(
            'float* g, const float* wx, const float* gloss, int m',
            '''
            float y;
            if (i % m == 0) {
              y = 1;
            } else {
              y = -1;
            }

            g[i] = -y * *gloss / (1.0f + __expf(wx[i] * y));
            ''',
            'negative_sampling_calculate_g'
        )(g, self.wx, gloss, self.sample_size + 1)
        gx = cuda.zeros_like(x)
        cuda.elementwise(
            '''float* gx, const float* g, const float* W, const int* k, int c,
            int m''',
            '''
            int d = i / c;
            g = &g[d * m];
            k = &k[d * m];
            float w = 0;
            for (int j = 0; j < m; ++j) {
              w += g[j] * W[k[j] * c + i % c];
            }
            gx[i] = w;
            ''',
            'negative_sampling_calculate_gx'
        )(gx, g, self.W, self.samples, n_in, self.sample_size + 1)
        cuda.elementwise(
            '''const float * g, const float* x, const int* k, float* gW, int c,
            int m''',
            '''
            x = &x[(i / m) * c];
            gW = &gW[k[i] * c];
            float gi = g[i];
            for (int j = 0; j < c; ++j) {
              atomicAdd(gW + j, gi * x[j]);
            }
            ''',
            'negative_sampling_calculate_gw'
        )(g, x, self.samples, self.gW, n_in, self.sample_size + 1)
        return gx, None
