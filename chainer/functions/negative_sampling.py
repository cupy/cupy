from itertools import izip

import numpy
from pycuda import gpuarray

from chainer import cuda, Function
from chainer.cuda import to_cpu, to_gpu, GPUArray

class WalkerAlias(object):
    def __init__(self, probs):
        prob = numpy.array(probs, numpy.float32)
        prob /= numpy.sum(prob)
        threshold = numpy.ndarray(len(probs), numpy.float32)
        values = numpy.ndarray(len(probs) * 2, numpy.int32)
        il, ir = 0, 0
        pairs = zip(prob, range(len(probs)))
        pairs.sort()
        for prob, i in pairs:
            p = prob * len(probs)
            while p > 1 and ir < len(threshold):
                values[ir * 2 + 1] = i
                p -= 1.0 - threshold[ir]
                ir += 1
            threshold[il] = p
            values[il * 2] = i
            il += 1
        # fill the rest
        for i in range(ir, len(probs)):
            values[i * 2 + 1] = 0

        assert((values < len(threshold)).all())
        self.threshold = threshold
        self.values = values
        self.use_gpu = False

    def to_gpu(self):
        if not self.use_gpu:
            self.threshold = to_gpu(self.threshold)
            self.values = to_gpu(self.values)
            self.use_gpu = True

    def sample(self, shape):
        if self.use_gpu:
            return self.sample_gpu(shape)
        else:
            return self.sample_cpu(shape)

    def sample_cpu(self, shape):
        ps = numpy.random.uniform(0, 1, shape)
        pb = ps * len(self.threshold)
        index = pb.astype(numpy.int32)
        left_right = (self.threshold[index] < pb - index).astype(numpy.int32)
        return self.values[index * 2 + left_right]

    def sample_gpu(self, shape):
        ps = cuda.empty(shape, numpy.float32)
        cuda.get_generator().fill_uniform(ps)
        vs = cuda.empty(shape, numpy.int32)
        cuda.elementwise(
            'int* vs, const float* ps, const float* threshold, const int* values, int b',
            '''
            float pb = ps[i] * b;
            int index = __float2int_rd(pb);
            // fill_uniform sometimes returns 1.0, so we need to check index
            if (index >= b) {
              index = 0;
            }
            int lr = threshold[index] < pb - index;
            vs[i] = values[index * 2 + lr];
            ''',
            'walker_alias_sample'
        )(vs, ps, self.threshold, self.values, len(self.threshold))
        return vs


class NegativeSampling(Function):
    """Negative sampling."""

    parameter_names = ('W',)
    gradient_names = ('gW',)

    def __init__(self, in_size, counts, sample_size, power=0.75):
        self.sample_size = sample_size
        # precision of float32 is not enough for `numpy.random.choice`
        p = numpy.array(counts, numpy.float64)
        p = numpy.power(p, power)
        self.sampler = WalkerAlias(p)
        
        vocab_size = len(counts)
        self.W = numpy.zeros((vocab_size, in_size)).astype(numpy.float32)
        #self.W = numpy.random.uniform(-1, 1, (vocab_size, in_size)).astype(numpy.float32)
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

    def to_gpu(self, device=None):
        Function.to_gpu(self, device)
        self.sampler.to_gpu()

    def forward_cpu(self, (x, t)):
        self._make_samples(t)

        loss = 0
        for i, (ix, k) in enumerate(izip(x, self.samples)):
            w = self.W[k]
            f = w.dot(ix)
            f[0] *= -1  # positive sample
            loss += numpy.sum(numpy.logaddexp(f, 0))
        return numpy.array([loss], numpy.float32),

    def forward_gpu(self, (x, t)):
        n_in = x.shape[1]
        self._make_samples(t)

        wx = cuda.empty((x.shape[0], self.sample_size + 1))
        cuda.elementwise(
            'float* wx, const float* W, const float* x, const int* k, int c, int m',
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
        loss = gpuarray.sum(y)
        return loss,

    def backward_cpu(self, (x, t), (gloss,)):
        gloss = numpy.sum(gloss)
        size = x.shape[0]
        gx = numpy.zeros_like(x)

        for i, (ix, k) in enumerate(izip(x, self.samples)):
            w = self.W[k]
            f = w.dot(ix)

            # g == -y * gloss / (1 + exp(yf))
            f[0] *= -1
            g = gloss / (1 + numpy.exp(-f))
            g[0] *= -1

            gx[i] = g.dot(w)
            for ik, ig in izip(k, g):
                self.gW[ik] += ig * ix
        return gx, None

    def backward_gpu(self, (x, t), (gloss,)):
        size = x.shape[0]
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
            'float* gx, const float* g, const float* W, const int* k, int c, int m',
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
            'const float * g, const float* x, const int* k, float* gW, int c, int m',
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
