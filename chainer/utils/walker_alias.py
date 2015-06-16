import numpy

from chainer import cuda


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
            self.threshold = cuda.to_gpu(self.threshold)
            self.values = cuda.to_gpu(self.values)
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
