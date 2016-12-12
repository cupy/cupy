import six

from chainer.functions.pooling import pooling_nd_kernel
from chainer.utils import conv_nd_kernel


class MaxPoolingNDKernelForward(pooling_nd_kernel.PoolingNDKernelForward):

    def name(self):
        # max_pool_{N}d_fwd
        return 'max'

    def out_params(self):
        # T out, S indexes
        return ['S indexes']

    def before(self):
        # 2D: T maxval = (T)-INFINITY;
        #     int argmax_0 = 0;
        #     int argmax_1 = 0;
        def aux(argmax):
            return 'int {} = 0;'.format(argmax)
        self.argmaxs = conv_nd_kernel.vars('argmax', self.ndim)
        argmax_decls = conv_nd_kernel.map_(aux, self.argmaxs)
        return '\n'.join(['T maxval = (T)-INFINITY;'] + argmax_decls)

    def main(self, offset, xs):
        # 2D: T v = in[offset_1];
        #     if (maxval < v) {
        #       maxval   = v;
        #       argmax_0 = x_0;
        #       argmax_1 = x_1;
        #     }
        w = conv_nd_kernel.Writer()
        w.write('T v = in[{}];'.format(offset))
        w.write('if (maxval < v) {', 'inc')
        w.write('maxval = v;')
        for argmax, x in six.moves.zip(self.argmaxs, xs):
            w.write('{} = {};'.format(argmax, x))
        w.write('}', 'dec')
        return w.get()

    def after(self, out_xs):
        # 2D: out = maxval;
        #     int argmax_k_0 = argmax_0 + p_0 - out_x_0 * s_0;
        #     int argmax_k_1 = argmax_1 + p_1 - out_x_1 * s_1;
        #     indexes = (argmax_k_1 + k_1 * argmax_k_0);
        def aux(argmax_k, argmax, p, out_x, s):
            return 'int {} = {} + {} - {} * {};'.format(
                argmax_k, argmax, p, out_x, s)
        argmax_ks = conv_nd_kernel.vars('argmax_k', self.ndim)
        argmax_k_decls = conv_nd_kernel.map_(
            aux, argmax_ks, self.argmaxs, self.ps, out_xs, self.ss)
        indexes_set = 'indexes = {};'.format(
            conv_nd_kernel.muladdexp(self.ks[1:], argmax_ks[1:], argmax_ks[0]))
        return '\n'.join(['out = maxval;'] + argmax_k_decls + [indexes_set])


class MaxPoolingNDKernelBackward(pooling_nd_kernel.PoolingNDKernelBackward):

    def name(self):
        # max_pool_{N}d_bwd
        return 'max'

    def in_params(self):
        # 2D: raw T gy, raw S indexes, int32 d_0, int32 d_1, int32 out_0,
        #     int32 out_1, int32 k_0, int32 k_1, int32 s_0, int32 s_1,
        #     int32 p_0, int32 p_1
        return (['raw S indexes'], [])

    def before(self):
        return 'T val = 0;'

    def main(self, offset, xs, out_xs):
        # 2D: int kx = (x_1 - out_x_1 * s_1 + k_1 *
        #              (x_0 - out_x_0 * s_0 + k_0 * 0));
        #     if (indexes[offset_1] == kx) {
        #       val = val + gy[offset_1];
        #     }
        def aux(x, out_x, s):
            return '{} - {} * {}'.format(x, out_x, s)
        w = conv_nd_kernel.Writer()
        w.write('int kx = {};'.format(
            conv_nd_kernel.muladdexp(self.ks, conv_nd_kernel.map_(
                aux, xs, out_xs, self.ss), '0')))
        w.write('if (indexes[{}] == kx) {{'.format(offset), 'inc')
        w.write('val = val + gy[{}];'.format(offset))
        w.write('}', 'dec')
        return w.get()

    def after(self, xs):
        return 'gx = val;'
