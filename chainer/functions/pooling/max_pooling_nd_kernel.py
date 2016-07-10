from __future__ import print_function

from chainer.utils.conv_nd_kernel import muladdexp
from chainer.utils.conv_nd_kernel import vars
from chainer.utils.conv_nd_kernel import writer
from chainer.functions.pooling import pooling_nd_kernel


class MaxPoolingNDKernelFwd(pooling_nd_kernel.PoolingNDKernelFwd):

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
        self.argmaxs = vars('argmax', self.ndim)
        argmax_decls = map(aux, self.argmaxs)
        return '\n'.join(['T maxval = (T)-INFINITY;'] + argmax_decls)

    def main(self, offset, xs):
        # 2D: T v = in[offset_1];
        #     if (maxval < v) {
        #       maxval   = v;
        #       argmax_0 = x_0;
        #       argmax_1 = x_1;
        #     }
        w = writer()
        w('T v = in[{}];'.format(offset))
        w('if (maxval < v) {', 'inc')
        w('maxval   = v;')
        for (argmax, x) in zip(self.argmaxs, xs):
            w('{} = {};'.format(argmax, x))
        w('}', 'dec')
        return w()

    def after(self, out_xs):
        # 2D: out = maxval;
        #     int argmax_k_0 = argmax_0 + p_0 - out_x_0 * s_0;
        #     int argmax_k_1 = argmax_1 + p_1 - out_x_1 * s_1;
        #     indexes = (argmax_k_1 + k_1 * argmax_k_0);
        def aux(argmax_k, argmax, p, out, s):
            return 'int {} = {} + {} - {} * {};'.format(
                argmax_k, argmax, p, out, s)
        argmax_ks = vars('argmax_k', self.ndim)
        argmax_k_decls = map(
            aux, argmax_ks, self.argmaxs, self.ps, out_xs, self.ss)
        indexes_set = 'indexes = {};'.format(
            muladdexp(self.ks[1:], argmax_ks[1:], argmax_ks[0]))
        return '\n'.join(['out = maxval;'] + argmax_k_decls + [indexes_set])


class MaxPoolingNDKernelBwd(pooling_nd_kernel.PoolingNDKernelBwd):

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

    def main(self, kx, out_xs):
        # 2D: if (indexes[offset] == kx_1) {
        #       val = val + gy[offset];
        #     }
        w = writer()
        w('int offset = {};'.format(muladdexp(self.outs, out_xs, 'c0')))
        w('if (indexes[offset] == {}) {{'.format(kx), 'inc')
        w('val = val + gy[offset];')
        w('}', 'dec')
        return w()

    def after(self, xs):
        return 'gx = val;'


# just for debug.
if __name__ == "__main__":
    ndim = 3

    print("MaxPoolingNDKernelFwd")
    print("----------------")
    print()
    for x in MaxPoolingNDKernelFwd(ndim).generate():
        print(x)
        print()

    print("MaxPoolingNDKernelBwd")
    print("----------------")
    print()
    for x in MaxPoolingNDKernelBwd(ndim).generate():
        print(x)
        print()
