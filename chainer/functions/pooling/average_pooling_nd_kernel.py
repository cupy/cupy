from chainer.functions.pooling import pooling_nd_kernel


class AveragePoolingNDKernelForward(pooling_nd_kernel.PoolingNDKernelForward):

    def name(self):
        # avg_pool_{N}d_fwd
        return 'avg'

    def in_params(self):
        # 2D: raw T in, int32 d_0, int32 d_1, int32 out_0, int32 out_1,
        #     int32 k_0, int32 k_1, int32 s_0, int32 s_1, int32 p_0,
        #     int32 p_1, T coeff
        return ['T coeff']

    def before(self):
        return 'T val = 0;'

    def main(self, offset, xs):
        # 2D: val = val + in[offset_1];
        return 'val = val + in[{}];'.format(offset)

    def after(self, out_xs):
        return 'out = val * coeff;'


class AveragePoolingNDKernelBackward(
        pooling_nd_kernel.PoolingNDKernelBackward):

    def name(self):
        # avg_pool_{N}d_bwd
        return 'avg'

    def in_params(self):
        # 2D: raw T gy, int32 d_0, int32 d_1, int32 out_0, int32 out_1,
        #     int32 k_0, int32 k_1, int32 s_0, int32 s_1, int32 p_0,
        #     int32 p_1, T coeff
        return ['T coeff']

    def before(self):
        return 'T val = 0;'

    def main(self, offset, xs, out_xs):
        # 2D: val = val + gy[offset_1];
        return 'val = val + gy[{}];'.format(offset)

    def after(self, xs):
        return 'gx = val * coeff;'
