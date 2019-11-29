import cupy
import cupy as cp
import numpy
from cupy.core.fusionx import _FusionXHistory, FusionX

import cupy_perf

from batchnorm.gen_input import gen_input

class PerfBatchNorm(cupy_perf.PerfCases):
    enable_line_profiler = False

    def run_elementwise(self, x, gamma, beta, decay, eps, running_mean, running_var, expander, adjust):
            mean = cp.mean(x, axis=(0,1,2))
            var = cp.var(x, axis=(0,1,2))
            inv_std = 1. / cp.sqrt(var)

            kern = cp.ElementwiseKernel(
                'T x, U mean, U inv_std, U gamma, U beta', 'T y',
                'y = gamma * (x - mean) * inv_std + beta', 'bn_fwd'
            )

            mean_ex = mean[expander]
            inv_std_ex = inv_std[expander]

            y = kern(x, mean_ex, inv_std_ex, gamma, beta)

            # update running statistics
            cp.ElementwiseKernel(
                'U mean, U var, U decay, U adjust',
                'U r_mean, U r_var',
                '''
                r_mean = r_mean * decay + mean * (1 - decay);
                r_var = r_var * decay + var * (1 - decay) * adjust;
                ''',
                'update_mean_var')(mean, var, decay, adjust,
                                    running_mean, running_var)

            return y, running_mean, running_var

    def setUp(self):
        x = gen_input()
        chan = x.shape[3]
        gamma = cp.ones(chan)
        beta = cp.zeros(chan)
        decay = 0.9
        decay_ = 1 - decay
        eps = 2e-5
        running_mean = cp.zeros(chan)
        running_var = cp.ones(chan)
        expander = [None, None, None, slice(None)]
        gamma_expand = gamma[expander]
        beta_expand = beta[expander]

        mean = cp.empty(chan)
        mean_expand = mean[expander]
        inv_std = cp.empty(chan)
        inv_std_expand = inv_std[expander]

        m = x.size // gamma.size
        adjust = m / max(m - 1., 1.)

        def run_nofuse(x, gamma, beta, decay, eps, running_mean, running_var, expander, adjust):
            mean = cp.mean(x, axis=(0,1,2))
            var = cp.var(x, axis=(0,1,2))
            inv_std = 1. / cp.sqrt(var + eps)
            y = gamma * (x - mean[expander]) * inv_std[expander] + beta

            # update running statistics
            running_mean = running_mean * decay + mean * (1 - decay)
            running_var = running_var * decay + var * (1 - decay) * adjust

            return y, running_mean, running_var

        def run_fuse(x, gamma, beta, decay, decay_, eps, running_mean, running_var, \
            mean, mean_expand, inv_std, inv_std_expand, m, adjust):
            # cp.mean(x, axis=(0,1,2), out=mean)
            cp.sum(x, axis=(0,1,2), out=mean)
            mean /= m

            # var = cp.var(x, axis=(0,1,2))
            tmp = x - mean_expand
            var = cp.sum(tmp * tmp, axis=(0,1,2))
            var /= m

            cp.true_divide(1, cp.sqrt(var + eps), out=inv_std)
            y = gamma * tmp * inv_std_expand + beta

            # update running statistics
            running_mean = running_mean * decay + mean * decay_
            running_var = running_var * decay + var * decay_ * adjust

            return y, running_mean, running_var

        def run_fuse2(x, gamma, beta, decay, decay_, eps, running_mean, running_var, \
            mean, mean_expand, inv_std, inv_std_expand, m, adjust):
            # cp.mean(x, axis=(0,1,2), out=mean)
            cp.sum(x, axis=(0,1,2), out=mean)
            mean /= m

            # var = cp.var(x, axis=(0,1,2))
            tmp = x - mean_expand
            var = cp.sum(tmp * tmp, axis=(0,1,2))
            var /= m

            cp.true_divide(1, cp.sqrt(var + eps), out=inv_std)
            y = gamma * tmp * inv_std_expand + beta

            # update running statistics
            running_mean = running_mean * decay + mean * decay_
            running_var = running_var * decay + var * decay_ * adjust

            return y, running_mean, running_var

        self.kern1 = cp.ElementwiseKernel(
                'T x, U mean, U inv_std, U gamma, U beta', 'T y',
                'y = gamma * (x - mean) * inv_std + beta', 'bn_fwd'
            )

        self.kern2 = cp.ElementwiseKernel(
                'U mean, U var, U decay, U adjust',
                'U r_mean, U r_var',
                '''
                r_mean = r_mean * decay + mean * (1 - decay);
                r_var = r_var * decay + var * (1 - decay) * adjust;
                ''',
                'update_mean_var')

        def run_elementwise2(x, gamma, beta, decay, eps, running_mean, running_var, expander, adjust, kern1, kern2):
            mean = cp.mean(x, axis=(0,1,2))
            var = cp.var(x, axis=(0,1,2))
            inv_std = 1. / cp.sqrt(var)

            y = kern1(x, mean[expander], inv_std[expander], gamma, beta)

            # update running statistics
            kern2(mean, var, decay, adjust,
                                    running_mean, running_var)

            return y, running_mean, running_var

        self.x = x
        self.gamma = gamma_expand
        self.beta = beta_expand
        self.decay = decay
        self.decay_ = decay_
        self.eps = eps
        self.running_mean = running_mean
        self.running_var = running_var
        self.expander = expander
        self.m = m
        self.adjust = adjust

        self.mean = mean
        self.mean_expand = mean_expand
        self.inv_std = inv_std
        self.inv_std_expand = inv_std_expand

        self.run_nofuse = run_nofuse
        self.run_fuse = cp.fusex(run_fuse)
        self.run_fuse2 = cp.fusex(run_fuse2)
        # self.run_elementwise = run_elementwise
        self.run_elementwise2 = run_elementwise2


    def perf_nofuse(self):
        self.run_nofuse(self.x, self.gamma, self.beta, self.decay, self.eps, \
            self.running_mean, self.running_var, self.expander, self.adjust)

    def perf_fuse(self):
        self.run_fuse(self.x, self.gamma, self.beta, self.decay, self.decay_, self.eps, \
            self.running_mean, self.running_var, self.mean, self.mean_expand, \
            self.inv_std, self.inv_std_expand, self.m, self.adjust)

    def perf_fuse2(self):
        self.run_fuse2(self.x, self.gamma, self.beta, self.decay, self.decay_, self.eps, \
            self.running_mean, self.running_var, self.mean, self.mean_expand, \
            self.inv_std, self.inv_std_expand, self.m, self.adjust, same_shape=[(1, 2), (6, 7), (8, 10), (9, 11)])

    def aperf_elementwise(self):
        self.run_elementwise(self.x, self.gamma, self.beta, self.decay, self.eps, \
            self.running_mean, self.running_var, self.expander, self.adjust)

    def perf_elementwise2(self):
        self.run_elementwise2(self.x, self.gamma, self.beta, self.decay, self.eps, \
            self.running_mean, self.running_var, self.expander, self.adjust, self.kern1, self.kern2)

# PerfBatchNorm.run_elementwise = profile(PerfBatchNorm.run_elementwise)
# PerfBatchNorm.perf_elementwise = profile(PerfBatchNorm.perf_elementwise)
# PerfBatchNorm.perf_fuse = profile(PerfBatchNorm.perf_fuse)
# _FusionXHistory.exec = profile(_FusionXHistory.exec)
# FusionX.__call__ = profile(FusionX.__call__)
cupy_perf.run(__name__)

# hoge = PerfBatchNorm()
# hoge.setUp()
# for i in range(1):
    # hoge.perf_fuse()
    # hoge.perf_fuse2()
    # hoge.perf_elementwise()
