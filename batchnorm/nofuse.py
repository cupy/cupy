from gen_input import gen_input
import cupy as cp

# assume axis=(0,2,3) and key_axis=(1,)
# decay=0.9, eps=2e-5

# x.shape=(n,c,h,w), gamma.shape=(c,), beta.shape=(c,), running_mean.shape=(c,), running_var.shape=(c,)
def batchnorm_forward(x):
    # functions/normalization/batch_normalization.py > GeneralBatchNormalization > forward
    chan = x.shape[1]
    gamma = cp.ones(chan)
    beta = cp.zeros(chan)
    decay = 0.9
    eps = 2e-5
    running_mean = cp.zeros(chan)
    running_var = cp.ones(chan)
    expander = [None, slice(None), None, None]
    gamma_expand = gamma[expander]
    beta_expand = beta[expander]

    m = x.size // gamma.size
    adjust = m / max(m - 1., 1.)


    # assume gamma, beta are already expanded
    def run(x, gamma, beta, decay, eps, running_mean, running_var, expander, adjust):
        mean = cp.mean(x, axis=(0,2,3))
        var = cp.var(x, axis=(0,2,3))
        inv_std = 1. / cp.sqrt(var + eps)
        y = gamma * (x - mean[expander]) * inv_std[expander] + beta

        # update running statistics
        running_mean = running_mean * decay + mean * (1 - decay)
        running_var = running_var * decay + var * (1 - decay) * adjust

        return y, running_mean, running_var

    return run(x, gamma_expand, beta_expand, decay, eps, running_mean, running_var, expander, adjust)

if __name__ == '__main__':
    x = gen_input()
    print(batchnorm_forward(x))
