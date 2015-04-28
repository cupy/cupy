import numpy
from chainer import cuda, Optimizer

class Adam(Optimizer):
    """Adam optimization algorithm.

    See: http://arxiv.org/abs/1412.6980

    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, lam=1-1e-8, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.lam   = lam
        self.eps   = eps

    def init_state_cpu(self, param, grad):
        return numpy.zeros_like(param), numpy.zeros_like(param)

    def init_state_gpu(self, param, grad):
        return cuda.zeros_like(param), cuda.zeros_like(param)

    def update_one_cpu(self, param, grad, state):
        m, v = state
        beta1_t = self.beta1 * (self.lam ** (self.t - 1))
        m *= beta1_t
        m += (1 - beta1_t) * grad
        v *= self.beta2
        v += (1 - self.beta2) * grad * grad
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        param -= self.lr * m_hat / (numpy.sqrt(v_hat) + self.eps)

    def update_one_gpu(self, param, grad, state):
        m, v = state
        cuda.elementwise(
            '''float* param, const float* grad, float* m, float* v,
               float lr, float beta1, float beta2, float lam, float eps, int t''',
            '''float beta1_t = beta1 * (__powf(lam, t - 1));
               m[i] = beta1_t * m[i] + (1 - beta1_t) * grad[i];
               v[i] = beta2   * v[i] + (1 - beta2)   * grad[i] * grad[i];
               float m_hat = m[i] / (1 - __powf(beta1, t));
               float v_hat = v[i] / (1 - __powf(beta2, t));
               param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);''',
            'adam')(param, grad, m, v, self.lr, self.beta1, self.beta2,
                    self.lam, self.eps, self.t)
