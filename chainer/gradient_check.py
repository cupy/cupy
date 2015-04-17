import numpy
import pycuda.cumath as cumath
from pycuda import gpuarray

def numerical_grad_cpu(f, inputs, grad_outputs, eps=1e-3):
    grads = tuple(numpy.zeros_like(x) for x in inputs)
    for x, gx in zip(inputs, grads):
        flat_x  = x.ravel()
        flat_gx = gx.ravel()
        for i in xrange(flat_x.size):
            orig = flat_x[i]
            flat_x[i] = orig + eps
            ys1 = f()
            flat_x[i] = orig - eps
            ys2 = f()
            flat_x[i] = orig

            for y1, y2, gy in zip(ys1, ys2, grad_outputs):
                if gy is not None:
                    dot = float(sum(((y1 - y2) * gy).ravel()))
                    flat_gx[i] += dot / (2 * eps)

    return grads

def numerical_grad_gpu(f, inputs, grad_outputs, eps=1e-3):
    grads = tuple(gpuarray.zeros_like(x) for x in inputs)
    for x, gx in zip(inputs, grads):
        x  = x.ravel()
        gx = gx.ravel()
        x_cpu  = x.get()
        gx_cpu = gx.get()
        for i in xrange(x_cpu.size):
            orig = x_cpu[i]
            x_cpu[i] = orig + eps
            x.set(x_cpu)
            ys1 = f()
            x_cpu[i] = orig - eps
            x.set(x_cpu)
            ys2 = f()
            x_cpu[i] = orig
            x.set(x_cpu)

            for y1, y2, gy in zip(ys1, ys2, grad_outputs):
                if gy is not None:
                    dot = sum(((y1 - y2) * gy).ravel()).get()
                    gx_cpu[i] += dot / (2 * eps)
        gx.set(gx_cpu)

    return grads

def numerical_grad(f, inputs, grad_outputs, eps=1e-3):
    """Compute numerical gradient by finite differences.

    This function is used to implement gradient check.

    """
    if any(type(x) == gpuarray.GPUArray for x in inputs):
        return numerical_grad_gpu(f, inputs, grad_outputs, eps)
    return numerical_grad_cpu(f, inputs, grad_outputs, eps)


def l_infty_dist(x, y):
    """Compute L_infty distance between x and y."""

    if type(x) == gpuarray.GPUArray:
        return float(gpuarray.max(cumath.fabs(x - y)).get())
    return numpy.abs(x - y).max()
