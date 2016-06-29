from __future__ import print_function

from chainer.utils.conv_nd_kernel import _vars
from chainer.utils.conv_nd_kernel import _writer
from chainer.utils.conv_nd_kernel import identity
from chainer.utils.conv_nd_kernel import maplist
from chainer.utils.conv_nd_kernel import muladdexp
from chainer.utils.conv_nd_kernel import mulexp


#
# Forward

def _forward_in_params(ds, outs, ks, ss, ps):
    # 2D: raw T in, int32 d_0, int32 d_1, int32 out_0, int32 out_1, int32 k_0,
    #     int32 k_1, int32 s_0, int32 s_1, int32 p_0, int32 p_1
    def aux(x):
        return 'int32 {}'.format(x)
    return ', '.join(['raw T in'] + map(aux, ds + outs + ks + ss + ps))


def _forward_out_params():
    return 'T out, S indexes'


def _forward_c0_decl(outs):
    # 2D: int c0 = i / (out_0 * out_1);
    return 'int c0      = i / ({});'.format(mulexp(outs))


def _forward_out_xs_decl(ndim, outs):
    # 2D: int outx_0 = i / (out_1) % out_0;
    #     int outx_1 = i % out_1;
    def aux(out_x, outs):
        head = outs[0]
        tail = outs[1:]
        if tail:
            return 'int {} = i / ({}) % {};'.format(out_x, mulexp(tail), head)
        else:
            return 'int {} = i % {};'.format(out_x, head)
    out_xs = _vars('out_x', ndim)
    out_xs_decl = map(aux, out_xs, maplist(identity, outs))
    return out_xs_decl, out_xs


def _forward_in_xs_decl(ndim, out_xs, ds, ks, ss, ps):
    # 2D: int in_x0_0 = max(0,   out_x_0 * s_0       - p_0);
    #     int in_x1_0 = min(d_0, out_x_0 * s_0 + k_0 - p_0);
    #     int in_x0_1 = max(0,   out_x_1 * s_1       - p_1);
    #     int in_x1_1 = min(d_1, out_x_1 * s_1 + k_1 - p_1);
    def aux(in_x0, in_x1, out, d, k, s, p):
        return [
            'int {} = max(0,   {} * {}       - {});'.format(in_x0, out, s, p),
            'int {} = min({}, {} * {} + {} - {});'.format(
                in_x1, d, out, s, k, p)]
    in_x0s = _vars('in_x0', ndim)
    in_x1s = _vars('in_x1', ndim)
    in_xs_decl = sum(map(aux, in_x0s, in_x1s, out_xs, ds, ks, ss, ps), [])
    return in_xs_decl, in_x0s, in_x1s


def _forward_val_max(ndim, in_x0s, in_x1s, ds):
    # 2D: T maxval = in[(in_x0_1 + d_1 * (in_x0_0 + d_0 * c0))];
    #     int argmax_0 = in_x0_0;
    #     int argmax_1 = in_x0_1;
    #     for (int x_0 = in_x0_0; x_0 < in_x1_0; ++x_0) {
    #       int offset_0 = d_1 * (x_0 + d_0 * c0);
    #       for (int x_1 = in_x0_1; x_1 < in_x1_1; ++x_1) {
    #         int offset_1 = 1 * (x_1 + offset_0);
    #         float v = in[offset_1];
    #         if (maxval < v) {
    #           maxval   = v;
    #           argmax_0 = x_0;
    #           argmax_1 = x_1;
    #         }
    #       }
    #     }
    w = _writer()

    # Initialize values for max operation.
    w('T maxval = in[{}];'.format(muladdexp(ds, in_x0s, 'c0')))
    argmaxs = _vars('argmax', ndim)
    for (argmax, in_x0) in zip(argmaxs, in_x0s):
        w('int {} = {};'.format(argmax, in_x0))

    # Loop openings.
    xs = _vars('x', ndim)
    offsets = _vars('offset', ndim)
    ds1 = ds[1:] + [1]
    offsets0 = ['d_0 * c0'] + offsets[:-1]
    for (x, in_x0, in_x1, offset, offset0, d1) in zip(
            xs, in_x0s, in_x1s, offsets, offsets0, ds1):
        w('for (int {} = {}; {} < {}; ++{}) {{'.format(
            x, in_x0, x, in_x1, x), 'inc')
        w('int {} = {} * ({} + {});'.format(offset, d1, x, offset0))

    # Max operation.
    w('float v = in[{}];'.format(offsets[-1]))
    w('if (maxval < v) {', 'inc')
    w('maxval   = v;')
    for (argmax, x) in zip(argmaxs, xs):
        w('{} = {};'.format(argmax, x))
    w('}', 'dec')

    # Loop closings.
    for _ in xs:
        w('}', 'dec')

    return [w()], argmaxs


def _forward_out_set():
    return 'out = maxval;'


def _forward_indexes_set(ndim, argmaxs, out_xs, ks, ss, ps):
    # 2D: int argmax_k_0 = argmax_0 + p_0 - out_0 * s_0;
    #     int argmax_k_1 = argmax_1 + p_1 - out_1 * s_1;
    #     indexes = (argmax_k_1 + k_1 * argmax_k_0);
    def aux(argmax_k, argmax, p, out, s):
        return 'int {} = {} + {} - {} * {};'.format(
            argmax_k, argmax, p, out, s)
    argmax_ks = _vars('argmax_k', ndim)
    argmax_ks_decl = map(aux, argmax_ks, argmaxs, ps, out_xs, ss)
    indexes_set = 'indexes = {};'.format(
        muladdexp(ks[1:], argmax_ks[1:], argmax_ks[0]))
    return argmax_ks_decl + [indexes_set]


def _forward_operation(ndim, ds, outs, ks, ss, ps):
    c0_decl = [_forward_c0_decl(outs)]
    out_xs_decl, out_xs = _forward_out_xs_decl(ndim, outs)
    in_xs_decl, in_x0s, in_x1s = _forward_in_xs_decl(
        ndim, out_xs, ds, ks, ss, ps)
    val_max, argmaxs = _forward_val_max(ndim, in_x0s, in_x1s, ds)
    out_set = [_forward_out_set()]
    indexes_set = _forward_indexes_set(ndim, argmaxs, out_xs, ks, ss, ps)
    return '\n'.join(
        c0_decl + out_xs_decl + in_xs_decl + val_max + out_set + indexes_set)


def generate_forward(ndim):
    ds = _vars('d', ndim)
    outs = _vars('out', ndim)
    ks = _vars('k', ndim)
    ss = _vars('s', ndim)
    ps = _vars('p', ndim)
    in_params = _forward_in_params(ds, outs, ks, ss, ps)
    out_params = _forward_out_params()
    operation = _forward_operation(ndim, ds, outs, ks, ss, ps)
    name = 'max_pool_{}d_fwd'.format(ndim)
    return in_params, out_params, operation, name


#
# Backward

def _backward_in_params(ds, outs, ks, ss, ps):
    # 2D: raw T gy, raw S indexes, int32 d_0, int32 d_1, int32 out_0,
    #     int32 out_1, int32 k_0, int32 k_1, int32 s_0, int32 s_1, int32 p_0,
    #     int32 p_1
    def aux(x):
        return 'int32 {}'.format(x)
    return ', '.join(['raw T gy, raw S indexes'] + map(
        aux, ds + outs + ks + ss + ps))


def _backward_out_params():
    return 'T gx'


def _backward_c0_decl(ds):
    # 2D: int c0  = i / (d_0 * d_1);
    return 'int c0  = i / ({});'.format(mulexp(ds))


def _backward_xs_decl(ndim, ds, ps):
    # 2D: int x_0 = i / (d_1) % d_0 + p_0;
    #     int x_1 = i % d_1 + p_1;
    def aux(x, ds, p):
        head = ds[0]
        tail = ds[1:]
        if tail:
            return 'int {} = i / ({}) % {} + {};'.format(
                x, mulexp(tail), head, p)
        else:
            return 'int {} = i % {} + {};'.format(x, head, p)
    xs = _vars('x', ndim)
    xs_decl = map(aux, xs, maplist(identity, ds), ps)
    return xs_decl, xs


def _backward_out_xs_decl(ndim, xs, outs, ks, ss):
    # 2D: int out_x0_0 = max(0,     (x_0 - k_0 + s_0) / s_0);
    #     int out_x1_0 = min(out_0, (x_0       + s_0) / s_0);
    #     int out_x0_1 = max(0,     (x_1 - k_1 + s_1) / s_1);
    #     int out_x1_1 = min(out_1, (x_1       + s_1) / s_1);
    def aux(out_x0, out_x1, x, out, k, s):
        return [
            'int {} = max(0,     ({} - {} + {}) / {});'.format(
                out_x0, x, k, s, s),
            'int {} = min({}, ({}       + {}) / {});'.format(
                out_x1, out, x, s, s)]
    out_x0s = _vars('out_x0', ndim)
    out_x1s = _vars('out_x1', ndim)
    out_xs_decl = sum(map(aux, out_x0s, out_x1s, xs, outs, ks, ss), [])
    return out_xs_decl, out_x0s, out_x1s


def _backward_val_accum(ndim, out_x0s, out_x1s, outs, xs, ks, ss):
    # 2D: T val = 0;
    #     for (int out_x_0 = out_x0_0; out_x_0 < out_x1_0; ++out_x_0) {
    #       int kx_0 = x_0 - out_x_0 * s_0;
    #       for (int out_x_1 = out_x0_1; out_x_1 < out_x1_1; ++out_x_1) {
    #         int kx_1 = x_1 - out_x_1 * s_1;
    #         int offset = (out_x_1 + out_1 * (out_x_0 + out_0 * c0));
    #         if (indexes[offset] == (kx_1 + k_1 * kx_0)) {
    #           val = val + gy[offset];
    #         }
    #       }
    #     }
    w = _writer()

    # Initialize values.
    w('T val = 0;')

    # Loop openings.
    out_xs = _vars('out_x', ndim)
    kxs = _vars('kx', ndim)
    for (out_x, out_x0, out_x1, kx, s, x) in zip(
            out_xs, out_x0s, out_x1s, kxs, ss, xs):
        w('for (int {} = {}; {} < {}; ++{}) {{'.format(
            out_x, out_x0, out_x, out_x1, out_x), 'inc')
        w('int {} = {} - {} * {};'.format(kx, x, out_x, s))

    # Accumulation.
    w('int offset = {};'.format(muladdexp(outs, out_xs, 'c0')))
    w('if (indexes[offset] == {}) {{'.format(
        muladdexp(ks[1:], kxs[1:], kxs[0])), 'inc')
    w('val = val + gy[offset];')
    w('}', 'dec')

    # Loop closings.
    for _ in xs:
        w('}', 'dec')

    return w()


def _backward_gx_set():
    return 'gx = val;'


def _backward_operation(ndim, ds, outs, ks, ss, ps):
    c0_decl = [_backward_c0_decl(ds)]
    xs_decl, xs = _backward_xs_decl(ndim, ds, ps)
    out_xs_decl, out_x0s, out_x1s = _backward_out_xs_decl(
        ndim, xs, outs, ks, ss)
    val_accum = [_backward_val_accum(ndim, out_x0s, out_x1s, outs, xs, ks, ss)]
    gx_set = [_backward_gx_set()]
    return '\n'.join(c0_decl + xs_decl + out_xs_decl + val_accum + gx_set)


def generate_backward(ndim):
    ds = _vars('d', ndim)
    outs = _vars('out', ndim)
    ks = _vars('k', ndim)
    ss = _vars('s', ndim)
    ps = _vars('p', ndim)
    in_params = _backward_in_params(ds, outs, ks, ss, ps)
    out_params = _backward_out_params()
    operation = _backward_operation(ndim, ds, outs, ks, ss, ps)
    name = 'max_pool_{}d_bwd'.format(ndim)
    return in_params, out_params, operation, name


# just for debug.
if __name__ == "__main__":
    ndim = 3

    print("max_pooling_nd_forward_kernel")
    print("----------------")
    print()
    for x in generate_forward(ndim):
        print(x)
        print()

    print("max_pooling_nd_backward_kernel")
    print("----------------")
    print()
    for x in generate_backward(ndim):
        print(x)
        print()
