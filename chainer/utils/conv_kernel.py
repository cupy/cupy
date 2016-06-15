from __future__ import print_function
import functools
import six


def identity(x):
    return x


def mulexp(xs, init=None):
    if init:
        return functools.reduce('{} * {}'.format, xs, init)
    else:
        return functools.reduce('{} * {}'.format, xs)


def andexp(xs, init=None):
    if init:
        return functools.reduce('{} && {}'.format, xs, init)
    else:
        return functools.reduce('{} && {}'.format, xs)


def muladdexp(xs, ys, init=None):
    def aux(exp, arg):
        x, y = arg
        return '({} + {} * {})'.format(y, x, exp)
    if init:
        return functools.reduce(aux, zip(xs, ys), init)
    else:
        return functools.reduce(aux, zip(xs, ys))


def maplist(fn, xs):
    # Imperative impl. because Python does not optimize tail recursion.
    ret = []
    while xs:
        ret += [fn(xs)]
        xs = xs[1:]
    return ret


def _vars(prefix, N):
    return ['{}_{}'.format(prefix, i) for i in six.moves.range(N)]


def _writer():
    _indent = [0]
    _lines = []

    def _aux(line=None, indent=None):
        if line is None:
            return '\n'.join(_lines)
        else:
            if indent == 'dec':
                _indent[0] -= 1
            _lines.append('  ' * _indent[0] + line)
            if indent == 'inc':
                _indent[0] += 1
    return _aux


#
# im2col

def _im2col_in_params(ds, outs, ks, ss, ps):
    # 2D: raw T img, int32 d_0, int32 d_1, int32 out_0, int32 out_1, int32 k_0,
    #     int32 k_1, int32 s_0, int32 s_1, int32 p_0, int32 p_1
    def aux(x):
        return 'int32 {}'.format(x)
    return ', '.join(['raw T img'] + map(aux, ds + outs + ks + ss + ps))


def _im2col_out_params():
    return 'T col'


def _im2col_c0_decl(outs, ks):
    # 2D: int c0 = i / (k_0 * k_1 * out_0 * out_1)
    return 'int c0 = i / ({});'.format(mulexp(ks + outs))


def _im2col_kxs_decl(N, outs, ks):
    # 2D: int kx_0 = i / (k_1 * out_0 * out_1) % k_0;
    #     int kx_1 = i / (out_0 * out_1) % k_1;
    def aux(kx, xs):
        head = xs[0]
        tail = xs[1:] + outs
        if tail:
            return 'int {} = i / ({}) % {};'.format(kx, mulexp(tail), head)
        else:
            return 'int {} = i % {};'.format(kx, head)
    kxs = _vars('kx', N)
    kxs_decl = map(aux, kxs, maplist(identity, ks))
    return kxs_decl, kxs


def _im2col_outxs_decl(N, outs):
    # 2D: int outx_0 = i / (out_1) % out_0;
    #     int outx_1 = i % out_1;
    def aux(outx, xs):
        head = xs[0]
        tail = xs[1:]
        if tail:
            return 'int {} = i / ({}) % {};'.format(outx, mulexp(tail), head)
        else:
            return 'int {} = i % {};'.format(outx, head)
    outxs = _vars('outx', N)
    outxs_decl = map(aux, outxs, maplist(identity, outs))
    return outxs_decl, outxs


def _im2col_ins_decl(N, kxs, outxs, ss, ps):
    # 2D: int in_0 = kx_0 + outx_0 * s_0 - p_0;
    #     int in_1 = kx_1 + outx_1 * s_1 - p_1;
    def aux(_in, kx, outx, s, p):
        return 'int {} = {} + {} * {} - {};'.format(_in, kx, outx, s, p)
    ins = _vars('in', N)
    ins_decl = map(aux, ins, kxs, outxs, ss, ps)
    return ins_decl, ins


def _im2col_col_set(ins, ds):
    # 2D: if (0 <= in_0 && in_0 < d_0 && 0 <= in_1 && in_1 < d_1) {
    #       col = img[(in_1 + d_1 * (in_0 + d_0 * c0))];
    #     } else {
    #       col = (T)0;
    #     }
    def rels_aux(_in, d):
        return '0 <= {} && {} < {}'.format(_in, _in, d)
    test = andexp(map(rels_aux, ins, ds))
    index = muladdexp(ds, ins, 'c0')
    col_set = 'col = img[{}];'.format(index)
    template = """if ({}) {{
  {}
}} else {{
  col = (T)0;
}}"""
    return template.format(test, col_set)


def _im2col_operation(N, ds, outs, ks, ss, ps):
    c0_decl = [_im2col_c0_decl(outs, ks)]
    kxs_decl, kxs = _im2col_kxs_decl(N, outs, ks)
    outxs_decl, outxs = _im2col_outxs_decl(N, outs)
    ins_decl, ins = _im2col_ins_decl(N, kxs, outxs, ss, ps)
    col_set = [_im2col_col_set(ins, ds)]
    return '\n'.join(c0_decl + kxs_decl + outxs_decl + ins_decl + col_set)


def generate_im2col_nd_kernel(N):
    ds = _vars('d', N)
    outs = _vars('out', N)
    ks = _vars('k', N)
    ss = _vars('s', N)
    ps = _vars('p', N)
    in_params = _im2col_in_params(ds, outs, ks, ss, ps)
    out_params = _im2col_out_params()
    operation = _im2col_operation(N, ds, outs, ks, ss, ps)
    name = 'im2col_{}d'.format(N)
    return in_params, out_params, operation, name


#
# col2im

def _col2im_in_params(ds, outs, ks, ss, ps):
    # 2D: raw T col, int32 d_0, int32 d_1, int32 out_0, int32 out_1, int32 k_0,
    #     int32 k_1, int32 s_0, int32 s_1, int32 p_0, int32 p_1
    def aux(x):
        return 'int32 {}'.format(x)
    return ', '.join(['raw T col'] + map(aux, ds + outs + ks + ss + ps))


def _col2im_out_params():
    return 'T img'


def _col2im_c0_decl(ds):
    # 2D: int c0  = i / (d_0 * d_1);
    return 'int c0  = i / ({});'.format(mulexp(ds))


def _col2im_xs_decl(N, ds, ps):
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
    xs = _vars('x', N)
    xs_decl = map(aux, xs, maplist(identity, ds), ps)
    return xs_decl, xs


def _col2im_outs_decl(N, outs, xs, ks, ss):
    # 2D: int outx_0_0 = max(0,     (x_0 - k_0 + s_0) / s_0);
    #     int outx_1_0 = min(out_0, (x_0       + s_0) / s_0);
    #     int outx_0_1 = max(0,     (x_1 - k_1 + s_1) / s_1);
    #     int outx_1_1 = min(out_1, (x_1       + s_1) / s_1);
    def aux(outx_0, outx_1, out, x, k, s):
        return [
            'int {} = max(0,     ({} - {} + {}) / {});'.format(
                outx_0, x, k, s, s),
            'int {} = min({}, ({}       + {}) / {});'.format(
                outx_1, out, x, s, s)]
    outxs_0 = _vars('outx_0', N)
    outxs_1 = _vars('outx_1', N)
    outs_decl = sum(map(aux, outxs_0, outxs_1, outs, xs, ks, ss), [])
    return outs_decl, outxs_0, outxs_1


def _col2im_val_accum(N, outxs_0, outxs_1, outs, ks, xs, ss):
    # 2D: T val = 0;
    #     for (int outx_0 = outx_0_0; outx_0 < outx_1_0; ++outx_0) {
    #       int kx_0 = x_0 - outx_0 * s_0;
    #       for (int outx_1 = outx_0_1; outx_1 < outx_1_1; ++outx_1) {
    #         int kx_1 = x_1 - outx_1 * s_1;
    #         val = val + col[(outx_1 + out_1 * (outx_0 + out_0 * (kx_1 + k_1
    #               * (kx_0 + k_0 * c0))))];
    #       }
    #     }
    w = _writer()
    w('T val = 0;')

    # Loop openings.
    outxs = _vars('outx', N)
    kxs = _vars('kx', N)
    for (outx, outx_0, outx_1, kx, x, s) in zip(
            outxs, outxs_0, outxs_1, kxs, xs, ss):
        w('for (int {} = {}; {} < {}; ++{}) {{'.format(
            outx, outx_0, outx, outx_1, outx), 'inc')
        w('int {} = {} - {} * {};'.format(kx, x, outx, s))

    # Accumulation.
    index = muladdexp(ks + outs, kxs + outxs, 'c0')
    w('val = val + col[{}];'.format(index))

    # Loop closings.
    for _ in outxs:
        w('}', 'dec')

    return w()


def _col2im_img_set():
    return 'img = val;'


def _col2im_operation(N, ds, outs, ks, ss, ps):
    c0_decl = [_col2im_c0_decl(ds)]
    xs_decl, xs = _col2im_xs_decl(N, ds, ps)
    outs_decl, outxs_0, outxs_1 = _col2im_outs_decl(N, outs, xs, ks, ss)
    val_accum = [_col2im_val_accum(N, outxs_0, outxs_1, outs, ks, xs, ss)]
    img_set = [_col2im_img_set()]
    return '\n'.join(c0_decl + xs_decl + outs_decl + val_accum + img_set)


def generate_col2im_nd_kernel(N):
    ds = _vars('d', N)
    outs = _vars('out', N)
    ks = _vars('k', N)
    ss = _vars('s', N)
    ps = _vars('p', N)
    in_params = _col2im_in_params(ds, outs, ks, ss, ps)
    out_params = _col2im_out_params()
    operation = _col2im_operation(N, ds, outs, ks, ss, ps)
    name = 'col2im_{}d'.format(N)
    return in_params, out_params, operation, name


# just for debug.
if __name__ == "__main__":
    N = 3

    print("im2col_nd_kernel")
    print("----------------")
    print()
    for x in generate_im2col_nd_kernel(N):
        print(x)
        print()

    print()
    print("col2im_nd_kernel")
    print("----------------")
    print()
    for x in generate_col2im_nd_kernel(N):
        print(x)
        print()
