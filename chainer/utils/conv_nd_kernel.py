import functools
import six

import chainer


def mulexp(xs, init=None):
    if init is not None:
        return functools.reduce('{} * {}'.format, xs, init)
    else:
        return functools.reduce('{} * {}'.format, xs)


def andexp(xs, init=None):
    if init is not None:
        return functools.reduce('{} && {}'.format, xs, init)
    else:
        return functools.reduce('{} && {}'.format, xs)


def muladdexp(xs, ys, init=None):
    def aux(exp, arg):
        x, y = arg
        return '({} + {} * {})'.format(y, x, exp)
    if init is not None:
        return functools.reduce(aux, six.moves.zip(xs, ys), init)
    else:
        return functools.reduce(aux, six.moves.zip(xs, ys))


def map_(fn, *lst):
    # For py2/py3 compatibility.
    return list(map(fn, *lst))


def succ_sublists(xs):
    # Returns successive sublists of xs.
    return [xs[i:] for i in six.moves.range(len(xs))]


def vars(prefix, n):
    return ['{}_{}'.format(prefix, i) for i in six.moves.range(n)]


class Writer(object):

    def __init__(self):
        self._indent = 0
        self._lines = []

    def write(self, line, indent=None):
        if indent == 'dec' or indent == 'decinc':
            self._indent -= 1
        self._lines.append('  ' * self._indent + line)
        if indent == 'inc' or indent == 'decinc':
            self._indent += 1

    def get(self):
        return '\n'.join(self._lines)


#
# im2col

class Im2colNDKernel(object):

    def _in_params(self, ds, outs, ks, ss, ps):
        # 2D: raw T img, int32 d_0, int32 d_1, int32 out_0, int32 out_1,
        #     int32 k_0, int32 k_1, int32 s_0, int32 s_1, int32 p_0, int32 p_1
        def aux(x):
            return 'int32 {}'.format(x)
        return ', '.join(['raw T img'] + map_(aux, ds + outs + ks + ss + ps))

    def _out_params(self):
        return 'T col'

    def _compile_c0(self, outs, ks):
        # 2D: int c0 = i / (k_0 * k_1 * out_0 * out_1)
        return ['int c0 = i / ({});'.format(mulexp(ks + outs))]

    def _compile_kx(self, ndim, outs, ks):
        # 2D: int kx_0 = i / (k_1 * out_0 * out_1) % k_0;
        #     int kx_1 = i / (out_0 * out_1) % k_1;
        def aux(kx, xs):
            head = xs[0]
            tail = xs[1:] + outs
            if tail:
                return 'int {} = i / ({}) % {};'.format(kx, mulexp(tail), head)
            else:
                return 'int {} = i % {};'.format(kx, head)
        kxs = vars('kx', ndim)
        kx_decls = map_(aux, kxs, succ_sublists(ks))
        return kx_decls, kxs

    def _compile_out_x(self, ndim, outs):
        # 2D: int out_x0 = i / (out_1) % out_0;
        #     int out_x1 = i % out_1;
        def aux(out_x, xs):
            head = xs[0]
            tail = xs[1:]
            if tail:
                return 'int {} = i / ({}) % {};'.format(
                    out_x, mulexp(tail), head)
            else:
                return 'int {} = i % {};'.format(out_x, head)
        out_xs = vars('out_x', ndim)
        out_x_decls = map_(aux, out_xs, succ_sublists(outs))
        return out_x_decls, out_xs

    def _compile_main(self, ndim, ds, ks, ss, ps, kxs, out_xs):
        # 2D: int in_0 = kx_0 + out_x_0 * s_0 - p_0;
        #     int in_1 = kx_1 + out_x_1 * s_1 - p_1;
        #     if (0 <= in_0 && in_0 < d_0 && 0 <= in_1 && in_1 < d_1) {
        #       int idx_0 = in_0 + d_0 * c0;
        #       int idx_1 = in_1 + d_1 * idx_0;
        #       col = img[idx_1];
        #     } else {
        #       col = (T)0;
        #     }
        w = Writer()

        ins = vars('in', ndim)
        for _in, kx, out_x, s, p in six.moves.zip(ins, kxs, out_xs, ss, ps):
            w.write('int {} = {} + {} * {} - {};'.format(_in, kx, out_x, s, p))

        def rel_aux(_in, d):
            return '0 <= {} && {} < {}'.format(_in, _in, d)
        w.write(
            'if ({}) {{'.format(andexp(map_(rel_aux, ins, ds))), indent='inc')

        idxs = vars('idx', ndim)
        idx0s = ['c0'] + idxs[:-1]
        for idx, _in, d, idx0 in six.moves.zip(idxs, ins, ds, idx0s):
            w.write('int {} = {} + {} * {};'.format(idx, _in, d, idx0))

        w.write('col = img[{}];'.format(idxs[-1]))
        w.write('} else {', indent='decinc')
        w.write('col = (T)0;')
        w.write('}', indent='dec')

        return [w.get()]

    def _operation(self, ndim, ds, outs, ks, ss, ps):
        c0 = self._compile_c0(outs, ks)
        kx, kxs = self._compile_kx(ndim, outs, ks)
        out_x, out_xs = self._compile_out_x(ndim, outs)
        main = self._compile_main(ndim, ds, ks, ss, ps, kxs, out_xs)
        return '\n'.join(c0 + kx + out_x + main)

    def _generate(self, ndim):
        ds = vars('d', ndim)
        outs = vars('out', ndim)
        ks = vars('k', ndim)
        ss = vars('s', ndim)
        ps = vars('p', ndim)

        in_params = self._in_params(ds, outs, ks, ss, ps)
        out_params = self._out_params()
        operation = self._operation(ndim, ds, outs, ks, ss, ps)
        name = name = 'im2col_{}d'.format(ndim)
        return in_params, out_params, operation, name

    @staticmethod
    @chainer.cuda.memoize()
    def generate(ndim):
        return _im2col_nd_kernel._generate(ndim)

_im2col_nd_kernel = Im2colNDKernel()


#
# col2im

class Col2imNDKernel(object):

    def _in_params(self, ds, outs, ks, ss, ps):
        # 2D: raw T col, int32 d_0, int32 d_1, int32 out_0, int32 out_1,
        #     int32 k_0, int32 k_1, int32 s_0, int32 s_1, int32 p_0, int32 p_1
        def aux(x):
            return 'int32 {}'.format(x)
        return ', '.join(['raw T col'] + map_(aux, ds + outs + ks + ss + ps))

    def _out_params(self):
        return 'T img'

    def _compile_c0(self, ds):
        # 2D: int c0 = i / (d_0 * d_1);
        return ['int c0 = i / ({});'.format(mulexp(ds))]

    def _compile_x(self, ndim, ds, ps):
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
        xs = vars('x', ndim)
        x_decls = map_(aux, xs, succ_sublists(ds), ps)
        return x_decls, xs

    def _compile_loop(self, ndim, outs, ks, ss, xs):
        # 2D: int out_x0_0 = max(0,     (x_0 - k_0 + s_0) / s_0);
        #     int out_x1_0 = min(out_0, (x_0       + s_0) / s_0);
        #     int out_x0_1 = max(0,     (x_1 - k_1 + s_1) / s_1);
        #     int out_x1_1 = min(out_1, (x_1       + s_1) / s_1);
        #     ... Before-part here ...
        #     for (int out_x_0 = out_x0_0; out_x_0 < out_x1_0; ++out_x_0) {
        #       int kx_0 = x_0 - out_x_0 * s_0 + k_0 * c0;
        #       for (int out_x_1 = out_x0_1; out_x_1 < out_x1_1; ++out_x_1) {
        #         int kx_1 = x_1 - out_x_1 * s_1 + k_1 * kx_0;
        #         ... Main-part here ...
        #       }
        #     }
        #     ... After-part here ...
        def aux(out_x0, out_x1, out, x, k, s):
            return [
                'int {} = max(0, ({} - {} + {}) / {});'.format(
                    out_x0, x, k, s, s),
                'int {} = min({}, ({} + {}) / {});'.format(
                    out_x1, out, x, s, s)]
        out_x0s = vars('out_x0', ndim)
        out_x1s = vars('out_x1', ndim)
        bounds = sum(map_(aux, out_x0s, out_x1s, outs, xs, ks, ss), [])

        def _loop_main(main, ndim, ks, ss):
            w = Writer()

            # Loop openings.
            out_xs = vars('out_x', ndim)
            kxs = vars('kx', ndim)
            kxs1 = ['c0'] + kxs[:-1]
            for out_x, out_x0, out_x1, kx, s, x, k, kx1 in six.moves.zip(
                    out_xs, out_x0s, out_x1s, kxs, ss, xs, ks, kxs1):
                w.write('for (int {} = {}; {} < {}; ++{}) {{'.format(
                    out_x, out_x0, out_x, out_x1, out_x), indent='inc')
                w.write('int {} = {} - {} * {} + {} * {};'.format(
                    kx, x, out_x, s, k, kx1))

            # Main-part.
            kx = kxs[-1]
            for l in main(kx, out_xs).split('\n'):
                w.write(l)

            # Loop closings.
            for _ in out_xs:
                w.write('}', indent='dec')

            return [w.get()]

        return bounds, _loop_main

    def _compile_procedure(self, outs, xs):
        # 2D: val = val + col[(out_x_1 + out_1 * (out_x_0 + out_0 * kx_1))];
        def _main(kx, out_xs):
            index = muladdexp(outs, out_xs, kx)
            return 'val = val + col[{}];'.format(index)
        before = ['T val = 0;']
        after = ['img = val;']
        return before, _main, after

    def _operation(self, ndim, ds, outs, ks, ss, ps):
        c0 = self._compile_c0(ds)
        x, xs = self._compile_x(ndim, ds, ps)
        loop_bounds, loop_main = self._compile_loop(ndim, outs, ks, ss, xs)
        before, main, after = self._compile_procedure(outs, xs)
        return '\n'.join(
            c0 + x + loop_bounds + before + loop_main(
                main, ndim, ks, ss) + after)

    def _generate(self, ndim):
        ds = vars('d', ndim)
        outs = vars('out', ndim)
        ks = vars('k', ndim)
        ss = vars('s', ndim)
        ps = vars('p', ndim)

        in_params = self._in_params(ds, outs, ks, ss, ps)
        out_params = self._out_params()
        operation = self._operation(ndim, ds, outs, ks, ss, ps)
        name = 'col2im_{}d'.format(ndim)
        return in_params, out_params, operation, name

    @staticmethod
    @chainer.cuda.memoize()
    def generate(ndim):
        return _col2im_nd_kernel._generate(ndim)

_col2im_nd_kernel = Col2imNDKernel()
