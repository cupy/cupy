# flake8: noqa: E402

import os
# Allow using cublas API while stream capture
os.environ["CUPY_EXPERIMENTAL_CUDA_LIB_GRAPH_CAPTURE"] = "1"  # noqa

import numpy
import cupy
import cupyx
from cupy.cuda.graph_functional_api import (
    GraphBuilder,
    MockGraphBuilder,
)
from scipy.sparse.linalg import lsmr as lsmr_cpu
from cupyx.scipy.sparse.linalg import lsmr as lsmr_gpu
from cupy import cublas
from cupyx.scipy.sparse.linalg import _interface
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--maxiter", type=int, default=None)
parser.add_argument("-N", type=int, default=1000)
parser.add_argument("--impl", type=str, default="normal")
args = parser.parse_args()


def lsmr_graph(A, b, x0=None, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8,
               maxiter=None, impl_name="graph"):
    A = _interface.aslinearoperator(A)
    b = b.squeeze()
    matvec = A.matvec
    rmatvec = A.rmatvec
    m, n = A.shape
    minDim = min([m, n])

    if maxiter is None:
        maxiter = minDim * 5

    def nrm2_graph_compatible(x, out=None):
        ret = cublas.dot(x, x, out)
        cupy.sqrt(ret, out=ret)
        return ret

    nrm2_fn = nrm2_graph_compatible

    u = b.copy()
    dtype_ = b.dtype
    normb = nrm2_fn(b)
    beta = normb.copy()
    if x0 is None:
        x = cupy.zeros((n,), dtype=A.dtype)
    else:
        if not (x0.shape == (n,) or x0.shape == (n, 1)):
            raise ValueError('x0 has incompatible dimensions')
        x = x0.astype(A.dtype).ravel()
        u -= matvec(x)
        beta = nrm2_fn(u)

    v = cupy.zeros(n, dtype=dtype_)
    alpha = cupy.zeros((), dtype=beta.dtype)

    if beta > 0:
        u /= beta
        v = rmatvec(u)
        alpha = nrm2_fn(v)

    if alpha > 0:
        v /= alpha

    damp = cupy.array([damp], dtype=dtype_).squeeze()

    # Initialize variables for 1st iteration.

    itn = cupy.zeros((), dtype=cupy.int32)
    zetabar = alpha * beta
    alphabar = alpha.copy()
    rho = cupy.ones((), dtype=dtype_)
    rhobar = cupy.ones((), dtype=dtype_)
    cbar = cupy.ones((), dtype=dtype_)
    sbar = cupy.zeros((), dtype=dtype_)

    h = v.copy()
    hbar = cupy.zeros(n)

    # Initialize variables for estimation of ||r||.

    betadd = beta.copy()
    betad = cupy.zeros((), dtype=dtype_)
    rhodold = cupy.ones((), dtype=dtype_)
    tautildeold = cupy.zeros((), dtype=dtype_)
    thetatilde = cupy.zeros((), dtype=dtype_)
    zeta = cupy.zeros((), dtype=dtype_)
    d = cupy.zeros((), dtype=dtype_)

    # Initialize variables for estimation of ||A|| and cond(A)

    normA2 = alpha * alpha
    maxrbar = cupy.zeros((), dtype=dtype_)
    minrbar = cupy.array([1e+100], dtype=dtype_)
    normA = alpha.copy()
    condA = cupy.ones((), dtype=dtype_)
    normx = cupy.zeros((), dtype=dtype_)

    # Items for use in stopping rules.
    istop = cupy.zeros((), dtype=cupy.int32)
    ctol = cupy.zeros((), dtype=dtype_)
    if conlim > 0:
        ctol = 1 / conlim
    normr = beta.copy()

    # Golub-Kahan process terminates when either alpha or beta is zero.
    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    normar = alpha * beta
    if normar == 0:
        return x, istop, itn, normr, normar, normA, condA, normx

    stop_flag = cupy.zeros((), dtype=cupy.bool_)

    def _symOrtho_raw(a, b):
        # print(a, b)
        _symOrtho_module = cupy.RawModule(code=r"""
        __device__ int _sign(double x) {
            if (x == 0) return 0;
            else return (x > 0) ? 1 : -1;
        }

        template<typename T>
        __global__ void symortho(T* a, T* b, T* c_out, T* s_out, T* r_out) {
            T a_val = a[0], b_val = b[0];
            T a_abs = fabs(a_val);
            T b_abs = fabs(b_val);
            if (b_val == 0) {
                c_out[0] = _sign(a_val);
                s_out[0] = 0;
                r_out[0] = a_abs;
            } else if (a_val == 0) {
                c_out[0] = 0;
                s_out[0] = _sign(b_val);
                r_out[0] = b_abs;
            } else if (b_abs > a_abs) {
                T tau = a_val / b_val;
                s_out[0] = _sign(b_val) / sqrt(1 + tau*tau);
                c_out[0] = s_out[0] * tau;
                r_out[0] = b_val / s_out[0];
            } else {
                T tau = b_val / a_val;
                c_out[0] = _sign(a_val) / sqrt(1 + tau*tau);
                s_out[0] = c_out[0] * tau;
                r_out[0] = a_val / c_out[0];
            }
        }
        """, name_expressions=("symortho<double>",))

        _symOrtho_kernel = _symOrtho_module.get_function("symortho<double>")
        c = cupy.empty((), dtype=cupy.float64)
        s = cupy.empty((), dtype=cupy.float64)
        r = cupy.empty((), dtype=cupy.float64)
        _symOrtho_kernel((1,), (1,), (a, b, c, s, r))
        return c, s, r

    _symOrtho = _symOrtho_raw

    if impl_name == "graph":
        gb = GraphBuilder()
    elif impl_name == "mock":
        gb = MockGraphBuilder()
    else:
        raise ValueError("impl_name is invalide")

    @gb.graphify
    def main_loop(
        itn,
        u,
        x,
        v,
        rho,
        rhobar,
        cbar,
        sbar,
        h,
        hbar,
        betadd,
        betad,
        rhodold,
        tautildeold,
        thetatilde,
        zeta,
        d,
        normA2,
        stop_flag,
    ):
        # Main iteration loop
        def while_fn(
            itn,
            u: cupy.ndarray,
            x: cupy.ndarray,
            v: cupy.ndarray,
            rho,
            rhobar,
            cbar,
            sbar,
            h: cupy.ndarray,
            hbar: cupy.ndarray,
            betadd,
            betad,
            rhodold,
            tautildeold,
            thetatilde,
            zeta,
            d,
            normA2,
            stop_flag,
        ):
            itn += 1

            # Perform the next step of the bidiagonalization to obtain the
            # next  beta, u, alpha, v.  These satisfy the relations
            #         beta*u  =  a*v   -  alpha*u,
            #        alpha*v  =  A'*u  -  beta*v.
            u *= -alpha
            u += matvec(v)
            nrm2_fn(u, out=beta)

            def beta_if():
                u[...] /= beta
                v[...] *= -beta
                v[...] += rmatvec(u)
                nrm2_fn(v, out=alpha)

                def alpha_if():
                    v[...] /= alpha
                gb.cond(
                    lambda: alpha > 0,
                    alpha_if,
                )
            gb.cond(
                lambda: beta > 0,
                beta_if,
            )

            # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

            # Construct rotation Qhat_{k,2k+1}.
            chat, shat, alphahat = _symOrtho(alphabar, damp)

            # Use a plane rotation (Q_i) to turn B_i to R_i

            rhoold = rho.copy()
            c, s, rho = _symOrtho(alphahat, beta)
            thetanew = s * alpha
            alphabar[...] = c * alpha

            # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

            rhobarold = rhobar.copy()
            zetaold = zeta.copy()
            thetabar = sbar * rho
            rhotemp = cbar * rho
            cbar, sbar, rhobar = _symOrtho(rhotemp, thetanew)
            zeta = cbar * zetabar
            zetabar[...] = - sbar * zetabar

            # Update h, h_hat, x.

            # hbar = h - (thetabar * rho / (rhoold * rhobarold)) * hbar
            # print(hbar)
            hbar *= -(thetabar * rho / (rhoold * rhobarold))
            hbar += h
            x += (zeta / (rho * rhobar)) * hbar
            # h = v - (thetanew / rho) * h
            h *= -(thetanew / rho)
            h += v

            # Estimate of ||r||.

            # Apply rotation Qhat_{k,2k+1}.
            betaacute = chat * betadd
            betacheck = -shat * betadd

            # Apply rotation Q_{k,k+1}.
            betahat = c * betaacute
            betadd = -s * betaacute

            # Apply rotation Qtilde_{k-1}.
            # betad = betad_{k-1} here.

            thetatildeold = thetatilde.copy()
            ctildeold, stildeold, rhotildeold = _symOrtho(rhodold, thetabar)
            thetatilde = stildeold * rhobar
            rhodold = ctildeold * rhobar
            betad = - stildeold * betad + ctildeold * betahat

            # betad   = betad_k here.
            # rhodold = rhod_k  here.

            tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
            taud = (zeta - thetatilde * tautildeold) / rhodold
            d = d + betacheck * betacheck
            cupy.sqrt(d + (betad - taud)**2 + betadd * betadd, out=normr)

            # Estimate ||A||.
            normA2 = normA2 + beta * beta
            cupy.sqrt(normA2, out=normA)
            normA2 = normA2 + alpha * alpha

            # Estimate cond(A).
            cupy.maximum(maxrbar, rhobarold, out=maxrbar)

            def itn_cond():
                minrbar[...] = cupy.minimum(minrbar, rhobarold)
            gb.cond(
                lambda: itn > 1,
                itn_cond
            )
            condA = cupy.maximum(maxrbar, rhotemp) / \
                cupy.minimum(minrbar, rhotemp)

            # Test for convergence.

            # Compute norms for convergence testing.
            normar = cupy.absolute(zetabar)
            nrm2_fn(x, out=normx)

            # Now use these norms to estimate certain other quantities,
            # some of which will be small near a solution.

            test1 = normr / normb
            test2 = cupy.empty((), dtype=cupy.float64)

            def true_fn():
                test2[...] = normar / normA * normr

            def false_fn():
                test2[...] = numpy.inf
            gb.multicond([
                (lambda: (normA * normr) != 0, true_fn),
                (None, false_fn)
            ])
            test3 = 1 / condA
            t1 = test1 / (1 + normA * normx / normb)
            rtol = btol + atol * normA * normx / normb

            def set_istop_fn(i):
                def fn():
                    istop[...] = i
                return fn
            gb.multicond([
                (lambda: test1 <= rtol, set_istop_fn(1)),
                (lambda: test2 <= atol, set_istop_fn(2)),
                (lambda: test3 <= ctol, set_istop_fn(3)),
                (lambda: 1 + t1 <= 1, set_istop_fn(4)),
                (lambda: 1 + test2 <= 1, set_istop_fn(5)),
                (lambda: 1 + test3 <= 1, set_istop_fn(6)),
                (lambda: itn >= maxiter, set_istop_fn(7)),
            ])

            stop_flag[...] = istop > 0

            return (
                itn,
                u,
                x,
                v,
                rho,
                rhobar,
                cbar,
                sbar,
                h,
                hbar,
                betadd,
                betad,
                rhodold,
                tautildeold,
                thetatilde,
                zeta,
                d,
                normA2,
                stop_flag,
            )
        return gb.while_loop(
            lambda itn, *_: (itn < maxiter) & (~stop_flag),
            while_fn,
            (
                itn,
                u,
                x,
                v,
                rho,
                rhobar,
                cbar,
                sbar,
                h,
                hbar,
                betadd,
                betad,
                rhodold,
                tautildeold,
                thetatilde,
                zeta,
                d,
                normA2,
                stop_flag,
            )
        )

    itn, u, x, v, *_ = main_loop(
        itn,
        u,
        x,
        v,
        rho,
        rhobar,
        cbar,
        sbar,
        h,
        hbar,
        betadd,
        betad,
        rhodold,
        tautildeold,
        thetatilde,
        zeta,
        d,
        normA2,
        stop_flag,
    )

    # The return type of SciPy is always float64. Therefore, x must be casted.
    x = x.astype(numpy.float64)

    return x, istop, itn, normr, normar, normA, condA, normx


N = args.N
dens = 0.1
cupy.random.seed(42)
numpy.random.seed(42)
A = cupyx.scipy.sparse.random(
    N,
    N,
    density=dens,
    format="csr",
    dtype=numpy.float64)
b = cupy.random.randn(N)
x0 = cupy.zeros(N, dtype=cupy.float64)

print("Start")

start = time.time()
impl_name = args.impl.lower()
if impl_name == "cpu":
    A_cpu = A.get()
    b_cpu = b.get()
    x0_cpu = x0.get()
    start = time.time()
    x, istop, itn, *_ = lsmr_cpu(A_cpu, b_cpu, x0=x0_cpu, maxiter=args.maxiter)
else:
    if impl_name == "normal":
        x, istop, itn, *_ = lsmr_gpu(A, b, x0=x0, maxiter=args.maxiter)
    elif impl_name == "graph":
        x, istop, itn, * \
            _ = lsmr_graph(
                A,
                b,
                x0=x0,
                maxiter=args.maxiter,
                impl_name="graph")
    elif impl_name == "mock":
        x, istop, itn, * \
            _ = lsmr_graph(A, b, x0=x0, maxiter=args.maxiter, impl_name="mock")
    else:
        raise ValueError("impl_name is invalide")
    cupy.cuda.runtime.deviceSynchronize()
end = time.time()

print(f"{itn=}")
print(f"{istop=}")
print(f"{end - start} sec")

x = cupy.asarray(x)
A_ = _interface.aslinearoperator(A)
Ax = A_.matvec(x)
print(f"{x[:10]=}")
print(f"{Ax[:10]=}")
print(f"{b[:10]=}")
print(f"||Ax - b|| = {cublas.nrm2(Ax - b)}")
