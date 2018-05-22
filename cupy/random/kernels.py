from cupy import core

_beta_kernel = None
_binomial_kernel = None
_chisquare_kernel = None
_f_kernel = None
_geometric_kernel = None
_gumbel_kernel = None
_laplace_kernel = None
_pareto_kernel = None
_poisson_kernel = None
_standard_cauchy_kernel = None
_standard_exponential_kernel = None
_standard_gamma_kernel = None
_standard_t_kernel = None


loggam_difinition = '''
/*
 * log-gamma function to support some of these distributions. The
 * algorithm comes from SPECFUN by Shanjie Zhang and Jianming Jin and their
 * book "Computation of Special Functions", 1996, John Wiley & Sons, Inc.
 */
static __device__ double loggam(double x)
{
    double x0, x2, xp, gl, gl0;
    long k, n;

    static double a[10] = {8.333333333333333e-02,-2.777777777777778e-03,
         7.936507936507937e-04,-5.952380952380952e-04,
         8.417508417508418e-04,-1.917526917526918e-03,
         6.410256410256410e-03,-2.955065359477124e-02,
         1.796443723688307e-01,-1.39243221690590e+00};
    x0 = x;
    n = 0;
    if ((x == 1.0) || (x == 2.0))
    {
        return 0.0;
    }
    else if (x <= 7.0)
    {
        n = (long)(7 - x);
        x0 = x + n;
    }
    x2 = 1.0/(x0*x0);
    xp = 2*M_PI;
    gl0 = a[9];
    for (k=8; k>=0; k--)
    {
        gl0 *= x2;
        gl0 += a[k];
    }
    gl = gl0/x0 + 0.5*log(xp) + (x0-0.5)*log(x0) - x0;
    if (x <= 7.0)
    {
        for (k=1; k<=n; k++)
        {
            gl -= log(x0-1.0);
            x0 -= 1.0;
        }
    }
    return gl;
}
'''


rk_state_difinition = '''
#define RK_STATE_LEN 624

typedef struct rk_state_
{
    unsigned long key[RK_STATE_LEN];
    int pos;
    int has_gauss; /* !=0: gauss contains a gaussian deviate */
    double gauss;

    /* The rk_state structure has been extended to store the following
     * information for the binomial generator. If the input values of n or p
     * are different than nsave and psave, then the other parameters will be
     * recomputed. RTK 2005-09-02 */

    int has_binomial; /* !=0: following parameters initialized for
                              binomial */
    double psave;
    long nsave;
    double r;
    double q;
    double fm;
    long m;
    double p1;
    double xm;
    double xl;
    double xr;
    double c;
    double laml;
    double lamr;
    double p2;
    double p3;
    double p4;

}
rk_state;
'''


rk_seed_definition = '''
__device__ void
rk_seed(unsigned long seed, rk_state *state)
{
    int pos;
    seed &= 0xffffffffUL;

    /* Knuth's PRNG as used in the Mersenne Twister reference implementation */
    for (pos = 0; pos < RK_STATE_LEN; pos++) {
        state->key[pos] = seed;
        seed = (1812433253UL * (seed ^ (seed >> 30)) + pos + 1) & 0xffffffffUL;
    }
    state->pos = RK_STATE_LEN;
    state->gauss = 0;
    state->has_gauss = 0;
    state->has_binomial = 0;
}
'''


rk_random_definition = '''
/* Magic Mersenne Twister constants */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL

/*
 * Slightly optimised reference implementation of the Mersenne Twister
 * Note that regardless of the precision of long, only 32 bit random
 * integers are produced
 */
__device__ unsigned long
rk_random(rk_state *state)
{
    unsigned long y;

    if (state->pos == RK_STATE_LEN) {
        int i;

        for (i = 0; i < N - M; i++) {
            y = (state->key[i] & UPPER_MASK) | (state->key[i+1] & LOWER_MASK);
            state->key[i] = state->key[i+M] ^ (y>>1) ^ (-(y & 1) & MATRIX_A);
        }
        for (; i < N - 1; i++) {
            y = (state->key[i] & UPPER_MASK) | (state->key[i+1] & LOWER_MASK);
            state->key[i]
                = state->key[i+(M-N)] ^ (y>>1) ^ (-(y & 1) & MATRIX_A);
        }
        y = (state->key[N - 1] & UPPER_MASK) | (state->key[0] & LOWER_MASK);
        state->key[N - 1]
            = state->key[M - 1] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);

        state->pos = 0;
    }
    y = state->key[state->pos++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}
'''


rk_double_definition = '''
__device__ double
rk_double(rk_state *state)
{
    /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
    long a = rk_random(state) >> 5, b = rk_random(state) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}
'''


rk_gauss_definition = '''
__device__ double
rk_gauss(rk_state *state)
{
    if (state->has_gauss) {
        const double tmp = state->gauss;
        state->gauss = 0;
        state->has_gauss = 0;
        return tmp;
    }
    else {
        double f, x1, x2, r2;

        do {
            x1 = 2.0*rk_double(state) - 1.0;
            x2 = 2.0*rk_double(state) - 1.0;
            r2 = x1*x1 + x2*x2;
        }
        while (r2 >= 1.0 || r2 == 0.0);

        /* Box-Muller transform */
        f = sqrt(-2.0*log(r2)/r2);
        /* Keep for next call */
        state->gauss = f*x1;
        state->has_gauss = 1;
        return f*x2;
    }
}
'''


rk_standard_exponential_definition = '''
__device__ double rk_standard_exponential(rk_state *state)
{
    /* We use -log(1-U) since U is [0, 1) */
    return -log(1.0 - rk_double(state));
}
'''


rk_standard_gamma_definition = '''
__device__ double rk_standard_gamma(rk_state *state, double shape)
{
    double b, c;
    double U, V, X, Y;

    if (shape == 1.0)
    {
        return rk_standard_exponential(state);
    }
    else if (shape < 1.0)
    {
        for (;;)
        {
            U = rk_double(state);
            V = rk_standard_exponential(state);
            if (U <= 1.0 - shape)
            {
                X = pow(U, 1./shape);
                if (X <= V)
                {
                    return X;
                }
            }
            else
            {
                Y = -log((1-U)/shape);
                X = pow(1.0 - shape + shape*Y, 1./shape);
                if (X <= (V + Y))
                {
                    return X;
                }
            }
        }
    }
    else
    {
        b = shape - 1./3.;
        c = 1./sqrt(9*b);
        for (;;)
        {
            do
            {
                X = rk_gauss(state);
                V = 1.0 + c*X;
            } while (V <= 0.0);

            V = V*V*V;
            U = rk_double(state);
            if (U < 1.0 - 0.0331*(X*X)*(X*X)) return (b*V);
            if (log(U) < 0.5*X*X + b*(1. - V + log(V))) return (b*V);
        }
    }
}
'''


rk_beta_definition = '''
__device__ double rk_beta(rk_state *state, double a, double b)
{
    double Ga, Gb;

    if ((a <= 1.0) && (b <= 1.0))
    {
        double U, V, X, Y;
        /* Use Johnk's algorithm */

        while (1)
        {
            U = rk_double(state);
            V = rk_double(state);
            X = pow(U, 1.0/a);
            Y = pow(V, 1.0/b);

            if ((X + Y) <= 1.0)
            {
                if (X +Y > 0)
                {
                    return X / (X + Y);
                }
                else
                {
                    double logX = log(U) / a;
                    double logY = log(V) / b;
                    double logM = logX > logY ? logX : logY;
                    logX -= logM;
                    logY -= logM;

                    return exp(logX - log(exp(logX) + exp(logY)));
                }
            }
        }
    }
    else
    {
        Ga = rk_standard_gamma(state, a);
        Gb = rk_standard_gamma(state, b);
        return Ga/(Ga + Gb);
    }
}
'''

rk_chisquare_definition = '''
__device__ double rk_chisquare(rk_state *state, double df)
{
    return 2.0*rk_standard_gamma(state, df/2.0);
}
'''

rk_f_definition = '''
__device__ double rk_f(rk_state *state, double dfnum, double dfden)
{
    return ((rk_chisquare(state, dfnum) * dfden) /
            (rk_chisquare(state, dfden) * dfnum));
}
'''

rk_standard_t_definition = '''
__device__ double rk_standard_t(rk_state *state, double df)
{
    return sqrt(df/2)*rk_gauss(state)/sqrt(rk_standard_gamma(state, df/2));
}
'''


rk_poisson_mult_definition = '''
__device__ long rk_poisson_mult(rk_state *state, double lam)
{
    long X;
    double prod, U, enlam;

    enlam = exp(-lam);
    X = 0;
    prod = 1.0;
    while (1)
    {
        U = rk_double(state);
        prod *= U;
        if (prod > enlam)
        {
            X += 1;
        }
        else
        {
            return X;
        }
    }
}
'''


rk_poisson_ptrs_definition = '''
/*
 * The transformed rejection method for generating Poisson random variables
 * W. Hoermann
 * Insurance: Mathematics and Economics 12, 39-45 (1993)
 */
#define LS2PI 0.91893853320467267
#define TWELFTH 0.083333333333333333333333
__device__ long rk_poisson_ptrs(rk_state *state, double lam)
{
    long k;
    double U, V, slam, loglam, a, b, invalpha, vr, us;

    slam = sqrt(lam);
    loglam = log(lam);
    b = 0.931 + 2.53*slam;
    a = -0.059 + 0.02483*b;
    invalpha = 1.1239 + 1.1328/(b-3.4);
    vr = 0.9277 - 3.6224/(b-2);

    while (1)
    {
        U = rk_double(state) - 0.5;
        V = rk_double(state);
        us = 0.5 - fabs(U);
        k = (long)floor((2*a/us + b)*U + lam + 0.43);
        if ((us >= 0.07) && (V <= vr))
        {
            return k;
        }
        if ((k < 0) ||
            ((us < 0.013) && (V > us)))
        {
            continue;
        }
        if ((log(V) + log(invalpha) - log(a/(us*us)+b)) <=
            (-lam + k*loglam - loggam(k+1)))
        {
            return k;
        }


    }

}
'''


rk_poisson_definition = '''
__device__ long rk_poisson(rk_state *state, double lam)
{
    if (lam >= 10)
    {
        return rk_poisson_ptrs(state, lam);
    }
    else if (lam == 0)
    {
        return 0;
    }
    else
    {
        return rk_poisson_mult(state, lam);
    }
}
'''


rk_standard_cauchy_definition = '''
__device__ double rk_standard_cauchy(rk_state *state)
{
    return rk_gauss(state) / rk_gauss(state);
}
'''


rk_geometric_search_definition = '''
__device__ long rk_geometric_search(rk_state *state, double p)
{
    double U;
    long X;
    double sum, prod, q;

    X = 1;
    sum = prod = p;
    q = 1.0 - p;
    U = rk_double(state);
    while (U > sum)
    {
        prod *= q;
        sum += prod;
        X++;
    }
    return X;
}
'''


rk_geometric_inversion_definition = '''
__device__ long rk_geometric_inversion(rk_state *state, double p)
{
    return (long)ceil(log(1.0-rk_double(state))/log(1.0-p));
}
'''


rk_geometric_definition = '''
__device__ long rk_geometric(rk_state *state, double p)
{
    if (p >= 0.333333333333333333333333)
    {
        return rk_geometric_search(state, p);
    } else
    {
        return rk_geometric_inversion(state, p);
    }
}
'''


rk_pareto_definition = '''
__device__ double rk_pareto(rk_state *state, double a)
{
    return exp(rk_standard_exponential(state)/a) - 1;
}
'''

rk_binomial_btpe_definition = '''
__device__ long rk_binomial_btpe(rk_state *state, long n, double p)
{
    double r,q,fm,p1,xm,xl,xr,c,laml,lamr,p2,p3,p4;
    double a,u,v,s,F,rho,t,A,nrq,x1,x2,f1,f2,z,z2,w,w2,x;
    long m,y,k,i;

    if (!(state->has_binomial) ||
         (state->nsave != n) ||
         (state->psave != p))
    {
        /* initialize */
        state->nsave = n;
        state->psave = p;
        state->has_binomial = 1;
        state->r = r = min(p, 1.0-p);
        state->q = q = 1.0 - r;
        state->fm = fm = n*r+r;
        state->m = m = (long)floor(state->fm);
        state->p1 = p1 = floor(2.195*sqrt(n*r*q)-4.6*q) + 0.5;
        state->xm = xm = m + 0.5;
        state->xl = xl = xm - p1;
        state->xr = xr = xm + p1;
        state->c = c = 0.134 + 20.5/(15.3 + m);
        a = (fm - xl)/(fm-xl*r);
        state->laml = laml = a*(1.0 + a/2.0);
        a = (xr - fm)/(xr*q);
        state->lamr = lamr = a*(1.0 + a/2.0);
        state->p2 = p2 = p1*(1.0 + 2.0*c);
        state->p3 = p3 = p2 + c/laml;
        state->p4 = p4 = p3 + c/lamr;
    }
    else
    {
        r = state->r;
        q = state->q;
        fm = state->fm;
        m = state->m;
        p1 = state->p1;
        xm = state->xm;
        xl = state->xl;
        xr = state->xr;
        c = state->c;
        laml = state->laml;
        lamr = state->lamr;
        p2 = state->p2;
        p3 = state->p3;
        p4 = state->p4;
    }

  /* sigh ... */
  Step10:
    nrq = n*r*q;
    u = rk_double(state)*p4;
    v = rk_double(state);
    if (u > p1) goto Step20;
    y = (long)floor(xm - p1*v + u);
    goto Step60;

  Step20:
    if (u > p2) goto Step30;
    x = xl + (u - p1)/c;
    v = v*c + 1.0 - fabs(m - x + 0.5)/p1;
    if (v > 1.0) goto Step10;
    y = (long)floor(x);
    goto Step50;

  Step30:
    if (u > p3) goto Step40;
    y = (long)floor(xl + log(v)/laml);
    if (y < 0) goto Step10;
    v = v*(u-p2)*laml;
    goto Step50;

  Step40:
    y = (long)floor(xr - log(v)/lamr);
    if (y > n) goto Step10;
    v = v*(u-p3)*lamr;

  Step50:
    k = labs(y - m);
    if ((k > 20) && (k < ((nrq)/2.0 - 1))) goto Step52;

    s = r/q;
    a = s*(n+1);
    F = 1.0;
    if (m < y)
    {
        for (i=m+1; i<=y; i++)
        {
            F *= (a/i - s);
        }
    }
    else if (m > y)
    {
        for (i=y+1; i<=m; i++)
        {
            F /= (a/i - s);
        }
    }
    if (v > F) goto Step10;
    goto Step60;

    Step52:
    rho = (k/(nrq))*((k*(k/3.0 + 0.625) + 0.16666666666666666)/nrq + 0.5);
    t = -k*k/(2*nrq);
    A = log(v);
    if (A < (t - rho)) goto Step60;
    if (A > (t + rho)) goto Step10;

    x1 = y+1;
    f1 = m+1;
    z = n+1-m;
    w = n-y+1;
    x2 = x1*x1;
    f2 = f1*f1;
    z2 = z*z;
    w2 = w*w;
    if (A > (xm*log(f1/x1)
           + (n-m+0.5)*log(z/w)
           + (y-m)*log(w*r/(x1*q))
           + (13680.-(462.-(132.-(99.-140./f2)/f2)/f2)/f2)/f1/166320.
           + (13680.-(462.-(132.-(99.-140./z2)/z2)/z2)/z2)/z/166320.
           + (13680.-(462.-(132.-(99.-140./x2)/x2)/x2)/x2)/x1/166320.
           + (13680.-(462.-(132.-(99.-140./w2)/w2)/w2)/w2)/w/166320.))
    {
        goto Step10;
    }

  Step60:
    if (p > 0.5)
    {
        y = n - y;
    }

    return y;
}
'''


rk_binomial_inversion_definition = '''
__device__ long rk_binomial_inversion(rk_state *state, long n, double p)
{
    double q, qn, np, px, U;
    long X, bound;

    if (!(state->has_binomial) ||
         (state->nsave != n) ||
         (state->psave != p))
    {
        state->nsave = n;
        state->psave = p;
        state->has_binomial = 1;
        state->q = q = 1.0 - p;
        state->r = qn = exp(n * log(q));
        state->c = np = n*p;
        state->m = bound = min((double)n, np + 10.0*sqrt(np*q + 1));
    } else
    {
        q = state->q;
        qn = state->r;
        np = state->c;
        bound = state->m;
    }
    X = 0;
    px = qn;
    U = rk_double(state);
    while (U > px)
    {
        X++;
        if (X > bound)
        {
            X = 0;
            px = qn;
            U = rk_double(state);
        } else
        {
            U -= px;
            px  = ((n-X+1) * p * px)/(X*q);
        }
    }
    return X;
}
'''

rk_binomial_definition = '''
__device__ long rk_binomial(rk_state *state, long n, double p)
{
    double q;

    if (p <= 0.5)
    {
        if (p*n <= 30.0)
        {
            return rk_binomial_inversion(state, n, p);
        }
        else
        {
            return rk_binomial_btpe(state, n, p);
        }
    }
    else
    {
        q = 1.0-p;
        if (q*n <= 30.0)
        {
            return n - rk_binomial_inversion(state, n, q);
        }
        else
        {
            return n - rk_binomial_btpe(state, n, q);
        }
    }

}
'''


def _get_beta_kernel():
    global _beta_kernel
    if _beta_kernel is None:
        definitions = \
            [rk_state_difinition, rk_seed_definition, rk_random_definition,
             rk_double_definition, rk_gauss_definition,
             rk_standard_exponential_definition, rk_standard_gamma_definition,
             rk_beta_definition]
        _beta_kernel = core.ElementwiseKernel(
            'T a, T b, T seed', 'T y',
            '''
            rk_seed(seed + i, &internal_state);
            y = rk_beta(&internal_state, a, b);
            ''',
            'beta_kernel',
            preamble=''.join(definitions),
            loop_prep="rk_state internal_state;"
        )
    return _beta_kernel


def _get_binomial_kernel():
    global _binomial_kernel
    if _binomial_kernel is None:
        definitions = \
            [rk_state_difinition, rk_seed_definition, rk_random_definition,
             rk_double_definition, rk_binomial_btpe_definition,
             rk_binomial_inversion_definition, rk_binomial_definition]
        _binomial_kernel = core.ElementwiseKernel(
            'T n, float64 p, T seed', 'T y',
            '''
            rk_seed(seed + i, &internal_state);
            y = rk_binomial(&internal_state, n, p);
            ''',
            'binomial_kernel',
            preamble=''.join(definitions),
            loop_prep="rk_state internal_state;"
        )
    return _binomial_kernel


def _get_chisquare_kernel():
    global _chisquare_kernel
    if _chisquare_kernel is None:
        definitions = \
            [rk_state_difinition, rk_seed_definition, rk_random_definition,
             rk_double_definition, rk_gauss_definition,
             rk_standard_exponential_definition, rk_standard_gamma_definition,
             rk_chisquare_definition]
        _chisquare_kernel = core.ElementwiseKernel(
            'T df, T seed', 'T y',
            '''
            rk_seed(seed + i, &internal_state);
            y = rk_chisquare(&internal_state, df);
            ''',
            'beta_kernel',
            preamble=''.join(definitions),
            loop_prep="rk_state internal_state;"
        )
    return _chisquare_kernel


def _get_f_kernel():
    global _f_kernel
    if _f_kernel is None:
        definitions = \
            [rk_state_difinition, rk_seed_definition, rk_random_definition,
             rk_double_definition, rk_gauss_definition,
             rk_standard_exponential_definition, rk_standard_gamma_definition,
             rk_chisquare_definition, rk_f_definition]
        _f_kernel = core.ElementwiseKernel(
            'T dfnum, T dfden, T seed', 'T y',
            '''
            rk_seed(seed + i, &internal_state);
            y = rk_f(&internal_state, dfnum, dfden);
            ''',
            'f_kernel',
            preamble=''.join(definitions),
            loop_prep="rk_state internal_state;"
        )
    return _f_kernel


def _get_geometric_kernel():
    global _geometric_kernel
    if _geometric_kernel is None:
        definitions = \
            [rk_state_difinition, rk_seed_definition, rk_random_definition,
             rk_double_definition, rk_geometric_search_definition,
             rk_geometric_inversion_definition, rk_geometric_definition]
        _geometric_kernel = core.ElementwiseKernel(
            'float64 p, T seed', 'T y',
            '''
            rk_seed(seed + i, &internal_state);
            y = rk_geometric(&internal_state, p);
            ''',
            'geometric_kernel',
            preamble=''.join(definitions),
            loop_prep="rk_state internal_state;"
        )
    return _geometric_kernel


def _get_gumbel_kernel():
    global _gumbel_kernel
    if _gumbel_kernel is None:
        _gumbel_kernel = core.ElementwiseKernel(
            'T x, T loc, T scale', 'T y',
            'y = loc - log(-log(1 - x)) * scale',
            'gumbel_kernel'
        )
    return _gumbel_kernel


def _get_laplace_kernel():
    global _laplace_kernel
    if _laplace_kernel is None:
        _laplace_kernel = core.ElementwiseKernel(
            'T x, T loc, T scale', 'T y',
            'y = (x < 0.5)? loc + scale * log(x + x):'
            ' loc - scale * log(2.0 - x - x)',
            'laplace_kernel'
        )
    return _laplace_kernel


def _get_pareto_kernel():
    global _pareto_kernel
    if _pareto_kernel is None:
        definitions = \
            [rk_state_difinition, rk_seed_definition, rk_random_definition,
             rk_double_definition, rk_standard_exponential_definition,
             rk_pareto_definition]
        _pareto_kernel = core.ElementwiseKernel(
            'T a, T seed', 'T y',
            '''
            rk_seed(seed + i, &internal_state);
            y = rk_pareto(&internal_state, a);
            ''',
            'pareto_kernel',
            preamble=''.join(definitions),
            loop_prep="rk_state internal_state;"
        )
    return _pareto_kernel


def _get_poisson_kernel():
    global _poisson_kernel
    if _poisson_kernel is None:
        definitions = \
            [rk_state_difinition, rk_seed_definition, rk_random_definition,
             rk_double_definition, loggam_difinition,
             rk_poisson_mult_definition, rk_poisson_ptrs_definition,
             rk_poisson_definition]
        _poisson_kernel = core.ElementwiseKernel(
            'float64 lam, T seed', 'T y',
            '''
            rk_seed(seed + i, &internal_state);
            y = rk_poisson(&internal_state, lam);
            ''',
            'poisson_kernel',
            preamble=''.join(definitions),
            loop_prep="rk_state internal_state;"
        )
    return _poisson_kernel


def _get_standard_cauchy_kernel():
    global _standard_cauchy_kernel
    if _standard_cauchy_kernel is None:
        definitions = \
            [rk_state_difinition, rk_seed_definition, rk_random_definition,
             rk_double_definition, rk_gauss_definition,
             rk_standard_cauchy_definition]
        _standard_cauchy_kernel = core.ElementwiseKernel(
            'T seed', 'T y',
            '''
            rk_seed(seed + i, &internal_state);
            y = rk_standard_cauchy(&internal_state);
            ''',
            'standard_gamma_kernel',
            preamble=''.join(definitions),
            loop_prep="rk_state internal_state;"
        )
    return _standard_cauchy_kernel


def _get_standard_exponential_kernel():
    global _standard_exponential_kernel
    if _standard_exponential_kernel is None:
        definitions = \
            [rk_state_difinition, rk_seed_definition, rk_random_definition,
             rk_double_definition, rk_standard_exponential_definition]
        _standard_exponential_kernel = core.ElementwiseKernel(
            'T seed', 'T y',
            '''
            rk_seed(seed + i, &internal_state);
            y = rk_standard_exponential(&internal_state);
            ''',
            'standard_exponential_kernel',
            preamble=''.join(definitions),
            loop_prep="rk_state internal_state;"
        )
    return _standard_exponential_kernel


def _get_standard_gamma_kernel():
    global _standard_gamma_kernel
    if _standard_gamma_kernel is None:
        definitions = \
            [rk_state_difinition, rk_seed_definition, rk_random_definition,
             rk_double_definition, rk_gauss_definition,
             rk_standard_exponential_definition, rk_standard_gamma_definition]
        _standard_gamma_kernel = core.ElementwiseKernel(
            'T shape, T seed', 'T y',
            '''
            rk_seed(seed + i, &internal_state);
            y = rk_standard_gamma(&internal_state, shape);
            ''',
            'standard_gamma_kernel',
            preamble=''.join(definitions),
            loop_prep="rk_state internal_state;"
        )
    return _standard_gamma_kernel


def _get_standard_t_kernel():
    global _standard_t_kernel
    if _standard_t_kernel is None:
        definitions = \
            [rk_state_difinition, rk_seed_definition, rk_random_definition,
             rk_double_definition, rk_gauss_definition,
             rk_standard_exponential_definition, rk_standard_gamma_definition,
             rk_standard_t_definition]
        _standard_t_kernel = core.ElementwiseKernel(
            'T df, T seed', 'T y',
            '''
            rk_seed(seed + i, &internal_state);
            y = rk_standard_t(&internal_state, df);
            ''',
            'standard_t_kernel',
            preamble=''.join(definitions),
            loop_prep="rk_state internal_state;"
        )
    return _standard_t_kernel
