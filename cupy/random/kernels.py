from cupy import core

_beta_kernel = None
_binomial_kernel = None
_chisquare_kernel = None
_gumbel_kernel = None
_laplace_kernel = None
_poisson_kernel = None
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


# TODO(YoshikawaMasashi): implementation of BTPE same as numpy
def _get_binomial_kernel():
    global _binomial_kernel
    if _binomial_kernel is None:
        _binomial_kernel = core.ElementwiseKernel(
            'T x, T n, T p', 'T y',
            '''
            y = 0.;
            T px = exp(n * log(1-p));
            while(x > px){
                y += 1.;
                x -= px;
                px = ((n-y+1) * p * px)/(y*(1-p));
            }
            ''',
            'binomial_kernel'
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


def _get_poisson_kernel():
    global _poisson_kernel
    if _poisson_kernel is None:
        definitions = \
            [rk_state_difinition, rk_seed_definition, rk_random_definition,
             rk_double_definition, loggam_difinition,
             rk_poisson_mult_definition, rk_poisson_ptrs_definition,
             rk_poisson_definition]
        _poisson_kernel = core.ElementwiseKernel(
            'T lam, T seed', 'T y',
            '''
            rk_seed(seed + i, &internal_state);
            y = rk_poisson(&internal_state, lam);
            ''',
            'poisson_kernel',
            preamble=''.join(definitions),
            loop_prep="rk_state internal_state;"
        )
    return _poisson_kernel


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