from cupy import core


rk_state_difinition = '''
#define RK_STATE_LEN 624
__shared__ typedef struct rk_state_ {
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
rk_seed(unsigned long seed, rk_state *state) {
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
rk_random(rk_state *state) {
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
rk_double(rk_state *state) {
    /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
    long a = rk_random(state) >> 5, b = rk_random(state) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}
'''

rk_binomial_btpe_definition = '''
__device__ long rk_binomial_btpe(rk_state *state, long n, double p) {
    double r,q,fm,p1,xm,xl,xr,c,laml,lamr,p2,p3,p4;
    double a,u,v,s,F,rho,t,A,nrq,x1,x2,f1,f2,z,z2,w,w2,x;
    long m,y,k,i;
    if (!(state->has_binomial) ||
         (state->nsave != n) ||
         (state->psave != p)) {
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
    } else {
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
    if (m < y) {
        for (i=m+1; i<=y; i++) {
            F *= (a/i - s);
        }
    } else if (m > y) {
        for (i=y+1; i<=m; i++) {
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
           + (13680.-(462.-(132.-(99.-140./w2)/w2)/w2)/w2)/w/166320.)) {
        goto Step10;
    }
  Step60:
    if (p > 0.5) {
        y = n - y;
    }
    return y;
}
'''

rk_binomial_inversion_definition = '''
__device__ long rk_binomial_inversion(rk_state *state, long n, double p) {
    double q, qn, np, px, U;
    long X, bound;
    if (!(state->has_binomial) ||
         (state->nsave != n) ||
         (state->psave != p)) {
        state->nsave = n;
        state->psave = p;
        state->has_binomial = 1;
        state->q = q = 1.0 - p;
        state->r = qn = exp(n * log(q));
        state->c = np = n*p;
        state->m = bound = min((double)n, np + 10.0*sqrt(np*q + 1));
    } else {
        q = state->q;
        qn = state->r;
        np = state->c;
        bound = state->m;
    }
    X = 0;
    px = qn;
    U = rk_double(state);
    while (U > px) {
        X++;
        if (X > bound) {
            X = 0;
            px = qn;
            U = rk_double(state);
        } else {
            U -= px;
            px  = ((n-X+1) * p * px)/(X*q);
        }
    }
    return X;
}
'''

rk_binomial_definition = '''
__device__ long rk_binomial(rk_state *state, long n, double p) {
    double q;
    if (p <= 0.5) {
        if (p*n <= 30.0) {
            return rk_binomial_inversion(state, n, p);
        } else {
            return rk_binomial_btpe(state, n, p);
        }
    } else {
        q = 1.0-p;
        if (q*n <= 30.0) {
            return n - rk_binomial_inversion(state, n, q);
        } else {
            return n - rk_binomial_btpe(state, n, q);
        }
    }
}
'''

rk_gauss_definition = '''
__device__ double rk_gauss(rk_state *state) {
    if (state->has_gauss) {
        const double tmp = state->gauss;
        state->gauss = 0;
        state->has_gauss = 0;
        return tmp;
    } else {
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
__device__ double rk_standard_exponential(rk_state *state) {
    /* We use -log(1-U) since U is [0, 1) */
    return -log(1.0 - rk_double(state));
}
'''

rk_standard_gamma_definition = '''
__device__ double rk_standard_gamma(rk_state *state, double shape) {
    double b, c;
    double U, V, X, Y;
    if (shape == 1.0) {
        return rk_standard_exponential(state);
    } else if (shape < 1.0) {
        for (;;) {
            U = rk_double(state);
            V = rk_standard_exponential(state);
            if (U <= 1.0 - shape) {
                X = pow(U, 1./shape);
                if (X <= V) {
                    return X;
                }
            } else {
                Y = -log((1-U)/shape);
                X = pow(1.0 - shape + shape*Y, 1./shape);
                if (X <= (V + Y)) {
                    return X;
                }
            }
        }
    } else {
        b = shape - 1./3.;
        c = 1./sqrt(9*b);
        for (;;) {
            do {
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
__device__ double rk_beta(rk_state *state, double a, double b) {
    double Ga, Gb;
    if ((a <= 1.0) && (b <= 1.0)) {
        double U, V, X, Y;
        /* Use Johnk's algorithm */
        while (1) {
            U = rk_double(state);
            V = rk_double(state);
            X = pow(U, 1.0/a);
            Y = pow(V, 1.0/b);
            if ((X + Y) <= 1.0) {
                if (X +Y > 0) {
                    return X / (X + Y);
                } else {
                    double logX = log(U) / a;
                    double logY = log(V) / b;
                    double logM = logX > logY ? logX : logY;
                    logX -= logM;
                    logY -= logM;
                    return exp(logX - log(exp(logX) + exp(logY)));
                }
            }
        }
    } else {
        Ga = rk_standard_gamma(state, a);
        Gb = rk_standard_gamma(state, b);
        return Ga/(Ga + Gb);
    }
}
'''

definitions = \
    [rk_state_difinition, rk_seed_definition, rk_random_definition,
     rk_double_definition, rk_binomial_btpe_definition,
     rk_binomial_inversion_definition, rk_binomial_definition]
binomial_kernel = core.ElementwiseKernel(
    'S n, T p, uint32 seed', 'Y y',
    '''
    rk_seed(seed + i, &internal_state);
    y = rk_binomial(&internal_state, n, p);
    ''',
    'binomial_kernel',
    preamble=''.join(definitions),
    loop_prep="rk_state internal_state;"
)

definitions = \
    [rk_state_difinition, rk_seed_definition, rk_random_definition,
     rk_double_definition, rk_gauss_definition,
     rk_standard_exponential_definition, rk_standard_gamma_definition]
standard_gamma_kernel = core.ElementwiseKernel(
    'T shape, uint32 seed', 'Y y',
    '''
    rk_seed(seed + i, &internal_state);
    y = rk_standard_gamma(&internal_state, shape);
    ''',
    'standard_gamma_kernel',
    preamble=''.join(definitions),
    loop_prep="rk_state internal_state;"
)

definitions = \
    [rk_state_difinition, rk_seed_definition, rk_random_definition,
     rk_double_definition, rk_gauss_definition,
     rk_standard_exponential_definition, rk_standard_gamma_definition,
     rk_beta_definition]
beta_kernel = core.ElementwiseKernel(
    'S a, T b, uint32 seed', 'Y y',
    '''
    rk_seed(seed + i, &internal_state);
    y = rk_beta(&internal_state, a, b);
    ''',
    'beta_kernel',
    preamble=''.join(definitions),
    loop_prep="rk_state internal_state;"
)
