#include <stdio.h>
#include <stdexcept>
#include <utility>
#include <iostream>
#include <stdint.h>
#include <type_traits>


#include "cupy_distributions.cuh"


// avoid explicit compilation for hip<4.3
#if !defined(CUPY_USE_HIP) || defined(COMPILE_FOR_HIP)

struct rk_state {

    __device__ virtual uint32_t rk_int() {
        return  0;
    }
    __device__ virtual double rk_double() {
        return  0.0;
    }
    __device__ virtual double rk_normal() {
        return  0.0;
    }
    __device__ virtual float rk_normal_float() {
        return  0.0;
    }
};

template<typename CURAND_TYPE>
struct curand_pseudo_state: rk_state {
    // Valid for  XORWOW and MRG32k3a
    CURAND_TYPE* _state;
    int _id;

    __device__ curand_pseudo_state(int id, intptr_t state) {
        _state = reinterpret_cast<CURAND_TYPE*>(state) + id;
        _id = id;
    }
    __device__ virtual uint32_t rk_int() {
        return curand(_state);
    }
    __device__ virtual double rk_double() {
        // Curand returns (0, 1] while the functions
        // below rely on [0, 1)
        double r = curand_uniform(_state);
        if (r >= 1.0) { 
           r = 0.0;
        }
        return r;
    }
    __device__ virtual double rk_normal() {
        return curand_normal_double(_state);
    }

    __device__ virtual float rk_normal_float() {
        return curand_normal(_state);
    }
};


// This design is the same as the dtypes one
template <typename F, typename... Ts>
void generator_dispatcher(int generator_id, F f, Ts&&... args) {
   switch(generator_id) {
       case CURAND_XOR_WOW: return f.template operator()<curand_pseudo_state<curandState>>(std::forward<Ts>(args)...);
       case CURAND_MRG32k3a: return f.template operator()<curand_pseudo_state<curandStateMRG32k3a>>(std::forward<Ts>(args)...);
       case CURAND_PHILOX_4x32_10: return f.template operator()<curand_pseudo_state<curandStatePhilox4_32_10_t>>(std::forward<Ts>(args)...);
       default: throw std::runtime_error("Unknown random generator");
   }
}


template<typename T>
__global__ void init_curand(intptr_t state, uint64_t seed, ssize_t size) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    T curand_state(id, state);
    if (id < size) {
        curand_init(seed, id, 0, curand_state._state);    
    }
}

struct initialize_launcher {
    initialize_launcher(ssize_t size, cudaStream_t stream) : _size(size), _stream(stream) {
    }
    template<typename T, typename... Args>
    void operator()(Args&&... args) { 
        int tpb = 256;
        int bpg =  (_size + tpb - 1) / tpb;
        init_curand<T><<<bpg, tpb, 0, _stream>>>(std::forward<Args>(args)...);
    }
    ssize_t _size;
    cudaStream_t _stream;
};

void init_curand_generator(int generator, intptr_t state_ptr, uint64_t seed, ssize_t size, intptr_t stream) {
    // state_ptr is a device ptr
    initialize_launcher launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state_ptr, seed, size);
}

template<typename T>
__device__ double rk_standard_exponential(T state) {
    /* We use -log(1-U) since U is [0, 1) */
    return -log(1.0 - state.rk_double());
}

template<typename T>
__device__ double rk_standard_normal(T state) {
    return state.rk_normal();
}

template<typename T>
__device__ float rk_standard_normal_float(T state) {
    return state.rk_normal_float();
}

template<typename T>
__device__ double rk_standard_gamma(T state, double shape) {
    double b, c;
    double U, V, X, Y;
    if (shape == 1.0) {
        return rk_standard_exponential(state);
    } else if (shape < 0.0) {
        return 0.0;
    } else if (shape < 1.0) {
        for (;;) {
            U = state.rk_double();
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
                X = state.rk_normal();
                V = 1.0 + c*X;
            } while (V <= 0.0);
            V = V*V*V;
            U = state.rk_double();
            if (U < 1.0 - 0.0331*(X*X)*(X*X)) return (b*V);
            if (log(U) < 0.5*X*X + b*(1. - V + log(V))) return (b*V);
        }
    }
}

template<typename T>
__device__ double rk_beta(T state, double a, double b) {
    double Ga, Gb;
    if ((a <= 1.0) && (b <= 1.0)) {
        double U, V, X, Y;
        /* Use Johnk's algorithm */
        while (1) {
            U = state.rk_double();
            V = state.rk_double();
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

template<typename T>
__device__ int64_t rk_geometric_search(T state, double p) {
    double U, sum, prod, q;
    int64_t X;
    X = 1;
    sum = prod = p;
    q = 1.0 - p;
    U = state.rk_double();
    while (U > sum) {
        prod *= q;
        sum += prod;
        X++;
    }
    return X;
}

template<typename T>
__device__ int64_t rk_geometric_inversion(T state, double p) {
    return ceil(log(1.0-state.rk_double())/log(1.0-p));
}

template<typename T>
__device__ int64_t rk_geometric(T state, double p) {
    if (p >= 0.333333333333333333333333) {
        return rk_geometric_search(state, p);
    } else {
        return rk_geometric_inversion(state, p);
    }
}

__device__ double loggam(double x) {
    double x0, x2, xp, gl, gl0;
    long k, n;
    double a[10] = {8.333333333333333e-02,-2.777777777777778e-03,
         7.936507936507937e-04,-5.952380952380952e-04,
         8.417508417508418e-04,-1.917526917526918e-03,
         6.410256410256410e-03,-2.955065359477124e-02,
         1.796443723688307e-01,-1.39243221690590e+00};
    x0 = x;
    n = 0;
    if ((x == 1.0) || (x == 2.0)) {
        return 0.0;
    } else if (x <= 7.0) {
        n = (long)(7 - x);
        x0 = x + n;
    }
    x2 = 1.0/(x0*x0);
    xp = 2*M_PI;
    gl0 = a[9];
    for (k=8; k>=0; k--) {
        gl0 *= x2;
        gl0 += a[k];
    }
    gl = gl0/x0 + 0.5*log(xp) + (x0-0.5)*log(x0) - x0;
    if (x <= 7.0) {
        for (k=1; k<=n; k++) {
            gl -= log(x0-1.0);
            x0 -= 1.0;
        }
    }
    return gl;
}

template<typename T>
__device__ int64_t rk_hypergeometric_hyp(T state, int64_t good, int64_t bad, int64_t sample) {
    int64_t d1, K, Z;
    double d2, U, Y;

    d1 = bad + good - sample;
    d2 = min(bad, good);

    Y = d2;
    K = sample;
    while (Y > 0.0)
    {
        U = state.rk_double();
        Y -= (int64_t)floor(U + Y/(d1 + K));
        K--;
        if (K == 0) break;
    }
    Z = (int64_t)(d2 - Y);
    if (good > bad) Z = sample - Z;
    return Z;
}


template<typename T>
__device__ int64_t rk_hypergeometric_hrua(T state, int64_t good, int64_t bad, int64_t sample) {

    int64_t mingoodbad, maxgoodbad, popsize, m, d9;
    int64_t Z;
    double d4, d5, d6, d7, d8, d10, d11, U, W, X, Y;
    double D1=1.7155277699214135, D2=0.8989161620588988;

    mingoodbad = min(good, bad);
    popsize = good + bad;
    maxgoodbad = max(good, bad);
    m = min(sample, popsize - sample);
    d4 = ((double)mingoodbad) / popsize;
    d5 = 1.0 - d4;
    d6 = m*d4 + 0.5;
    d7 = sqrt((double)(popsize - m) * sample * d4 * d5 / (popsize - 1) + 0.5);
    d8 = D1*d7 + D2;
    d9 = (int64_t)floor((double)(m + 1) * (mingoodbad + 1) / (popsize + 2));
    d10 = (loggam(d9+1) + loggam(mingoodbad-d9+1) + loggam(m-d9+1) +
	   loggam(maxgoodbad-m+d9+1));
    d11 = min(min(m, mingoodbad)+1.0, floor(d6+16*d7));
    /* 16 for 16-decimal-digit precision in D1 and D2 */

    while (1) {
        X = state.rk_double();
        Y = state.rk_double();
        W = d6 + d8*(Y- 0.5)/X;

        if ((W < 0.0) || (W >= d11)) continue;

        Z = (int64_t)floor(W);
        U = d10 - (loggam(Z+1) + loggam(mingoodbad-Z+1) + loggam(m-Z+1) +
                   loggam(maxgoodbad-m+Z+1));

        if ((X*(4.0-X)-3.0) <= U) break;

        if (X*(X-U) >= 1) continue;

        if (2.0*log(X) <= U) break;
    }

    if (good > bad) Z = m - Z;
    if (m < sample) Z = good - Z;

    return Z;
}

template<typename T>
__device__ int64_t rk_hypergeometric(T state, int64_t good, int64_t bad, int64_t sample) {
    if (sample > 10) {
        return rk_hypergeometric_hrua(state, good, bad, sample);
    }
    else {
        return rk_hypergeometric_hyp(state, good, bad, sample);
    }
}

template<typename T>
__device__ int64_t rk_logseries(T state, double p)
{
    double q, r, U, V;
    int64_t result;

    r = log(1.0 - p);

    while (1) {
        V = state.rk_double();
        if (V >= p) {
            return 1;
        }
        U = state.rk_double();
        q = 1.0 - exp(r*U);
        if (V <= q*q) {
            result = floor(1 + log(V)/log(q));
            if (result < 1) {
                continue;
            }
            else {
                return result;
            }
        }
        if (V >= q) {
            return 1;
        }
        return 2;
    }
}

template<typename T>
__device__ int64_t rk_poisson_mult(T state, double lam) {
    int64_t X;
    double prod, U, enlam;
    enlam = exp(-lam);
    X = 0;
    prod = 1.0;
    while (1) {
        U = state.rk_double();
        prod *= U;
        if (prod > enlam) {
            X += 1;
        } else {
            return X;
        }
    }
}

template<typename T>
__device__ int64_t rk_poisson_ptrs(T state, double lam) {
    int64_t k;
    double U, V, slam, loglam, a, b, invalpha, vr, us;
    slam = sqrt(lam);
    loglam = log(lam);
    b = 0.931 + 2.53*slam;
    a = -0.059 + 0.02483*b;
    invalpha = 1.1239 + 1.1328/(b-3.4);
    vr = 0.9277 - 3.6224/(b-2);
    while (1) {
        U = state.rk_double() - 0.5;
        V = state.rk_double();
        us = 0.5 - fabs(U);
        k = (int64_t)floor((2*a/us + b)*U + lam + 0.43);
        if ((us >= 0.07) && (V <= vr)) {
            return k;
        }
        if ((k < 0) ||
            ((us < 0.013) && (V > us))) {
            continue;
        }
        if ((log(V) + log(invalpha) - log(a/(us*us)+b)) <=
            (-lam + k*loglam - loggam(k+1))) {
            return k;
        }
    }
}

template<typename T>
__device__ int64_t rk_poisson(T state, double lam) {
    if (lam >= 10) {
        return rk_poisson_ptrs(state, lam);
    } else if (lam == 0) {
        return 0;
    } else {
        return rk_poisson_mult(state, lam);
    }
}

template<typename T>
__device__ uint32_t rk_raw(T state) {
    return state.rk_int();
}

template<typename T>
__device__ double rk_random_uniform(T state) {
    return state.rk_double();
}

template<typename T>
__device__ uint32_t rk_interval_32(T state, uint32_t mx, uint32_t mask) {
    uint32_t sampled = state.rk_int() & mask;
    while(sampled > mx)  {
        sampled = state.rk_int() & mask;
    }
    return sampled;
}

template<typename T>
__device__ uint64_t rk_interval_64(T state, uint64_t  mx, uint64_t mask) {
    uint32_t hi= state.rk_int();
    uint32_t lo= state.rk_int();
    uint64_t sampled = (static_cast<uint64_t>(hi) << 32 | lo)  & mask;
    while(sampled > mx)  {
        hi= state.rk_int();
        lo= state.rk_int();
        sampled = (static_cast<uint64_t>(hi) << 32 | lo) & mask;
    }
    return sampled;
}

template<typename T>
__device__ int64_t rk_binomial_btpe(T state, long n, double p, rk_binomial_state *binomial_state) {
    double r,q,fm,p1,xm,xl,xr,c,laml,lamr,p2,p3,p4;
    double a,u,v,s,F,rho,t,A,nrq,x1,x2,f1,f2,z,z2,w,w2,x;
    int m,y,k,i;

    if (!(binomial_state->initialized) ||
         (binomial_state->nsave != n) ||
         (binomial_state->psave != p)) {
        /* initialize */
        binomial_state->nsave = n;
        binomial_state->psave = p;
        binomial_state->initialized = 1;
        binomial_state->r = r = min(p, 1.0-p);
        binomial_state->q = q = 1.0 - r;
        binomial_state->fm = fm = n*r+r;
        binomial_state->m = m = (long)floor(binomial_state->fm);
        binomial_state->p1 = p1 = floor(2.195*sqrt(n*r*q)-4.6*q) + 0.5;
        binomial_state->xm = xm = m + 0.5;
        binomial_state->xl = xl = xm - p1;
        binomial_state->xr = xr = xm + p1;
        binomial_state->c = c = 0.134 + 20.5/(15.3 + m);
        a = (fm - xl)/(fm-xl*r);
        binomial_state->laml = laml = a*(1.0 + a/2.0);
        a = (xr - fm)/(xr*q);
        binomial_state->lamr = lamr = a*(1.0 + a/2.0);
        binomial_state->p2 = p2 = p1*(1.0 + 2.0*c);
        binomial_state->p3 = p3 = p2 + c/laml;
        binomial_state->p4 = p4 = p3 + c/lamr;
    } else {
        r = binomial_state->r;
        q = binomial_state->q;
        fm = binomial_state->fm;
        m = binomial_state->m;
        p1 = binomial_state->p1;
        xm = binomial_state->xm;
        xl = binomial_state->xl;
        xr = binomial_state->xr;
        c = binomial_state->c;
        laml = binomial_state->laml;
        lamr = binomial_state->lamr;
        p2 = binomial_state->p2;
        p3 = binomial_state->p3;
        p4 = binomial_state->p4;
    }
  /* sigh ... */
  Step10:
    nrq = n*r*q;
    u = state.rk_double()*p4;
    v = state.rk_double();
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

template<typename T>
__device__ int64_t rk_binomial_inversion(T state, int n, double p, rk_binomial_state *binomial_state) {
    double q, qn, np, px, U;
    int X, bound;

    if (!(binomial_state->initialized) ||
         (binomial_state->nsave != n) ||
         (binomial_state->psave != p)) {
        binomial_state->nsave = n;
        binomial_state->psave = p;
        binomial_state->initialized = 1;
        binomial_state->q = q = 1.0 - p;
        binomial_state->r = qn = exp(n * log(q));
        binomial_state->c = np = n*p;
        binomial_state->m = bound = min((double)n, np + 10.0*sqrt(np*q + 1));
    } else {
        q = binomial_state->q;
        qn = binomial_state->r;
        np = binomial_state->c;
        bound = binomial_state->m;
    }
    X = 0;
    px = qn;
    U = state.rk_double();
    while (U > px) {
        X++;
        if (X > bound) {
            X = 0;
            px = qn;
            U = state.rk_double();
        } else {
            U -= px;
            px  = ((n-X+1) * p * px)/(X*q);
        }
    }
    return X;
}

template<typename T>
__device__ int64_t rk_binomial(T state, int n, double p, rk_binomial_state *binomial_state) {
    double q;

    if (p <= 0.5) {
        if (p*n <= 30.0) {
            return rk_binomial_inversion(state, n, p, binomial_state);
        } else {
            return rk_binomial_btpe(state, n, p, binomial_state);
        }
    } else {
        q = 1.0-p;
        if (q*n <= 30.0) {
            return n - rk_binomial_inversion(state, n, q, binomial_state);
        } else {
            return n - rk_binomial_btpe(state, n, q, binomial_state);
        }
    }
}


struct raw_functor {
    template<typename... Args>
    __device__ uint32_t operator () (Args&&... args) {
        return rk_raw(args...);
    }
};


struct random_uniform_functor {
    template<typename... Args>
    __device__ double operator () (Args&&... args) {
        return rk_random_uniform(args...);
    }
};


struct interval_32_functor {
    template<typename... Args>
    __device__ uint32_t operator () (Args&&... args) {
        return rk_interval_32(args...);
    }
};

struct interval_64_functor {
    template<typename... Args>
    __device__ uint64_t operator () (Args&&... args) {
        return rk_interval_64(args...);
    }
};

struct beta_functor {
    template<typename... Args>
    __device__ double operator () (Args&&... args) {
        return rk_beta(args...);
    }
};

struct poisson_functor {
    template<typename... Args>
    __device__ int64_t operator () (Args&&... args) {
        return rk_poisson(args...);
    }
};

// There are several errors when trying to do this a full template
struct exponential_functor {
    template<typename... Args>
    __device__ double operator () (Args&&... args) {
        return rk_standard_exponential(args...);
    }
};

struct geometric_functor {
    template<typename... Args>
    __device__ int64_t operator () (Args&&... args) {
        return rk_geometric(args...);
    }
};

struct hypergeometric_functor {
    template<typename... Args>
    __device__ int64_t operator () (Args&&... args) {
        return rk_hypergeometric(args...);
    }
};

struct logseries_functor {
    template<typename... Args>
    __device__ int64_t operator () (Args&&... args) {
        return rk_logseries(args...);
    }
};

struct standard_normal_functor {
    template<typename... Args>
    __device__ double operator () (Args&&... args) {
        return rk_standard_normal(args...);
    }
};

struct standard_normal_float_functor {
    template<typename... Args>
    __device__ float operator () (Args&&... args) {
        return rk_standard_normal_float(args...);
    }
};

// There are several errors when trying to do this a full template
struct standard_gamma_functor {
    template<typename... Args>
    __device__ double operator () (Args&&... args) {
        return rk_standard_gamma(args...);
    }
};

struct binomial_functor {
    template<typename... Args>
    __device__ int64_t operator () (Args&&... args) {
        return rk_binomial(args...);
    }
};

// The following templates are used to unwrap arrays into an elementwise
// approach, the array is `_array_data` in `cupy/random/_generator_api.pyx`.
// When a pointer to `array_data<T>` is present in the variadic Args, it will
// be replaced by the value of pointer[thread_id]

template<typename T>
struct array_data {};  // opaque type always used as a pointer type

template<typename T>
__device__ T get_index(array_data<T> *value, int id) {
    int64_t* data = reinterpret_cast<int64_t*>(value);
    intptr_t ptr = reinterpret_cast<intptr_t>(data[0]);
    int ndim = data[1];
    ptrdiff_t offset = 0;
    for (int dim = ndim; --dim >= 0; ) {
        offset += data[ndim + dim + 2] * (id % data[dim + 2]);
        id /= data[dim + 2];
    }
    return *reinterpret_cast<T*>(ptr + offset);
}

template<typename T>
__device__ typename std::enable_if<std::is_arithmetic<T>::value, T>::type get_index(T value, int id) {
    return value;
}

__device__ rk_binomial_state* get_index(rk_binomial_state *value, int id) {
    return value + id;
}

template<typename F, typename T, typename R, typename... Args>
__global__ void execute_dist(intptr_t state, intptr_t out, ssize_t size, Args... args) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    R* out_ptr = reinterpret_cast<R*>(out);
    if (id < size) {
        T random(id, state);
        F func;
        // need to pass it by copy due to hip issues with templating
        out_ptr[id] = func(random, (get_index(args, id))...);
    }
    return;
}

template <typename F, typename R>
struct kernel_launcher {
    kernel_launcher(ssize_t size, cudaStream_t stream) : _size(size), _stream(stream) {
    }
    template<typename T, typename... Args>
    void operator()(Args&&... args) { 
        int tpb = 256;
        int bpg =  (_size + tpb - 1) / tpb;
        execute_dist<F, T, R><<<bpg, tpb, 0, _stream>>>(std::forward<Args>(args)...);
    }
    ssize_t _size;
    cudaStream_t _stream;
};

//These functions will take the generator_id as a parameter
void raw(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {
    kernel_launcher<raw_functor, int32_t> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size);
}

void random_uniform(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {
    kernel_launcher<random_uniform_functor, double> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size);
}

//These functions will take the generator_id as a parameter
void interval_32(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, int32_t mx, int32_t mask) {
    kernel_launcher<interval_32_functor, int32_t> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size, static_cast<uint32_t>(mx), static_cast<uint32_t>(mask));
}

void interval_64(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, int64_t mx, int64_t mask) {
    kernel_launcher<interval_64_functor, int64_t> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size, static_cast<uint64_t>(mx), static_cast<uint64_t>(mask));
}

void beta(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t a, intptr_t b) {
    kernel_launcher<beta_functor, double> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size, reinterpret_cast<array_data<double>*>(a), reinterpret_cast<array_data<double>*>(b));
}

void exponential(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {
    kernel_launcher<exponential_functor, double> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size);
}

void geometric(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t p) {
    kernel_launcher<geometric_functor, int64_t> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size, reinterpret_cast<array_data<double>*>(p));
}

void hypergeometric(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t ngood, intptr_t nbad, intptr_t nsample) {
    kernel_launcher<hypergeometric_functor, int64_t> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size, reinterpret_cast<array_data<int64_t>*>(ngood), reinterpret_cast<array_data<int64_t>*>(nbad), reinterpret_cast<array_data<int64_t>*>(nsample));
}

void logseries(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t p) {
    kernel_launcher<logseries_functor, int64_t> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size, reinterpret_cast<array_data<double>*>(p));
}

void poisson(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t lam) {
    kernel_launcher<poisson_functor, int64_t> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size, reinterpret_cast<array_data<double>*>(lam));
}

void standard_normal(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {
    kernel_launcher<standard_normal_functor, double> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size);
}

void standard_normal_float(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {
    kernel_launcher<standard_normal_float_functor, float> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size);
}

void standard_gamma(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t shape) {
    kernel_launcher<standard_gamma_functor, double> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size, reinterpret_cast<array_data<double>*>(shape));
}

void binomial(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t n, intptr_t p, intptr_t binomial_state) {
    kernel_launcher<binomial_functor, int64_t> launcher(size, reinterpret_cast<cudaStream_t>(stream));
    generator_dispatcher(generator, launcher, state, out, size, reinterpret_cast<array_data<int>*>(n), reinterpret_cast<array_data<double>*>(p), reinterpret_cast<rk_binomial_state*>(binomial_state));
}

#else
// the stubs need to be redeclared here for HIP versions less than 4.3 to avoid redeclarations in cython when importing the headers
// No cuda will not compile the .cu file, so the definition needs to be done here explicitly
void init_curand_generator(int generator, intptr_t state_ptr, uint64_t seed, ssize_t size, intptr_t stream) {}
void random_uniform(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {}
void raw(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {}
void interval_32(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, int32_t mx, int32_t mask) {}
void interval_64(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, int64_t mx, int64_t mask) {}
void beta(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t a, intptr_t b) {}
void exponential(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {}
void geometric(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t p) {}
void hypergeometric(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t ngood, intptr_t nbad, intptr_t nsample) {}
void logseries(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t p) {}
void poisson(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t lam) {}
void standard_normal(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {}
void standard_normal_float(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream) {}
void standard_gamma(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t shape) {}
void binomial(int generator, intptr_t state, intptr_t out, ssize_t size, intptr_t stream, intptr_t n, intptr_t p, intptr_t binomial_state) {}

#endif
