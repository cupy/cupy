"""
The source code here is an adaptation with minimal changes from the following
files in SciPy's bundled Cephes library:

https://github.com/scipy/scipy/blob/master/scipy/special/cephes/const.c
https://github.com/scipy/scipy/blob/master/scipy/special/cephes/igam.c
https://github.com/scipy/scipy/blob/master/scipy/special/cephes/igam.h
https://github.com/scipy/scipy/blob/master/scipy/special/cephes/igami.c
https://github.com/scipy/scipy/blob/master/scipy/special/cephes/lanczos.c
https://github.com/scipy/scipy/blob/master/scipy/special/cephes/lanczos.h
https://github.com/scipy/scipy/blob/master/scipy/special/cephes/mconf.h
https://github.com/scipy/scipy/blob/master/scipy/special/cephes/polevl.h
https://github.com/scipy/scipy/blob/master/scipy/special/cephes/unity.c

Cephes Math Library Release 2.0:  April, 1987
Copyright 1984, 1987 by Stephen L. Moshier
Direct inquiries to 30 Frost Street, Cambridge, MA 02140
"""

from cupy import _core

from cupyx.scipy.special._digamma import polevl_definition
from cupyx.scipy.special._gamma import gamma_definition
from cupyx.scipy.special._zeta import zeta_definition

_zeta_c = zeta_definition
# comment out duplicate MACHEP definition
_zeta_c = _zeta_c.replace(
    "__constant__ double MACHEP", "// __constant__ double MACHEP"
)


# TODO: finish

# See the following on error states:
#    docs.scipy.org/doc/scipy/reference/generated/scipy.special.errstate.html
#    scipy/special/sf_error.h

# TODO: which value for MACHEP, MAXLOG from const.c to use?

_misc_preamble = """

// include for CUDART_NAN, CUDART_INF
#include <cupy/math_constants.h>

// defines from /scipy/special/cephes/const.c
__constant__ double MACHEP = 1.11022302462515654042E-16;
__constant__ double MAXLOG = 7.09782712893383996732E2;

// defines from numpy/core/include/numpy/npy_math.h
#define NPY_EULER     0.577215664901532860606512090082402431  /* Euler constant */
#define NPY_PI        3.141592653589793238462643383279502884  /* pi */
#define NPY_SQRT2     1.414213562373095048801688724209698079  /* sqrt(2) */
#define NPY_SQRT1_2   0.707106781186547524400844362104849039  /* 1/sqrt(2) */

// from /scipy/special/cephes/mconf.h
#ifdef MAXITER
#undef MAXITER
#endif

#define MAXITER 2000
#define IGAM 1
#define IGAMC 0
#define SMALL 20
#define LARGE 200
#define SMALLRATIO 0.3
#define LARGERATIO 4.5

__constant__ double big = 4.503599627370496e15;
__constant__ double biginv = 2.22044604925031308085e-16;

"""


# Note: p1evl -> followed the template style used for polevl in digamma.py
p1evl_definition = """
template<int N> static __device__ double p1evl(double x, const double coef[])
{
    double ans;
    const double *p;
    int i;

    p = coef;
    ans = x + *p++;

    for (i = 0; i < N - 1; ++i){
        ans = ans * x + *p++;
    }

    return ans;
}
"""

lgam_definition = """
// define lgam, using CUDA's lgamma as done for cupyx.scipy.special.gammaln
__device__ double lgam(double x)
{
    if (isinf(x) && x < 0) {
        return -1.0 / 0.0;
    } else {
        return lgamma(x);
    }
}
"""

# lgam1p from unity.c
_unity_c_partial = (
    _zeta_c
    + lgam_definition
    + polevl_definition
    + p1evl_definition
    + """
// part of cephes/unity.c


/* log(1 + x) - x */
#ifdef __HIP_DEVICE_COMPILE__
// For some reason ROCm 4.3 crashes with the noinline in this function
__device__ double log1pmx(double x)
#else
__noinline__ __device__ double log1pmx(double x)
#endif
{
    if (fabs(x) < 0.5) {
        int n;
        double xfac = x;
        double term;
        double res = 0;

        for(n = 2; n < MAXITER; n++) {
            xfac *= -x;
            term = xfac / n;
            res += term;
            if (fabs(term) < MACHEP * fabs(res)) {
                break;
            }
        }
        return res;
    }
    else {
        return log1p(x) - x;
    }
}


/* Compute lgam(x + 1) around x = 0 using its Taylor series. */
__noinline__ __device__ double lgam1p_taylor(double x)
{
    int n;
    double xfac, coeff, res;

    if (x == 0) {
        return 0;
    }
    res = -NPY_EULER * x;
    xfac = -x;
    for (n = 2; n < 42; n++) {
        xfac *= -x;
        coeff = zeta(n, 1) * xfac / n;
        res += coeff;
        if (fabs(coeff) < MACHEP * fabs(res)) {
                break;
        }
    }

    return res;
}


/* Compute lgam(x + 1). */
__noinline__ __device__ double lgam1p(double x)
{
    if (fabs(x) <= 0.5) {
        return lgam1p_taylor(x);
    } else if (fabs(x - 1) < 0.5) {
        return log(x) + lgam1p_taylor(x - 1);
    } else {
        return lgam(x + 1);
    }
}


"""
)

ratevl_definition = """
// from scipy/special/cephes/polevl.h

/* Evaluate a rational function. See [1]. */

static __device__ double ratevl(double x, const double num[], int M,
                                const double denom[], int N)
{
    int i, dir;
    double y, num_ans, denom_ans;
    double absx = fabs(x);
    const double *p;

    if (absx > 1) {
        /* Evaluate as a polynomial in 1/x. */
        dir = -1;
        p = num + M;
        y = 1 / x;
    } else {
        dir = 1;
        p = num;
        y = x;
    }

    /* Evaluate the numerator */
    num_ans = *p;
    p += dir;
    for (i = 1; i <= M; i++) {
        num_ans = num_ans * y + *p;
        p += dir;
    }

    /* Evaluate the denominator */
    if (absx > 1) {
        p = denom + N;
    } else {
        p = denom;
    }

    denom_ans = *p;
    p += dir;
    for (i = 1; i <= N; i++) {
        denom_ans = denom_ans * y + *p;
        p += dir;
    }

    if (absx > 1) {
        i = N - M;
        return pow(x, (double)i) * num_ans / denom_ans;
    } else {
        return num_ans / denom_ans;
    }
}
"""

_lanczos_h = """
// from scipy/special/cephes/lanczos.h

__constant__ double lanczos_num[13] = {
    2.506628274631000270164908177133837338626,
    210.8242777515793458725097339207133627117,
    8071.672002365816210638002902272250613822,
    186056.2653952234950402949897160456992822,
    2876370.628935372441225409051620849613599,
    31426415.58540019438061423162831820536287,
    248874557.8620541565114603864132294232163,
    1439720407.311721673663223072794912393972,
    6039542586.35202800506429164430729792107,
    17921034426.03720969991975575445893111267,
    35711959237.35566804944018545154716670596,
    42919803642.64909876895789904700198885093,
    23531376880.41075968857200767445163675473
};

__constant__ double lanczos_denom[13] = {
    1,
    66,
    1925,
    32670,
    357423,
    2637558,
    13339535,
    45995730,
    105258076,
    150917976,
    120543840,
    39916800,
    0
};

__constant__ double lanczos_sum_expg_scaled_num[13] = {
    0.006061842346248906525783753964555936883222,
    0.5098416655656676188125178644804694509993,
    19.51992788247617482847860966235652136208,
    449.9445569063168119446858607650988409623,
    6955.999602515376140356310115515198987526,
    75999.29304014542649875303443598909137092,
    601859.6171681098786670226533699352302507,
    3481712.15498064590882071018964774556468,
    14605578.08768506808414169982791359218571,
    43338889.32467613834773723740590533316085,
    86363131.28813859145546927288977868422342,
    103794043.1163445451906271053616070238554,
    56906521.91347156388090791033559122686859
};

__constant__ double lanczos_sum_expg_scaled_denom[13] = {
    1,
    66,
    1925,
    32670,
    357423,
    2637558,
    13339535,
    45995730,
    105258076,
    150917976,
    120543840,
    39916800,
    0
};

__constant__ double lanczos_sum_near_1_d[12] = {
    0.3394643171893132535170101292240837927725e-9,
    -0.2499505151487868335680273909354071938387e-8,
    0.8690926181038057039526127422002498960172e-8,
    -0.1933117898880828348692541394841204288047e-7,
    0.3075580174791348492737947340039992829546e-7,
    -0.2752907702903126466004207345038327818713e-7,
    -0.1515973019871092388943437623825208095123e-5,
    0.004785200610085071473880915854204301886437,
    -0.1993758927614728757314233026257810172008,
    1.483082862367253753040442933770164111678,
    -3.327150580651624233553677113928873034916,
    2.208709979316623790862569924861841433016
};

__constant__ double lanczos_sum_near_2_d[12] = {
    0.1009141566987569892221439918230042368112e-8,
    -0.7430396708998719707642735577238449585822e-8,
    0.2583592566524439230844378948704262291927e-7,
    -0.5746670642147041587497159649318454348117e-7,
    0.9142922068165324132060550591210267992072e-7,
    -0.8183698410724358930823737982119474130069e-7,
    -0.4506604409707170077136555010018549819192e-5,
    0.01422519127192419234315002746252160965831,
    -0.5926941084905061794445733628891024027949,
    4.408830289125943377923077727900630927902,
    -9.8907772644920670589288081640128194231,
    6.565936202082889535528455955485877361223
};

__constant__ double lanczos_g = 6.024680040776729583740234375;
"""

_lanczos_preamble = (
    ratevl_definition
    + _lanczos_h
    + """

// from scipy/special/cephes/lanczos.c

__device__ double lanczos_sum(double x)
{
    return ratevl(x, lanczos_num,
          sizeof(lanczos_num) / sizeof(lanczos_num[0]) - 1,
          lanczos_denom,
          sizeof(lanczos_denom) / sizeof(lanczos_denom[0]) - 1);
}


__device__ double lanczos_sum_expg_scaled(double x)
{
    return ratevl(x, lanczos_sum_expg_scaled_num,
          sizeof(lanczos_sum_expg_scaled_num)
              / sizeof(lanczos_sum_expg_scaled_num[0]) - 1,
          lanczos_sum_expg_scaled_denom,
          sizeof(lanczos_sum_expg_scaled_denom)
              / sizeof(lanczos_sum_expg_scaled_denom[0]) - 1);
}


__device__ double lanczos_sum_near_1(double dx)
{
    double result = 0;
    unsigned k;

    for (k = 1;
         k <= sizeof(lanczos_sum_near_1_d) / sizeof(lanczos_sum_near_1_d[0]);
         ++k) {
    result += (-lanczos_sum_near_1_d[k-1]*dx)/(k*dx + k*k);
    }
    return result;
}


__device__ double lanczos_sum_near_2(double dx)
{
    double result = 0;
    double x = dx + 2;
    unsigned k;

    for(k = 1;
        k <= sizeof(lanczos_sum_near_2_d) / sizeof(lanczos_sum_near_2_d[0]);
        ++k) {
    result += (-lanczos_sum_near_2_d[k-1]*dx)/(x + k*x + k*k - 1);
    }
    return result;
}

"""
)

# add predefined array from igam.h
_igam_h = """

// from scipy/special/cephes/igam.h

#define d_K 25
#define d_N 25

__device__ static const double d[d_K][d_N] =
{{-3.3333333333333333e-1, 8.3333333333333333e-2, -1.4814814814814815e-2, 1.1574074074074074e-3, 3.527336860670194e-4, -1.7875514403292181e-4, 3.9192631785224378e-5, -2.1854485106799922e-6, -1.85406221071516e-6, 8.296711340953086e-7, -1.7665952736826079e-7, 6.7078535434014986e-9, 1.0261809784240308e-8, -4.3820360184533532e-9, 9.1476995822367902e-10, -2.551419399494625e-11, -5.8307721325504251e-11, 2.4361948020667416e-11, -5.0276692801141756e-12, 1.1004392031956135e-13, 3.3717632624009854e-13, -1.3923887224181621e-13, 2.8534893807047443e-14, -5.1391118342425726e-16, -1.9752288294349443e-15},
{-1.8518518518518519e-3, -3.4722222222222222e-3, 2.6455026455026455e-3, -9.9022633744855967e-4, 2.0576131687242798e-4, -4.0187757201646091e-7, -1.8098550334489978e-5, 7.6491609160811101e-6, -1.6120900894563446e-6, 4.6471278028074343e-9, 1.378633446915721e-7, -5.752545603517705e-8, 1.1951628599778147e-8, -1.7543241719747648e-11, -1.0091543710600413e-9, 4.1627929918425826e-10, -8.5639070264929806e-11, 6.0672151016047586e-14, 7.1624989648114854e-12, -2.9331866437714371e-12, 5.9966963656836887e-13, -2.1671786527323314e-16, -4.9783399723692616e-14, 2.0291628823713425e-14, -4.13125571381061e-15},
{4.1335978835978836e-3, -2.6813271604938272e-3, 7.7160493827160494e-4, 2.0093878600823045e-6, -1.0736653226365161e-4, 5.2923448829120125e-5, -1.2760635188618728e-5, 3.4235787340961381e-8, 1.3721957309062933e-6, -6.298992138380055e-7, 1.4280614206064242e-7, -2.0477098421990866e-10, -1.4092529910867521e-8, 6.228974084922022e-9, -1.3670488396617113e-9, 9.4283561590146782e-13, 1.2872252400089318e-10, -5.5645956134363321e-11, 1.1975935546366981e-11, -4.1689782251838635e-15, -1.0940640427884594e-12, 4.6622399463901357e-13, -9.905105763906906e-14, 1.8931876768373515e-17, 8.8592218725911273e-15},
{6.4943415637860082e-4, 2.2947209362139918e-4, -4.6918949439525571e-4, 2.6772063206283885e-4, -7.5618016718839764e-5, -2.3965051138672967e-7, 1.1082654115347302e-5, -5.6749528269915966e-6, 1.4230900732435884e-6, -2.7861080291528142e-11, -1.6958404091930277e-7, 8.0994649053880824e-8, -1.9111168485973654e-8, 2.3928620439808118e-12, 2.0620131815488798e-9, -9.4604966618551322e-10, 2.1541049775774908e-10, -1.388823336813903e-14, -2.1894761681963939e-11, 9.7909989511716851e-12, -2.1782191880180962e-12, 6.2088195734079014e-17, 2.126978363279737e-13, -9.3446887915174333e-14, 2.0453671226782849e-14},
{-8.618882909167117e-4, 7.8403922172006663e-4, -2.9907248030319018e-4, -1.4638452578843418e-6, 6.6414982154651222e-5, -3.9683650471794347e-5, 1.1375726970678419e-5, 2.5074972262375328e-10, -1.6954149536558306e-6, 8.9075075322053097e-7, -2.2929348340008049e-7, 2.956794137544049e-11, 2.8865829742708784e-8, -1.4189739437803219e-8, 3.4463580499464897e-9, -2.3024517174528067e-13, -3.9409233028046405e-10, 1.8602338968504502e-10, -4.356323005056618e-11, 1.2786001016296231e-15, 4.6792750266579195e-12, -2.1492464706134829e-12, 4.9088156148096522e-13, -6.3385914848915603e-18, -5.0453320690800944e-14},
{-3.3679855336635815e-4, -6.9728137583658578e-5, 2.7727532449593921e-4, -1.9932570516188848e-4, 6.7977804779372078e-5, 1.419062920643967e-7, -1.3594048189768693e-5, 8.0184702563342015e-6, -2.2914811765080952e-6, -3.252473551298454e-10, 3.4652846491085265e-7, -1.8447187191171343e-7, 4.8240967037894181e-8, -1.7989466721743515e-14, -6.3061945000135234e-9, 3.1624176287745679e-9, -7.8409242536974293e-10, 5.1926791652540407e-15, 9.3589442423067836e-11, -4.5134262161632782e-11, 1.0799129993116827e-11, -3.661886712685252e-17, -1.210902069055155e-12, 5.6807435849905643e-13, -1.3249659916340829e-13},
{5.3130793646399222e-4, -5.9216643735369388e-4, 2.7087820967180448e-4, 7.9023532326603279e-7, -8.1539693675619688e-5, 5.6116827531062497e-5, -1.8329116582843376e-5, -3.0796134506033048e-9, 3.4651553688036091e-6, -2.0291327396058604e-6, 5.7887928631490037e-7, 2.338630673826657e-13, -8.8286007463304835e-8, 4.7435958880408128e-8, -1.2545415020710382e-8, 8.6496488580102925e-14, 1.6846058979264063e-9, -8.5754928235775947e-10, 2.1598224929232125e-10, -7.6132305204761539e-16, -2.6639822008536144e-11, 1.3065700536611057e-11, -3.1799163902367977e-12, 4.7109761213674315e-18, 3.6902800842763467e-13},
{3.4436760689237767e-4, 5.1717909082605922e-5, -3.3493161081142236e-4, 2.812695154763237e-4, -1.0976582244684731e-4, -1.2741009095484485e-7, 2.7744451511563644e-5, -1.8263488805711333e-5, 5.7876949497350524e-6, 4.9387589339362704e-10, -1.0595367014026043e-6, 6.1667143761104075e-7, -1.7562973359060462e-7, -1.2974473287015439e-12, 2.695423606288966e-8, -1.4578352908731271e-8, 3.887645959386175e-9, -3.8810022510194121e-17, -5.3279941738772867e-10, 2.7437977643314845e-10, -6.9957960920705679e-11, 2.5899863874868481e-17, 8.8566890996696381e-12, -4.403168815871311e-12, 1.0865561947091654e-12},
{-6.5262391859530942e-4, 8.3949872067208728e-4, -4.3829709854172101e-4, -6.969091458420552e-7, 1.6644846642067548e-4, -1.2783517679769219e-4, 4.6299532636913043e-5, 4.5579098679227077e-9, -1.0595271125805195e-5, 6.7833429048651666e-6, -2.1075476666258804e-6, -1.7213731432817145e-11, 3.7735877416110979e-7, -2.1867506700122867e-7, 6.2202288040189269e-8, 6.5977038267330006e-16, -9.5903864974256858e-9, 5.2132144922808078e-9, -1.3991589583935709e-9, 5.382058999060575e-16, 1.9484714275467745e-10, -1.0127287556389682e-10, 2.6077347197254926e-11, -5.0904186999932993e-18, -3.3721464474854592e-12},
{-5.9676129019274625e-4, -7.2048954160200106e-5, 6.7823088376673284e-4, -6.4014752602627585e-4, 2.7750107634328704e-4, 1.8197008380465151e-7, -8.4795071170685032e-5, 6.105192082501531e-5, -2.1073920183404862e-5, -8.8585890141255994e-10, 4.5284535953805377e-6, -2.8427815022504408e-6, 8.7082341778646412e-7, 3.6886101871706965e-12, -1.5344695190702061e-7, 8.862466778790695e-8, -2.5184812301826817e-8, -1.0225912098215092e-14, 3.8969470758154777e-9, -2.1267304792235635e-9, 5.7370135528051385e-10, -1.887749850169741e-19, -8.0931538694657866e-11, 4.2382723283449199e-11, -1.1002224534207726e-11},
{1.3324454494800656e-3, -1.9144384985654775e-3, 1.1089369134596637e-3, 9.932404122642299e-7, -5.0874501293093199e-4, 4.2735056665392884e-4, -1.6858853767910799e-4, -8.1301893922784998e-9, 4.5284402370562147e-5, -3.127053674781734e-5, 1.044986828530338e-5, 4.8435226265680926e-11, -2.1482565873456258e-6, 1.329369701097492e-6, -4.0295693092101029e-7, -1.7567877666323291e-13, 7.0145043163668257e-8, -4.040787734999483e-8, 1.1474026743371963e-8, 3.9642746853563325e-18, -1.7804938269892714e-9, 9.7480262548731646e-10, -2.6405338676507616e-10, 5.794875163403742e-18, 3.7647749553543836e-11},
{1.579727660730835e-3, 1.6251626278391582e-4, -2.0633421035543276e-3, 2.1389686185689098e-3, -1.0108559391263003e-3, -3.9912705529919201e-7, 3.6235025084764691e-4, -2.8143901463712154e-4, 1.0449513336495887e-4, 2.1211418491830297e-9, -2.5779417251947842e-5, 1.7281818956040463e-5, -5.6413773872904282e-6, -1.1024320105776174e-11, 1.1223224418895175e-6, -6.8693396379526735e-7, 2.0653236975414887e-7, 4.6714772409838506e-14, -3.5609886164949055e-8, 2.0470855345905963e-8, -5.8091738633283358e-9, -1.332821287582869e-16, 9.0354604391335133e-10, -4.9598782517330834e-10, 1.3481607129399749e-10},
{-4.0725121195140166e-3, 6.4033628338080698e-3, -4.0410161081676618e-3, -2.183732802866233e-6, 2.1740441801254639e-3, -1.9700440518418892e-3, 8.3595469747962458e-4, 1.9445447567109655e-8, -2.5779387120421696e-4, 1.9009987368139304e-4, -6.7696499937438965e-5, -1.4440629666426572e-10, 1.5712512518742269e-5, -1.0304008744776893e-5, 3.304517767401387e-6, 7.9829760242325709e-13, -6.4097794149313004e-7, 3.8894624761300056e-7, -1.1618347644948869e-7, -2.816808630596451e-15, 1.9878012911297093e-8, -1.1407719956357511e-8, 3.2355857064185555e-9, 4.1759468293455945e-20, -5.0423112718105824e-10},
{-5.9475779383993003e-3, -5.4016476789260452e-4, 8.7910413550767898e-3, -9.8576315587856125e-3, 5.0134695031021538e-3, 1.2807521786221875e-6, -2.0626019342754683e-3, 1.7109128573523058e-3, -6.7695312714133799e-4, -6.9011545676562133e-9, 1.8855128143995902e-4, -1.3395215663491969e-4, 4.6263183033528039e-5, 4.0034230613321351e-11, -1.0255652921494033e-5, 6.612086372797651e-6, -2.0913022027253008e-6, -2.0951775649603837e-13, 3.9756029041993247e-7, -2.3956211978815887e-7, 7.1182883382145864e-8, 8.925574873053455e-16, -1.2101547235064676e-8, 6.9350618248334386e-9, -1.9661464453856102e-9},
{1.7402027787522711e-2, -2.9527880945699121e-2, 2.0045875571402799e-2, 7.0289515966903407e-6, -1.2375421071343148e-2, 1.1976293444235254e-2, -5.4156038466518525e-3, -6.3290893396418616e-8, 1.8855118129005065e-3, -1.473473274825001e-3, 5.5515810097708387e-4, 5.2406834412550662e-10, -1.4357913535784836e-4, 9.9181293224943297e-5, -3.3460834749478311e-5, -3.5755837291098993e-12, 7.1560851960630076e-6, -4.5516802628155526e-6, 1.4236576649271475e-6, 1.8803149082089664e-14, -2.6623403898929211e-7, 1.5950642189595716e-7, -4.7187514673841102e-8, -6.5107872958755177e-17, 7.9795091026746235e-9},
{3.0249124160905891e-2, 2.4817436002649977e-3, -4.9939134373457022e-2, 5.9915643009307869e-2, -3.2483207601623391e-2, -5.7212968652103441e-6, 1.5085251778569354e-2, -1.3261324005088445e-2, 5.5515262632426148e-3, 3.0263182257030016e-8, -1.7229548406756723e-3, 1.2893570099929637e-3, -4.6845138348319876e-4, -1.830259937893045e-10, 1.1449739014822654e-4, -7.7378565221244477e-5, 2.5625836246985201e-5, 1.0766165333192814e-12, -5.3246809282422621e-6, 3.349634863064464e-6, -1.0381253128684018e-6, -5.608909920621128e-15, 1.9150821930676591e-7, -1.1418365800203486e-7, 3.3654425209171788e-8},
{-9.9051020880159045e-2, 1.7954011706123486e-1, -1.2989606383463778e-1, -3.1478872752284357e-5, 9.0510635276848131e-2, -9.2828824411184397e-2, 4.4412112839877808e-2, 2.7779236316835888e-7, -1.7229543805449697e-2, 1.4182925050891573e-2, -5.6214161633747336e-3, -2.39598509186381e-9, 1.6029634366079908e-3, -1.1606784674435773e-3, 4.1001337768153873e-4, 1.8365800754090661e-11, -9.5844256563655903e-5, 6.3643062337764708e-5, -2.076250624489065e-5, -1.1806020912804483e-13, 4.2131808239120649e-6, -2.6262241337012467e-6, 8.0770620494930662e-7, 6.0125912123632725e-16, -1.4729737374018841e-7},
{-1.9994542198219728e-1, -1.5056113040026424e-2, 3.6470239469348489e-1, -4.6435192311733545e-1, 2.6640934719197893e-1, 3.4038266027147191e-5, -1.3784338709329624e-1, 1.276467178337056e-1, -5.6213828755200985e-2, -1.753150885483011e-7, 1.9235592956768113e-2, -1.5088821281095315e-2, 5.7401854451350123e-3, 1.0622382710310225e-9, -1.5335082692563998e-3, 1.0819320643228214e-3, -3.7372510193945659e-4, -6.6170909729031985e-12, 8.4263617380909628e-5, -5.5150706827483479e-5, 1.7769536448348069e-5, 3.8827923210205533e-14, -3.53513697488768e-6, 2.1865832130045269e-6, -6.6812849447625594e-7},
{7.2438608504029431e-1, -1.3918010932653375, 1.0654143352413968, 1.876173868950258e-4, -8.2705501176152696e-1, 8.9352433347828414e-1, -4.4971003995291339e-1, -1.6107401567546652e-6, 1.9235590165271091e-1, -1.6597702160042609e-1, 6.8882222681814333e-2, 1.3910091724608687e-8, -2.146911561508663e-2, 1.6228980898865892e-2, -5.9796016172584256e-3, -1.1287469112826745e-10, 1.5167451119784857e-3, -1.0478634293553899e-3, 3.5539072889126421e-4, 8.1704322111801517e-13, -7.7773013442452395e-5, 5.0291413897007722e-5, -1.6035083867000518e-5, 1.2469354315487605e-14, 3.1369106244517615e-6},
{1.6668949727276811, 1.165462765994632e-1, -3.3288393225018906, 4.4692325482864037, -2.6977693045875807, -2.600667859891061e-4, 1.5389017615694539, -1.4937962361134612, 6.8881964633233148e-1, 1.3077482004552385e-6, -2.5762963325596288e-1, 2.1097676102125449e-1, -8.3714408359219882e-2, -7.7920428881354753e-9, 2.4267923064833599e-2, -1.7813678334552311e-2, 6.3970330388900056e-3, 4.9430807090480523e-11, -1.5554602758465635e-3, 1.0561196919903214e-3, -3.5277184460472902e-4, 9.3002334645022459e-14, 7.5285855026557172e-5, -4.8186515569156351e-5, 1.5227271505597605e-5},
{-6.6188298861372935, 1.3397985455142589e+1, -1.0789350606845146e+1, -1.4352254537875018e-3, 9.2333694596189809, -1.0456552819547769e+1, 5.5105526029033471, 1.2024439690716742e-5, -2.5762961164755816, 2.3207442745387179, -1.0045728797216284, -1.0207833290021914e-7, 3.3975092171169466e-1, -2.6720517450757468e-1, 1.0235252851562706e-1, 8.4329730484871625e-10, -2.7998284958442595e-2, 2.0066274144976813e-2, -7.0554368915086242e-3, 1.9402238183698188e-12, 1.6562888105449611e-3, -1.1082898580743683e-3, 3.654545161310169e-4, -5.1290032026971794e-11, -7.6340103696869031e-5},
{-1.7112706061976095e+1, -1.1208044642899116, 3.7131966511885444e+1, -5.2298271025348962e+1, 3.3058589696624618e+1, 2.4791298976200222e-3, -2.061089403411526e+1, 2.088672775145582e+1, -1.0045703956517752e+1, -1.2238783449063012e-5, 4.0770134274221141, -3.473667358470195, 1.4329352617312006, 7.1359914411879712e-8, -4.4797257159115612e-1, 3.4112666080644461e-1, -1.2699786326594923e-1, -2.8953677269081528e-10, 3.3125776278259863e-2, -2.3274087021036101e-2, 8.0399993503648882e-3, -1.177805216235265e-9, -1.8321624891071668e-3, 1.2108282933588665e-3, -3.9479941246822517e-4},
{7.389033153567425e+1, -1.5680141270402273e+2, 1.322177542759164e+2, 1.3692876877324546e-2, -1.2366496885920151e+2, 1.4620689391062729e+2, -8.0365587724865346e+1, -1.1259851148881298e-4, 4.0770132196179938e+1, -3.8210340013273034e+1, 1.719522294277362e+1, 9.3519707955168356e-7, -6.2716159907747034, 5.1168999071852637, -2.0319658112299095, -4.9507215582761543e-9, 5.9626397294332597e-1, -4.4220765337238094e-1, 1.6079998700166273e-1, -2.4733786203223402e-8, -4.0307574759979762e-2, 2.7849050747097869e-2, -9.4751858992054221e-3, 6.419922235909132e-6, 2.1250180774699461e-3},
{2.1216837098382522e+2, 1.3107863022633868e+1, -4.9698285932871748e+2, 7.3121595266969204e+2, -4.8213821720890847e+2, -2.8817248692894889e-2, 3.2616720302947102e+2, -3.4389340280087117e+2, 1.7195193870816232e+2, 1.4038077378096158e-4, -7.52594195897599e+1, 6.651969984520934e+1, -2.8447519748152462e+1, -7.613702615875391e-7, 9.5402237105304373, -7.5175301113311376, 2.8943997568871961, -4.6612194999538201e-7, -8.0615149598794088e-1, 5.8483006570631029e-1, -2.0845408972964956e-1, 1.4765818959305817e-4, 5.1000433863753019e-2, -3.3066252141883665e-2, 1.5109265210467774e-2},
{-9.8959643098322368e+2, 2.1925555360905233e+3, -1.9283586782723356e+3, -1.5925738122215253e-1, 1.9569985945919857e+3, -2.4072514765081556e+3, 1.3756149959336496e+3, 1.2920735237496668e-3, -7.525941715948055e+2, 7.3171668742208716e+2, -3.4137023466220065e+2, -9.9857390260608043e-6, 1.3356313181291573e+2, -1.1276295161252794e+2, 4.6310396098204458e+1, -7.9237387133614756e-6, -1.4510726927018646e+1, 1.1111771248100563e+1, -4.1690817945270892, 3.1008219800117808e-3, 1.1220095449981468, -7.6052379926149916e-1, 3.6262236505085254e-1, 2.216867741940747e-1, 4.8683443692930507e-1}};

"""

# add functions from igam.c
_igam_preamble = (
    _misc_preamble
    + _unity_c_partial
    + _lanczos_preamble
    + _igam_h
    + """

// from scipy/special/cephes/igam.c

/* Compute igam/igamc using DLMF 8.12.3/8.12.4. */
__noinline__ __device__ double  asymptotic_series(double a, double x, int func)
{
    int k, n, sgn;
    int maxpow = 0;
    double lambda = x / a;
    double sigma = (x - a) / a;
    double eta, res, ck, ckterm, term, absterm;
    double absoldterm = CUDART_INF;
    double etapow[d_N] = {1};
    double sum = 0;
    double afac = 1;

    if (func == IGAM) {
        sgn = -1;
    } else {
        sgn = 1;
    }

    if (lambda > 1) {
        eta = sqrt(-2 * log1pmx(sigma));
    } else if (lambda < 1) {
        eta = -sqrt(-2 * log1pmx(sigma));
    } else {
        eta = 0;
    }
    res = 0.5 * erfc(sgn * eta * sqrt(a / 2));

    for (k = 0; k < d_K; k++) {
        ck = d[k][0];
        for (n = 1; n < d_N; n++) {
            if (n > maxpow) {
                etapow[n] = eta * etapow[n-1];
                maxpow += 1;
            }
            ckterm = d[k][n]*etapow[n];
            ck += ckterm;
            if (fabs(ckterm) < MACHEP * fabs(ck)) {
                break;
            }
        }
        term = ck * afac;
        absterm = fabs(term);
        if (absterm > absoldterm) {
            break;
        }
        sum += term;
        if (absterm < MACHEP * fabs(sum)) {
            break;
        }
        absoldterm = absterm;
        afac /= a;
    }
    res += sgn * exp(-0.5 * a * eta * eta) * sum / sqrt(2 * NPY_PI * a);

    return res;
}


/* Compute igamc using DLMF 8.7.3. This is related to the series in
 * igam_series but extra care is taken to avoid cancellation.
 */
__noinline__ __device__ double  igamc_series(double a, double x)
{
    int n;
    double fac = 1;
    double sum = 0;
    double term, logx;

    for (n = 1; n < MAXITER; n++) {
        fac *= -x / n;
        term = fac / (a + n);
        sum += term;
        if (fabs(term) <= MACHEP * fabs(sum)) {
            break;
        }
    }

    logx = log(x);
    term = -expm1(a * logx - lgam1p(a));
    return term - exp(a * logx - lgam(a)) * sum;
}


/* Compute
 *
 * x^a * exp(-x) / gamma(a)
 *
 * corrected from (15) and (16) in [2] by replacing exp(x - a) with
 * exp(a - x).
 */
__noinline__ __device__ double igam_fac(double a, double x)
{
    double ax, fac, res, num;

    if (fabs(a - x) > 0.4 * fabs(a)) {
        ax = a * log(x) - x - lgam(a);
        if (ax < -MAXLOG) {
            // sf_error("igam", SF_ERROR_UNDERFLOW, NULL);
            return 0.0;
        }
        return exp(ax);
    }

    fac = a + lanczos_g - 0.5;
    res = sqrt(fac / exp(1.)) / lanczos_sum_expg_scaled(a);

    if ((a < 200) && (x < 200)) {
        res *= exp(a - x) * pow(x / fac, a);
    } else {
        num = x - a - lanczos_g + 0.5;
        res *= exp(a * log1pmx(num / fac) + x * (0.5 - lanczos_g) / fac);
    }

    return res;
}


/* Compute igamc using DLMF 8.9.2. */
__noinline__ __device__ double  igamc_continued_fraction(double a, double x)
{
    int i;
    double ans, ax, c, yc, r, t, y, z;
    double pk, pkm1, pkm2, qk, qkm1, qkm2;

    ax = igam_fac(a, x);
    if (ax == 0.0) {
        return 0.0;
    }

    /* continued fraction */
    y = 1.0 - a;
    z = x + y + 1.0;
    c = 0.0;
    pkm2 = 1.0;
    qkm2 = x;
    pkm1 = x + 1.0;
    qkm1 = z * x;
    ans = pkm1 / qkm1;

    for (i = 0; i < MAXITER; i++) {
        c += 1.0;
        y += 1.0;
        z += 2.0;
        yc = y * c;
        pk = pkm1 * z - pkm2 * yc;
        qk = qkm1 * z - qkm2 * yc;
        if (qk != 0) {
            r = pk / qk;
            t = fabs((ans - r) / r);
            ans = r;
        }
        else
            t = 1.0;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;
        if (fabs(pk) > big) {
            pkm2 *= biginv;
            pkm1 *= biginv;
            qkm2 *= biginv;
            qkm1 *= biginv;
        }
        if (t <= MACHEP) {
            break;
        }
    }

    return (ans * ax);
}


/* Compute igam using DLMF 8.11.4. */
__noinline__ __device__ double  igam_series(double a, double x)
{
    int i;
    double ans, ax, c, r;

    ax = igam_fac(a, x);
    if (ax == 0.0) {
        return 0.0;
    }

    /* power series */
    r = a;
    c = 1.0;
    ans = 1.0;

    for (i = 0; i < MAXITER; i++) {
        r += 1.0;
        c *= x / r;
        ans += c;
        if (c <= MACHEP * ans) {
            break;
        }
    }

    return (ans * ax / a);
}

__noinline__ __device__ double igamc(double a, double x)
{
    double absxma_a;

    if (x < 0 || a < 0) {
        // sf_error("gammaincc", SF_ERROR_DOMAIN, NULL);
        return CUDART_NAN;
    } else if (a == 0) {
        if (x > 0) {
            return 0;
        } else {
            return CUDART_NAN;
        }
    } else if (x == 0) {
        return 1;
    } else if (isinf(a)) {
        if (isinf(x)) {
            return CUDART_NAN;
        }
        return 1;
    } else if (isinf(x)) {
        return 0;
    }

    /* Asymptotic regime where a ~ x; see [2]. */
    absxma_a = fabs(x - a) / a;
    if ((a > SMALL) && (a < LARGE) && (absxma_a < SMALLRATIO)) {
        return asymptotic_series(a, x, IGAMC);
    } else if ((a > LARGE) && (absxma_a < LARGERATIO / sqrt(a))) {
        return asymptotic_series(a, x, IGAMC);
    }

    /* Everywhere else; see [2]. */
    if (x > 1.1) {
        if (x < a) {
            return 1.0 - igam_series(a, x);
        } else {
            return igamc_continued_fraction(a, x);
        }
    } else if (x <= 0.5) {
        if (-0.4 / log(x) < a) {
            return 1.0 - igam_series(a, x);
        } else {
            return igamc_series(a, x);
        }
    } else {
        if (x * 1.1 < a) {
            return 1.0 - igam_series(a, x);
        } else {
            return igamc_series(a, x);
        }
    }
}


__noinline__ __device__ double igam(double a, double x)
{
    double absxma_a;

    if (x < 0 || a < 0) {
        // sf_error("gammainc", SF_ERROR_DOMAIN, NULL);
        return CUDART_NAN;
    } else if (a == 0) {
        if (x > 0) {
            return 1;
        } else {
            return CUDART_NAN;
        }
    } else if (x == 0) {
        /* Zero integration limit */
        return 0;
    } else if (isinf(a)) {
        if (isinf(x)) {
            return CUDART_NAN;
        }
        return 0;
    } else if (isinf(x)) {
        return 1;
    }

    /* Asymptotic regime where a ~ x; see [2]. */
    absxma_a = fabs(x - a) / a;
    if ((a > SMALL) && (a < LARGE) && (absxma_a < SMALLRATIO)) {
        return asymptotic_series(a, x, IGAM);
    } else if ((a > LARGE) && (absxma_a < LARGERATIO / sqrt(a))) {
        return asymptotic_series(a, x, IGAM);
    }

    if ((x > 1.0) && (x > a)) {
        return (1.0 - igamc(a, x));
    }

    return igam_series(a, x);
}


"""
)


_igami_preamble = (
    _igam_preamble
    + gamma_definition
    + """

// from scipy/special/cephes/igami.c

__noinline__ __device__ double igamci(double a, double q);
__noinline__ __device__ double igami(double a, double q);

__noinline__ __device__ double find_inverse_s(double p, double q)
{
    /*
     * Computation of the Incomplete Gamma Function Ratios and their Inverse
     * ARMIDO R. DIDONATO and ALFRED H. MORRIS, JR.
     * ACM Transactions on Mathematical Software, Vol. 12, No. 4,
     * December 1986, Pages 377-393.
     *
     * See equation 32.
     */
    double s, t;
    double a[4] = {0.213623493715853, 4.28342155967104,
           11.6616720288968, 3.31125922108741};
    double b[5] = {0.3611708101884203e-1, 1.27364489782223,
           6.40691597760039, 6.61053765625462, 1};

    if (p < 0.5) {
        t = sqrt(-2 * log(p));
    }
    else {
        t = sqrt(-2 * log(q));
    }
    s = t - polevl<3>(t, a) / polevl<4>(t, b);
    if(p < 0.5)
        s = -s;
    return s;
}


__noinline__ __device__ double didonato_SN(double a, double x, unsigned N, double tolerance)
{
    /*
     * Computation of the Incomplete Gamma Function Ratios and their Inverse
     * ARMIDO R. DIDONATO and ALFRED H. MORRIS, JR.
     * ACM Transactions on Mathematical Software, Vol. 12, No. 4,
     * December 1986, Pages 377-393.
     *
     * See equation 34.
     */
    double sum = 1.0;

    if (N >= 1) {
    unsigned i;
        double partial = x / (a + 1);

        sum += partial;
        for(i = 2; i <= N; ++i) {
            partial *= x / (a + i);
            sum += partial;
            if(partial < tolerance) {
                break;
            }
        }
    }
    return sum;
}


__noinline__ __device__ double find_inverse_gamma(double a, double p, double q)
{
    /*
     * In order to understand what's going on here, you will
     * need to refer to:
     *
     * Computation of the Incomplete Gamma Function Ratios and their Inverse
     * ARMIDO R. DIDONATO and ALFRED H. MORRIS, JR.
     * ACM Transactions on Mathematical Software, Vol. 12, No. 4,
     * December 1986, Pages 377-393.
     */
    double result;

    if (a == 1) {
        if (q > 0.9) {
            result = -log1p(-p);
        }
        else {
            result = -log(q);
        }
    }
    else if (a < 1) {
        double g = Gamma(a);
        double b = q * g;

        if ((b > 0.6) || ((b >= 0.45) && (a >= 0.3))) {
            /* DiDonato & Morris Eq 21:
             *
             * There is a slight variation from DiDonato and Morris here:
             * the first form given here is unstable when p is close to 1,
             * making it impossible to compute the inverse of Q(a,x) for small
             * q. Fortunately the second form works perfectly well in this case.
             */
            double u;
            if((b * q > 1e-8) && (q > 1e-5)) {
                u = pow(p * g * a, 1 / a);
            }
            else {
                u = exp((-q / a) - NPY_EULER);
            }
            result = u / (1 - (u / (a + 1)));
        }
        else if ((a < 0.3) && (b >= 0.35)) {
            /* DiDonato & Morris Eq 22: */
            double t = exp(-NPY_EULER - b);
            double u = t * exp(t);
            result = t * exp(u);
        }
        else if ((b > 0.15) || (a >= 0.3)) {
            /* DiDonato & Morris Eq 23: */
            double y = -log(b);
            double u = y - (1 - a) * log(y);
            result = y - (1 - a) * log(u) - log(1 + (1 - a) / (1 + u));
        }
        else if (b > 0.1) {
            /* DiDonato & Morris Eq 24: */
            double y = -log(b);
            double u = y - (1 - a) * log(y);
            result = y - (1 - a) * log(u)
                - log((u * u + 2 * (3 - a) * u + (2 - a) * (3 - a))
                      / (u * u + (5 - a) * u + 2));
        }
        else {
            /* DiDonato & Morris Eq 25: */
            double y = -log(b);
            double c1 = (a - 1) * log(y);
            double c1_2 = c1 * c1;
            double c1_3 = c1_2 * c1;
            double c1_4 = c1_2 * c1_2;
            double a_2 = a * a;
            double a_3 = a_2 * a;

            double c2 = (a - 1) * (1 + c1);
            double c3 = (a - 1) * (-(c1_2 / 2)
                                   + (a - 2) * c1
                                   + (3 * a - 5) / 2);
            double c4 = (a - 1) * ((c1_3 / 3) - (3 * a - 5) * c1_2 / 2
                                   + (a_2 - 6 * a + 7) * c1
                                   + (11 * a_2 - 46 * a + 47) / 6);
            double c5 = (a - 1)
                        * (-(c1_4 / 4)
                        + (11 * a - 17) * c1_3 / 6
                        + (-3 * a_2 + 13 * a -13) * c1_2
                        + (2 * a_3 - 25 * a_2 + 72 * a - 61) * c1 / 2
                        + (25 * a_3 - 195 * a_2 + 477 * a - 379) / 12);

            double y_2 = y * y;
            double y_3 = y_2 * y;
            double y_4 = y_2 * y_2;
            result = y + c1 + (c2 / y) + (c3 / y_2) + (c4 / y_3) + (c5 / y_4);
        }
    }
    else {
        /* DiDonato and Morris Eq 31: */
        double s = find_inverse_s(p, q);

        double s_2 = s * s;
        double s_3 = s_2 * s;
        double s_4 = s_2 * s_2;
        double s_5 = s_4 * s;
        double ra = sqrt(a);

        double w = a + s * ra + (s_2 - 1) / 3;
        w += (s_3 - 7 * s) / (36 * ra);
        w -= (3 * s_4 + 7 * s_2 - 16) / (810 * a);
        w += (9 * s_5 + 256 * s_3 - 433 * s) / (38880 * a * ra);

        if ((a >= 500) && (fabs(1 - w / a) < 1e-6)) {
            result = w;
        }
        else if (p > 0.5) {
            if (w < 3 * a) {
                result = w;
            }
            else {
                double D = fmax(2, a * (a - 1));
                double lg = lgam(a);
                double lb = log(q) + lg;
                if (lb < -D * 2.3) {
                    /* DiDonato and Morris Eq 25: */
                    double y = -lb;
                    double c1 = (a - 1) * log(y);
                    double c1_2 = c1 * c1;
                    double c1_3 = c1_2 * c1;
                    double c1_4 = c1_2 * c1_2;
                    double a_2 = a * a;
                    double a_3 = a_2 * a;

                    double c2 = (a - 1) * (1 + c1);
                    double c3 = (a - 1) * (-(c1_2 / 2)
                               + (a - 2) * c1
                               + (3 * a - 5) / 2);
                    double c4 = (a - 1) * ((c1_3 / 3)
                               - (3 * a - 5) * c1_2 / 2
                               + (a_2 - 6 * a + 7) * c1
                               + (11 * a_2 - 46 * a + 47) / 6);
                    double c5 = (a - 1) * (-(c1_4 / 4)
                               + (11 * a - 17) * c1_3 / 6
                               + (-3 * a_2 + 13 * a -13) * c1_2
                               + (2 * a_3 - 25 * a_2 + 72 * a - 61) * c1 / 2
                               + (25 * a_3 - 195 * a_2 + 477 * a - 379) / 12);

                    double y_2 = y * y;
                    double y_3 = y_2 * y;
                    double y_4 = y_2 * y_2;
                    result = y + c1 + (c2 / y) + (c3 / y_2) + (c4 / y_3)
                             + (c5 / y_4);
                }
                else {
                    /* DiDonato and Morris Eq 33: */
                    double u = -lb + (a - 1) * log(w)
                               - log(1 + (1 - a) / (1 + w));
                    result = -lb + (a - 1) * log(u)
                             - log(1 + (1 - a) / (1 + u));
                }
            }
        }
        else {
            double z = w;
            double ap1 = a + 1;
            double ap2 = a + 2;
            if (w < 0.15 * ap1) {
                /* DiDonato and Morris Eq 35: */
                double v = log(p) + lgam(ap1);
                z = exp((v + w) / a);
                s = log1p(z / ap1 * (1 + z / ap2));
                z = exp((v + z - s) / a);
                s = log1p(z / ap1 * (1 + z / ap2));
                z = exp((v + z - s) / a);
                s = log1p(z / ap1 * (1 + z / ap2 * (1 + z / (a + 3))));
                z = exp((v + z - s) / a);
            }

            if ((z <= 0.01 * ap1) || (z > 0.7 * ap1)) {
                result = z;
            }
            else {
                /* DiDonato and Morris Eq 36: */
                double ls = log(didonato_SN(a, z, 100, 1e-4));
                double v = log(p) + lgam(ap1);
                z = exp((v + z - ls) / a);
                result = z * (1 - (a * log(z) - z - v + ls) / (a - z));
            }
        }
    }
    return result;
}


__noinline__ __device__ double igamci(double a, double q)
{
    int i;
    double x, fac, f_fp, fpp_fp;

    if (isnan(a) || isnan(q)) {
        return CUDART_NAN;
    }
    else if ((a < 0.0) || (q < 0.0) || (q > 1.0)) {
        // sf_error("gammainccinv", SF_ERROR_DOMAIN, NULL);
    }
    else if (q == 0.0) {
        return CUDART_INF;
    }
    else if (q == 1.0) {
        return 0.0;
    }
    else if (q > 0.9) {
        return igami(a, 1 - q);
    }

    x = find_inverse_gamma(a, 1 - q, q);
    for (i = 0; i < 3; i++) {
        fac = igam_fac(a, x);
        if (fac == 0.0) {
            return x;
        }
        f_fp = (igamc(a, x) - q) * x / (-fac);
        fpp_fp = -1.0 + (a - 1) / x;
        if (isinf(fpp_fp)) {
            x = x - f_fp;
        }
        else {
            x = x - f_fp / (1.0 - 0.5 * f_fp * fpp_fp);
        }
    }

    return x;
}


__noinline__ __device__ double igami(double a, double p)
{
    int i;
    double x, fac, f_fp, fpp_fp;

    if (isnan(a) || isnan(p)) {
        return CUDART_NAN;
    }
    else if ((a < 0) || (p < 0) || (p > 1)) {
        // sf_error("gammaincinv", SF_ERROR_DOMAIN, NULL);
    }
    else if (p == 0.0) {
        return 0.0;
    }
    else if (p == 1.0) {
        return CUDART_INF;
    }
    else if (p > 0.9) {
        return igamci(a, 1 - p);
    }
    x = find_inverse_gamma(a, p, 1 - p);
    /* Halley's method */
    for (i = 0; i < 3; i++) {
        fac = igam_fac(a, x);
        if (fac == 0.0) {
            return x;
        }
        f_fp = (igam(a, x) - p) * x / fac;
        // The ratio of the first and second derivatives simplifies
        fpp_fp = -1.0 + (a - 1) / x;
        if (isinf(fpp_fp)) {
            // Resort to Newton's method in the case of overflow
            x = x - f_fp;
        }
        else {
            x = x - f_fp / (1.0 - 0.5 * f_fp * fpp_fp);
        }
    }

    return x;
}

"""
)


gammaincc = _core.create_ufunc(
    "cupyx_scipy_gammaincc",
    ("ff->f", "dd->d"),
    "out0 = out0_type(igamc(in0, in1));",
    preamble=_igam_preamble,
    doc="""Elementwise function for scipy.special.gammaincc

    Regularized upper incomplete gamma function.

    .. seealso:: :meth:`scipy.special.gammaincc`
    """,
)


gammainc = _core.create_ufunc(
    "cupyx_scipy_gammainc",
    ("ff->f", "dd->d"),
    "out0 = out0_type(igam(in0, in1));",
    preamble=_igam_preamble,
    doc="""Elementwise function for scipy.special.gammainc

    Regularized lower incomplete gamma function.

    .. seealso:: :meth:`scipy.special.gammainc`
    """,
)


gammainccinv = _core.create_ufunc(
    "cupyx_scipy_gammainccinv",
    ("ff->f", "dd->d"),
    "out0 = out0_type(igamci(in0, in1));",
    preamble=_igami_preamble,
    doc="""Elementwise function for scipy.special.gammainccinv

    Inverse to gammaincc.

    .. seealso:: :meth:`scipy.special.gammainccinv`
    """,
)


gammaincinv = _core.create_ufunc(
    "cupyx_scipy_gammaincinv",
    ("ff->f", "dd->d"),
    "out0 = out0_type(igami(in0, in1));",
    preamble=_igami_preamble,
    doc="""Elementwise function for scipy.special.gammaincinv

    Inverse to gammainc.

    .. seealso:: :meth:`scipy.special.gammaincinv`
    """,
)
