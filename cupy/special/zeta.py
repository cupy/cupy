import cupy
from cupy import core


_zeta_kernel = None


def _get_zeta_kernel():
    global _zeta_kernel
    if _zeta_kernel is None:
        _zeta_kernel = core.ElementwiseKernel(
            'T x_, T q_', 'T y',
            """
            int j;
            float a, b, k, s, t, w;
            float MACHEP =  1.38777878078144567553E-17;
            float A[] = {
                12.0,
                -720.0,
                30240.0,
                -1209600.0,
                47900160.0,
                -1.8924375803183791606e9,	/*1.307674368e12/691 */
                7.47242496e10,
                -2.950130727918164224e12,	/*1.067062284288e16/3617 */
                1.1646782814350067249e14,	/*5.109094217170944e18/43867 */
                -4.5979787224074726105e15,	/*8.028576626982912e20/174611 */
                1.8152105401943546773e17,	/*1.5511210043330985984e23/854513 */
                -7.1661652561756670113e18
                    /*1.6938241367317436694528e27/236364091 */
            };

            float x = x_;
            float q = q_;
            if (x == 1.0){
                y = 1.0 / 0.0;
                return;
            }

            if (x < 1.0) {
                y = nanf("");
                return;
            }

            if (q <= 0.0) {
                if (q == floor(q)) {
                    y = 1.0 / 0.0;
                    return;
                }
                if (x != floor(x)){
                    y = nanf("");
                    return;
                }
            }

            if (q > 1e8) {
                y = (1/(x - 1) + 1/(2*q)) * powf(q, 1 - x);
                return;
            }

            s = pow(q, -x);
            a = q;
            j = 0;
            b = 0.0;
            while ((j < 9) || (a <= 9.0)) {
                j += 1;
                a += 1.0;
                b = pow(a, -x);
                s += b;
                if (fabs(b / s) < MACHEP){
                    y = s;
                    return;
                }
            }

            w = a;
            s += b * w / (x - 1.0);
            s -= 0.5 * b;
            a = 1.0;
            k = 0.0;
            for (j = 0; j < 12; j++) {
                a *= x + k;
                b /= w;
                t = a * b / A[j];
                s = s + t;
                t = fabs(t / s);
                if (t < MACHEP){
                    y = s;
                    return;
                }
                k += 1.0;
                a *= x + k;
                b /= w;
                k += 1.0;
            }
            """,
            'zeta_kernel'
        )
    return _zeta_kernel


def zeta(x, q):
    y = cupy.zeros_like(x)
    _get_zeta_kernel()(x, q, y)
    return y
