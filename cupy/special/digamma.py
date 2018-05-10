import cupy
from cupy import core


_digamma_kernel = None


def _get_digamma_kernel():
    global _digamma_kernel
    if _digamma_kernel is None:
        _digamma_kernel = core.ElementwiseKernel(
            'T x_', 'T y',
            """
            float PI = 3.141592653589793;
            float x = x_;
            y = 0.;
            if (isnan(x)){
                y = x;
                return;
            }
            else if (isinf(x)){
                if(x > 0){
                    y = x;
                    return;
                }else{
                    y = 1./0.;
                    return;
                }
            }
            else if (x == 0.){
                y = 1./0.;
                return;
            }
            else if(x < 0.){
                float r = fmodf(x, 1.);
                if (r == 0.){
                    y = 1./0.;
                    return;
                }
                y = - PI / tan(PI * r);
                x = 1.0 - x;
            }

            if(x <= 10. && x == floor(x)){
                float EULER = 0.5772156649015329;
                int n = (int)x;
                for(int j = 1; j < n; j++){
                    y += 1. / j;
                }
                y -= EULER;
                return;
            }

            if(x < 1.){
                y -= 1. / x;
                x += 1.;
            }
            else if(x < 10.){
                while(x > 2.){
                    x -= 1.;
                    y += 1. / x;
                }
            }

            if( x >= 1. && x <= 2.){
                float Y = 0.99558162689208984;

                float root1 = 1569415565.0 / 1073741824.0;
                float root2 = (381566830.0 / 1073741824.0) / 1073741824.0;
                float root3 = 0.9016312093258695918615325266959189453125e-19;

                float P[] = {
                   -0.0020713321167745952,
                   -0.045251321448739056,
                   -0.28919126444774784,
                   -0.65031853770896507,
                   -0.32555031186804491,
                   0.25479851061131551
                    };
                float Q[] = {
                   -0.55789841321675513e-6,
                   0.0021284987017821144,
                   0.054151797245674225,
                   0.43593529692665969,
                   1.4606242909763515,
                   2.0767117023730469,
                   1.0
                   };

                float g = x - root1;
                g -= root2;
                g -= root3;

                float u_polevl, d_polevl;
                int j;
                float *p;

                p = P;
                u_polevl = *p++;
                j = 5;

                do{
                    u_polevl = u_polevl * (x-1.) + *p++;
                }while(--j);

                p = Q;
                d_polevl = *p++;
                j = 6;

                do{
                    d_polevl = d_polevl * (x-1.) + *p++;
                }while(--j);

                float r = u_polevl / d_polevl;
                y += g * Y + g * r;
                return;
            }

            float A[] = {
                8.33333333333333333333E-2,
                -2.10927960927960927961E-2,
                7.57575757575757575758E-3,
                -4.16666666666666666667E-3,
                3.96825396825396825397E-3,
                -8.33333333333333333333E-3,
                8.33333333333333333333E-2
            };

            float w;
            if (x < 1e17){
                float z = 1.0 / (x * x);

                int j;
                float *p;

                p = A;
                w = *p++;
                j = 6;

                do{
                    w = w * z + *p++;
                }while(--j);

                w *= z;
            }else{
                w = 0.;
            }

            y += log(x) - (0.5 / x) - w;
            """,
            'digamma_kernel'
        )
    return _digamma_kernel


def digamma(x):
    y = cupy.zeros_like(x)
    _get_digamma_kernel()(x, y)
    return y
