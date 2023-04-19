"""
A collection of basic statistical functions for Python.

References
----------
.. [CRCProbStat2000] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
   Probability and Statistics Tables and Formulae. Chapman & Hall: New
   York. 2000.
"""
import cupy as cp

from cupy._core import _routines_statistics as _statistics

# tmin_kernel =  cp.ReductionKernel(
#         'S x1 , S limit, bool inclusive, bool propagate_nan',
#         'S y',
#         'x1',
#         'tmin_helper(a,b,limit,inclusive,propagate_nan)',
#         'y=a',
#         'CUDART_INF',
#         name='tmin',
#         reduce_dims=True,
#         preamble="""    template <typename S>
#                         __device__ S tmin_helper(S a , S b ,S limit, bool inclusive , bool propagate_nan ){
#                             if (propagate_nan){
#                                 return ((b!=b || a!=a  ) ? nan("")  :
#                                 (inclusive ? (b>=limit ? min(a,b) : a) : (b>limit ? min(a,b) : a)));
#                             }
#                             else{
#                                 return inclusive ? (b>=limit ? min(a,b) : a) : (b>limit ? min(a,b) : a);
#                             }
#                         }


#                         """


#     )

tmin_kernel =  cp.ReductionKernel(
        'S x1 ,T limit, bool inclusive, bool propagate_nan',
        'S y',
        'x1',
        'tmin_helper(a,b,limit,inclusive,propagate_nan)',
        'y=a',
        'CUDART_INF',
        name='tmin',
        reduce_dims=True,
        preamble="""   
                        template <typename S,typename T>
                        __device__ S tmin_helper(S a , S b ,T limit , bool inclusive , bool propagate_nan ){
                            
                            if (propagate_nan){
                                return ((b!=b || a!=a  ) ? nan("")  :
                                (inclusive ? (a>=(S) limit ? min(a,b) : b) : (a>(S) limit ? min(a,b) : b)));
                            }
                            else{
                                printf("a=%d,b=%d  ",a,b);
                                return inclusive ? (a>=(S) limit ? min(a,b) : b) : (a>(S) limit ? min(a,b) : b);
                            }
                        }


                        """


    )

# tmin_kernel =  cp.ReductionKernel(
#         'S x1 , T limit, bool inclusive, bool propagate_nan',
#         'S y',
#         'x1',
#         'tmin_helper(a,b,limit,inclusive,propagate_nan)',
#         'y=a',
#         'CUDART_INF',
#         name='tmin',
#         reduce_dims=True,
#         preamble="""    template <typename S>
#                         __device__ S tmin_helper(S a , S b ,T limit, bool inclusive , bool propagate_nan ){
#                             if (propagate_nan){
                                
#                                 if ((b!=b) || (a!=a)) {
#                                     return nan("");
#                                 }
#                                 else{
#                                     if (inclusive) {
#                                         return (b>=limit ? min(a,b) : a);
#                                     } 
#                                     else{ 
#                                         return (b>limit ? min(a,b) : a);
#                                     }
#                                 }
#                             }
#                             else{
                                 
#                                  if (inclusive) {
#                                         printf("%d   ",a);
#                                         return (b>=limit ? min(a,b) : a);
#                                     } 
#                                     else{ 
#                                         return (b>limit ? min(a,b) : a);
#                                     }
#                             }
#                         }


#                         """


#     )

# tmin_kernel=cp.ReductionKernel(
#             'T x1 ,T limit,bool inclusive,bool propagate_nan',
#             'T y , T within_limits',
#             'x1',
#             '(tmin_helper(a,b,limit,inclusive,propagate_nan),10)',
#             '(y=a) , within_limits=0',
#             '(CUDART_INF,0)',
#             name='tmin',
#             reduce_dims=True,
#             preamble="""
#                     bool within_limits=false;
#                     __device__ T tmin_helper(T a , T b ,T limit, bool inclusive , bool propagate_nan ){

#                         if (propagate_nan){
#                             return  ((isnan(b) || isnan(a)  ) ? nan("")  :
#                             (inclusive ? (b>=limit ? min(a,b) : a) : (b>limit ? min(a,b) : a)));
#                         }
#                         else{
#                             return inclusive ? (b>=limit ? min(a,b) : a) : (b>limit ? min(a,b) : a);
#                         }
#                     }


#                     """

# )
# params = ['input','limit',]
# arginfos = [('input', cp.ndarray),
#                 ('limit', 'float64')]
# tmin_kernel= _core.create_reduction_func(
#     'cupy_tmin',

#     ('e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
#     ('T in0 ,T limit,bool inclusive,bool propagate_nan',
#      'in0', 'tmin_helper(a, b,in1,in2,in3)',
#      'out0 = a'), 'CUDART_INF' ,
#      preamble="""
#             __device__ T tmin_helper(T a , T b ,T limit, bool inclusive , bool propagate_nan ){
#                 if (propagate_nan){
#                     return ((isnan(a) || isnan(b) ) ? nan("")  :
#                     (inclusive ? (b>=limit ? min(a,b) : a) : (b>limit ? min(a,b) : a)));
#                 }
#                 else{
#                     return inclusive ? (b>=limit ? min(a,b) : a) : (b>limit ? min(a,b) : a);
#                 }
#             }


#             """

#         )

def trim_mean(a, proportiontocut, axis=0):
    """Return mean of array after trimming distribution from both tails.

    If `proportiontocut` = 0.1, slices off 'leftmost' and 'rightmost' 10% of
    scores. The input is sorted before slicing. Slices off less if proportion
    results in a non-integer slice index (i.e., conservatively slices off
    `proportiontocut` ).

    Parameters
    ----------
    a : cupy.ndarray
        Input array.
    proportiontocut : float
        Fraction to cut off of both tails of the distribution.
    axis : int or None, optional
        Axis along which the trimmed means are computed. Default is 0.
        If None, compute over the whole array `a`.

    Returns
    -------
    trim_mean : ndarray
        Mean of trimmed array.

    See Also
    --------
    trimboth
    tmean : Compute the trimmed mean ignoring values outside given `limits`.

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyx.scipy import stats
    >>> x = cp.arange(20)
    >>> stats.trim_mean(x, 0.1)
    array(9.5)
    >>> x2 = x.reshape(5, 4)
    >>> x2
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])
    >>> stats.trim_mean(x2, 0.25)
    array([ 8.,  9., 10., 11.])
    >>> stats.trim_mean(x2, 0.25, axis=1)
    array([ 1.5,  5.5,  9.5, 13.5, 17.5])
    """
    if a.size == 0:
        return cp.nan

    if axis is None:
        a = a.ravel()
        axis = 0

    nobs = a.shape[axis]
    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if (lowercut > uppercut):
        raise ValueError("Proportion too big.")

    atmp = cp.partition(a, (lowercut, uppercut - 1), axis)

    sl = [slice(None)] * atmp.ndim
    sl[axis] = slice(lowercut, uppercut)
    return cp.mean(atmp[tuple(sl)], axis=axis)


def tmin(a, lowerlimit=None, axis=0, inclusive=True, nan_policy='propagate'):
    
    if a.size == 0:
        raise ValueError("No array values within given limits")
    if axis is None:
        a = a.ravel()
        axis = 0
    if a.ndim == 0:
        a = cp.atleast_1d(a)
   
    policies = ['propagate', 'raise', 'omit']

    if nan_policy not in policies:
        raise ValueError("nan_policy must be one of {%s}" %
                         ', '.join("'%s'" % s for s in policies))


    contains_nan = cp.isnan(cp.sum(a))
    
    
    if contains_nan :
            if nan_policy != 'propagate':
                if nan_policy == 'raise':
                    raise ValueError("The input contains nan values")
                return _statistics._tmin(a,cp.atleast_1d( -cp.inf if lowerlimit is None else lowerlimit), inclusive, False, axis=axis)
            return _statistics._tmin(a, cp.atleast_1d(-cp.inf if lowerlimit is None else lowerlimit), inclusive, True, axis=axis)
    else:
        if lowerlimit is None:
            return cp.amin(a, axis=axis)
        max_element= cp.max(a)
        if max_element>lowerlimit or (max_element==lowerlimit and inclusive):
            return _statistics._tmin(a, cp.atleast_1d(lowerlimit), inclusive, False, axis=axis)
        raise ValueError("No array values within given limits")
            
            # if cp.all(solution==cp.iinfo(solution.dtype).max):
            #     raise ValueError("No array values within given limits")
            