.. module:: cupyx.scipy.special

Special functions (:mod:`cupyx.scipy.special`)
===============================================

.. Hint:: `SciPy API Reference: Special functions (scipy.special) <https://docs.scipy.org/doc/scipy/reference/special.html>`_

Bessel functions
----------------

.. autosummary::
   :toctree: generated/

   j0
   j1
   k0
   k0e
   k1
   k1e
   y0
   y1
   yn
   i0
   i0e
   i1
   i1e
   spherical_yn


Raw statistical functions
-------------------------

.. seealso:: :mod:`cupyx.scipy.stats`

.. autosummary::
   :toctree: generated/

   bdtr
   bdtrc
   bdtri
   btdtr
   btdtri
   fdtr
   fdtrc
   fdtri
   gdtr
   gdtrc
   nbdtr
   nbdtrc
   nbdtri
   pdtr
   pdtrc
   pdtri
   chdtr
   chdtrc
   chdtri
   ndtr
   log_ndtr
   ndtri
   logit
   expit
   log_expit
   boxcox
   boxcox1p
   inv_boxcox
   inv_boxcox1p


Information Theory functions
----------------------------

.. autosummary::
   :toctree: generated/

   entr
   rel_entr
   kl_div
   huber
   pseudo_huber


Gamma and related functions
---------------------------

.. autosummary::
   :toctree: generated/

   gamma
   gammaln
   loggamma
   gammainc
   gammaincinv
   gammaincc
   gammainccinv
   beta
   betaln
   betainc
   betaincinv
   psi
   rgamma
   polygamma
   multigammaln
   digamma
   poch


Elliptic integrals
------------------

   ellipk
   ellipkm1
   ellipj


Error function and Fresnel integrals
------------------------------------

.. autosummary::
   :toctree: generated/

   erf
   erfc
   erfcx
   erfinv
   erfcinv


Legendre functions
---------------------------

.. autosummary::
   :toctree: generated/

   lpmv
   sph_harm


Other special functions
-----------------------

.. autosummary::
   :toctree: generated/

   exp1
   expi
   expn
   exprel
   softmax
   log_softmax
   zeta
   zetac


Convenience functions
-----------------------

.. autosummary::
   :toctree: generated/

   cbrt
   exp10
   exp2
   radian
   cosdg
   sindg
   tandg
   cotdg
   log1p
   expm1
   cosm1
   round
   xlogy
   xlog1py
   logsumexp
   sinc
