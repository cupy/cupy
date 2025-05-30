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
   wright_bessel


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
   gammasgn
   gammainc
   gammaincinv
   gammaincc
   gammainccinv
   beta
   betaln
   betainc
   # betaincc
   betaincinv
   # betainccinv
   psi
   rgamma
   polygamma
   multigammaln
   digamma
   poch


Elliptic functions and integrals
--------------------------------

.. autosummary::
   :toctree: generated/

   ellipj
   ellipk
   ellipkm1
   ellipkinc
   # ellipe
   ellipeinc
   # elliprc
   # elliprd
   # elliprf
   # elliprg
   # elliprj


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


Lambert W and related functions
-------------------------------

.. autosummary::
   :toctree: generated/

   lambertw


Other special functions
-----------------------

.. autosummary::
   :toctree: generated/

   # agm
   # bernoulli
   binom
   # diric
   # euler
   expn
   exp1
   expi
   # factorial
   # factorial2
   # factorialk
   shichi
   sici
   softmax
   log_softmax
   # spence
   zeta
   zetac
   # softplus


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
   # powm1
   round
   xlogy
   xlog1py
   logsumexp
   exprel
   sinc
