Tips and FAQs
=============

It takes too long time to compile a computational graph. Can I skip it?
-----------------------------------------------------------------------

Chainer does not compile computational graphs, so you cannot skip it, or, I mean, you have already skipped it :).

It seems you have actually seen on-the-fly compilations of CUDA kernels.
CuPy compiles kernels on demand to make kernels optimized to the number of dimensions and element types of input arguments.
Pre-compilation is not available, because we have to compile an exponential number of kernels to support all CuPy functionalities.
This restriction is unavoidable because Python cannot call CUDA/C++ template functions in generic way.
Note that every framework using CUDA require compilation at some point; the difference between other statically-compiled frameworks (such as cutorch) and Chainer is whether a kernel is compiled at installation or at the first use.

These compilations should run only at the first use of the kernels.
The compiled binaries are cached to the ``$(HOME)/.cupy/kernel_cache`` directory by default.
If you see that compilations run every time you run the same script, then the caching is failed.
Please check that the directory is kept as is between multiple executions of the script.
If your home directory is not suited to caching the kernels (e.g. in case that it uses NFS), change the kernel caching directory by setting the ``CUPY_CACHE_DIR`` environment variable to an appropriate path.
See :ref:`cupy-overview` for more details.


mnist example does not converge in CPU mode on Mac OS X
-------------------------------------------------------

Many users reported that mnist example does not work correctly on Mac OS X.
We are suspecting it is caused by vecLib, that is a default BLAS library installed on Mac OS X.

.. note::

   Mac OS X is not officially supported.
   I mean it is not tested continuously on our test server.

We recommend to use other BLAS libraries such as `OpenBLAS <http://www.openblas.net/>`_.
We empirically found that it fixes this problem.
It is necessary to reinstall NumPy to use replaced BLAS library.
Here is an instruction to install NumPy with OpneBLAS using `Homebrew <http://brew.sh/>`_.

::

   $ brew tap homebrew/science
   $ brew install openblas
   $ brew install numpy --with-openblas

If you want to install NumPy with pip, use `site.cfg <https://github.com/numpy/numpy/blob/master/site.cfg.example>`_ file.

You can check if NumPy uses OpenBLAS with ``numpy.show_config`` method.
Check if `blas_opt_info` refers to `openblas`.

::

   >>> import numpy
   >>> numpy.show_config()
   lapack_opt_info:
       libraries = ['openblas', 'openblas']
       library_dirs = ['/usr/local/opt/openblas/lib']
       define_macros = [('HAVE_CBLAS', None)]
       language = c
   blas_opt_info:
       libraries = ['openblas', 'openblas']
       library_dirs = ['/usr/local/opt/openblas/lib']
       define_macros = [('HAVE_CBLAS', None)]
       language = c
   openblas_info:
       libraries = ['openblas', 'openblas']
       library_dirs = ['/usr/local/opt/openblas/lib']
       define_macros = [('HAVE_CBLAS', None)]
       language = c
   openblas_lapack_info:
       libraries = ['openblas', 'openblas']
       library_dirs = ['/usr/local/opt/openblas/lib']
       define_macros = [('HAVE_CBLAS', None)]
       language = c
   blas_mkl_info:
       NOT AVAILABLE

See detail about this problem in `issue #704 <https://github.com/pfnet/chainer/issues/704>`_.
