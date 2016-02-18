Tips and FAQs
=============

It takes too long time to compile a computational graph. Can I skip it?
-----------------------------------------------------------------------

Chainer does not compile computational graphs, so you cannot skip it, or, I mean, you have already skipped it :).

It seems you have actually seen on-the-fly compilations of CUDA kernels.
CuPy compiles kernels on demand to make kernels optimized to the number of dimensions and element types of input arguments.
Precompilation is not available, because we have to compile an exponential number of kernels to support all CuPy functionalities.
This restriction is unavoidable because Python cannot call CUDA/C++ template functions in generic way.
Note that every framework using CUDA require compilation at some point; the difference between other statically-compiled frameworks (such as cutorch) and Chainer is whether a kernel is compiled at installation or at the first use.

These compilations should run only at the first use of the kernels.
The compiled binaries are cached to the ``$(HOME)/.cupy/kernel_cache`` directory by default.
If you see that compilations run everytime you run the same script, then the caching is failed.
Please check that the directory is kept as is between multiple executions of the script.
If your home directory is not suited to caching the kernels (e.g. in case that it uses NFS), change the kernel caching directory by setting the ``CUPY_CACHE_DIR`` environment variable to an appropriate path.
See :ref:`cupy-overview` for more details.
