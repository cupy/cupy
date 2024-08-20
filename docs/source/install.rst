Installation
============

Requirements
------------

* `NVIDIA CUDA GPU <https://developer.nvidia.com/cuda-gpus>`_ with the Compute Capability 3.0 or larger.

* `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_: v11.2 / v11.3 / v11.4 / v11.5 / v11.6 / v11.7 / v11.8 / v12.0 / v12.1 / v12.2 / v12.3 / v12.4 / v12.5 / v12.6

    * If you have multiple versions of CUDA Toolkit installed, CuPy will automatically choose one of the CUDA installations.
      See :ref:`install_cuda` for details.

    * This requirement is optional if you install CuPy from ``conda-forge``. However, you still need to have a compatible
      driver installed for your GPU. See :ref:`install_cupy_from_conda_forge` for details.

* `Python <https://python.org/>`_: v3.9 / v3.10 / v3.11 / v3.12

.. note::

   Currently, CuPy is tested against  `Ubuntu <https://www.ubuntu.com/>`_ 20.04 LTS / 22.04 LTS (x86_64), `CentOS <https://www.centos.org/>`_ 7 / 8 (x86_64) and Windows Server 2016 (x86_64).

Python Dependencies
~~~~~~~~~~~~~~~~~~~

NumPy/SciPy-compatible API in CuPy v13 is based on NumPy 1.26 and SciPy 1.11, and has been tested against the following versions:

* `NumPy <https://numpy.org/>`_: v1.22 / v1.23 / v1.24 / v1.25 / v1.26 / v2.0

* `SciPy <https://scipy.org/>`_ (*optional*): v1.7 / v1.8 / v1.9 / v1.10 / v1.11

    * Required only when copying sparse matrices from GPU to CPU (see :doc:`../reference/scipy_sparse`.)

* `Optuna <https://optuna.org/>`_ (*optional*): v3.x

    * Required only when using :ref:`kernel_param_opt`.

.. note::

   SciPy and Optuna are optional dependencies and will not be installed automatically.

.. note::

   Before installing CuPy, we recommend you to upgrade ``setuptools`` and ``pip``::

    $ python -m pip install -U setuptools pip

Additional CUDA Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~

Part of the CUDA features in CuPy will be activated only when the corresponding libraries are installed.

* `cuTENSOR <https://developer.nvidia.com/cutensor>`_: v2.0

    * The library to accelerate tensor operations. See :doc:`../reference/environment` for the details.

* `NCCL <https://developer.nvidia.com/nccl>`_: v2.16 / v2.17

    * The library to perform collective multi-GPU / multi-node computations.

* `cuDNN <https://developer.nvidia.com/cudnn>`_: v8.8

    * The library to accelerate deep neural network computations.

* `cuSPARSELt <https://docs.nvidia.com/cuda/cusparselt/>`_: v0.2.0

    * The library to accelerate sparse matrix-matrix multiplication.


Installing CuPy
---------------

Installing CuPy from PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~

Wheels (precompiled binary packages) are available for Linux and Windows.
Package names are different depending on your CUDA Toolkit version.

.. list-table::
   :header-rows: 1

   * - CUDA
     - Command
   * - **v11.2 ~ 11.8** (x86_64 / aarch64)
     - ``pip install cupy-cuda11x``
   * - **v12.x** (x86_64 / aarch64)
     - ``pip install cupy-cuda12x``

.. note::

   To enable features provided by additional CUDA libraries (cuTENSOR / NCCL / cuDNN), you need to install them manually.
   If you installed CuPy via wheels, you can use the installer command below to setup these libraries in case you don't have a previous installation::

    $ python -m cupyx.tools.install_library --cuda 11.x --library cutensor

.. note::

   Append ``--pre -U -f https://pip.cupy.dev/pre`` options to install pre-releases (e.g., ``pip install cupy-cuda11x --pre -U -f https://pip.cupy.dev/pre``).


When using wheels, please be careful not to install multiple CuPy packages at the same time.
Any of these packages and ``cupy`` package (source installation) conflict with each other.
Please make sure that only one CuPy package (``cupy`` or ``cupy-cudaXX`` where XX is a CUDA version) is installed::

  $ pip freeze | grep cupy


.. _install_cupy_from_conda_forge:

Installing CuPy from Conda-Forge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conda is a cross-language, cross-platform package management solution widely used in scientific computing and other fields.
The above ``pip install`` instruction is compatible with ``conda`` environments. Alternatively, for both Linux (x86_64,
ppc64le, aarch64-sbsa) and
Windows once the CUDA driver is correctly set up, you can also install CuPy from the ``conda-forge`` channel::

    $ conda install -c conda-forge cupy

and ``conda`` will install a pre-built CuPy binary package for you, along with the CUDA runtime libraries
(``cudatoolkit`` for CUDA 11 and below, or ``cuda-XXXXX`` for CUDA 12 and above). It is not necessary to install CUDA Toolkit in advance.

If you aim at minimizing the installation footprint, you can install the ``cupy-core`` package::

    $ conda install -c conda-forge cupy-core

which only depends on ``numpy``. None of the CUDA libraries will be installed this way, and it is your responsibility to install the needed
dependencies yourself, either from conda-forge or elsewhere. This is equivalent of the ``cupy-cudaXX`` wheel installation.

Conda has a built-in mechanism to determine and install the latest version of ``cudatoolkit`` or any other CUDA components supported by your driver.
However, if for any reason you need to force-install a particular CUDA version (say 11.8), you can do::

    $ conda install -c conda-forge cupy cuda-version=11.8

.. note::

    cuDNN, cuTENSOR, and NCCL are available on ``conda-forge`` as optional dependencies. The following command can install them all at once::

        $ conda install -c conda-forge cupy cudnn cutensor nccl

    Each of them can also be installed separately as needed.

.. note::

    If you encounter any problem with CuPy installed from ``conda-forge``, please feel free to report to `cupy-feedstock
    <https://github.com/conda-forge/cupy-feedstock/issues>`_, and we will help investigate if it is just a packaging
    issue in ``conda-forge``'s recipe or a real issue in CuPy.

.. note::

    If you did not install CUDA Toolkit by yourself, for CUDA 11 and below the ``nvcc`` compiler might not be available, as
    the ``cudatoolkit`` package from ``conda-forge`` does not include the ``nvcc`` compiler toolchain. If you would like to use
    it from a local CUDA installation, you need to make sure the version of CUDA Toolkit matches that of ``cudatoolkit`` to
    avoid surprises. For CUDA 12 and above, ``nvcc`` can be installed on a per-``conda`` environment basis via

        $ conda install -c conda-forge cuda-nvcc


.. _install_cupy_from_source:

Installing CuPy from Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use of wheel packages is recommended whenever possible.
However, if wheels cannot meet your requirements (e.g., you are running non-Linux environment or want to use a version of CUDA / cuDNN / NCCL not supported by wheels), you can also build CuPy from source.

.. note::

   CuPy source build requires ``g++-6`` or later.
   For Ubuntu 18.04, run ``apt-get install g++``.
   For Ubuntu 16.04, CentOS 6 or 7, follow the instructions :ref:`here <install_gcc6>`.

.. note::

   When installing CuPy from source, features provided by additional CUDA libraries will be disabled if these libraries are not available at the build time.
   See :ref:`install_cudnn` for the instructions.

.. note::

   If you upgrade or downgrade the version of CUDA Toolkit, cuDNN, NCCL or cuTENSOR, you may need to reinstall CuPy.
   See :ref:`install_reinstall` for details.

You can install the latest stable release version of the `CuPy source package <https://pypi.python.org/pypi/cupy>`_ via ``pip``.

::

  $ pip install cupy

If you want to install the latest development version of CuPy from a cloned Git repository::

  $ git clone --recursive https://github.com/cupy/cupy.git
  $ cd cupy
  $ pip install .

.. note::

   Cython 0.29.22 or later is required to build CuPy from source.
   It will be automatically installed during the build process if not available.


Uninstalling CuPy
-----------------

Use ``pip`` to uninstall CuPy::

  $ pip uninstall cupy

.. note::

   If you are using a wheel, ``cupy`` shall be replaced with ``cupy-cudaXX`` (where XX is a CUDA version number).

.. note::

   If CuPy is installed via ``conda``, please do ``conda uninstall cupy`` instead.


Upgrading CuPy
---------------

Just use ``pip install`` with ``-U`` option::

  $ pip install -U cupy

.. note::

   If you are using a wheel, ``cupy`` shall be replaced with ``cupy-cudaXX`` (where XX is a CUDA version number).


.. _install_reinstall:


Reinstalling CuPy
-----------------

To reinstall CuPy, please uninstall CuPy and then install it.
When reinstalling CuPy, we recommend using ``--no-cache-dir`` option as ``pip`` caches the previously built binaries::

  $ pip uninstall cupy
  $ pip install cupy --no-cache-dir

.. note::

   If you are using a wheel, ``cupy`` shall be replaced with ``cupy-cudaXX`` (where XX is a CUDA version number).


Using CuPy inside Docker
------------------------

We are providing the `official Docker images <https://hub.docker.com/r/cupy/cupy/>`_.
Use `NVIDIA Container Toolkit <https://github.com/NVIDIA/nvidia-docker>`_ to run CuPy image with GPU.
You can login to the environment with bash, and run the Python interpreter::

  $ docker run --gpus all -it cupy/cupy /bin/bash

Or run the interpreter directly::

  $ docker run --gpus all -it cupy/cupy /usr/bin/python3


FAQ
---

.. _install_error:

``pip`` fails to install CuPy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please make sure that you are using the latest ``setuptools`` and ``pip``::

  $ pip install -U setuptools pip

Use ``-vvvv`` option with ``pip`` command.
This will display all logs of installation::

  $ pip install cupy -vvvv

If you are using ``sudo`` to install CuPy, note that ``sudo`` command does not propagate environment variables.
If you need to pass environment variable (e.g., ``CUDA_PATH``), you need to specify them inside ``sudo`` like this::

  $ sudo CUDA_PATH=/opt/nvidia/cuda pip install cupy

If you are using certain versions of conda, it may fail to build CuPy with error ``g++: error: unrecognized command line option ‘-R’``.
This is due to a bug in conda (see `conda/conda#6030 <https://github.com/conda/conda/issues/6030>`_ for details).
If you encounter this problem, please upgrade your conda.

.. _install_cudnn:

Installing cuDNN and NCCL
~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend installing cuDNN and NCCL using binary packages (i.e., using ``apt`` or ``yum``) provided by NVIDIA.

If you want to install tar-gz version of cuDNN and NCCL, we recommend installing it under the ``CUDA_PATH`` directory.
For example, if you are using Ubuntu, copy ``*.h`` files to ``include`` directory and ``*.so*`` files to ``lib64`` directory::

  $ cp /path/to/cudnn.h $CUDA_PATH/include
  $ cp /path/to/libcudnn.so* $CUDA_PATH/lib64

The destination directories depend on your environment.

If you want to use cuDNN or NCCL installed in another directory, please use ``CFLAGS``, ``LDFLAGS`` and ``LD_LIBRARY_PATH`` environment variables before installing CuPy::

  $ export CFLAGS=-I/path/to/cudnn/include
  $ export LDFLAGS=-L/path/to/cudnn/lib
  $ export LD_LIBRARY_PATH=/path/to/cudnn/lib:$LD_LIBRARY_PATH

.. _install_cuda:

Working with Custom CUDA Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have installed CUDA on the non-default directory or multiple CUDA versions on the same host, you may need to manually specify the CUDA installation directory to be used by CuPy.

CuPy uses the first CUDA installation directory found by the following order.

#. ``CUDA_PATH`` environment variable.
#. The parent directory of ``nvcc`` command. CuPy looks for ``nvcc`` command from ``PATH`` environment variable.
#. ``/usr/local/cuda``

For example, you can build CuPy using non-default CUDA directory by ``CUDA_PATH`` environment variable::

  $ CUDA_PATH=/opt/nvidia/cuda pip install cupy

.. note::

   CUDA installation discovery is also performed at runtime using the rule above.
   Depending on your system configuration, you may also need to set ``LD_LIBRARY_PATH`` environment variable to ``$CUDA_PATH/lib64`` at runtime.

CuPy always raises ``cupy.cuda.compiler.CompileException``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If CuPy raises a ``CompileException`` for almost everything, it is possible that CuPy cannot detect CUDA installed on your system correctly.
The following are error messages commonly observed in such cases.

* ``nvrtc: error: failed to load builtins``
* ``catastrophic error: cannot open source file "cuda_fp16.h"``
* ``error: cannot overload functions distinguished by return type alone``
* ``error: identifier "__half_raw" is undefined``

Please try setting ``LD_LIBRARY_PATH`` and ``CUDA_PATH`` environment variable.
For example, if you have CUDA installed at ``/usr/local/cuda-9.2``::

  $ export CUDA_PATH=/usr/local/cuda-9.2
  $ export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

Also see :ref:`install_cuda`.

.. _install_gcc6:

Build fails on Ubuntu 16.04, CentOS 6 or 7
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to build CuPy from source on systems with legacy GCC (g++-5 or earlier), you need to manually set up g++-6 or later and configure ``NVCC`` environment variable.

On Ubuntu 16.04::

  $ sudo add-apt-repository ppa:ubuntu-toolchain-r/test
  $ sudo apt update
  $ sudo apt install g++-6
  $ export NVCC="nvcc --compiler-bindir gcc-6"

On CentOS 6 / 7::

  $ sudo yum install centos-release-scl
  $ sudo yum install devtoolset-7-gcc-c++
  $ source /opt/rh/devtoolset-7/enable
  $ export NVCC="nvcc --compiler-bindir gcc"


Using CuPy on AMD GPU (experimental)
====================================

CuPy has an experimental support for AMD GPU (ROCm).

Requirements
------------

* `AMD GPU supported by ROCm <https://github.com/RadeonOpenCompute/ROCm#Hardware-and-Software-Support>`_

* `ROCm <https://rocmdocs.amd.com/en/latest/index.html>`_: v4.3 / v5.0
    * See the `ROCm Installation Guide <https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html>`_ for details.

The following ROCm libraries are required:

::

  $ sudo apt install hipblas hipsparse rocsparse rocrand rocthrust rocsolver rocfft hipcub rocprim rccl

Environment Variables
---------------------

When building or running CuPy for ROCm, the following environment variables are effective.

* ``ROCM_HOME``: directory containing the ROCm software (e.g., ``/opt/rocm``).

Docker
------

You can try running CuPy for ROCm using Docker.

::

  $ docker run -it --device=/dev/kfd --device=/dev/dri --group-add video cupy/cupy-rocm

.. _install_hip:

Installing Binary Packages
--------------------------

Wheels (precompiled binary packages) are available for Linux (x86_64).
Package names are different depending on your ROCm version.

.. list-table::
   :header-rows: 1

   * - ROCm
     - Command
   * - v4.3
     - ``$ pip install cupy-rocm-4-3``
   * - v5.0
     - ``$ pip install cupy-rocm-5-0``

Building CuPy for ROCm From Source
----------------------------------

To build CuPy from source, set the ``CUPY_INSTALL_USE_HIP``, ``ROCM_HOME``, and ``HCC_AMDGPU_TARGET`` environment variables.
(``HCC_AMDGPU_TARGET`` is the ISA name supported by your GPU.
Run ``rocminfo`` and use the value displayed in ``Name:`` line (e.g., ``gfx900``).
You can specify a comma-separated list of ISAs if you have multiple GPUs of different architectures.)

::

  $ export CUPY_INSTALL_USE_HIP=1
  $ export ROCM_HOME=/opt/rocm
  $ export HCC_AMDGPU_TARGET=gfx906
  $ pip install cupy

.. note::

  If you don't specify the ``HCC_AMDGPU_TARGET`` environment variable, CuPy will be built for the GPU architectures available on the build host.
  This behavior is specific to ROCm builds; when building CuPy for NVIDIA CUDA, the build result is not affected by the host configuration.

Limitations
-----------

The following features are not available due to the limitation of ROCm or because that they are specific to CUDA:

* CUDA Array Interface
* cuTENSOR
* Handling extremely large arrays whose size is around 32-bit boundary (HIP is known to fail with sizes `2**32-1024`)
* Atomic addition in FP16 (``cupy.ndarray.scatter_add`` and ``cupyx.scatter_add``)
* Multi-GPU FFT and FFT callback
* Some random number generation algorithms
* Several options in RawKernel/RawModule APIs: Jitify, dynamic parallelism
* Per-thread default stream

The following features are not yet supported:

* Sparse matrices (``cupyx.scipy.sparse``)
* cuDNN (hipDNN)
* Hermitian/symmetric eigenvalue solver (``cupy.linalg.eigh``)
* Polynomial roots (uses Hermitian/symmetric eigenvalue solver)
* Splines in ``cupyx.scipy.interpolate`` (``make_interp_spline``, spline modes of ``RegularGridInterpolator``/``interpn``), as they depend on sparse matrices.

The following features may not work in edge cases (e.g., some combinations of dtype):

.. note::
   We are investigating the root causes of the issues. They are not necessarily
   CuPy's issues, but ROCm may have some potential bugs.

* ``cupy.ndarray.__getitem__`` (`#4653 <https://github.com/cupy/cupy/pull/4653>`_)
* ``cupy.ix_`` (`#4654 <https://github.com/cupy/cupy/pull/4654>`_)
* Some polynomial routines (`#4758 <https://github.com/cupy/cupy/pull/4758>`_, `#4759 <https://github.com/cupy/cupy/pull/4759>`_)
* ``cupy.broadcast`` (`#4662 <https://github.com/cupy/cupy/pull/4662>`_)
* ``cupy.convolve`` (`#4668 <https://github.com/cupy/cupy/pull/4668>`_)
* ``cupy.correlate`` (`#4781 <https://github.com/cupy/cupy/pull/4781>`_)
* Some random sampling routines (``cupy.random``, `#4770 <https://github.com/cupy/cupy/pull/4770>`_)
* ``cupy.linalg.einsum``
* ``cupyx.scipy.ndimage`` and ``cupyx.scipy.signal`` (`#4878 <https://github.com/cupy/cupy/pull/4878>`_, `#4879 <https://github.com/cupy/cupy/pull/4879>`_, `#4880 <https://github.com/cupy/cupy/pull/4880>`_)
