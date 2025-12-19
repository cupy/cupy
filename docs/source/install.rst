Installation
============

Requirements
------------

* `NVIDIA CUDA GPU <https://developer.nvidia.com/cuda-gpus>`_ with the Compute Capability 3.0 or larger.

* `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_: v12.0 / v12.1 / v12.2 / v12.3 / v12.4 / v12.5 / v12.6 / v12.8 / v12.9 / v13.0

    * If you have multiple versions of CUDA Toolkit installed, CuPy will automatically choose one of the CUDA installations.
      See :ref:`install_cuda` for details.

    * This requirement is optional if you install both CUDA and CuPy from either PyPI or conda-forge. However, you still need to have a compatible
      driver installed for your GPU. See :ref:`install_cupy_from_pypi` and :ref:`install_cupy_from_conda_forge` for details.

* `Python <https://python.org/>`_: v3.10 / v3.11 / v3.12 / v3.13 / v3.14

.. note::

   Currently, CuPy is tested against  `Ubuntu <https://www.ubuntu.com/>`_ 20.04 LTS / 22.04 LTS (x86_64), `CentOS <https://www.centos.org/>`_ 7 / 8 (x86_64) and Windows Server 2016 (x86_64).

Python Dependencies
~~~~~~~~~~~~~~~~~~~

NumPy/SciPy-compatible API in CuPy v14 is based on NumPy 2.3 and SciPy 1.16, and has been tested against the following versions:

* `NumPy <https://numpy.org/>`_: v2.0 / v2.1 / v2.2 / v2.3

* `SciPy <https://scipy.org/>`_ (*optional*): v1.14 / v1.15 / v1.16

    * Required only when copying sparse matrices from GPU to CPU (see :doc:`../reference/scipy_sparse`.)

* `Optuna <https://optuna.org/>`_ (*optional*): v3.x / v4.x

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

* `NCCL <https://developer.nvidia.com/nccl>`_: v2.16 / v2.17 / v2.18 / v2.19 / v2.20 / v2.21 / v2.22 / v2.25 / v2.26 / v2.27

    * The library to perform collective multi-GPU / multi-node computations.

* `cuSPARSELt <https://docs.nvidia.com/cuda/cusparselt/>`_: v0.8.0 / v0.8.1

    * The library to accelerate sparse matrix-matrix multiplication.


Installing CuPy
---------------

.. _install_cupy_from_pypi:

Installing CuPy from PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~

Wheels (precompiled binary packages) are available for Linux and Windows.
Package names are different depending on your CUDA Toolkit version.

.. list-table::
   :header-rows: 1

   * - CUDA
     - Command
   * - **v12.x** (x86_64 / aarch64)
     - ``pip install cupy-cuda12x``
   * - **v13.x** (x86_64 / aarch64)
     - ``pip install cupy-cuda13x``

By default, the above command only installs CuPy itself, assuming a CUDA Toolkit is already installed on the system. To use NVIDIA's CUDA component wheels
(so as to quickly spinning up a fresh virtual environment without installing a system-wide CUDA Toolkit -- only the CUDA driver is needed -- and allowing
smaller installation footprint and better interoperability with other Python GPU libraries), you can pass ``[ctk]`` to install them all as
optional dependencies, e.g.::

   $ pip install "cupy-cuda12x[ctk]"

.. note::

   To enable features provided by additional CUDA libraries (cuTENSOR / NCCL), you need to install them manually.
   If you installed CuPy via PyPI, the easiest way to setup these libraries is to use ``cutensor-cuXX`` and ``nvidia-nccl-cuXX`` PyPI packages, e.g.:::

    $ pip install "cutensor-cu13==2.3.*"
    $ pip install "nvidia-nccl-cu13==2.27.*"

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
The above ``pip install`` instruction is compatible with ``conda`` environments. Alternatively, for both Linux (x86_64, aarch64) and
Windows once the CUDA driver is correctly set up, you can also install CuPy from the conda-forge channel::

    $ conda install -c conda-forge cupy

and ``conda`` will install a pre-built CuPy binary package for you, along with all needed CUDA runtime libraries.
It is not necessary to install CUDA Toolkit in advance, and is equivalent to the wheel counterpart ``pip install cupy-cudaXX[ctk]``,
but with everything installed from conda-forge instead of PyPI.

If you aim at minimizing the installation footprint, you can install the ``cupy-core`` package::

    $ conda install -c conda-forge cupy-core

which only depends on ``numpy``. None of the CUDA libraries will be installed this way, and it is your responsibility to install the needed
dependencies yourself, either from conda-forge or elsewhere. This is equivalent to the wheel counterpart ``pip install cupy-cudaXX`` (without any extras).

Conda has a built-in mechanism to determine and install the latest version of any CUDA components supported by your CUDA driver.
However, if for any reason you need to force-install a particular CUDA version (say 12.9), you can do::

    $ conda install -c conda-forge cupy cuda-version=12.9

.. note::

    cuTENSOR and NCCL are available on conda-forge as optional dependencies. The following command can install them all at once::

        $ conda install -c conda-forge cupy cutensor nccl

    Each of them can also be installed separately as needed.

.. note::

    If you encounter any problem with CuPy installed from conda-forge, please feel free to report to `cupy-feedstock
    <https://github.com/conda-forge/cupy-feedstock/issues>`_, and we will help investigate if it is just a packaging
    issue in conda-forge's recipe or a real issue in CuPy.

.. note::

    If you did not install CUDA Toolkit by yourself, for CUDA 12 and above, ``nvcc`` can be installed on a per-``conda`` environment basis via

        $ conda install -c conda-forge cuda-nvcc


.. _install_cupy_from_source:

Installing CuPy from Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use of wheel packages is recommended whenever possible.
However, if wheels cannot meet your requirements (e.g., you are running non-Linux environment or want to use a version of CUDA / NCCL not supported by wheels), you can also build CuPy from source.

.. note::

   When installing CuPy from source, features provided by additional CUDA libraries will be disabled if these libraries are not available at the build time.
   See :ref:`install_nccl` for the instructions.

.. note::

   If you upgrade or downgrade the version of CUDA Toolkit, NCCL or cuTENSOR, you may need to reinstall CuPy.
   See :ref:`install_reinstall` for details.

You can install the latest stable release version of the `CuPy source package <https://pypi.python.org/pypi/cupy>`_ via ``pip``.

::

  $ pip install cupy

If you want to install the latest development version of CuPy from a cloned Git repository::

  $ git clone --recursive https://github.com/cupy/cupy.git
  $ cd cupy
  $ pip install .

.. note::

   Cython 3 is required to build CuPy from source.
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

.. _install_nccl:

Installing NCCL
~~~~~~~~~~~~~~~

We recommend installing NCCL using binary packages (i.e., using ``apt`` or ``yum``) provided by NVIDIA.

If you want to install tar-gz version of NCCL, we recommend installing it under the ``CUDA_PATH`` directory.
For example, if you are using Ubuntu, copy ``*.h`` files to ``include`` directory and ``*.so*`` files to ``lib64`` directory.

The destination directories depend on your environment.

If you want to use NCCL installed in another directory, please use ``CFLAGS``, ``LDFLAGS`` and ``LD_LIBRARY_PATH`` environment variables before installing CuPy::

  $ export CFLAGS=-I/path/to/nccl/include
  $ export LDFLAGS=-L/path/to/nccl/lib
  $ export LD_LIBRARY_PATH=/path/to/nccl/lib:$LD_LIBRARY_PATH

.. _install_cuda:

Working with Custom CUDA Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have installed CUDA on the non-default directory or multiple CUDA versions on the same host, you may need to manually specify the CUDA installation directory to be used by CuPy.

CuPy uses the first CUDA installation directory found by the following order.

#. ``cuda-pathfinder``'s `documented search order <https://nvidia.github.io/cuda-python/cuda-pathfinder/latest/generated/cuda.pathfinder.load_nvidia_dynamic_lib.html>`_.
#. ``CUDA_PATH`` environment variable.
#. The parent directory of ``nvcc`` command. CuPy looks for ``nvcc`` command from ``PATH`` environment variable.
#. ``/usr/local/cuda``

For example, you can build CuPy using non-default CUDA directory by ``CUDA_PATH`` environment variable::

  $ CUDA_PATH=/opt/nvidia/cuda pip install cupy

.. note::

   CUDA installation discovery is also performed at runtime using the rule above.
   Depending on your system configuration, you may also need to set ``LD_LIBRARY_PATH`` environment variable to ``$CUDA_PATH/lib64`` at runtime.

CuPy always raises ``NVRTC_ERROR_COMPILATION (6)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On CUDA 12.2 or later, CUDA Runtime header files are required to compile kernels in CuPy.
If CuPy raises a ``NVRTC_ERROR_COMPILATION`` with the error message saying ``catastrophic error: cannot open source file "vector_types.h"`` for almost everything, it is possible that CuPy cannot find the header files on your system correctly.

This problem does not happen if you have installed CuPy from conda-forge (i.e., ``conda install -c conda-forge cupy``), as the package ``cuda-cudart-dev_<platform>`` that contains the needed headers is correctly installed as a dependency.
Please report to the CuPy repository if you encounter issues with Conda-installed CuPy.

If you have installed CuPy from PyPI (i.e., ``pip install cupy-cuda12x``), you can install CUDA headers by running ``pip install "nvidia-cuda-runtime-cu12==12.X.*"`` where ``12.X`` is the version of your CUDA installation.
Once headers from the package is recognized, ``cupy.show_config()`` will display the path as ``CUDA Extra Include Dirs``:

.. code:: console

  $ python -c 'import cupy; cupy.show_config()'
  ...
  CUDA Extra Include Dirs      : []
  ...
  NVRTC Version                : (12, 6)
  ...
  $ pip install "nvidia-cuda-runtime-cu12==12.6.*"
  ...
  $ python -c 'import cupy; cupy.show_config()'
  ...
  CUDA Extra Include Dirs      : ['.../site-packages/nvidia/cuda_runtime/include']
  ...

Alternatively, you can install CUDA headers system-wide (``/usr/local/cuda``) using NVIDIA's Apt (or DNF) repository.
Install the ``cuda-cudart-dev-12-X`` package where ``12-X`` is the version of your ``cuda-cudart`` package, e.g.:

.. code:: console

  $ apt list "cuda-cudart-*"
  cuda-cudart-12-6/now 12.6.68-1 amd64 [installed,local]
  $ sudo apt install "cuda-cudart-dev-12-6"

CuPy always raises ``cupy.cuda.compiler.CompileException``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If CuPy raises a ``CompileException`` for almost everything, it is possible that CuPy cannot detect CUDA installed on your system correctly.
The following are error messages commonly observed in such cases.

* ``nvrtc: error: failed to load builtins``
* ``catastrophic error: cannot open source file "cuda_fp16.h"``
* ``error: cannot overload functions distinguished by return type alone``
* ``error: identifier "__half_raw" is undefined``
* ``error: no instance of overloaded function "__half::__half" matches the specified type``

Please try setting ``LD_LIBRARY_PATH`` and ``CUDA_PATH`` environment variable.
For example, if you have CUDA installed at ``/usr/local/cuda-12.6``::

  $ export CUDA_PATH=/usr/local/cuda-12.6
  $ export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

Also see :ref:`install_cuda`.


Using CuPy on AMD GPU (experimental)
------------------------------------

CuPy has an experimental support for AMD GPU (ROCm).

Requirements
~~~~~~~~~~~~

* `AMD GPU supported by ROCm <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_

* `ROCm <https://rocm.docs.amd.com/en/latest/>`_ 7.x
    * See the `Installation Guide <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html>`_ for details.

The following ROCm libraries are required:

::

  $ sudo apt install hipblas hipsparse rocsparse rocrand hiprand rocthrust rocsolver rocfft hipfft hipcub rocprim rccl roctracer-dev

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

When building or running CuPy for ROCm, the following environment variables are necessary to set.

* ``ROCM_HOME``: directory containing the ROCm software (e.g., ``/opt/rocm``).

.. note::
    It is recommended to always have ROCm installed to `/opt/rocm`. Non standard install locations have a tendency
    to break some functionality.

Docker
~~~~~~

You can try running CuPy for ROCm using Docker.

::

  $ docker run -it --device=/dev/kfd --device=/dev/dri --group-add video cupy/cupy-rocm

.. _install_hip:

Installing Binary Packages
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   ROCm binary packages (wheels) and ROCm Docker images are unavailable in recent CuPy versions (v13.4.0+).
   AMD is currently hosting ROCm 6.4 wheels and can be installed with `pip install amd-cupy --extra-index-url=https://pypi.amd.com/simple`.
   This wheel supports PTDS, CAI, and other misc bug fixes in addition to other v13.4 functionality.
   We are currently working on improving packaging to improve this situation. Follow `#8607 <https://github.com/cupy/cupy/issues/8607>`_ for the latest status.


Building CuPy for ROCm From Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build CuPy from source, set the ``CUPY_INSTALL_USE_HIP``, ``ROCM_HOME``, and ``HCC_AMDGPU_TARGET`` environment variables.
(``HCC_AMDGPU_TARGET`` is the ISA name supported by your GPU.
Run ``rocminfo`` and use the value displayed in ``Name:`` line (e.g., ``gfx942``).
You can specify a comma-separated list of ISAs if you have multiple GPUs of different architectures.)

::

  $ export CUPY_INSTALL_USE_HIP=1
  $ export ROCM_HOME=/opt/rocm
  $ export HCC_AMDGPU_TARGET=gfx908
  $ pip install cupy

.. note::

  If you don't specify the ``HCC_AMDGPU_TARGET`` environment variable, CuPy will be built for the GPU architectures available on the build host.
  This behavior is specific to ROCm builds; when building CuPy for NVIDIA CUDA, the build result is not affected by the host configuration.

Limitations
~~~~~~~~~~~

The following features are not available due to the limitation of ROCm or because that they are specific to CUDA:

* cuTENSOR
* Handling extremely large arrays whose size is around 32-bit boundary (HIP is known to fail with sizes `2**32-1024`)
* Atomic addition in FP16 (``cupy.ndarray.scatter_add`` and ``cupyx.scatter_add``)
* Multi-GPU FFT and FFT callback
* Some random number generation algorithms
* Several options in RawKernel/RawModule APIs: Jitify, dynamic parallelism

The following features are not yet supported:

* Sparse matrices (``cupyx.scipy.sparse``)
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
