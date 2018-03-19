Installation Guide
==================

.. contents:: :local:

Recommended Environments
------------------------

We recommend these Linux distributions.

* `Ubuntu <https://www.ubuntu.com/>`_ 14.04/16.04 LTS 64bit
* `CentOS <https://www.centos.org/>`_ 7 64bit

The following versions of Python can be used: 2.7.6+, 3.4.3+, 3.5.1+, and 3.6.0+.

.. warning::

   If you are using certain versions of conda, it may fail to build CuPy with error
   ``g++: error: unrecognized command line option ‘-R’``.
   This is due to a bug in conda (see `conda/conda#6030 <https://github.com/conda/conda/issues/6030>`_ for details).
   If you encounter this problem, please downgrade or upgrade it.

.. note::

   We are testing CuPy automatically with Jenkins, where all the above *recommended* environments are tested.
   We cannot guarantee that CuPy works on other environments including Windows and macOS, even if CuPy looks running correctly.

CuPy uses C++ compiler such as g++.
You need to install it before installing CuPy.
This is typical installation method for each platform::

  # Ubuntu 14.04
  $ apt-get install g++

  # CentOS 7
  $ yum install gcc-c++

If you use old ``setuptools``, upgrade it::

  $ pip install -U setuptools


Dependencies
------------

Before installing CuPy, we recommend to upgrade ``setuptools`` if you are using an old one::

  $ pip install -U setuptools

The following Python packages are required to install CuPy.
The latest version of each package will automatically be installed if missing.

* `NumPy <http://www.numpy.org/>`_ 1.9, 1.10, 1.11, 1.12, 1.13
* `Six <https://pythonhosted.org/six/>`_ 1.9+

In addition, you need to install `CUDA <https://developer.nvidia.com/cuda-zone>`_.
The following versions of CUDA can be used: 7.0, 7.5, 8.0, 9.0 and 9.1.
You need a GPU with Compute Capability of at least 3.0.

Optional Libraries
~~~~~~~~~~~~~~~~~~

The following libraries are optional dependencies.
CuPy will enable these features only if they are installed.

* `cuDNN <https://developer.nvidia.com/cudnn>`_ v4, v5, v5.1, v6, v7
* `NCCL <https://github.com/NVIDIA/nccl>`_ v1.3+

Install CuPy
------------

Install CuPy via pip
~~~~~~~~~~~~~~~~~~~~

We recommend to install CuPy via pip::

  $ pip install cupy

.. note::

   All optional CUDA related libraries, cuDNN and NCCL, need to be installed before installing CuPy.
   After you update these libraries, please reinstall CuPy because you need to compile and link to the newer version of them.


Install CuPy using wheel
~~~~~~~~~~~~~~~~~~~~~~~~

We are experimentally providing wheels (precompiled binary packages) for Linux (x86_64) environment.
Package names are different depending on the CUDA version you have installed on your host::

  (For CUDA 8.0)
  $ pip install cupy-cuda80

  (For CUDA 9.0)
  $ pip install cupy-cuda90

  (For CUDA 9.1)
  $ pip install cupy-cuda91

When using wheels, please be careful not to install multiple CuPy packages at the same time.
These packages and ``cupy`` package conflict to each other.

.. note::

   The latest version of cuDNN and NCCL are included in these wheels.


Install CuPy from source
~~~~~~~~~~~~~~~~~~~~~~~~

The tarball of the source tree is available via ``pip download cupy`` or from `the release notes page <https://github.com/cupy/cupy/releases>`_.
You can install CuPy from the tarball::

  $ pip install cupy-x.x.x.tar.gz

You can also install the development version of CuPy from a cloned Git repository::

  $ git clone https://github.com/cupy/cupy.git
  $ cd cupy
  $ pip install .


.. _install_error:

When an error occurs...
~~~~~~~~~~~~~~~~~~~~~~~

Use ``-vvvv`` option with ``pip`` command.
That shows all logs of installation.
It may help you::

  $ pip install cupy -vvvv

If you are using wheel, make sure that you don't have multiple CuPy packages installed.
Only one cupy package (``cupy`` or ``cupy-cudaXX`` where XX is a CUDA version) can be installed::

  $ pip freeze | grep cupy

.. _install_cuda:

Install CuPy with CUDA
~~~~~~~~~~~~~~~~~~~~~~

You need to install CUDA Toolkit before installing CuPy.
If you have CUDA in a default directory or set ``CUDA_PATH`` correctly, CuPy installer finds CUDA automatically::

  $ pip install cupy


.. note::

   CuPy installer looks up ``CUDA_PATH`` environment variable first.
   If it is empty, the installer looks for ``nvcc`` command from ``PATH`` environment variable and use its parent directory as the root directory of CUDA installation.
   If ``nvcc`` command is also not found, the installer tries to use the default directory for Ubuntu ``/usr/local/cuda``.


If you installed CUDA into a non-default directory, you need to specify the directory with ``CUDA_PATH`` environment variable::

  $ CUDA_PATH=/opt/nvidia/cuda pip install cupy


If you want to use a custom ``nvcc`` compiler (For example, to use ``ccache`` ), please set ``NVCC`` environment variables before installing CuPy::

  export NVCC='ccache nvcc'


.. warning::

   If you want to use ``sudo`` to install CuPy, note that ``sudo`` command initializes all environment variables.
   Please specify ``CUDA_PATH`` environment variable inside ``sudo`` like this::

      $ sudo CUDA_PATH=/opt/nvidia/cuda pip install cupy


.. _install_cudnn:

Install CuPy with cuDNN and NCCL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cuDNN is a library for Deep Neural Networks that NVIDIA provides.
NCCL is a library for collective multi-GPU communication.
CuPy can use cuDNN and NCCL.
If you want to enable these libraries, install them before installing CuPy.
We recommend you to install developer library of deb package of cuDNN and NCCL.

If you want to install tar-gz version of cuDNN, we recommend you to install it to CUDA directory.
For example if you uses Ubuntu Linux, copy ``.h`` files to ``include`` directory and ``.so`` files to ``lib64`` directory::

  $ cp /path/to/cudnn.h $CUDA_PATH/include
  $ cp /path/to/libcudnn.so* $CUDA_PATH/lib64

The destination directories depend on your environment.

If you want to use cuDNN or NCCL installed in other directory, please use ``CFLAGS``, ``LDFLAGS`` and ``LD_LIBRARY_PATH`` environment variables before installing CuPy::

  export CFLAGS=-I/path/to/cudnn/include
  export LDFLAGS=-L/path/to/cudnn/lib
  export LD_LIBRARY_PATH=/path/to/cudnn/lib:$LD_LIBRARY_PATH

.. note::

   Use full paths for the environment variables.
   ``distutils`` that is used in the setup script does not parse the home directory mark ``~``.


Install CuPy for developers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuPy uses Cython (>=0.26.1).
Developers need to use Cython to regenerate C++ sources from ``pyx`` files.
We recommend to use ``pip`` with ``-e`` option for editable mode::

  $ pip install -U cython
  $ cd /path/to/cupy/source
  $ pip install -e .

Users need not to install Cython as a distribution package of CuPy only contains generated sources.


Uninstall CuPy
--------------

Use pip to uninstall CuPy::

  $ pip uninstall cupy

.. note::

   When you upgrade Chainer, ``pip`` sometimes install the new version without removing the old one in ``site-packages``.
   In this case, ``pip uninstall`` only removes the latest one.
   To ensure that Chainer is completely removed, run the above command repeatedly until ``pip`` returns an error.

.. note::

   If you installed CuPy using wheel, use ``pip uninstall cupy-cudaXX`` (where XX is a CUDA version number) instead.

Upgrade CuPy
------------

Just use ``pip`` with ``-U`` option::

  $ pip install -U cupy


Reinstall CuPy
--------------

If you want to reinstall CuPy, please uninstall CuPy and then install it.
We recommend to use ``--no-cache-dir`` option as ``pip`` sometimes uses cache::

  $ pip uninstall cupy
  $ pip install cupy --no-cache-dir

When you install CuPy without CUDA, and after that you want to use CUDA, please reinstall CuPy.
You need to reinstall CuPy when you want to upgrade CUDA.


Run CuPy with Docker
--------------------

We are providing the official Docker image.
Use `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ command to run CuPy image with GPU.
You can login to the environment with bash, and run the Python interpreter::

  $ nvidia-docker run -it cupy/cupy /bin/bash

Or run the interpreter directly::

  $ nvidia-docker run -it cupy/cupy /usr/bin/python


FAQ
---

Warning message "cuDNN is not enabled" appears
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You failed to build CuPy with cuDNN.
If you don't need cuDNN, ignore this message.
Otherwise, retry to install CuPy with cuDNN.
``-vvvv`` option helps you.
See :ref:`install_cudnn`.
