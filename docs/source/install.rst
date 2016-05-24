Install Guide
=============

.. _before_install:

Before installing Chainer
-------------------------

We recommend these platforms.

* `Ubuntu <http://www.ubuntu.com/>`_ 14.04 LTS 64bit
* `CentOS <https://www.centos.org/>`_ 7 64bit

Chainer is supported on Python 2.7.6+, 3.4.3+, 3.5.1+.
Chainer uses C++ compiler such as g++.
You need to install it before installing Chainer.
This is typical installation method for each platform::

  # Ubuntu 14.04
  $ apt-get install g++

  # CentOS 7
  $ yum install gcc-c++

If you use old ``setuptools``, upgrade it::

  $ pip install -U setuptools


Install Chainer
---------------

Chainer depends on these Python packages:

* `NumPy <http://www.numpy.org/>`_ 1.9, 1.10, 1.11
* `Six <https://pythonhosted.org/six/>`_ 1.9

CUDA support

* `CUDA <https://developer.nvidia.com/cuda-zone>`_ 6.5, 7.0, 7.5
* `filelock <https://filelock.readthedocs.org>`_

cuDNN support

* `cuDNN <https://developer.nvidia.com/cudnn>`_ v2, v3, v4, v5

Caffe model support

* `Protocol Buffers <https://developers.google.com/protocol-buffers/>`_
* protobuf>=3.0.0 is required for Py3

All these libraries are automatically installed with ``pip`` or ``setup.py``.

HDF5 serialization is optional

* `h5py <http://www.h5py.org/>`_ 2.5.0


Install Chainer via pip
~~~~~~~~~~~~~~~~~~~~~~~

We recommend to install Chainer via pip::

  $ pip install chainer


Install Chainer from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use ``setup.py`` to install Chainer from source::

  $ tar zxf chainer-x.x.x.tar.gz
  $ cd chainer-x.x.x
  $ python setup.py install


.. _install_error:

When an error occurs...
~~~~~~~~~~~~~~~~~~~~~~~

Use ``-vvvv`` option with ``pip`` command.
That shows all logs of installation. It may helps you::

  $ pip install chainer -vvvv


Install Chainer with CUDA
~~~~~~~~~~~~~~~~~~~~~~~~~

You need to install CUDA Toolkit before installing Chainer.
If you have CUDA in a default directory or set ``CUDA_PATH`` correctly, Chainer installer finds CUDA automatically::

  $ pip install chainer


.. note::

   Chainer installer looks up ``CUDA_PATH`` environment variable first.
   If it is empty, the installer looks for ``nvcc`` command from ``PATH`` environment variable and use its parent directory as the root directory of CUDA installation.
   If ``nvcc`` command is also not found, the installer tries to use the default directory for Ubuntu ``/usr/local/cuda``.


If you installed CUDA into a non-default directory, you need to specify the directory with ``CUDA_PATH`` environment variable::

  $ CUDA_PATH=/opt/nvidia/cuda pip install chainer


.. warning::

   If you want to use ``sudo`` to install Chainer, note that ``sudo`` command initializes all environment variables.
   Please specify ``CUDA_PATH`` environment variable inside ``sudo`` like this::

      $ sudo CUDA_PATH=/opt/nvidia/cuda pip install chainer


.. _install_cudnn:

Install Chainer with CUDA and cuDNN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cuDNN is a library for Deep Neural Networks that NVIDIA provides.
Chainer can use cuDNN.
If you want to enable cuDNN, install cuDNN and CUDA before installing Chainer.
We recommend you to install cuDNN to CUDA directory.
For example if you uses Ubuntu Linux, copy ``.h`` files to ``include`` directory and ``.so`` files to ``lib64`` directory::

  $ cp /path/to/cudnn.h $CUDA_PATH/include
  $ cp /path/to/libcudnn.so* $CUDA_PATH/lib64

The destination directories depend on your environment.


Install Chainer for developers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chainer uses Cython (>=0.23).
Developers need to use Cython to regenerate C++ sources from ``pyx`` files.
We recommend to use ``pip`` with ``-e`` option for editable mode::

  $ pip install -U cython
  $ cd /path/to/chainer/source
  $ pip install -e .

Users need not to install Cython as a distribution package of Chainer only contains generated sources.


Support HDF5 serialization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Install h5py manually to activate HDF5 serialization.
This feature is optional::

  $ pip install h5py

Before installing h5py, you need to install libhdf5.
It depends on your environment::

  # Ubuntu 14.04
  $ apt-get install libhdf5-dev

  # CentOS 7
  $ yum -y install epel-release
  $ yum install hdf5-devel


Uninstall Chainer
-----------------

Use pip to uninstall Chainer::

  $ pip uninstall chainer

.. note::

   When you upgrade Chainer, ``pip`` sometimes installed various version of Chainer in ``site-packages``.
   Please uninstall it repeatedly until ``pip`` returns an error.


Upgrade Chainer
---------------

Just use ``pip`` with ``-U`` option::

  $ pip install -U chainer


Reinstall Chainer
-----------------

If you want to reinstall Chainer, please uninstall Chainer and then install it.
We recommend to use ``--no-cache-dir`` option as ``pip`` sometimes uses cache::

  $ pip uninstall chainer
  $ pip install chainer --no-cache-dir

When you install Chainer without CUDA, and after that you want to use CUDA, please reinstall Chainer.
You need to reinstall Chainer when you want to upgrade CUDA.


What "recommend" means?
-----------------------

We tests Chainer automatically with Jenkins.
All supported environments are tested in this environment.
We cannot guarantee that Chainer works on other environments.


FAQ
---

The installer says "hdf5.h is not found"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You don't have libhdf5.
Please install hdf5.
See :ref:`before_install`.


MemoryError happens
~~~~~~~~~~~~~~~~~~~

You maybe failed to install Cython.
Please install it manually.
See :ref:`install_error`.


Examples says "cuDNN is not enabled"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You failed to build Chainer with cuDNN.
If you don't need cuDNN, ignore this message.
Otherwise, retry to install Chainer with cuDNN.
``-vvvv`` option helps you.
See :ref:`install_cudnn`.
