Tips and FAQs
=============

Installation
------------

I cannot install pycuda
~~~~~~~~~~~~~~~~~~~~~~~

You need to set ``PATH`` to CUDA bin path if you get the error below when you use ``pip install chainer-cuda-deps``::

   src/cpp/cuda.hpp:14:18: fatal error: cuda.h: No such file or directory
    #include <cuda.h>
                     ^
   compilation terminated.
   error: command 'x86_64-linux-gnu-gcc' failed with exit status 1

``chainer-cuda-deps`` only installs ``pycuda`` and other dependent libraries.
In ``setup.py`` of ``pycuda``, it checks the path of ``nvcc`` command and guesses the path of CUDA (https://github.com/inducer/pycuda/blob/v2015.1.2/setup.py#L30).
If ``setup.py`` couldn't find CUDA, it causes an error like that.

Please try to set ``PATH`` before ``pip install chainer-cuda-deps``.
If you use NVIDIA's official installer, ``nvcc`` command is located at ``/usr/local/cuda/bin``::

   $ export PATH=/usr/local/cuda/bin:$PATH
   $ pip install chainer-cuda-deps


I cannot install pycuda with ``sudo``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``sudo`` changes ``PATH`` environment variable for security.
You need to set ``PATH`` inside ``sudo``.
For example use ``sh`` command::

  $ sudo sh -c "PATH=/usr/local/cuda/bin:\$PATH pip install chainer-cuda-deps"

Or, install as a root user::

  $ su - root
  % export PATH=/usr/local/cuda/bin:$PATH
  % pip install chainer-cuda-deps

We recommend to install Chainer in your local environment with ``--user`` option if possible.
You don't need to use ``sudo`` in this case::

  $ pip install --user chainer-cuda-deps

You can also use `pyenv <https://github.com/yyuu/pyenv>`_ to create your local environment.
