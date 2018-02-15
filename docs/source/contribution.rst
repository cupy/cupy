Contribution Guide
==================

This is a guide for all contributions to CuPy.
The development of CuPy is running on `the official repository at GitHub <https://github.com/cupy/cupy>`_.
Anyone that wants to register an issue or to send a pull request should read through this document.

Classification of Contributions
-------------------------------

There are several ways to contribute to CuPy community:

1. Registering an issue
2. Sending a pull request (PR)
3. Sending a question to `CuPy User Group <https://groups.google.com/forum/#!forum/cupy>`_
4. Writing a post about CuPy

This document mainly focuses on 1 and 2, though other contributions are also appreciated.

Release and Milestone
---------------------

We are using `GitHub Flow <https://scottchacon.com/2011/08/31/github-flow.html>`_ as our basic working process.
In particular, we are using the master branch for our development, and releases are made as tags.

Releases are classified into three groups: major, minor, and revision.
This classification is based on following criteria:

- **Major update** contains disruptive changes that break the backward compatibility.
- **Minor update** contains additions and extensions to the APIs keeping the supported backward compatibility.
- **Revision update** contains improvements on the API implementations without changing any API specification.

The release classification is reflected into the version number x.y.z, where x, y, and z corresponds to major, minor, and revision updates, respectively.

We set a milestone for an upcoming release.
The milestone is of name 'vX.Y.Z', where the version number represents a revision release at the outset.
If at least one *feature* PR is merged in the period, we rename the milestone to represent a minor release (see the next section for the PR types).

See also :doc:`compatibility`.

Issues and PRs
--------------

Issues and PRs are classified into following categories:

* **Bug**: bug reports (issues) and bug fixes (PRs)
* **Enhancement**: implementation improvements without breaking the interface
* **Feature**: feature requests (issues) and their implementations (PRs)
* **NoCompat**: disrupts backward compatibility
* **Test**: test fixes and updates
* **Document**: document fixes and improvements
* **Example**: fixes and improvements on the examples
* **Install**: fixes installation script
* **Contribution-Welcome**: issues that we request for contribution (only issues are categorized to this)
* **Other**: other issues and PRs

Issues and PRs are labeled by these categories.
This classification is often reflected into its corresponding release category: Feature issues/PRs are contained into minor/major releases and NoCompat issues/PRs are contained into major releases, while other issues/PRs can be contained into any releases including revision ones.

On registering an issue, write precise explanations on what you want CuPy to be.
Bug reports must include necessary and sufficient conditions to reproduce the bugs.
Feature requests must include **what** you want to do (and **why** you want to do, if needed).
You can contain your thoughts on **how** to realize it into the feature requests, though **what** part is most important for discussions.

.. warning::

   If you have a question on usages of CuPy, it is highly recommended to send a post to `CuPy User Group <https://groups.google.com/forum/#!forum/cupy>`_ instead of the issue tracker.
   The issue tracker is not a place to share knowledge on practices.
   We may redirect question issues to CuPy User Group.

If you can write code to fix an issue, send a PR to the master branch.
Before writing your code for PRs, read through the :ref:`coding-guide`.
The description of any PR must contain a precise explanation of **what** and **how** you want to do; it is the first documentation of your code for developers, a very important part of your PR.

Once you send a PR, it is automatically tested on `Travis CI <https://travis-ci.org/cupy/cupy/>`_ for Linux and Mac OS X, and on `AppVeyor <https://ci.appveyor.com/project/cupy/cupy>`_ for Windows.
Your PR need to pass at least the test for Linux on Travis CI.
After the automatic test passes, some of the core developers will start reviewing your code.
Note that this automatic PR test only includes CPU tests.

.. note::

   We are also running continuous integration with GPU tests for the master branch.
   Since this service is running on our internal server, we do not use it for automatic PR tests to keep the server secure.


Even if your code is not complete, you can send a pull request as a *work-in-progress PR* by putting the ``[WIP]`` prefix to the PR title.
If you write a precise explanation about the PR, core developers and other contributors can join the discussion about how to proceed the PR.

.. _coding-guide:

Coding Guidelines
-----------------

We use `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ and a part of `OpenStack Style Guidelines <https://docs.openstack.org/developer/hacking/>`_ related to general coding style as our basic style guidelines.

To check your code, use ``autopep8`` and ``flake8`` command installed by ``hacking`` package::

  $ pip install autopep8 hacking
  $ autopep8 --global-config .pep8 path/to/your/code.py
  $ flake8 path/to/your/code.py

To check Cython code, use ``.flake8.cython`` configuration file::

  $ flake8 --config=.flake8.cython path/to/your/cython/code.pyx

The ``autopep8`` supports automatically correct Python code to conform to the PEP 8 style guide::

  $ autopep8 --in-place --global-config .pep8 path/to/your/code.py

The ``flake8`` command lets you know the part of your code not obeying our style guidelines.
Before sending a pull request, be sure to check that your code passes the ``flake8`` checking.

Note that ``flake8`` command is not perfect.
It does not check some of the style guidelines.
Here is a (not-complete) list of the rules that ``flake8`` cannot check.

* Relative imports are prohibited. [H304]
* Importing non-module symbols is prohibited.
* Import statements must be organized into three parts: standard libraries, third-party libraries, and internal imports. [H306]

In addition, we restrict the usage of *shortcut symbols* in our code base.
They are symbols imported by packages and sub-packages of ``cupy``.
For example, ``cupy.cuda.Device`` is a shortcut of ``cupy.cuda.device.Device``.
**It is not allowed to use such shortcuts in the ``cupy`` library implementation**.
Note that you can still use them in ``tests`` and ``examples`` directories.

Once you send a pull request, your coding style is automatically checked by `Travis-CI <https://travis-ci.org/cupy/cupy/>`_.
The reviewing process starts after the check passes.

The CuPy is designed based on NumPy's API design. CuPy's source code and documents contain the original NumPy ones.
Please note the followings when writing the document.

* In order to identify overlapping parts, it is preferable to add some remarks
  that this document is just copied or altered from the original one. It is
  also preferable to briefly explain the specification of the function in a
  short paragraph, and refer to the corresponding function in NumPy so that
  users can read the detailed document. However, it is possible to include a
  complete copy of the document with such a remark if users cannot summarize
  in such a way.
* If a function in CuPy only implements a limited amount of features in the
  original one, users should explicitly describe only what is implemented in
  the document.


Testing Guidelines
------------------

Testing is one of the most important part of your code.
You must test your code by unit tests following our testing guidelines.

Note that we are using pytest and mock package for testing, so install them before writing your code::

  $ pip install pytest mock

In order to run unit tests at the repository root, you first have to build Cython files in place by running the following command::

  $ python setup.py develop

.. note::

  When you modify ``*.pxd`` files, before running ``python setup.py develop``, you must clean ``*.cpp`` and ``*.so`` files once with the following command, because Cython does not automatically rebuild those files nicely::

    $ git clean -fdx

.. note::

  It's not officially supported, but you can use `ccache <https://ccache.samba.org/>`_ to reduce compilation time.
  On Ubuntu 16.04, you can set up as follows::

    $ sudo apt-get install ccache
    $ export PATH=/usr/lib/ccache:$PATH

  See `ccache <https://ccache.samba.org/>`_ for details.

  If you want to use ccache for nvcc, please install ccache v3.3 or later.
  You also need to set environment variable ``NVCC='ccache nvcc'``.

Once Cython modules are built, you can run unit tests by running the following command at the repository root::

  $ python -m pytest

CUDA must be installed to run unit tests.

Some GPU tests require cuDNN to run.
In order to skip unit tests that require cuDNN, specify ``-m='not cudnn'`` option::

  $ python -m pytest path/to/your/test.py -m='not cudnn'

Some GPU tests involve multiple GPUs.
If you want to run GPU tests with insufficient number of GPUs, specify the number of available GPUs to ``CUPY_TEST_GPU_LIMIT``.
For example, if you have only one GPU, launch ``pytest`` by the following command to skip multi-GPU tests::

  $ export CUPY_TEST_GPU_LIMIT=1
  $ python -m pytest path/to/gpu/test.py

Tests are put into the ``tests/cupy_tests`` and ``tests/install_tests`` directories.
These have the same structure as that of ``cupy`` and ``install`` directories, respectively.
In order to enable test runner to find test scripts correctly, we are using special naming convention for the test subdirectories and the test scripts.

* The name of each subdirectory of ``tests`` must end with the ``_tests`` suffix.
* The name of each test script must start with the ``test_`` prefix.

Following this naming convention, you can run all the tests by running the following command at the repository root::

  $ python -m pytest

Or you can also specify a root directory to search test scripts from::

  $ python -m pytest tests/cupy_tests     # to just run tests of CuPy
  $ python -m pytest tests/install_tests  # to just run tests of installation modules

If you modify the code related to existing unit tests, you must run appropriate commands.

There are many examples of unit tests under the ``tests`` directory.
They simply use the ``unittest`` package of the standard library.

Even if your patch includes GPU-related code, your tests should not fail without GPU capability.
Test functions that require CUDA must be tagged by the ``cupy.testing.attr.gpu``::

  import unittest
  from cupy.testing import attr

  class TestMyFunc(unittest.TestCase):
      ...

      @attr.gpu
      def test_my_gpu_func(self):
          ...

The functions tagged by the ``gpu`` decorator are skipped if ``CUPY_TEST_GPU_LIMIT=0`` environment variable is set.
We also have the ``cupy.testing.attr.cudnn`` decorator to let ``pytest`` know that the test depends on cuDNN.
The test functions decorated by ``cudnn`` are skipped if ``-m='not cudnn'`` is given.

The test functions decorated by ``gpu`` must not depend on multiple GPUs.
In order to write tests for multiple GPUs, use ``cupy.testing.attr.multi_gpu()`` or ``cupy.testing.attr.multi_gpu()`` decorators instead::

  import unittest
  from cupy.testing import attr

  class TestMyFunc(unittest.TestCase):
      ...

      @attr.multi_gpu(2)  # specify the number of required GPUs here
      def test_my_two_gpu_func(self):
          ...

Once you send a pull request, `Travis-CI <https://travis-ci.org/cupy/cupy/>`_ automatically checks if your code meets our coding guidelines described above.
Since Travis-CI does not support CUDA, we cannot run unit tests automatically.
The reviewing process starts after the automatic check passes.
Note that reviewers will test your code without the option to check CUDA-related code.

We leverage doctest as well. You can run doctest by typing ``make doctest`` at the ``docs`` directory::

  $ cd docs
  $ make doctest
