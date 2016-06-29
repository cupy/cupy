Chainer Contribution Guide
==========================

This is a guide for all contributions to Chainer.
The development of Chainer is running on `the official repository at GitHub <https://github.com/pfnet/chainer>`_.
Anyone that wants to register an issue or to send a pull request should read through this document.

Classification of Contributions
-------------------------------

There are several ways to contribute to Chainer community:

1. Registering an issue
2. Sending a pull request (PR)
3. Sending a question to `Chainer User Group <https://groups.google.com/forum/#!forum/chainer>`_
4. Open-sourcing an external example
5. Writing a post about Chainer

This document mainly focuses on 1 and 2, though other contributions are also appreciated.

Release and Milestone
---------------------

We are using `GitHub Flow <http://scottchacon.com/2011/08/31/github-flow.html>`_ as our basic working process.
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
* **Other**: other issues and PRs

Issues and PRs are labeled by these categories.
This classification is often reflected into its corresponding release category: Feature issues/PRs are contained into minor/major releases and NoCompat issues/PRs are contained into major releases, while other issues/PRs can be contained into any releases including revision ones.

On registering an issue, write precise explanations on what you want Chainer to be.
Bug reports must include necessary and sufficient conditions to reproduce the bugs.
Feature requests must include **what** you want to do (and **why** you want to do, if needed).
You can contain your thoughts on **how** to realize it into the feature requests, though **what** part is most important for discussions.

.. warning::

   If you have a question on usages of Chainer, it is highly recommended to send a post to `Chainer User Group <https://groups.google.com/forum/#!forum/chainer>`_ instead of the issue tracker.
   The issue tracker is not a place to share knowledge on practices.
   We may redirect question issues to Chainer User Group.

If you can write code to fix an issue, send a PR to the master branch.
Before writing your code for PRs, read through the :ref:`coding-guide`.
The description of any PR must contain a precise explanation of **what** and **how** you want to do; it is the first documentation of your code for developers, a very important part of your PR.

Once you send a PR, it is automatically tested on `Travis CI <https://travis-ci.org/pfnet/chainer/>`_ for Linux and Mac OS X, and on `AppVeyor <https://ci.appveyor.com/project/pfnet/chainer>`_ for Windows.
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

We use `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ and a part of `OpenStack Style Guidelines <http://docs.openstack.org/developer/hacking/>`_ related to general coding style as our basic style guidelines.

To check your code, use ``flake8`` command installed by ``hacking`` package::

  $ pip install hacking
  $ flake8 path/to/your/code.py

The ``flake8`` command lets you know the part of your code not obeying our style guidelines.
Before sending a pull request, be sure to check that your code passes the ``flake8`` checking.

Note that ``flake8`` command is not perfect.
It does not check some of the style guidelines.
Here is a (not-complete) list of the rules that ``flake8`` cannot check.

* Relative imports are prohibited. [H304]
* Importing non-module symbols is prohibited.
* Import statements must be organized into three parts: standard libraries, third-party libraries, and internal imports. [H306]

In addition, we restrict the usage of *shortcut symbols* in our code base.
They are symbols imported by packages and sub-packages of ``chainer``.
For example, ``chainer.Variable`` is a shortcut of ``chainer.variable.Variable``.
**It is not allowed to use such shortcuts in the ``chainer`` library implementation**.
Note that you can still use them in ``tests`` and ``examples`` directories.
Also note that you should use shortcut names of CuPy APIs in Chainer implementation.

Once you send a pull request, your coding style is automatically checked by `Travis-CI <https://travis-ci.org/pfnet/chainer/>`_.
The reviewing process starts after the check passes.


Testing Guidelines
------------------

Testing is one of the most important part of your code.
You must test your code by unit tests following our testing guidelines.
Note that we are using the nose package and the mock package for testing, so install nose and mock before writing your code::

  $ pip install nose mock

In order to run unit tests at the repository root, you first have to build Cython files in place by running the following command::

  $ python setup.py develop

Once the Cython modules are built, you can run unit tests simply by running ``nosetests`` command at the repository root::

  $ nosetests

It requires CUDA by default.
In order to run unit tests that do not require CUDA, pass ``--attr='!gpu'`` option to the ``nosetests`` command::

  $ nosetests path/to/your/test.py --attr='!gpu'

Some GPU tests involve multiple GPUs.
If you want to run GPU tests with insufficient number of GPUs, specify the number of available GPUs by ``--eval-attr='gpu<N'`` where ``N`` is a concrete integer.
For example, if you have only one GPU, launch ``nosetests`` by the following command to skip multi-GPU tests::

  $ nosetests path/to/gpu/test.py --eval-attr='gpu<2'

Tests are put into the ``tests/chainer_tests``, ``tests/cupy_tests`` and ``tests/install_tests`` directories.
These have the same structure as that of ``chainer``, ``cupy`` and ``install`` directories, respectively.
In order to enable test runner to find test scripts correctly, we are using special naming convention for the test subdirectories and the test scripts.

* The name of each subdirectory of ``tests`` must end with the ``_tests`` suffix.
* The name of each test script must start with the ``test_`` prefix.

Following this naming convention, you can run all the tests by just typing ``nosetests`` at the repository root::

  $ nosetests

Or you can also specify a root directory to search test scripts from::

  $ nosetests tests/chainer_tests  # to just run tests of Chainer
  $ nosetests tests/cupy_tests     # to just run tests of CuPy
  $ nosetests tests/install_tests  # to just run tests of installation modules

If you modify the code related to existing unit tests, you must run appropriate commands.

.. note::
   CuPy tests include type-exhaustive test functions which take long time to execute.
   If you are running tests on a multi-core machine, you can parallelize the tests by following options::

     $ nosetests --processes=12 --process-timeout=1000 tests/cupy_tests

   The magic numbers can be modified for your usage.
   Note that some tests require many CUDA compilations, which require a bit long time.
   Without the ``process-timeout`` option, the timeout is set shorter, causing timeout failures for many test cases.

There are many examples of unit tests under the ``tests`` directory.
They simply use the ``unittest`` package of the standard library.

Even if your patch includes GPU-related code, your tests should not fail without GPU capability.
Test functions that require CUDA must be tagged by the ``chainer.testing.attr.gpu`` decorator (or ``cupy.testing.attr.gpu`` for testing CuPy APIs)::

  import unittest
  from chainer.testing import attr

  class TestMyFunc(unittest.TestCase):
      ...

      @attr.gpu
      def test_my_gpu_func(self):
          ...

The functions tagged by the ``gpu`` decorator are skipped if ``--attr='!gpu'`` is given.
We also have the ``chainer.testing.attr.cudnn`` decorator to let ``nosetests`` know that the test depends on cuDNN.

The test functions decorated by ``gpu`` must not depend on multiple GPUs.
In order to write tests for multiple GPUs, use ``chainer.testing.attr.multi_gpu()`` or ``cupy.testing.attr.multi_gpu()`` decorators instead::

  import unittest
  from chainer.testing import attr

  class TestMyFunc(unittest.TestCase):
      ...

      @attr.multi_gpu(2)  # specify the number of required GPUs here
      def test_my_two_gpu_func(self):
          ...

Once you send a pull request, your code is automatically tested by `Travis-CI <https://travis-ci.org/pfnet/chainer/>`_ **with --attr='!gpu' option**.
Since Travis-CI does not support CUDA, we cannot check your CUDA-related code automatically.
The reviewing process starts after the test passes.
Note that reviewers will test your code without the option to check CUDA-related code.

.. note::
   Some of numerically unstable tests might cause errors irrelevant to your changes.
   In such a case, we ignore the failures and go on to the review process, so do not worry about it.
