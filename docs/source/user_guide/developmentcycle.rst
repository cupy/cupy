
Development Cycle
-----------------

This section explains the development process of CuPy.
Before contributing to CuPy, it is strongly recommended to understand the development cycle.

Versioning
~~~~~~~~~~

The versioning of CuPy follows `PEP 440 <https://www.python.org/dev/peps/pep-0440/>`_ and a part of `Semantic versioning <https://semver.org/>`_.
The version number consists of three or four parts: ``X.Y.Zw`` where ``X`` denotes the **major version**, ``Y`` denotes the **minor version**, ``Z`` denotes the **revision number**, and the optional ``w`` denotes the prelease suffix.
While the major, minor, and revision numbers follow the rule of semantic versioning, the pre-release suffix follows PEP 440 so that the version string is much friendly with Python eco-system.

**Note that a major update basically does not contain compatibility-breaking changes from the last release candidate (RC).**
This is not a strict rule, though; if there is a critical API bug that we have to fix for the major version, we may add breaking changes to the major version up.

As for the backward compatibility, see :doc:`user_guide/compatibility`.


.. _contrib-release-cycle:

Release Cycle
~~~~~~~~~~~~~

The first one is the track of **stable versions**, which is a series of revision updates for the latest major version.
The second one is the track of **development versions**, which is a series of pre-releases for the upcoming major version.

Consider that ``X.0.0`` is the latest major version and ``Y.0.0``, ``Z.0.0`` are the succeeding major versions.
Then, the timeline of the updates is depicted by the following table.

========== =========== =========== ============
   Date       ver X       ver Y       ver Z
========== =========== =========== ============
  0 weeks    X.0.0rc1    --         --
  4 weeks    X.0.0       Y.0.0a1    --
  8 weeks    X.1.0*      Y.0.0b1    --
 12 weeks    X.2.0*      Y.0.0rc1   --
 16 weeks    --          Y.0.0      Z.0.0a1
========== =========== =========== ============

(* These might be revision releases)

The dates shown in the left-most column are relative to the release of ``X.0.0rc1``.
In particular, each revision/minor release is made four weeks after the previous one of the same major version, and the pre-release of the upcoming major version is made at the same time.
Whether these releases are revision or minor is determined based on the contents of each update.

Note that there are only three stable releases for the versions ``X.x.x``.
During the parallel development of ``Y.0.0`` and ``Z.0.0a1``, the version ``Y`` is treated as an **almost-stable version** and ``Z`` is treated as a development version.

If there is a critical bug found in ``X.x.x`` after stopping the development of version ``X``, we may release a hot-fix for this version at any time.

We create a milestone for each upcoming release at GitHub.
The GitHub milestone is basically used for collecting the issues and PRs resolved in the release.

.. _contrib-git-branches:

Git Branches
~~~~~~~~~~~~

The ``master`` branch is used to develop pre-release versions.
It means that **alpha, beta, and RC updates are developed at the** ``master`` **branch**.
This branch contains the most up-to-date source tree that includes features newly added after the latest major version.

The stable version is developed at the individual branch named as ``vN`` where "N" reflects the version number (we call it a *versioned branch*).
For example, v1.0.0, v1.0.1, and v1.0.2 will be developed at the ``v1`` branch.

**Notes for contributors:**
When you send a pull request, you basically have to send it to the ``master`` branch.
If the change can also be applied to the stable version, a core team member will apply the same change to the stable version so that the change is also included in the next revision update.

If the change is only applicable to the stable version and not to the ``master`` branch, please send it to the versioned branch.
We basically only accept changes to the latest versioned branch (where the stable version is developed) unless the fix is critical.

If you want to make a new feature of the ``master`` branch available in the current stable version, please send a *backport PR* to the stable version (the latest ``vN`` branch).
See the next section for details.

*Note: a change that can be applied to both branches should be sent to the* ``master`` *branch.*
*Each release of the stable version is also merged to the development version so that the change is also reflected to the next major version.*

Feature Backport PRs
~~~~~~~~~~~~~~~~~~~~

We basically do not backport any new features of the development version to the stable versions.
If you desire to include the feature to the current stable version and you can work on the backport work, we welcome such a contribution.
In such a case, you have to send a backport PR to the latest ``vN`` branch.
**Note that we do not accept any feature backport PRs to older versions because we are not running quality assurance workflows (e.g. CI) for older versions so that we cannot ensure that the PR is correctly ported.**

There are some rules on sending a backport PR.

- Start the PR title from the prefix **[backport]**.
- Clarify the original PR number in the PR description (something like "This is a backport of #XXXX").
- (optional) Write to the PR description the motivation of backporting the feature to the stable version.

Please follow these rules when you create a feature backport PR.

Note: PRs that do not include any changes/additions to APIs (e.g. bug fixes, documentation improvements) are usually backported by core dev members.
It is also appreciated to make such a backport PR by any contributors, though, so that the overall development proceeds more smoothly!
