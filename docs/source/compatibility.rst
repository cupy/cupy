API Compatibility Policy
========================

This document expresses the design policy on compatibilities of Chainer APIs.
Development team should obey this policy on deciding to add, extend, and change APIs and their behaviors.

This document is written for both users and developers.
Users can decide the level of dependencies on Chainer’s implementations in their codes based on this document.
Developers should read through this document before creating pull requests that contain changes on the interface.
Note that this document may contain ambiguities on the level of supported compatibilities.


Targeted Versions
-----------------

This policy is applied to Chainer of versions v1.5.1 and higher.
Note that this policy is not applied to Chainer of lower versions.


Versioning and Backward Compatibilities
---------------------------------------

The updates of Chainer are classified into three levels: major, minor, and revision.
These types have distinct levels of backward compatibilities.

- **Major update** contains disruptive changes that break the backward compatibility.
- **Minor update** contains addition and extension to the APIs keeping the supported backward compatibility.
- **Revision update** contains improvements on the API implementations without changing any API specifications.

Note that we do not support full backward compatibility, which is almost infeasible for Python-based APIs, since there is no way to completely hide the implementation details.


Processes to Break Backward Compatibilities
-------------------------------------------

Deprecation, Dropping, and Its Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any APIs may be *deprecated* at some minor updates.
In such a case, the deprecation note is added to the API documentation, and the API implementation is changed to fire deprecation warning (if possible).
There should be another way to reimplement the same things previously written with the deprecated APIs.

Any APIs may be marked as *to be dropped in the future*.
In such a case, the dropping is stated in the documentation with the major version number on which the API is planned to be dropped, and the API implementation is changed to fire the future warning (if possible).

The actual dropping should be done through the following steps:

- Make the API deprecated.
  At this point, users should not need the deprecated API in their new application codes.
- After that, mark the API as *to be dropped in the future*.
  It must be done in the minor update different from that of the deprecation.
- At the major version announced in the above update, drop the API.

Consequently, it takes at least two minor versions to drop any APIs after the first deprecation.

API Changes and Its Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any APIs may be marked as *to be changed in the future* for changes without backward compatibility.
In such a case, the change is stated in the documentation with the version number on which the API is planned to be changed, and the API implementation is changed to fire the future warning on the certain usages.

The actual change should be done in the following steps:

- Announce that the API will be changed in the future.
  At this point, the actual version of change need not be accurate.
- After the announcement, mark the API as *to be changed in the future* with version number of planned changes.
  At this point, users should not use the marked API in their new application codes.
- At the major update announced in the above update, change the API.


Supported Backward Compatibility
--------------------------------

This section defines backward compatibilities that minor updates must maintain.

Documented Interface
~~~~~~~~~~~~~~~~~~~~

Chainer has the official API documentation.
Many applications can be written based on the documented features.
We support backward compatibilities of documented features.
In other words, codes only based on the documented features run correctly with minor/revision-updated versions.

Developers are encouraged to use apparent names for objects of implementation details.
For example, attributes outside of the documented APIs should have one or more underscores at the prefix of their names.

Undocumented behaviors
~~~~~~~~~~~~~~~~~~~~~~

Behaviors of Chainer implementation not stated in the documentation are undefined.
Undocumented behaviors are not guaranteed to be stable between different minor/revision versions.

Minor update may contain changes to undocumented behaviors.
For example, suppose an API X is added at the minor update.
In the previous version, attempts to use X cause AttributeError.
This behavior is not stated in the documentation, so this is undefined.
Thus, adding the API X in minor version is permissible.

Revision update may also contain changes to undefined behaviors.
Typical example is a bug fix.
Another example is an improvement on implementation, which may change the internal object structures not shown in the documentation.
As a consequence, **even revision updates do not support compatibility of pickling, unless the full layout of pickled objects is clearly documented.**

Documentation Error
~~~~~~~~~~~~~~~~~~~

Compatibility is basically determined based on the documentation, though it sometimes contains errors.
It may make the APIs confusing to assume the documentation always stronger than the implementations.
We therefore may fix the documentation errors in any updates that may break the compatibility in regard to the documentation.

.. note::
   Developers MUST NOT fix the documentation and implementation of the same functionality at the same time in revision updates as "bug fix".
   Such a change completely breaks the backward compatibility.
   If you want to fix the bugs in both sides, first fix the documentation to fit it into the implementation, and start the API changing procedure described above.

Object Attributes and Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Object attributes and properties are sometimes replaced by each other at minor updates.
It does not break the user codes, except the codes depend on how the attributes and properties are implemented.

Functions and Methods
~~~~~~~~~~~~~~~~~~~~~

Methods may be replaced by callable attributes keeping the compatibility of parameters and return values in minor updates.
It does not break the user codes, except the codes depend on how the methods and callable attributes are implemented.

Exceptions and Warnings
~~~~~~~~~~~~~~~~~~~~~~~

The specifications of raising exceptions are considered as a part of standard backward compatibilities.
No exception is raised in the future versions with correct usages that the documentation allows, unless the API changing process is completed.

On the other hand, warnings may be added at any minor updates for any APIs.
It means minor updates do not keep backward compatibility of warnings.

Model Format Compatibility
--------------------------

Objects serialized by official serializers that Chainer provides are correctly loaded with the higher (future) versions.
They might not be correctly loaded with Chainer of the lower versions.

.. note::
   Current serialization APIs do not support versioning (at least in v1.6.1).
   It prevents us from introducing changes in the layout of objects that support serialization.
   We are discussing about introducing versioning in serialization APIs.

Installation Compatibility
--------------------------

The installation process is another concern of compatibilities.
We support environmental compatibilities in the following ways.

- Any changes of dependent libraries that force modifications on the existing environments must be done in major updates.
  Such changes include following cases:

  - dropping supported versions of dependent libraries (e.g. dropping cuDNN v2)
  - adding new mandatory dependencies (e.g. adding h5py to setup_requires)

- Supporting optional packages/libraries may be done in minor updates (e.g. supporting h5py in optional features).

.. note::
   The installation compatibility does not guarantee that all the features of Chainer correctly run on supported environments.
   It may contain bugs that only occurs in certain environments.
   Such bugs should be fixed in some updates.
