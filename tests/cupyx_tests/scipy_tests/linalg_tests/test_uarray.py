from cupy import testing
import cupyx.scipy.linalg._uarray


@testing.with_requires('scipy>=1.5')
def test_implements_names():
    # With the newest SciPy, the decorator `@implements` must find the
    # matching scipy functions.
    assert not cupyx.scipy.linalg._uarray._notfound
