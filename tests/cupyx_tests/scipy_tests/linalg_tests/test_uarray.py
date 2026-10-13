from __future__ import annotations

from cupy import testing
import cupyx.scipy.linalg._uarray


@testing.with_requires('scipy')
def test_implements_names():
    # With the newest SciPy, the decorator `@implements` must find the
    # matching scipy functions.
    notfound = list(cupyx.scipy.linalg._uarray._notfound)
    if not testing.installed('scipy<1.17'):
        # scipy 1.17 removed scipy.linalg.kron; cupyx.scipy.linalg.kron stays.
        notfound = [n for n in notfound if n != 'kron']
    assert not notfound
