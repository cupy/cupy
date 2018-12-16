# TODO(kmaehashi) this is to avoid DeprecationWarning raised during import
# not being filtered in Python 3.4. Remove this when we drop Python 3.4.

import sys
import warnings


ver = sys.version_info
if ver.major == 3 and ver.minor == 4:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # This imports `numpy.testing.*` module, which emits
        # DeprecationWarning during import (in NumPy 1.15+).
        try:
            from scipy import stats  # NOQA
        except ImportError:
            pass
