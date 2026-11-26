from __future__ import annotations

import os
import sys
import warnings

import cupy


if sys.version_info >= (3, 12):
    _skip_prefixes = (os.path.dirname(cupy.__file__),)

    def warn_external(message, category, stacklevel=1):
        # Keep this wrapper in Python so warnings.warn sees its frame before
        # applying the minimum stacklevel of 2 for skip_file_prefixes.
        warnings.warn(message, category, stacklevel=stacklevel,
                      skip_file_prefixes=_skip_prefixes)

else:
    # Python 3.11 doesn't support skip_file_prefixes
    warn_external = warnings.warn
