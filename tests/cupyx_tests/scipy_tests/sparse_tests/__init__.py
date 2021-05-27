import pytest

import cupy


if cupy.cuda.runtime.is_hip:
    pytest.skip('HIP sparse support is not yet ready',
                allow_module_level=True)
