
from cupy.xpu.function cimport Module

cpdef Module compile_with_cache(str source, tuple options=*, arch=*,
                                cachd_dir=*, prepend_cupy_headers=*,
                                backend=*, translate_cucomplex=*,
                                enable_cooperative_groups=*,
                                name_expressions=*, log_stream=*,
                                bint jitify=*)
