"""Backend ``libs`` package (cuBLAS / cuSOLVER / cuSPARSE ... bindings).

Most of these libraries have no direct CUDA counterpart on Ascend: the CANN
``ops-*`` packages provide API-compatible replacements that live under
:mod:`cupy.backends.ascend.api`.  ``__getattr__`` (PEP 562) maps the CUDA module
names onto those replacements lazily, so code written against the CUDA backend
keeps working unchanged on both backends, e.g.
``from cupy_backends.cuda.libs import cusparse``.  Modules that are compiled for
the active backend are found normally and never reach this hook.
"""

#: CUDA module name -> Ascend implementation
#: (see ``cupy/backends/ascend/api/`` for the available modules).
_ASCEND_ALIASES = {
    # ops-sparse: emulates the cuSPARSE generic + legacy APIs
    'cusparse': 'cupy.backends.ascend.api.acl_sparse',
}


def __getattr__(name):
    target = _ASCEND_ALIASES.get(name)
    if target is not None:
        from cupy.backends.backend import is_ascend
        if is_ascend:
            import importlib
            import sys
            module = importlib.import_module(target)
            # Cache as a regular package attribute and register the submodule
            # under both spellings (`cupy.` and the legacy `cupy_backends.`).
            globals()[name] = module
            for key in (f'{__name__}.{name}', f'cupy_backends.cuda.libs.{name}'):
                sys.modules.setdefault(key, module)
            return module
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
