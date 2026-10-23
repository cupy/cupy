"""Header-cache policy tests that do not require built CuPy extensions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.thread_unsafe(reason='Changes header-cache policy')


_ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    '_cupy_header_cache', _ROOT / 'cupy/_core/_header_cache.py')
header_cache = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(header_cache)


def _write(root, name, source):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


@pytest.fixture
def include_dir(tmp_path):
    _write(tmp_path, 'cupy/carray.cuh', '// common\n')
    for dtype in ('float16', 'bfloat16', 'structview', 'unstructured_void'):
        _write(tmp_path, f'cupy/{dtype}.cuh', '#include "cupy/carray.cuh"\n')
    _write(tmp_path, 'cupy/xsf/function.h', '// xsf\n')
    return tmp_path


@pytest.fixture
def cache(include_dir):
    return header_cache.build_header_cache(include_dir)


@pytest.mark.parametrize('group', [
    'xsf', 'float16', 'bfloat16', 'structview', 'unstructured_void',
])
def test_optional_update(include_dir, group):
    before = header_cache.build_header_cache(include_dir)
    name = 'cupy/xsf/function.h' if group == 'xsf' else f'cupy/{group}.cuh'
    with (include_dir / name).open('a') as f:
        f.write('// changed\n')
    after = header_cache.build_header_cache(include_dir)
    assert before['keys']['common'] == after['keys']['common']
    assert before['keys'][group] != after['keys'][group]
    assert before['keys']['full'] != after['keys']['full']
    source = f'#include <{name}>'
    assert (header_cache.get_cache_key(source, before)
            != header_cache.get_cache_key(source, after))
    assert (header_cache.get_cache_key('', before)
            == header_cache.get_cache_key('', after))
    # External wrappers must invalidate even though their contents are unknown.
    source = '#include "user_wrapper.h"'
    assert (header_cache.get_cache_key(source, before)
            != header_cache.get_cache_key(source, after))


def test_transitive_dependencies_and_cycles(include_dir):
    _write(include_dir, 'cupy/wrapper.h', '#include "detail/other.h"\n')
    _write(include_dir, 'cupy/detail/other.h',
           '#if SOME_MACRO\n#include "../float16.cuh"\n#endif\n')
    _write(include_dir, 'cupy/float16.cuh', '#include "bfloat16.cuh"\n')
    _write(include_dir, 'cupy/bfloat16.cuh',
           '#include "float16.cuh"\n#include "xsf/function.h"\n')
    before = header_cache.build_header_cache(include_dir)
    _write(include_dir, 'cupy/xsf/function.h', '// changed\n')
    after = header_cache.build_header_cache(include_dir)
    for group in ('common', 'float16', 'bfloat16', 'xsf'):
        assert before['keys'][group] != after['keys'][group]


def test_build_only_and_metadata(include_dir):
    _write(include_dir, 'cupy/_dlpack/dlpack.h', '// dlpack\n')
    _write(include_dir, 'cupy/type_dispatcher.cuh', '// dispatcher\n')
    before = header_cache.build_header_cache(include_dir)
    _write(include_dir, 'cupy/README.md', 'Documentation\n')
    _write(include_dir, 'cupy/_dlpack/LICENSE', 'License\n')
    assert before == header_cache.build_header_cache(include_dir)
    _write(include_dir, 'cupy/_dlpack/dlpack.h', '// changed\n')
    after = header_cache.build_header_cache(include_dir)
    assert before['keys']['common'] == after['keys']['common']
    assert before['keys']['build'] != after['keys']['build']
    assert before['keys']['full'] != after['keys']['full']


@pytest.mark.parametrize('directive', [
    '#include <cupy/float16.cuh>',
    '  # include "cupy/float16.cuh" // comment',
    '/* comment */ #include <cupy/float16.cuh>',
    '%:include <cupy/float16.cuh>',
    '??=include <cupy/float16.cuh>',
])
def test_literal_includes(cache, directive):
    key = header_cache.get_cache_key(directive, cache)
    assert f'float16:{cache["keys"]["float16"]}' in key
    assert 'bfloat16:' not in key


def test_comments(cache):
    source = ('/*\n#include <cupy/bfloat16.cuh>\n*/\n'
              '#include <cupy/float16.cuh>\n')
    key = header_cache.get_cache_key(source, cache)
    assert f'float16:{cache["keys"]["float16"]}' in key
    assert f'bfloat16:{cache["keys"]["bfloat16"]}' in key


@pytest.mark.parametrize('source', [
    '#include "user_wrapper.h"',
    '#define HEADER "user.h"\n#include HEADER',
    '#include_next <stdint.h>',
    '#include <cupy/unknown_header.h>',
    '#include "stdint.h"',
    '#include </tmp/wrapper.h>',
    '#include <C:/headers/wrapper.h>',
    r'#include <C:\headers\wrapper.h>',
    r'#include <\\server\share\wrapper.h>',
    '#include <C:wrapper.h>',
    '#include <../wrapper.h>',
    '#include <dir/../wrapper.h>',
    '#include <dir//wrapper.h>',
    '#include <a b.h>',
    '#/**/include <cupy/float16.cuh>',
    '#include /* comment */ <cupy/float16.cuh>',
    '#include "cupy/float16.cuh" "cupy/bfloat16.cuh"',
    '#include <cupy/float16.\ncuh>',
    '#include \\\n<cupy/float16.cuh>',
    '#include \\\r\n<cupy/float16.cuh>',
])
def test_unknown_include(cache, source):
    assert header_cache.get_cache_key(source, cache) == cache['keys']['full']


def test_strict_mode(cache, monkeypatch):
    monkeypatch.setattr(header_cache, '_error_on_unknown_header', True)
    with pytest.raises(header_cache._UnknownHeaderError,
                       match='Cannot classify kernel include'):
        header_cache.get_cache_key('#include "user.h"', cache)


def test_unknown_options(cache, monkeypatch):
    options = ('--pre-include=user.h',)
    assert (header_cache.get_cache_key('', cache, options)
            == cache['keys']['full'])
    monkeypatch.setattr(header_cache, '_error_on_unknown_header', True)
    with pytest.raises(header_cache._UnknownHeaderError,
                       match='compiler option'):
        header_cache.get_cache_key('', cache, options)


def test_known_options(cache):
    source = '#include <stdint.h>\n#include <cupy/float16.cuh>'
    options = ('-std=c++20', '--std=c++23', '-O3', '-Ofast', '-w',
               '-DUSER_MACRO=1', '--define-macro=OTHER=2')
    assert (header_cache.get_cache_key(source, cache, options)
            == header_cache.get_cache_key(source, cache))


def test_toolchain_header_names(cache):
    source = ('#include <new_system_header.h>\n'
              '#include <cuda/unknown_header>\n'
              '#include <dir/header++.h>\n')
    assert (header_cache.get_cache_key(source, cache)
            == header_cache.get_cache_key('', cache))


def test_jitify_update(include_dir):
    _write(include_dir, 'cupy/_jitify/jitify.hpp', '// jitify\n')
    before = header_cache.build_header_cache(include_dir)
    _write(include_dir, 'cupy/_jitify/jitify.hpp', '// changed\n')
    after = header_cache.build_header_cache(include_dir)
    assert (header_cache.get_cache_key('', before)
            == header_cache.get_cache_key('', after))
    assert (header_cache.get_cache_key('', before, jitify=True)
            != header_cache.get_cache_key('', after, jitify=True))


def test_unknown_owned_header(include_dir):
    _write(include_dir, 'cupy/wrapper.h', '#include "unknown.h"')
    with pytest.raises(header_cache._UnknownHeaderError,
                       match='cupy/wrapper.h:.*unknown.h'):
        header_cache.build_header_cache(include_dir)
