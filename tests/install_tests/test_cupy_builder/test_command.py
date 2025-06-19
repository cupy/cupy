from cupy_builder._command import filter_files_by_extension


def test_filter_files_by_extension():
    sources_cpp = ['a.cpp', 'b.cpp']
    sources_pyx = ['c.pyx']
    sources = sources_cpp + sources_pyx
    assert filter_files_by_extension(
        sources, '.cpp') == (sources_cpp, sources_pyx)
    assert filter_files_by_extension(
        sources, '.pyx') == (sources_pyx, sources_cpp)
    assert filter_files_by_extension(
        sources, '.cu') == ([], sources)
    assert filter_files_by_extension(
        sources_cpp, '.cpp') == (sources_cpp, [])
    assert filter_files_by_extension(
        sources_cpp, '.pyx') == ([], sources_cpp)
