import os
import argparse
import re

TEST_ROOT = os.path.dirname("./")
CUPY_TESTS = [
    'array_api_tests',
    'binary_tests',
    'core_tests',
    'creation_tests',
    'cuda_tests',
    'fft_tests',
    'functional_tests',
    'indexing_tests',
    'io_tests',
    'lib_tests',
    'linalg_tests',
    'logic_tests',
    'manipulation_tests',
    'math_tests',
    'misc_tests',
    'padding_tests',
    'polynomial_tests',
    'prof_tests',
    'random_tests',
    'sorting_tests',
    'statistics_tests',
    'test_cublas.py',
    'testing_tests',
    'test_init.py',
    'test_ndim.py',
    'test_numpy_interop.py',
    'test_type_routines.py',
    'test_typing.py',
]

CUPYX_TESTS = [
    'distributed_tests',
    'fallback_mode_tests',
    'jit_tests',
    'linalg_tests',
    'profiler_tests',
    'scipy_tests/fftpack_tests',
    'scipy_tests/fft_tests',
    'scipy_tests/interpolate_tests',
    'scipy_tests/linalg_tests',
    'scipy_tests/ndimage_tests',
    'scipy_tests/signal_tests',
    'scipy_tests/sparse_tests',
    'scipy_tests/spatial_tests',
    'scipy_tests/special_tests',
    'scipy_tests/stats_tests',
    'scipy_tests/test_get_array_module.py',
    'signal_tests',
    'test_cudnn.py',
    'test_cupyx.py',
    'test_cusolver.py',
    'test_cusparse.py',
    'test_cutensor.py',
    'test_lapack.py',
    'test_optimize.py',
    'test_pinned_array.py',
    'test_rsqrt.py',
    'test_runtime.py',
    'test_time.py',
    'tools_tests',
]

TEST_SUITES = [
    'cupy_tests',
    'cupyx_tests',
    'example_tests',
    'install_tests',
    'import_tests',
    'typing_tests'
]

def run_all_tests(pytest_opts):
    initial_cmd = 'CUPY_TEST_GPU_LIMIT=4 CUPY_INSTALL_USE_HIP=1 ' + \
        'python3 -m pytest -k "not compile_cuda" ' + pytest_opts
    for test_suite in TEST_SUITES:
        if test_suite == "cupy_tests":
            for cupy_test in CUPY_TESTS:
                cmd = initial_cmd + TEST_ROOT + "/cupy_tests/" + cupy_test
                print("Running : " + cmd)
                os.system(cmd)
        elif test_suite == "cupyx_tests":
            for cupyx_test in CUPYX_TESTS:
                cmd = initial_cmd + TEST_ROOT + "/cupyx_tests/" + cupyx_test
                print("Running : " + cmd)
                os.system(cmd)
        else:
            cmd = initial_cmd + TEST_ROOT + "/" + test_suite
            print("Running : " + cmd)
            os.system(cmd)

def main():
    all_tests = args.all_tests
    if all_tests:
        run_all_tests(args.pytest_opts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-tests", action="store_true",
                        default=True, required=False, help="Run all tests")
    parser.add_argument("--pytest-opts", type=str,
                        default="", required=False, help="Options for pytest")
    args = parser.parse_args()

    main()
