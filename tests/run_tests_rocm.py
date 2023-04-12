import os
import sys
import argparse
import re

TEST_ROOT = os.path.dirname(os.path.abspath(__file__))
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
]

def parse_test_log_and_get_summary(test_name):
    fs = open("/root/.cache/cupy-tests/cupy_test.log", 'r')
    lines = fs.readlines()
    fs.close()

    count = ""
    summary = ""
    pattern = "^=*"
    for j in range(len(lines)):
        line = lines[j].rstrip()
        if ("collecting ..." in line):
            count = line.split("collected")[1].split("items")[0].strip()
        if ("==" in line):
            summary = re.split(pattern, line)[1].split("=")[0]
    test_data = test_name + "|" + count + "|" + summary
    return test_data

def run_all_tests():
    initial_cmd = 'CUPY_TEST_GPU_LIMIT=4 CUPY_INSTALL_USE_HIP=1 pytest -vvv -k "not compile_cuda and not fft_allocate" -m "not slow" '
    os.system("mkdir -p ~/.cache/cupy-tests")
    test_summary = []
    for test_suite in TEST_SUITES:
        if test_suite == "cupy_tests":
            for cupy_test in CUPY_TESTS:
                cmd = initial_cmd + TEST_ROOT + "/cupy_tests/" + cupy_test + " | tee ~/.cache/cupy-tests/cupy_test.log"
                print ("Running : " + cmd)
                os.system(cmd)
                test_name = "tests/cupy_tests/" + cupy_test
                test_summary.append(parse_test_log_and_get_summary(test_name))
        elif test_suite == "cupyx_tests":
            for cupyx_test in CUPYX_TESTS:
                cmd = initial_cmd + TEST_ROOT + "/cupyx_tests/" + cupyx_test + " | tee ~/.cache/cupy-tests/cupy_test.log"
                print ("Running : " + cmd)
                os.system(cmd)
                test_name = "tests/cupyx_tests/" + cupyx_test
                test_summary.append(parse_test_log_and_get_summary(test_name))
        else:
            cmd = initial_cmd + TEST_ROOT + "/" + test_suite + " | tee ~/.cache/cupy-tests/cupy_test.log"
            print (cmd)
            os.system(cmd)
            test_name = "tests/" + test_suite
            test_summary.append(parse_test_log_and_get_summary(test_name))

    return test_summary

def main():
    all_tests = args.all_tests
    if all_tests:
        test_summary = run_all_tests()
        print ("---------------------- TEST SUMMARY ------------------")
        for j in range(len(test_summary)):
            print (test_summary[j])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-tests", action="store_true", default=True, required=False, help="Run all tests");
    args = parser.parse_args()

    main()
