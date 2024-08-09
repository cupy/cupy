Param(
    [String]$cuda,
    [String]$python,
    [String]$test
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_error_handler.ps1"

. "$PSScriptRoot\_flexci.ps1"


$cache_gcs_dir = "gs://tmp-asia-pfn-public-ci/cupy-ci/cache"

function DownloadCache([String]$cupy_kernel_cache_file) {
    pushd $Env:USERPROFILE
    echo "Downloading kernel cache..."
    gsutil -m -q cp "$cache_gcs_dir/$cupy_kernel_cache_file" .
    if (-not $?) {
        echo "*** Kernel cache unavailable"
    } else {
        echo "Extracting kernel cache..."
        RunOrDie 7z x $cupy_kernel_cache_file
        rm $cupy_kernel_cache_file
    }
    popd
}

function UploadCache([String]$cupy_kernel_cache_file) {
    # Maximum 1 GB
    echo "Trimming kernel cache..."
    RunOrDie python .pfnci\trim_cupy_kernel_cache.py --max-size 1000000000 --rm

    pushd $Env:USERPROFILE
    # -mx=0 ... no compression
    # -mtc=on ... preserve timestamp
    echo "Compressing kernel cache..."
    RunOrDie 7z a -tzip -mx=0 -mtc=on $cupy_kernel_cache_file .cupy
    echo "Uploading kernel cache..."
    RunOrDie gsutil -m -q cp $cupy_kernel_cache_file $cache_gcs_dir/
    popd
}

function PublishTestResults {
    # Upload test results
    echo "Uploading test results..."
    $artifact_id = $Env:CI_JOB_ID
    RunOrDie gsutil -m -q cp cupy_build_log.txt cupy_test_log.txt "gs://chainer-artifacts-pfn-public-ci/cupy-ci/$artifact_id/"
    echo "Build Log:"
    echo "https://storage.googleapis.com/chainer-artifacts-pfn-public-ci/cupy-ci/$artifact_id/cupy_build_log.txt"
    echo "Test Log:"
    echo "https://storage.googleapis.com/chainer-artifacts-pfn-public-ci/cupy-ci/$artifact_id/cupy_test_log.txt"
}

function Main {
    PrioritizeFlexCIDaemon
    EnableLongPaths

    # Enable symbolic links and re-checkout
    git config core.symlinks true
    git reset --hard

    # Setup environment
    echo "Using CUDA $cuda and Python $python"
    ActivateCUDA $cuda
    if ($cuda -eq "10.2") {
        ActivateCuDNN "8.6" $cuda
    } else {
        ActivateCuDNN "8.8" $cuda
    }
    ActivateNVTX1
    ActivatePython $python

    # Setup build environment variables
    $Env:CUPY_NUM_BUILD_JOBS = "16"
    $Env:CUPY_NVCC_GENERATE_CODE = "current"
    echo "Environment:"
    RunOrDie cmd.exe /C set

    # Build
    echo "Setting up test environment"
    RunOrDie python -V
    RunOrDie python -m pip install -U pip setuptools wheel
    RunOrDie python -m pip freeze

    echo "Building..."
    $build_retval = 0
    RunOrDie python -m pip install -U "numpy" "scipy==1.12.*"
    python -m pip install ".[all,test]" -v > cupy_build_log.txt
    if (-not $?) {
        $build_retval = $LastExitCode
    }

    echo "------------------------------------------------------------------------------------------"
    echo "Last 10 lines from the build output:"
    Get-Content cupy_build_log.txt -Tail 10
    echo "------------------------------------------------------------------------------------------"

    if ($build_retval -ne 0) {
        echo "n/a" > cupy_test_log.txt
        PublishTestResults
        throw "Build failed with status $build_retval"
    }

    $Env:CUPY_TEST_GPU_LIMIT = $Env:GPU
    $Env:CUPY_DUMP_CUDA_SOURCE_ON_ERROR = "1"

    # Unit test
    if ($test -eq "build") {
        return
    } elseif ($test -eq "test") {
        $pytest_opts = "-m", '"not slow"'
    } elseif ($test -eq "slow") {
        $pytest_opts = "-m", "slow"
    } else {
        throw "Unsupported test target: $target"
    }

    $base_branch = (Get-Content .pfnci\BRANCH)
    $is_pull_request = IsPullRequestTest
    $cache_archive = "windows-cuda${cuda}-${base_branch}.zip"

    DownloadCache "${cache_archive}"

    if (-Not $is_pull_request) {
        $Env:CUPY_TEST_FULL_COMBINATION = "1"
    }

    # Install dependency for cuDNN 8.3+
    echo ">> Installing zlib"
    InstallZLIB

    pushd tests
    echo "CuPy Configuration:"
    RunOrDie python -c "import cupy; print(cupy); cupy.show_config()"
    echo "Running test..."
    $test_retval = RunWithTimeout -timeout 32767 -output ../cupy_test_log.txt -- python -m pytest -rfEX @pytest_opts .
    popd

    if (-Not $is_pull_request) {
        UploadCache "${cache_archive}"
    }

    echo "------------------------------------------------------------------------------------------"
    echo "Last 10 lines from the test output:"
    Get-Content cupy_test_log.txt -Tail 10
    echo "------------------------------------------------------------------------------------------"

    PublishTestResults
    if ($test_retval -ne 0) {
        throw "Test failed with status $test_retval"
    }
}

Main
