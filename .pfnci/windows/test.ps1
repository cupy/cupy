Param(
    [String]$cuda,
    [String]$python,
    [String]$numpy,
    [String]$scipy,
    [String]$test
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_error_handler.ps1"

. "$PSScriptRoot\_flexci.ps1"


function DownloadCache([String]$gcs_dir, [String]$cupy_kernel_cache_file) {
    pushd $Env:USERPROFILE
    echo "Downloading kernel cache..."
    gsutil -m -q cp "$gcs_dir/$cupy_kernel_cache_file" .
    if (-not $?) {
        echo "*** Kernel cache unavailable ($gcs_dir/$cupy_kernel_cache_file)"
    } else {
        echo "Extracting kernel cache..."
        RunOrDie 7z x -aoa $cupy_kernel_cache_file
        rm $cupy_kernel_cache_file
    }
    popd
}

function UploadCache([String]$gcs_dir, [String]$cupy_kernel_cache_file) {
    # Limit *.cubin cache by total size.
    # Note: --expiry cannot be used as access time is not updated on Windows.
    # TODO: Also clean up ~/.cupy/callback_cache/*.ltoir
    echo "Trimming kernel cache..."
    RunOrDie python .pfnci\trim_cupy_kernel_cache.py --max-size 4831838208 --rm

    pushd $Env:USERPROFILE
    # -mx=0 ... no compression
    # -mtc=on ... preserve timestamp
    echo "Compressing kernel cache..."
    RunOrDie 7z a -tzip -mx=0 -mtc=on $cupy_kernel_cache_file .cupy
    echo "Uploading kernel cache..."
    RunOrDie gsutil -m -q cp $cupy_kernel_cache_file $gcs_dir/
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
    RunOrDie python -m pip install "numpy==$numpy.*" "scipy==$scipy.*" "Cython==3.*"
    python -m pip install --no-build-isolation ".[all,test]" -v > cupy_build_log.txt
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
    $cache_gcs_dir = "gs://tmp-asia-pfn-public-ci/cupy-ci/cache"
    $cache_pr_gcs_dir = "${cache_gcs_dir}-pr-" + (GetPullRequestNumber)

    DownloadCache "${cache_gcs_dir}" "${cache_archive}"
    if ($is_pull_request) {
        DownloadCache "${cache_pr_gcs_dir}" "${cache_archive}"
    }

    if (-Not $is_pull_request) {
        $Env:CUPY_TEST_FULL_COMBINATION = "1"
    }
    # Skip full test for these CUDA versions as compilation seems so slow
    if (($cuda -eq "12.0") -or ($cuda -eq "12.1") -or ($cuda -eq "12.2")) {
        $Env:CUPY_TEST_FULL_COMBINATION = "0"
    }

    pushd tests
    echo "CuPy Configuration:"
    RunOrDie python -c "import cupy; print(cupy); cupy.show_config()"
    echo "Running test..."
    # TODO(leofang): allow larger/adjustable timeout?
    $test_retval = RunWithTimeout -timeout 18000 -output ../cupy_test_log.txt -- python -m pytest -rfEX @pytest_opts .
    popd

    if ($is_pull_request) {
        UploadCache "${cache_pr_gcs_dir}" "${cache_archive}"
    } else {
        UploadCache "${cache_gcs_dir}" "${cache_archive}"
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
