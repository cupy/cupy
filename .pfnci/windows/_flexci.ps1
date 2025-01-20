function ActivatePython($version) {
    if ($version -eq "3.5") {
        $pydir = "Python35"
    } elseif ($version -eq "3.6") {
        $pydir = "Python36"
    } elseif ($version -eq "3.7") {
        $pydir = "Python37"
    } elseif ($version -eq "3.8") {
        $pydir = "Python38"
    } elseif ($version -eq "3.9") {
        $pydir = "Python39"
    } elseif ($version -eq "3.10") {
        $pydir = "Python310"
    } elseif ($version -eq "3.11") {
        $pydir = "Python311"
    } elseif ($version -eq "3.12") {
        $pydir = "Python312"
    } else {
        throw "Unsupported Python version: $version"
    }
    $Env:PATH = "C:\Development\Python\$pydir;C:\Development\Python\$pydir\Scripts;" + $Env:PATH
}

function ActivateCUDA($version) {
    if ($version -eq "10.2") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V10_2
    } elseif ($version -eq "11.0") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V11_0
    } elseif ($version -eq "11.1") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V11_1
    } elseif ($version -eq "11.2") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V11_2
    } elseif ($version -eq "11.3") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V11_3
    } elseif ($version -eq "11.4") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V11_4
    } elseif ($version -eq "11.5") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V11_5
    } elseif ($version -eq "11.6") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V11_6
    } elseif ($version -eq "11.7") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V11_7
    } elseif ($version -eq "11.8") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V11_8
    } elseif ($version -eq "11.x") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V11_8
    } elseif ($version -eq "12.0") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_0
    } elseif ($version -eq "12.1") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_1
    } elseif ($version -eq "12.2") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_2
    } elseif ($version -eq "12.3") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_3
    } elseif ($version -eq "12.4") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_4
    } elseif ($version -eq "12.5") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_5
    } elseif ($version -eq "12.6") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_6
    } elseif ($version -eq "12.x") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_6
    } else {
        throw "Unsupported CUDA version: $version"
    }
    $Env:PATH = "$Env:CUDA_PATH\bin;" + $Env:PATH
}

function ActivateCuDNN($cudnn_version, $cuda_version) {
    if ($cudnn_version -eq "8.6") {
        $cudnn = "v8.6.0"
    } elseif ($cudnn_version -eq "8.8") {
        $cudnn = "v8.8.1"
    } elseif ($cudnn_version -eq "8.9") {
        $cudnn = "v8.9.3"
    } else {
        throw "Unsupported cuDNN version: $cudnn_version"
    }

    if ($cuda_version -eq "10.2") {
        $cuda = "10"
    } elseif ($cuda_version.startswith("11.")) {
        $cuda = "11"
    } elseif ($cuda_version.startswith("12.")) {
        $cuda = "12"
    } else {
        throw "Unsupported CUDA version: $cuda_version"
    }

    $base = "C:\Development\cuDNN\$cudnn\cuda$cuda"
    $Env:CL = "-I$base\include " + $Env:CL
    $Env:LINK = "/LIBPATH:$base\lib\x64 " + $Env:LINK
    $Env:PATH = "$base\bin;" + $Env:PATH
}

function ActivateNVTX1() {
    $base = "C:\Development\NvToolsExt"
    $Env:NVTOOLSEXT_PATH = "C:\Development\NvToolsExt"
    $Env:CL = "-I$base\include " + $Env:CL
    $Env:LINK = "/LIBPATH:$base\lib\x64 " + $Env:LINK
    $Env:PATH = "$base\bin\x64;" + $Env:PATH
}

function InstallZLIB() {
    Copy-Item -Path "C:\Development\ZLIB\zlibwapi.dll" -Destination "C:\Windows\System32"
}

function IsPullRequestTest() {
    return ${Env:FLEXCI_BRANCH} -ne $null -and ${Env:FLEXCI_BRANCH}.StartsWith("refs/pull/")
}

function PrioritizeFlexCIDaemon() {
    echo "Prioritizing FlexCI daemon process..."
    wmic.exe process where 'name="imosci.exe"' CALL setpriority realtime
    if (-not $?) {
        throw "Failed to change priority of daemon (exit code = $LastExitCode)"
    }
}

function EnableLongPaths() {
    Set-ItemProperty "Registry::HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -value 1
}
