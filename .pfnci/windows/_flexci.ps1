function ActivatePython($version) {
    if ($version -eq "3.9") {
        $pydir = "Python39"
    } elseif ($version -eq "3.10") {
        $pydir = "Python310"
    } elseif ($version -eq "3.11") {
        $pydir = "Python311"
    } elseif ($version -eq "3.12") {
        $pydir = "Python312"
    } elseif ($version -eq "3.13") {
        $pydir = "Python313"
    } else {
        throw "Unsupported Python version: $version"
    }
    $Env:PATH = "C:\Development\Python\$pydir;C:\Development\Python\$pydir\Scripts;" + $Env:PATH
}

function ActivateCUDA($version) {
    if ($version -eq "11.2") {
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
    } elseif ($version -eq "12.8") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_8
    } elseif ($version -eq "12.9") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_9
    } elseif ($version -eq "12.x") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V12_9
    } elseif ($version -eq "13.0") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V13_0
    } elseif ($version -eq "13.x") {
        $Env:CUDA_PATH = $Env:CUDA_PATH_V13_0
    } else {
        throw "Unsupported CUDA version: $version"
    }
    $Env:PATH = "$Env:CUDA_PATH\bin;$Env:CUDA_PATH\bin\x64;" + $Env:PATH
}

function ActivateCuDNN($cudnn_version, $cuda_version) {
    # Only supports CUDA 11 and 12.
    if ($cudnn_version -eq "8.8") {
        $cudnn = "v8.8.1"
    } else {
        throw "Unsupported cuDNN version: $cudnn_version"
    }

    if ($cuda_version.startswith("11.")) {
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

function GetPullRequestNumber() {
    if (${Env:FLEXCI_BRANCH} -match "refs/pull/(\d+)/") {
        return $matches[1]
    }
    return 0
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
