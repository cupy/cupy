function Download ($filename, $url) {
    $webclient = New-Object System.Net.WebClient

    $basedir = $pwd.Path + "\"
    $filepath = $basedir + $filename
    if (Test-Path $filename) {
        Write-Host "Reusing" $filepath
        return $filepath
    }

    # Download and retry up to 3 times in case of network transient errors.
    Write-Host "Downloading" $filename "from" $url
    $retry_attempts = 2
    for ($i = 0; $i -lt $retry_attempts; $i++) {
        try {
            $webclient.DownloadFile($url, $filepath)
            break
        }
        Catch [Exception]{
            Start-Sleep 1
        }
    }
    if (Test-Path $filepath) {
        Write-Host "File saved at" $filepath
    } else {
        # Retry once to get the error message if any at the last try
        $webclient.DownloadFile($url, $filepath)
    }
    return $filepath
}

function RunCommand ($command, $command_args) {
    Write-Host $command $command_args
    Start-Process -FilePath $command -ArgumentList $command_args -Wait -Passthru
}

function Unzip ($zipfile, $outpath) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($zipfile, $outpath)
}

function main () {
    $CUDA_VERSION = "9.1"
    $CUDA_RELEASE = "9.1.85"
    $CUDNN_VERSION = "7"
    $CUDNN_RELEASE = "7.0.5"
    $CUDA_PATH = "${Env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA\v${CUDA_VERSION}"

    # Download & Install CUDA
    $filename = "cuda_${CUDA_RELEASE}_windows.exe"
    $url = "https://developer.nvidia.com/compute/cuda/${CUDA_VERSION}/Prod/local_installers/cuda_${CUDA_RELEASE}_windows"
    $filepath = Download $filename $url
    RunCommand $filepath "-s"

    # Download & Install cuDNN
    $filename = "cudnn-${CUDA_VERSION}-windows7-x64-v${CUDNN_VERSION}.zip"
    $url = "http://developer.download.nvidia.com/compute/redist/cudnn/v${CUDNN_RELEASE}/$filename"
    $filepath = Download $filename $url
    Unzip $filepath $pwd.Path
    Move-Item "cuda\bin\cudnn64_${CUDNN_VERSION}.dll"   "${CUDA_PATH}\bin"
    Move-Item "cuda\lib\x64\cudnn.lib"                  "${CUDA_PATH}\lib\x64"
    Move-Item "cuda\include\cudnn.h"                    "${CUDA_PATH}\include"
}

main
