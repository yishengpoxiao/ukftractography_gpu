# Install only build-time CUDA components into a private, new directory.
[CmdletBinding()]
param([string]$Version = "12.9.1", [Parameter(Mandatory=$true)][string]$Directory)
$ErrorActionPreference = "Stop"
if (Test-Path $Directory) { throw "Choose a new CUDA directory; existing environments are preserved." }
$Base = "https://developer.download.nvidia.com/compute/cuda/redist/"
$Manifest = Invoke-RestMethod "$($Base)redistrib_$Version.json"
New-Item -ItemType Directory -Path $Directory | Out-Null
$Downloads = Join-Path $env:RUNNER_TEMP "cuda-downloads-$Version"
New-Item -ItemType Directory -Path $Downloads -Force | Out-Null
foreach ($Component in @("cuda_nvcc", "cuda_cudart", "cuda_cccl", "cuda_nvrtc", "cuda_profiler_api")) {
    $Package = $Manifest.$Component.'windows-x86_64'
    if (-not $Package) { throw "No Windows x64 $Component in NVIDIA CUDA $Version manifest." }
    $Zip = Join-Path $Downloads "$Component.zip"
    Invoke-WebRequest "$Base$($Package.relative_path)" -OutFile $Zip
    if ((Get-FileHash $Zip -Algorithm SHA256).Hash.ToLower() -ne $Package.sha256) { throw "SHA256 mismatch: $Component" }
    $Extract = Join-Path $Downloads $Component
    Expand-Archive $Zip $Extract
    $Source = (Get-ChildItem $Extract -Directory | Select-Object -First 1).FullName
    foreach ($File in Get-ChildItem $Source -File -Recurse) {
        $Relative = $File.FullName.Substring($Source.Length + 1)
        $Destination = Join-Path $Directory $Relative
        if (Test-Path $Destination) {
            if ((Get-FileHash $Destination).Hash -eq (Get-FileHash $File.FullName).Hash) { continue }
            if ($Relative -match 'LICENSE|EULA') { $Destination = Join-Path $Directory "$Component-$($File.Name)" }
            else { throw "Conflicting CUDA component file: $Relative" }
        }
        New-Item -ItemType Directory -Path (Split-Path $Destination) -Force | Out-Null
        Copy-Item $File.FullName $Destination
    }
}
& (Join-Path $Directory "bin/nvcc.exe") --version
if ($LASTEXITCODE -ne 0) { throw "Installed nvcc failed." }
"CUDA_PATH=$Directory" | Out-File -Append -Encoding utf8 $env:GITHUB_ENV
(Join-Path $Directory "bin") | Out-File -Append -Encoding utf8 $env:GITHUB_PATH
