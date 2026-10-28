# Run from an x64 Visual Studio Developer PowerShell with CUDA Toolkit on PATH.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$BuildDirectory,
    [Parameter(Mandatory=$true)][string]$PackageDirectory,
    [string]$SlicerVersion = "5.6",
    [string]$CudaArchitectures,
    [string]$CudaPath = $env:CUDA_PATH,
    [string]$MsvcRuntimeDirectory,
    [string]$Python = "python",
    [int]$Jobs = 4
)
$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") { throw "This script requires native Windows x64." }
$Repository = Split-Path -Parent $PSScriptRoot
foreach ($tool in @("cl.exe", "cmake.exe", "ninja.exe", "git.exe", "dumpbin.exe", $Python)) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "Missing $tool. Use an x64 Visual Studio Developer PowerShell." }
}
if ($env:VSCMD_ARG_TGT_ARCH -ne "x64") { throw "Select the x64 Visual Studio developer environment." }
if (-not $CudaPath) { throw "Set -CudaPath to the installed CUDA Toolkit directory." }
$Nvcc = Join-Path $CudaPath "bin/nvcc.exe"
if (-not (Test-Path $Nvcc)) { throw "CUDA Toolkit nvcc.exe was not found. Set -CudaPath (the driver UMD version is not a Toolkit install)." }
if (-not $MsvcRuntimeDirectory) {
    if (-not $env:VCToolsRedistDir) { throw "Set -MsvcRuntimeDirectory to the x64 MSVC redistributable directory." }
    $MsvcRuntimeDirectory = Join-Path $env:VCToolsRedistDir "x64/Microsoft.VC143.CRT"
}
if (-not (Test-Path (Join-Path $MsvcRuntimeDirectory "msvcp140.dll"))) {
    throw "Set -MsvcRuntimeDirectory to the x64 Microsoft.VC143.CRT redistributable directory."
}
if (Test-Path $PackageDirectory) { throw "PackageDirectory must be new; existing packages are never overwritten." }
if (-not $CudaArchitectures) {
    $Codes = & $Nvcc --list-gpu-code
    if ($LASTEXITCODE -ne 0) { throw "nvcc could not list supported GPU architectures." }
    $Supported = @($Codes | Where-Object { $_ -match '^sm_(\d+)$' } |
        ForEach-Object { [int]($_ -replace '^sm_', '') } | Sort-Object -Unique)
    $Selected = @($Supported | Where-Object { $_ -ge 60 })
    if ($Selected.Count -eq 0) { throw "No supported baseline architectures found; set -CudaArchitectures explicitly." }
    $CudaArchitectures = ((@($Selected | ForEach-Object { "$_-real" }) + "$($Selected[0])-virtual") -join ';')
}
Write-Host "CUDA targets: $CudaArchitectures"
$BuildDirectory = [System.IO.Path]::GetFullPath($BuildDirectory)
function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}
Invoke-Checked "cmake" @("-S", "$Repository/platforms/superbuild", "-B", $BuildDirectory,
    "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_COMPILER=$Nvcc",
    "-DUKF_CUDA_ARCHITECTURES=$CudaArchitectures")
# Child ExternalProject builds also obey this bound.
$PreviousParallelLevel = $env:CMAKE_BUILD_PARALLEL_LEVEL
try {
    $env:CMAKE_BUILD_PARALLEL_LEVEL = "$Jobs"
    Invoke-Checked "cmake" @("--build", $BuildDirectory, "--parallel", "$Jobs")
    # ExternalProject stamps do not track edits in our local backend sources.
    Invoke-Checked "cmake" @("-S", "$Repository/platforms/standalone", "-B", "$BuildDirectory/backend")
    Invoke-Checked "cmake" @("--build", "$BuildDirectory/backend", "--parallel", "$Jobs")
} finally {
    $env:CMAKE_BUILD_PARALLEL_LEVEL = $PreviousParallelLevel
}
$Binary = Join-Path $BuildDirectory "backend/bin/UKFTractography_gpu.exe"
$GpuLibrary = Join-Path $BuildDirectory "backend/bin/ukf_gpu.dll"
$CudaRuntime = Get-ChildItem (Join-Path $CudaPath "bin/cudart64_*.dll") | Select-Object -First 1
if (-not $CudaRuntime) { throw "CUDA runtime DLL not found in the selected Toolkit." }
# This output confirms the DLL exports; a missing ABI entry is a build failure.
$Exports = & dumpbin.exe /exports $GpuLibrary
if ($LASTEXITCODE -ne 0 -or -not ($Exports | Select-String "gpu_compute")) {
    throw "ukf_gpu.dll does not export the GPU API."
}
Invoke-Checked $Python @("$Repository/scripts/package_slicer_extension.py", "--platform", "windows",
    "--binary", $Binary, "--gpu-library", $GpuLibrary,
    "--cuda-runtime", $CudaRuntime.FullName, "--runtime-dir", $MsvcRuntimeDirectory,
    "--slicer-version", $SlicerVersion, "--output", $PackageDirectory)
$PackagedBinary = [System.IO.Path]::GetFullPath((Join-Path $PackageDirectory "UKFGPU/UKFGPU/Runtime/bin/UKFTractography_gpu.exe"))
$PreviousPath = $env:PATH
try {
    $env:PATH = "$(Split-Path $PackagedBinary);$env:SystemRoot\System32;$env:SystemRoot"
    Invoke-Checked $PackagedBinary @("--help")
} finally {
    $env:PATH = $PreviousPath
}
Invoke-Checked $Python @("$Repository/scripts/package_portable_installer.py",
    "--runtime", "windows-x86_64=$(Join-Path $PackageDirectory 'UKFGPU/UKFGPU/Runtime')",
    "--output", (Join-Path $PackageDirectory "UKFGPU-Installer-windows-x86_64.zip"))
Write-Host "Package built. GPU tracking accuracy must be checked on the target Windows NVIDIA hardware."
