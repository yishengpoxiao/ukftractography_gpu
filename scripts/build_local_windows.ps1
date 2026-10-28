[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$SlicerPath,
    [string]$BuildDirectory,
    [int]$Jobs = 4,
    [string]$CudaPath,
    [switch]$CheckOnly,
    [switch]$NoInstall
)
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw 'Run this script in Windows PowerShell.' }
if ($Jobs -lt 1) { throw '-Jobs must be at least 1.' }

$Repository = Split-Path -Parent $PSScriptRoot
if (-not $BuildDirectory) { $BuildDirectory = Join-Path (Split-Path -Parent $Repository) 'build-local-windows' }
$BuildDirectory = [IO.Path]::GetFullPath($BuildDirectory)

$SlicerItem = Get-Item -LiteralPath $SlicerPath -ErrorAction Stop
if ($SlicerItem.PSIsContainer) {
    $SlicerExe = Get-ChildItem -LiteralPath $SlicerItem.FullName -Filter Slicer.exe -File -Recurse |
        Select-Object -First 1 -ExpandProperty FullName
} else { $SlicerExe = $SlicerItem.FullName }
if (-not $SlicerExe -or (Split-Path -Leaf $SlicerExe) -ne 'Slicer.exe') {
    throw "Slicer.exe was not found. Pass its path or the Slicer installation directory: -SlicerPath 'C:\Program Files\Slicer 5.6.2'"
}
$PythonSlicer = Get-ChildItem -LiteralPath (Split-Path -Parent $SlicerExe) -Filter PythonSlicer.exe -File -Recurse |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $PythonSlicer) { throw "PythonSlicer.exe was not found under $(Split-Path -Parent $SlicerExe). Repair or reinstall Slicer, then retry with -SlicerPath '$SlicerExe'." }
$VersionText = $null
$SlicerRoot = Split-Path -Parent $SlicerExe
for ($Depth = 0; $Depth -lt 3 -and $SlicerRoot -and -not $VersionText; $Depth++) {
    $LibDirectory = Join-Path $SlicerRoot 'lib'
    if (Test-Path $LibDirectory) {
        $VersionDirectory = Get-ChildItem $LibDirectory -Directory -Filter 'Slicer-*' | Select-Object -First 1
        if ($VersionDirectory -and $VersionDirectory.Name -match '^Slicer-(\d+\.\d+)') { $VersionText = $Matches[1] }
    }
    $SlicerRoot = Split-Path -Parent $SlicerRoot
}
if (-not $VersionText) { $VersionText = (Get-Item -LiteralPath $SlicerExe).VersionInfo.ProductVersion }
if (-not $VersionText) { $VersionText = $SlicerExe }
if ($VersionText -notmatch '(?<!\d)(\d+)\.(\d+)(?:\.\d+)?') {
    throw "Could not determine the Slicer version from $SlicerExe. Use a versioned Slicer installation and retry with -SlicerPath."
}
$SlicerVersion = "$($Matches[1]).$($Matches[2])"
if ([version]$SlicerVersion -lt [version]'5.2') { throw "Slicer $SlicerVersion is unsupported; install Slicer 5.2 or newer." }

if (-not $CudaPath -and $env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH 'bin/nvcc.exe'))) { $CudaPath = $env:CUDA_PATH }
if (-not $CudaPath) {
    $NvccCommand = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($NvccCommand) { $CudaPath = Split-Path -Parent (Split-Path -Parent $NvccCommand.Source) }
}
if (-not $CudaPath) {
    $CudaRoot = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit/CUDA'
    if (Test-Path $CudaRoot) {
        $CudaPath = Get-ChildItem $CudaRoot -Directory | Sort-Object {
            try { [version]($_.Name -replace '^v','') } catch { [version]'0.0' }
        } -Descending | Where-Object { Test-Path (Join-Path $_.FullName 'bin/nvcc.exe') } |
            Select-Object -First 1 -ExpandProperty FullName
    }
}
if (-not $CudaPath -or -not (Test-Path (Join-Path $CudaPath 'bin/nvcc.exe'))) {
    throw "CUDA Toolkit nvcc.exe was not found. Install the CUDA Toolkit from https://developer.nvidia.com/cuda-downloads, then retry: .\scripts\build_local_windows.ps1 -SlicerPath '$SlicerExe' -CudaPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y'"
}
$CudaPath = [IO.Path]::GetFullPath($CudaPath)

if ($env:VSCMD_ARG_TGT_ARCH -ne 'x64' -or -not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $VsWhere = $null
    if (${env:ProgramFiles(x86)}) { $VsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe' }
    if (-not (Test-Path $VsWhere)) {
        throw "Visual Studio C++ x64 tools are required. Install Visual Studio 2022 Build Tools with the 'Desktop development with C++' workload, then retry: .\scripts\build_local_windows.ps1 -SlicerPath '$SlicerExe'"
    }
    $VsInstall = & $VsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $VsInstall) {
        throw "Visual Studio C++ x64 tools were not found. Install Visual Studio 2022 Build Tools with the C++ workload, then retry with -SlicerPath '$SlicerExe'."
    }
    $VsDevCmd = Join-Path $VsInstall 'Common7/Tools/VsDevCmd.bat'
    if (-not (Test-Path $VsDevCmd)) {
        throw "Visual Studio C++ x64 tools were not found. Install Visual Studio 2022 Build Tools with the C++ workload, then retry with -SlicerPath '$SlicerExe'."
    }
    $CmdLine = '"' + $VsDevCmd + '" -no_logo -arch=x64 -host_arch=x64 && set'
    $EnvironmentLines = & $env:ComSpec /d /s /c ('"' + $CmdLine + '"')
    if ($LASTEXITCODE -ne 0) { throw "VsDevCmd failed. Open an x64 Developer PowerShell or repair Visual Studio, then retry with -SlicerPath '$SlicerExe'." }
    foreach ($Line in $EnvironmentLines) {
        if ($Line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
    }
}
if ($env:VSCMD_ARG_TGT_ARCH -ne 'x64') { throw 'Could not activate the x64 Visual Studio environment; use an x64 Developer PowerShell.' }

$VsInstall = $env:VSINSTALLDIR
if (-not $VsInstall -and $VsWhere -and (Test-Path $VsWhere)) {
    $VsInstall = (& $VsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
}
$VsCmakeRoot = if ($VsInstall) { Join-Path $VsInstall 'Common7/IDE/CommonExtensions/Microsoft/CMake' }
foreach ($ToolName in @('cmake.exe','ninja.exe')) {
    if (-not (Get-Command $ToolName -ErrorAction SilentlyContinue) -and $VsCmakeRoot -and (Test-Path $VsCmakeRoot)) {
        $Candidate = Get-ChildItem $VsCmakeRoot -Filter $ToolName -File -Recurse | Select-Object -First 1 -ExpandProperty FullName
        if ($Candidate) { $env:PATH = (Split-Path -Parent $Candidate) + ';' + $env:PATH }
    }
}
foreach ($ToolName in @('cmake.exe','ninja.exe','git.exe','dumpbin.exe')) {
    if (-not (Get-Command $ToolName -ErrorAction SilentlyContinue)) { throw "Missing $ToolName. Install it or add it to PATH, then retry with -SlicerPath '$SlicerExe'." }
}
$CmakeVersionText = (& cmake.exe --version | Select-Object -First 1)
if ($CmakeVersionText -notmatch 'cmake version (\d+\.\d+)') { throw 'Could not determine the installed CMake version.' }
if ([version]$Matches[1] -lt [version]'3.22') { throw "CMake 3.22 or newer is required; found $($Matches[1])." }
foreach ($ToolPath in @($PythonSlicer, (Get-Command cl.exe).Source, (Get-Command cmake.exe).Source,
        (Get-Command ninja.exe).Source, (Get-Command git.exe).Source, (Get-Command dumpbin.exe).Source)) {
    if (-not (Test-Path $ToolPath)) { throw "Required build tool was not found: $ToolPath" }
}

$CudaArchitectures = $null
$Nvcc = Join-Path $CudaPath 'bin/nvcc.exe'
try {
    $Supported = @(& $Nvcc --list-gpu-code | ForEach-Object {
        if ($_ -match '^sm_(\d+)$') { [int]$Matches[1] }
    } | Where-Object { $_ -ge 60 } | Sort-Object -Unique)
    $Caps = @(& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | ForEach-Object {
        if ($_ -match '^\s*(\d+)\.(\d+)\s*$') { [int]("$($Matches[1])$($Matches[2])") }
    } | Where-Object { $Supported -contains $_ } | Sort-Object -Unique)
    if ($LASTEXITCODE -eq 0 -and $Caps.Count -gt 0 -and $Supported.Count -gt 0) {
        $CudaArchitectures = ((@($Caps | ForEach-Object { "$_-real" }) + "$($Supported[0])-virtual") -join ';')
        Write-Host "Using detected local GPU CUDA targets: $CudaArchitectures"
    } else { Write-Host 'GPU capability detection unavailable; using the build script supported-architecture defaults.' }
} catch { Write-Host 'GPU capability detection unavailable; using the build script supported-architecture defaults.' }

Write-Host "Slicer: $SlicerVersion ($SlicerExe)"
Write-Host "PythonSlicer: $PythonSlicer"
Write-Host "CUDA Toolkit: $CudaPath"
Write-Host "Visual Studio: $VsInstall (x64)"
Write-Host "CMake: $CmakeVersionText"
Write-Host "Ninja: $((Get-Command ninja.exe).Source)"
Write-Host "Git: $((Get-Command git.exe).Source)"
Write-Host "MSVC: $((Get-Command cl.exe).Source)"
Write-Host "Jobs: $Jobs"
if (-not $CudaArchitectures) { Write-Host 'CUDA targets: build_windows_gpu.ps1 supported-architecture defaults' }
if ($CheckOnly) { Write-Host 'Preflight succeeded; no build or installation was started.'; exit 0 }

$Stage = Join-Path $BuildDirectory ('.staging-' + [guid]::NewGuid().ToString('N'))
$PackageDirectory = Join-Path $Stage 'package'
$BuildScript = Join-Path $PSScriptRoot 'build_windows_gpu.ps1'
$BuildArgs = @{
    BuildDirectory = $BuildDirectory; PackageDirectory = $PackageDirectory
    CudaPath = $CudaPath; Python = $PythonSlicer; SlicerVersion = $SlicerVersion; Jobs = $Jobs
}
if ($CudaArchitectures) { $BuildArgs.CudaArchitectures = $CudaArchitectures }
New-Item -ItemType Directory -Force $BuildDirectory, $Stage | Out-Null
try {
    & $BuildScript @BuildArgs
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw "Windows build or packaging failed with exit code $LASTEXITCODE." }
    $StagedZip = Join-Path $PackageDirectory 'UKFGPU-Installer-windows-x86_64.zip'
    $StagedRoot = Join-Path $Stage 'extract'
    New-Item -ItemType Directory $StagedRoot | Out-Null
    Expand-Archive -LiteralPath $StagedZip -DestinationPath $StagedRoot
    $StagedBundle = Join-Path $StagedRoot 'UKFGPU-Installer'
    if (-not (Test-Path (Join-Path $StagedBundle 'manifest.json')) -or -not (Test-Path (Join-Path $StagedBundle 'install.py'))) {
        throw "The staged installer is missing manifest.json or install.py: $StagedZip"
    }
    $InstallerZip = Join-Path $BuildDirectory 'UKFGPU-Installer-windows-x86_64.zip'
    $InstallerBundle = Join-Path $BuildDirectory 'UKFGPU-Installer'
    if (Test-Path $InstallerBundle) {
        if (-not (Test-Path (Join-Path $InstallerBundle 'manifest.json')) -or -not (Test-Path (Join-Path $InstallerBundle 'install.py'))) {
            throw "Existing $InstallerBundle is not a recognized UKFGPU installer bundle; it was left untouched. Move it yourself and retry."
        }
        Remove-Item -LiteralPath $InstallerBundle -Recurse -Force
    }
    Move-Item -LiteralPath $StagedBundle -Destination $InstallerBundle
    Copy-Item -LiteralPath $StagedZip -Destination (Join-Path $Stage 'UKFGPU-Installer-windows-x86_64.zip')
    Move-Item -LiteralPath (Join-Path $Stage 'UKFGPU-Installer-windows-x86_64.zip') -Destination $InstallerZip -Force
    $InstallScript = Join-Path $InstallerBundle 'install.py'
    Write-Host "Build cache: $BuildDirectory"
    Write-Host "Installer ZIP: $InstallerZip"
    Write-Host "Extracted installer and runpy entry point: $InstallScript"
    if (-not $NoInstall) {
        Write-Host "Launching Slicer $SlicerVersion to install the freshly built extension. Restart Slicer after installation."
        Start-Process -FilePath $SlicerExe -ArgumentList @('--python-script', ('"' + $InstallScript + '"')) | Out-Null
    }
} finally {
    if (Test-Path $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
}
