[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Bundle,
    [Parameter(Mandatory=$true)][string]$WorkDirectory,
    [string[]]$Versions = @('5.6.2', 'latest')
)
$ErrorActionPreference = 'Stop'
if (Test-Path $WorkDirectory) { throw 'WorkDirectory must be new.' }
New-Item -ItemType Directory $WorkDirectory | Out-Null
foreach ($Version in $Versions) {
    $Installer = Join-Path $WorkDirectory "Slicer-$Version.exe"
    python "$PSScriptRoot/fetch_slicer.py" --version $Version --output $Installer
    if ($LASTEXITCODE -ne 0) { throw 'Slicer download/checksum verification failed.' }
    $Application = Join-Path $WorkDirectory "Slicer-$Version"
    $Setup = Start-Process $Installer -ArgumentList "/S /D=$Application" -Wait -PassThru
    if ($Setup.ExitCode -ne 0) { throw "Slicer installer failed: $($Setup.ExitCode)" }
    $Slicer = Join-Path $Application 'Slicer.exe'
    if (-not (Test-Path $Slicer)) { throw "Slicer.exe not installed at $Application" }
    python "$PSScriptRoot/verify_slicer_installation.py" --slicer $Slicer --bundle $Bundle --output "$WorkDirectory/check-$Version"
    if ($LASTEXITCODE -ne 0) { throw "Slicer $Version extension installation/loading failed." }
}
