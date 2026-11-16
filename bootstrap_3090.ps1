param(
    [string]$DataSource = "",
    [string]$DataZip = "",
    [string]$BasePython = "python",
    [string]$EnvDir = (Join-Path $env:USERPROFILE "reservoir8-env"),
    [switch]$SkipEnvironmentSetup
)
$ErrorActionPreference = "Stop"
$python = Join-Path $EnvDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = $BasePython }
& $python -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), 'Use Python 3.10-3.13'"
if ($LASTEXITCODE -ne 0) { throw "Select a compatible -BasePython interpreter" }
$arguments = @((Join-Path $PSScriptRoot "initialize_data.py"))
if ($DataSource) { $arguments += @("--source-dir", $DataSource) }
if ($DataZip) { $arguments += @("--archive", $DataZip) }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "Dataset import/check failed; existing files were not overwritten" }
if (-not $SkipEnvironmentSetup) {
    & (Join-Path $PSScriptRoot "setup_3090.ps1") -BasePython $BasePython -EnvDir $EnvDir
}
Write-Host "Project initialized. Data remains outside Git; update code with git pull --ff-only."
Write-Host "Next: .\run_3090.ps1 -Stage smoke"
