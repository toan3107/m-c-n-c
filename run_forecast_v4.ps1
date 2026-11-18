param(
    [string]$EnvDir = (Join-Path $env:USERPROFILE "reservoir8-env"),
    [string]$OutputRoot = (Join-Path $PSScriptRoot "runs_forecast_v4"),
    [switch]$Resume,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$arguments = @{
    Config = "configs/method_forecast_v4.json"
    EnvDir = $EnvDir
    OutputRoot = $OutputRoot
}
if ($DryRun) { $arguments.DryRun = $true }
& (Join-Path $PSScriptRoot "run_3090.ps1") -Stage smoke @arguments
if ($Resume) { $arguments.Resume = $true }
& (Join-Path $PSScriptRoot "run_3090.ps1") -Stage confirm @arguments
