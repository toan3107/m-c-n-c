param(
    [ValidateSet("smoke", "overfit", "pilot", "confirm", "final-test")]
    [string]$Stage = "smoke",
    [string]$EnvDir = (Join-Path $env:USERPROFILE "reservoir8-env"),
    [int]$BatchSize = 2,
    [int]$ImageSize = 512,
    [int[]]$Seeds = @(42),
    [int]$Epochs = 0,
    [string]$OutputRoot = (Join-Path $PSScriptRoot "runs"),
    [switch]$Resume,
    [switch]$ConfirmFinalTest
)
$ErrorActionPreference = "Stop"
$python = Join-Path $EnvDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Run setup_3090.ps1 first" }
$arguments = @((Join-Path $PSScriptRoot "run_experiments.py"), "--stage", $Stage,
               "--device", "cuda", "--batch-size", $BatchSize, "--image-size", $ImageSize,
               "--output-root", $OutputRoot)
$arguments += "--seeds"
$arguments += @($Seeds | ForEach-Object { [string]$_ })
if ($Epochs -gt 0) { $arguments += @("--epochs", [string]$Epochs) }
if ($Resume) { $arguments += "--resume" }
if ($ConfirmFinalTest) { $arguments += "--confirm-final-test" }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "Experiment stage failed; see console.log; no automatic retry was attempted" }
