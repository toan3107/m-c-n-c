param(
    [ValidateSet("smoke", "overfit", "pilot", "confirm", "final-test", "refit", "refit-test")]
    [string]$Stage = "smoke",
    [string]$EnvDir = (Join-Path $env:USERPROFILE "reservoir8-env"),
    [int]$BatchSize = 0,
    [int]$ImageSize = 0,
    [int[]]$Seeds = @(),
    [string]$Config = "",
    [string[]]$Experiments = @(),
    [int]$ContextLength = 0,
    [int]$BaseChannels = 0,
    [int]$Epochs = 0,
    [string]$OutputRoot = (Join-Path $PSScriptRoot "runs"),
    [switch]$Resume,
    [switch]$DryRun,
    [switch]$ConfirmFinalTest
)
$ErrorActionPreference = "Stop"
$python = Join-Path $EnvDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Run setup_3090.ps1 first" }
$arguments = @((Join-Path $PSScriptRoot "run_experiments.py"), "--stage", $Stage,
               "--device", "cuda",
               "--output-root", $OutputRoot)
if ($Config) { $arguments += @("--config", (Join-Path $PSScriptRoot $Config)) }
if ($BatchSize -gt 0) { $arguments += @("--batch-size", [string]$BatchSize) }
if ($ImageSize -gt 0) { $arguments += @("--image-size", [string]$ImageSize) }
if ($ContextLength -gt 0) { $arguments += @("--context-length", [string]$ContextLength) }
if ($BaseChannels -gt 0) { $arguments += @("--base-channels", [string]$BaseChannels) }
if ($Seeds.Count) {
    $arguments += "--seeds"
    $arguments += @($Seeds | ForEach-Object { [string]$_ })
}
if ($Experiments.Count) { $arguments += "--experiments"; $arguments += $Experiments }
if ($Epochs -gt 0) { $arguments += @("--epochs", [string]$Epochs) }
if ($Resume) { $arguments += "--resume" }
if ($DryRun) { $arguments += "--dry-run" }
if ($ConfirmFinalTest) { $arguments += "--confirm-final-test" }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "Experiment stage failed; see console.log; no automatic retry was attempted" }
