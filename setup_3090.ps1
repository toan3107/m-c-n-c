param(
    [string]$BasePython = "python",
    [string]$EnvDir = (Join-Path $env:USERPROFILE "reservoir8-env")
)
$ErrorActionPreference = "Stop"
$python = Join-Path $EnvDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    & $BasePython -m venv $EnvDir
    if ($LASTEXITCODE -ne 0) { throw "Cannot create the isolated reservoir8 environment" }
}
& $python -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), 'Use Python 3.10-3.13 for these pinned wheels'"
if ($LASTEXITCODE -ne 0) { throw "Unsupported Python; select a compatible BasePython" }
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed" }
& $python -m pip install --upgrade torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed" }
& $python -m pip install -r (Join-Path $PSScriptRoot "requirements-3090.txt")
if ($LASTEXITCODE -ne 0) { throw "Scientific dependencies installation failed" }
& $python (Join-Path $PSScriptRoot "audit_dataset.py")
if ($LASTEXITCODE -ne 0) { throw "Dataset integrity audit failed" }
& $python -m unittest discover -s $PSScriptRoot -p "test_*.py" -v
if ($LASTEXITCODE -ne 0) { throw "Unit tests failed" }
& $python (Join-Path $PSScriptRoot "gpu_preflight.py") --device cuda --output-dir (Join-Path $PSScriptRoot "runs\preflight")
if ($LASTEXITCODE -ne 0) { throw "GPU preflight failed; do not start the experiments" }
Write-Host "SETUP COMPLETE. Isolated Python: $python"
Write-Host "Next: .\run_3090.ps1 -Stage smoke"
