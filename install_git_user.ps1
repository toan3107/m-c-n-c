param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "Programs\ReservoirGit"),
    [switch]$SessionOnly
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$candidates = @()
$command = Get-Command git.exe -CommandType Application -ErrorAction SilentlyContinue
if ($command) { $candidates += $command.Source }
$candidates += @(
    (Join-Path $env:ProgramFiles "Git\cmd\git.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Git\cmd\git.exe"),
    (Join-Path $InstallDir "cmd\git.exe")
)
$git = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1

if (-not $git) {
    $InstallDir = [IO.Path]::GetFullPath($InstallDir)
    if (Test-Path -LiteralPath $InstallDir) {
        if (-not (Test-Path -LiteralPath $InstallDir -PathType Container)) {
            throw "Git destination is not a directory: $InstallDir"
        }
        if (Get-ChildItem -LiteralPath $InstallDir -Force | Select-Object -First 1) {
            throw "Git destination is nonempty; refusing to overwrite: $InstallDir"
        }
    }

    # Official MinGit release; verify before extracting or running any executable.
    $url = "https://github.com/git-for-windows/git/releases/download/v2.56.1.windows.1/MinGit-2.56.1-64-bit.zip"
    $sha256 = "4a92850b6e917da4fd010669b614b55dccf5b0ec333a534cc4be4700dfefdafd"
    $archive = Join-Path ([IO.Path]::GetTempPath()) ("reservoir8-mingit-" + [guid]::NewGuid().ToString("N") + ".zip")
    try {
        Write-Host "Downloading official MinGit (about 40 MB); no administrator rights needed."
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $archive -TimeoutSec 300
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $sha256) {
            throw "Git archive SHA256 mismatch; nothing will be extracted"
        }
        Expand-Archive -LiteralPath $archive -DestinationPath $InstallDir
    }
    finally {
        if (Test-Path -LiteralPath $archive -PathType Leaf) {
            Remove-Item -LiteralPath $archive -Force
        }
    }
    $git = Join-Path $InstallDir "cmd\git.exe"
}

& $git --version
if ($LASTEXITCODE -ne 0) { throw "Git executable check failed: $git" }
$gitDir = Split-Path -Parent $git
if (($env:Path -split ";") -notcontains $gitDir) {
    $env:Path = "$gitDir;$env:Path"
}
if (-not $SessionOnly) {
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (($userPath -split ";") -notcontains $gitDir) {
        $updated = if ([string]::IsNullOrWhiteSpace($userPath)) { $gitDir } else { "$gitDir;$userPath" }
        [Environment]::SetEnvironmentVariable("Path", $updated, "User")
    }
}
Write-Host "Git ready: $git"
Write-Host "Current PowerShell PATH updated. Future sessions may need a terminal/SSH restart."
