param([int]$Port = 8088, [string]$ModelConfig = '', [string]$Python = '', [switch]$Offline)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $env:PYTHONIOENCODING = 'utf-8'
    if (!$Python) { $Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
    if (!(Test-Path -LiteralPath $Python)) {
        throw 'CloudCare Python environment is missing. Run scripts/bootstrap.ps1 -CreateEnv -CreateVenv -InstallDependencies -StartServices first.'
    }
    $appArguments = @((Join-Path $PSScriptRoot 'app.py'), '--port', $Port)
    if ($ModelConfig) { $appArguments += @('--model-config', $ModelConfig) }
    if ($Offline) { $appArguments += '--offline' }
    & $Python @appArguments
    exit $LASTEXITCODE
} finally { Pop-Location }
