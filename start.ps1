param([int]$Port = 8088, [string]$ModelConfig = '')
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $env:PYTHONIOENCODING = 'utf-8'
    if ($ModelConfig) { python app.py --port $Port --model-config $ModelConfig }
    else { python app.py --port $Port }
} finally { Pop-Location }
