[CmdletBinding()]
param(
    [string]$Python = '',
    [switch]$CreateEnv,
    [switch]$CreateVenv,
    [switch]$InstallDependencies,
    [ValidateSet('cpu', 'cu126')]
    [string]$TorchRuntime = 'cpu',
    [switch]$StartServices,
    [switch]$UseExistingImages,
    [switch]$ProbeModels,
    [switch]$ProbeOCR,
    [switch]$ProbeLLM,
    [switch]$SkipDoctor,
    [string]$Report = ''
)

$ErrorActionPreference = 'Stop'
if ($SkipDoctor -and ($ProbeModels -or $ProbeOCR -or $ProbeLLM -or $Report)) {
    throw '-SkipDoctor cannot be combined with probe or report options.'
}
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$localPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$sharedPython = Join-Path (Split-Path -Parent $projectRoot) '.venv\Scripts\python.exe'
$envFile = Join-Path $projectRoot '.env'
$composeFile = Join-Path $projectRoot 'compose.yaml'
$requirementsFile = Join-Path $projectRoot 'requirements-full.txt'

function New-LocalSecret {
    $bytes = New-Object byte[] 24
    $generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $generator.GetBytes($bytes) } finally { $generator.Dispose() }
    return ([BitConverter]::ToString($bytes)).Replace('-', '').ToLowerInvariant()
}

function Read-ProjectEnvironment {
    $entries = @{}
    if (Test-Path -LiteralPath $envFile) {
        foreach ($line in [IO.File]::ReadAllLines($envFile)) {
            if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$') {
                $entries[$Matches[1]] = $Matches[2].Trim().Trim('"').Trim("'")
            }
        }
    }
    return $entries
}

if ($CreateEnv) {
    $settings = Read-ProjectEnvironment
    $defaults = [ordered]@{
        CLOUDCARE_MYSQL_HOST = '127.0.0.1'
        CLOUDCARE_MYSQL_PORT = '3308'
        CLOUDCARE_MYSQL_DATABASE = 'cloudcare_support'
        CLOUDCARE_MYSQL_USER = 'cloudcare'
        CLOUDCARE_MYSQL_PASSWORD = (New-LocalSecret)
        CLOUDCARE_MYSQL_ROOT_PASSWORD = (New-LocalSecret)
        CLOUDCARE_REDIS_PORT = '6381'
        CLOUDCARE_REDIS_URL = 'redis://127.0.0.1:6381/0'
        CLOUDCARE_REDIS_PREFIX = 'cloudcare:v2:'
        CLOUDCARE_MILVUS_PORT = '19531'
        CLOUDCARE_MILVUS_URI = 'http://127.0.0.1:19531'
        CLOUDCARE_MILVUS_COLLECTION = 'cloudcare_support_v2'
        CLOUDCARE_MINIO_ACCESS_KEY = ('cloudcare' + (New-LocalSecret).Substring(0, 16))
        CLOUDCARE_MINIO_SECRET_KEY = (New-LocalSecret)
    }
    $newLines = [Collections.Generic.List[string]]::new()
    foreach ($name in $defaults.Keys) {
        if (!$settings.ContainsKey($name) -or [string]::IsNullOrWhiteSpace([string]$settings[$name])) {
            $existingValue = [Environment]::GetEnvironmentVariable($name)
            $entryValue = if ($existingValue) { $existingValue } else { $defaults[$name] }
            $newLines.Add($name + '=' + $entryValue)
        }
    }
    if ($newLines.Count -gt 0) {
        if (!(Test-Path -LiteralPath $envFile)) {
            [IO.File]::WriteAllText($envFile, "# Local credentials; ignored by Git.`n", [Text.UTF8Encoding]::new($false))
        }
        [IO.File]::AppendAllText($envFile, "`n" + ($newLines -join "`n") + "`n", [Text.UTF8Encoding]::new($false))
    }
    Write-Host 'Local configuration prepared; existing values preserved.'
}

if ($CreateVenv) {
    if (!$Python) { $Python = (Get-Command python -ErrorAction Stop).Source }
    if (!(Test-Path -LiteralPath $localPython)) {
        & $Python -m venv (Join-Path $projectRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Creating the project environment failed.' }
    }
    $Python = $localPython
}
if (!$Python) {
    if (Test-Path -LiteralPath $localPython) { $Python = $localPython }
    elseif (Test-Path -LiteralPath $sharedPython) { $Python = $sharedPython }
    else { $Python = (Get-Command python -ErrorAction Stop).Source }
}

if ($InstallDependencies) {
    $resolvedPython = [IO.Path]::GetFullPath($Python)
    $allowedRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot '.venv')) + [IO.Path]::DirectorySeparatorChar
    if (!$resolvedPython.StartsWith($allowedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Dependency installation is restricted to the new project .venv. Use -CreateVenv first.'
    }
    if ($TorchRuntime -eq 'cu126') {
        & $Python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126
        if ($LASTEXITCODE -ne 0) { throw 'Installing the pinned CUDA 12.6 PyTorch runtime failed.' }
    }
    & $Python -m pip install --requirement $requirementsFile
    if ($LASTEXITCODE -ne 0) { throw 'Installing the pinned project dependencies failed.' }
}

if ($StartServices) {
    if (!(Test-Path -LiteralPath $envFile)) { throw 'Create local credentials with -CreateEnv before starting services.' }
    $settings = Read-ProjectEnvironment
    $requiredSecrets = @('CLOUDCARE_MYSQL_PASSWORD', 'CLOUDCARE_MYSQL_ROOT_PASSWORD', 'CLOUDCARE_MINIO_ACCESS_KEY', 'CLOUDCARE_MINIO_SECRET_KEY')
    foreach ($name in $requiredSecrets) {
        if (!$settings[$name] -and ![Environment]::GetEnvironmentVariable($name)) { throw "Missing local setting: $name" }
    }
    & docker info --format '{{.ServerVersion}}' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Docker engine is unavailable. Start Docker Desktop and rerun.' }
    $dockerArgs = @('compose', '--project-name', 'cloudcare-v2', '--env-file', $envFile, '--file', $composeFile)
    & docker @dockerArgs config --quiet
    if ($LASTEXITCODE -ne 0) { throw 'The independent service configuration is invalid.' }
    $ownContainers = @(& docker @dockerArgs ps --status running --format json | ForEach-Object { $_ | ConvertFrom-Json })
    $ownedPorts = @($ownContainers | ForEach-Object { $_.Publishers | ForEach-Object { $_.PublishedPort } })
    $portDefaults = @{ CLOUDCARE_MYSQL_PORT = 3308; CLOUDCARE_REDIS_PORT = 6381; CLOUDCARE_MILVUS_PORT = 19531; CLOUDCARE_MILVUS_HEALTH_PORT = 9092 }
    $listeners = [Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    foreach ($name in $portDefaults.Keys) {
        $fromProcess = [Environment]::GetEnvironmentVariable($name)
        $selectedPort = if ($fromProcess) { [int]$fromProcess } elseif ($settings[$name]) { [int]$settings[$name] } else { $portDefaults[$name] }
        if (($listeners.Port -contains $selectedPort) -and ($ownedPorts -notcontains $selectedPort)) {
            throw "Port $selectedPort is occupied by a different service. Set $name to an unused port and update its corresponding URL."
        }
    }
    $startArgs = @('up', '--detach', '--wait', '--wait-timeout', '240')
    if ($UseExistingImages) { $startArgs += @('--no-build', '--pull', 'never') }
    & docker @dockerArgs @startArgs
    if ($LASTEXITCODE -ne 0) { throw 'Starting the independent services failed; inspect CloudCare container health.' }
}

if ($SkipDoctor) {
    Write-Host 'Environment/service setup finished. Train the support router, initialize the corpus, then run doctor for acceptance.'
    exit 0
}

$doctorArgs = @('-X', 'utf8', (Join-Path $PSScriptRoot 'doctor.py'))
if ($ProbeModels) { $doctorArgs += '--probe-models' }
if ($ProbeOCR) { $doctorArgs += '--probe-ocr' }
if ($ProbeLLM) { $doctorArgs += '--probe-llm' }
if ($Report) { $doctorArgs += @('--output', $Report) }
& $Python @doctorArgs
exit $LASTEXITCODE
