[CmdletBinding()]
param(
    [string]$PythonExecutable = "python",
    [string]$CloudflaredExecutable = "cloudflared",
    [ValidateRange(1,65535)][int]$Port = 8123,
    [ValidateSet('off', 'qwen')][string]$VlmMode = 'off',
    [string]$QwenModelPath,
    [string]$QwenManifestPath,
    [string]$WorkerConfigPath,
    [switch]$RestartApiOnly
)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RunRoot = Join-Path $RepoRoot "tmp/public-demo"
$KeyPath = Join-Path $RepoRoot "secrets/public-demo-api-key.txt"
$ModelRoot = Join-Path $RepoRoot "artifacts/ml/bundles/baseline-audited-v4"
$ExpectedSha = "00ae541eac23249dd4751093ce276bfa9c14afca2d5f15f4fec9719c009ae52d"
$PythonPath = (Get-Command $PythonExecutable -ErrorAction Stop).Source
$TunnelPath = (Get-Command $CloudflaredExecutable -ErrorAction Stop).Source
if ($WorkerConfigPath) {
    $WorkerConfig = Get-Content -LiteralPath $WorkerConfigPath -Raw | ConvertFrom-Json
    if (-not $WorkerConfig.url -or -not $WorkerConfig.key_file) { throw 'Worker config requires url and key_file.' }
    $env:QAITU_WORKER_URL = $WorkerConfig.url
    $env:QAITU_WORKER_KEY_FILE = (Resolve-Path -LiteralPath $WorkerConfig.key_file).Path
} else {
    Remove-Item Env:QAITU_WORKER_URL -ErrorAction SilentlyContinue
    Remove-Item Env:QAITU_WORKER_KEY_FILE -ErrorAction SilentlyContinue
}
if ($VlmMode -eq 'qwen' -and -not $WorkerConfigPath) {
    if (-not $QwenModelPath -or -not (Test-Path -LiteralPath $QwenModelPath -PathType Container)) {
        throw 'Qwen mode requires -QwenModelPath pointing to a local snapshot directory.'
    }
    if (-not $QwenManifestPath -or -not (Test-Path -LiteralPath $QwenManifestPath -PathType Leaf)) {
        throw 'Qwen mode requires -QwenManifestPath for the exact delivered snapshot.'
    }
    $env:QAITU_VLM_MODEL_PATH = (Resolve-Path -LiteralPath $QwenModelPath).Path
    $env:QAITU_VLM_MANIFEST_PATH = (Resolve-Path -LiteralPath $QwenManifestPath).Path
}
if ($RestartApiOnly) {
    $PriorSession = Get-Content -LiteralPath (Join-Path $RunRoot 'session.json') -Raw | ConvertFrom-Json
    if ($PriorSession.port -ne $Port) { throw 'Existing demo uses a different port.' }
    $PriorApi = Get-Process -Id $PriorSession.api_pid -ErrorAction Stop
    $PriorTunnel = Get-Process -Id $PriorSession.tunnel_pid -ErrorAction Stop
    if ($PriorApi.StartTime.ToUniversalTime().Ticks -ne ([datetime]$PriorSession.api_started).ToUniversalTime().Ticks -or
        $PriorTunnel.StartTime.ToUniversalTime().Ticks -ne ([datetime]$PriorSession.tunnel_started).ToUniversalTime().Ticks) {
        throw 'Existing demo process identities do not match; refusing restart.'
    }
} elseif (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is occupied. Stop the intended demo explicitly before restarting."
}
if ((Get-FileHash (Join-Path $ModelRoot "model.onnx") -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedSha) {
    throw "Expected ONNX model checksum does not match."
}
New-Item -ItemType Directory -Force -Path $RunRoot, (Split-Path -Parent $KeyPath) | Out-Null
if (-not (Test-Path -LiteralPath $KeyPath)) {
    $KeyBytes = New-Object byte[] 32
    $Generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $Generator.GetBytes($KeyBytes) } finally { $Generator.Dispose() }
    [IO.File]::WriteAllText($KeyPath, [Convert]::ToBase64String($KeyBytes))
}
$ApiKey = [IO.File]::ReadAllText($KeyPath).Trim()
if ($ApiKey.Length -lt 32 -or $ApiKey -eq "demo-secret") { throw "Public demo requires a strong API key." }
$env:FORTEBANK_API_KEY = $ApiKey
$env:PYTHONPATH = Join-Path $RepoRoot "src"
$env:QAITU_INFERENCE_MODE = "onnx"
$env:QAITU_VLM_MODE = $VlmMode
$env:QAITU_VLM_MAX_CONCURRENT_JOBS = '1'
$env:QAITU_MODEL_BUNDLE = $ModelRoot
$env:QAITU_ALLOW_DEVELOPMENT_MODEL = "true"
$env:QAITU_EXPECTED_MODEL_SHA256 = $ExpectedSha
$env:QAITU_MAX_CONCURRENT_JOBS = "1"
$env:QAITU_PREFER_CUDA = "false"
$env:FORTEBANK_UPLOAD_ROOT = Join-Path $RunRoot "uploads"
$env:QAITU_EVIDENCE_ROOT = Join-Path $RunRoot "evidence"
$ApiProcess = $null
$TunnelProcess = $null
try {
    if ($RestartApiOnly) {
        Stop-Process -InputObject $PriorApi
        $PriorApi.WaitForExit()
    }
    $ApiProcess = Start-Process -FilePath $PythonPath -ArgumentList @("-m", "uvicorn", "fortebank_audit.api:app", "--host", "127.0.0.1", "--port", "$Port") -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $RunRoot "api.out.log") -RedirectStandardError (Join-Path $RunRoot "api.err.log")
    $Ready = $false
    $StartupAttempts = if ($VlmMode -eq 'qwen') { 600 } else { 60 }
    for ($Attempt = 0; $Attempt -lt $StartupAttempts; $Attempt++) {
        if ($ApiProcess.HasExited) { throw "API exited. Inspect tmp/public-demo/api.err.log." }
        try {
            $Health = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2
            if ($Health.predictor_class -eq "OnnxPredictor" -and $Health.model_sha256 -eq $ExpectedSha -and $Health.semantic_mode -eq $VlmMode) { $Ready = $true; break }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    if (-not $Ready) { throw "Expected ONNX API did not become ready." }
    $Denied = $false
    try { Invoke-WebRequest "http://127.0.0.1:$Port/internal/api/v1/rules" -TimeoutSec 5 | Out-Null }
    catch { $Denied = [int]$_.Exception.Response.StatusCode -eq 401 }
    if (-not $Denied) { throw "Unauthenticated API request was not rejected." }
    $Catalog = Invoke-RestMethod "http://127.0.0.1:$Port/internal/api/v1/rules" -Headers @{"X-API-Key"=$ApiKey} -TimeoutSec 5
    if ($Catalog.rules.Count -ne 3) { throw "Rule catalog preflight failed." }
    if ($RestartApiOnly) {
        $TunnelProcess = $PriorTunnel
        $PublicUrl = $PriorSession.url
    } else {
    $TunnelProcess = Start-Process -FilePath $TunnelPath -ArgumentList @("tunnel", "--no-autoupdate", "--protocol", "http2", "--url", "http://127.0.0.1:$Port") -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $RunRoot "tunnel.out.log") -RedirectStandardError (Join-Path $RunRoot "tunnel.err.log")
    $PublicUrl = $null
    for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
        if ($TunnelProcess.HasExited) { throw "Tunnel exited. Inspect tmp/public-demo/tunnel.err.log." }
        $Log = Get-Content (Join-Path $RunRoot "tunnel.err.log") -Raw -ErrorAction SilentlyContinue
        if ($Log -match 'https://[a-z0-9-]+\.trycloudflare\.com') { $PublicUrl = $Matches[0]; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $PublicUrl) { throw "Tunnel did not issue an HTTPS URL." }
    }
    $State = @{
        url=$PublicUrl; port=$Port
        api_pid=$ApiProcess.Id; api_started=$ApiProcess.StartTime.ToUniversalTime().ToString("o")
        tunnel_pid=$TunnelProcess.Id; tunnel_started=$TunnelProcess.StartTime.ToUniversalTime().ToString("o")
    }
    $State | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunRoot "session.json") -Encoding utf8
    Write-Host "Public demo: $PublicUrl"
    Write-Host "Key is saved in secrets/public-demo-api-key.txt; share separately with the jury."
    Write-Host "Temporary URL: keep this computer awake. Stop using scripts/stop_public_demo.ps1."
} catch {
    if (-not $RestartApiOnly -and $TunnelProcess -and -not $TunnelProcess.HasExited) { $TunnelProcess.Kill() }
    if ($ApiProcess -and -not $ApiProcess.HasExited) { $ApiProcess.Kill() }
    throw
}
