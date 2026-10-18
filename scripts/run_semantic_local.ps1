param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$Bundle,
    [Parameter(Mandatory = $true)][string]$QwenModel,
    [Parameter(Mandatory = $true)][string]$QwenManifest,
    [ValidateRange(1, 65535)][int]$Port = 8010,
    [switch]$AllowDevelopmentModel,
    [switch]$ReviewedRules
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $env:FORTEBANK_API_KEY) { throw 'Set FORTEBANK_API_KEY before launch.' }
$env:QAITU_INFERENCE_MODE = 'onnx'
$env:QAITU_MODEL_BUNDLE = (Resolve-Path -LiteralPath $Bundle).Path
$env:QAITU_ALLOW_DEVELOPMENT_MODEL = if ($AllowDevelopmentModel) { 'true' } else { 'false' }
$env:QAITU_PREFER_CUDA = 'false'
$env:QAITU_MAX_CONCURRENT_JOBS = '1'
$env:QAITU_VLM_MODE = 'qwen'
$env:QAITU_VLM_MODEL_PATH = (Resolve-Path -LiteralPath $QwenModel).Path
$env:QAITU_VLM_MANIFEST_PATH = (Resolve-Path -LiteralPath $QwenManifest).Path
$env:QAITU_VLM_REVIEWED_RULES = if ($ReviewedRules) { '1' } else { '0' }
$env:QAITU_VLM_MAX_CONCURRENT_JOBS = '1'
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONPATH = (Join-Path $repoRoot 'src') + [IO.Path]::PathSeparator + $env:PYTHONPATH
Push-Location -LiteralPath $repoRoot
try {
    & $Python -m uvicorn fortebank_audit.api:app --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Semantic API exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
