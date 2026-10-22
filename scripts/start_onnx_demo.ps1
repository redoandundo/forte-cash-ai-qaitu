[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$ArchivePath,

    [Parameter(Mandatory = $false)]
    [string]$PythonExecutable = "python",

    [Parameter(Mandatory = $false)]
    [ValidateRange(1, 65535)]
    [int]$Port = 8123,

    [Parameter(Mandatory = $false)]
    [switch]$PreferCpu
)

$ErrorActionPreference = "Stop"

$ExpectedArchiveSha256 = "c387bdea9bd80a124c5587fccfcec3243d2550658ee09ca6834acaa9e54a1241"
$ExpectedModelSha256 = "00ae541eac23249dd4751093ce276bfa9c14afca2d5f15f4fec9719c009ae52d"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$BundleRoot = Join-Path $RepoRoot "artifacts\ml\bundles"
$BundlePath = Join-Path $BundleRoot "baseline-audited-v4"
$ModelPath = Join-Path $BundlePath "model.onnx"

function Get-LowerSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-ExpectedHash {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Expected,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $Actual = Get-LowerSha256 -Path $Path
    if ($Actual -ne $Expected) {
        throw "$Label SHA-256 mismatch. Expected $Expected, got $Actual. File: $Path"
    }

    Write-Host "$Label SHA-256 verified: $Actual" -ForegroundColor Green
}

if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
    if ([string]::IsNullOrWhiteSpace($ArchivePath)) {
        throw "The verified ONNX bundle is absent. Pass -ArchivePath with baseline-audited-v4.zip. Expected archive SHA-256: $ExpectedArchiveSha256"
    }

    $ResolvedArchivePath = (Resolve-Path -LiteralPath $ArchivePath).Path
    Assert-ExpectedHash -Path $ResolvedArchivePath -Expected $ExpectedArchiveSha256 -Label "Archive"

    New-Item -ItemType Directory -Path $BundleRoot -Force | Out-Null
    $ExtractionRoot = Join-Path $BundleRoot (".baseline-audited-v4-extract-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $ExtractionRoot | Out-Null

    try {
        Expand-Archive -LiteralPath $ResolvedArchivePath -DestinationPath $ExtractionRoot
        $Models = @(Get-ChildItem -LiteralPath $ExtractionRoot -Filter "model.onnx" -File -Recurse)
        if ($Models.Count -ne 1) {
            throw "Expected exactly one model.onnx in the archive, found $($Models.Count)."
        }

        $ExtractedBundlePath = $Models[0].Directory.FullName
        Assert-ExpectedHash -Path $Models[0].FullName -Expected $ExpectedModelSha256 -Label "model.onnx"

        if (Test-Path -LiteralPath $BundlePath) {
            throw "Bundle destination already exists but has no model.onnx: $BundlePath. Refusing to overwrite it."
        }

        Move-Item -LiteralPath $ExtractedBundlePath -Destination $BundlePath
        Write-Host "Bundle installed: $BundlePath" -ForegroundColor Green
    }
    finally {
        if (Test-Path -LiteralPath $ExtractionRoot) {
            $BundleRootFull = [IO.Path]::GetFullPath($BundleRoot).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
            $ExtractionRootFull = [IO.Path]::GetFullPath($ExtractionRoot)
            if (-not $ExtractionRootFull.StartsWith($BundleRootFull, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Refusing to remove an extraction directory outside the bundle root: $ExtractionRootFull"
            }
            Remove-Item -LiteralPath $ExtractionRoot -Recurse -Force
        }
    }
}

Assert-ExpectedHash -Path $ModelPath -Expected $ExpectedModelSha256 -Label "model.onnx"

$PythonCommand = Get-Command $PythonExecutable -ErrorAction Stop
$SourcePath = Join-Path $RepoRoot "src"
$env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $SourcePath
} else {
    "$SourcePath$([IO.Path]::PathSeparator)$env:PYTHONPATH"
}
if ([string]::IsNullOrWhiteSpace($env:FORTEBANK_API_KEY)) {
    $env:FORTEBANK_API_KEY = "demo-secret"
}
$env:QAITU_INFERENCE_MODE = "onnx"
$env:QAITU_MODEL_BUNDLE = $BundlePath
$env:QAITU_ALLOW_DEVELOPMENT_MODEL = "true"
$env:QAITU_EXPECTED_MODEL_SHA256 = $ExpectedModelSha256
$env:QAITU_PREFER_CUDA = if ($PreferCpu) { "false" } else { "true" }
$env:QAITU_MAX_CONCURRENT_JOBS = "1"

Write-Host "Starting verified ONNX demo at http://127.0.0.1:$Port/" -ForegroundColor Cyan
Write-Host "The API will fail closed if the bundle contract or ONNX runtime is invalid." -ForegroundColor Cyan

Push-Location $RepoRoot
try {
    & $PythonCommand.Source -m uvicorn fortebank_audit.api:app --host 127.0.0.1 --port $Port
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
