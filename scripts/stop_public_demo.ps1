$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$StatePath = Join-Path $RepoRoot "tmp/public-demo/session.json"
if (-not (Test-Path -LiteralPath $StatePath)) { throw "No saved public-demo session." }
$State = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
foreach ($Kind in @("tunnel", "api")) {
    $SavedId = $State."${Kind}_pid"
    $SavedStart = $State."${Kind}_started"
    $Process = Get-Process -Id $SavedId -ErrorAction SilentlyContinue
    if ($Process) {
        if ($Process.StartTime.ToUniversalTime().ToString("o") -ne $SavedStart) {
            throw "PID was reused; refusing to stop an unrelated process."
        }
        Stop-Process -Id $SavedId
    }
}
Write-Host "Public tunnel and demo API stopped. The saved URL is no longer active."
