$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$SessionPath = Join-Path $RepoRoot 'tmp/gpu-server/session.json'
if (-not (Test-Path -LiteralPath $SessionPath)) { Write-Host 'No managed GPU server session.'; exit 0 }
$Session = Get-Content -LiteralPath $SessionPath -Raw | ConvertFrom-Json
foreach ($Kind in @('tunnel','api')) {
    $Process = Get-Process -Id $Session."${Kind}_pid" -ErrorAction SilentlyContinue
    if (-not $Process) { continue }
    if ($Process.StartTime.ToUniversalTime().ToString('o') -ne $Session."${Kind}_started") { throw 'Process identity changed; refusing to stop an unrelated process.' }
    if ($Kind -eq 'api') {
        Get-CimInstance Win32_Process -Filter "ParentProcessId=$($Process.Id)" | Where-Object { $_.Name -eq 'python.exe' } | ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
    }
    Stop-Process -InputObject $Process
}
Remove-Item -LiteralPath $SessionPath
Write-Host 'GPU server and tunnel stopped.'
