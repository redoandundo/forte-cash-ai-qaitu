[CmdletBinding()]
param([int]$Port = 8001)
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RunRoot = Join-Path $RepoRoot 'tmp/gpu-server'
$SecretsRoot = Join-Path $RepoRoot 'secrets'
New-Item -ItemType Directory -Force -Path $RunRoot,$SecretsRoot | Out-Null
$Python = Join-Path $RepoRoot '.venv-ml/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) { $Python = Join-Path $RepoRoot '.venv-run/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $Python)) { throw 'GPU environment missing. Run scripts/setup_local.py --qwen on this server.' }
$Tunnel = Join-Path $RunRoot 'cloudflared.exe'
if (-not (Test-Path -LiteralPath $Tunnel)) {
    # Download a fixed official release, never an executable supplied by a site visitor.
    Invoke-WebRequest 'https://github.com/cloudflare/cloudflared/releases/download/2026.10.1/cloudflared-windows-amd64.exe' -OutFile $Tunnel
}
$SessionPath = Join-Path $RunRoot 'session.json'
if (Test-Path -LiteralPath $SessionPath) {
    $Prior = Get-Content -LiteralPath $SessionPath -Raw | ConvertFrom-Json
    $Api = Get-Process -Id $Prior.api_pid -ErrorAction SilentlyContinue
    $Proxy = Get-Process -Id $Prior.tunnel_pid -ErrorAction SilentlyContinue
    if ($Api -and $Proxy -and
        $Api.StartTime.ToUniversalTime().ToString('o') -eq $Prior.api_started -and
        $Proxy.StartTime.ToUniversalTime().ToString('o') -eq $Prior.tunnel_started) {
        Write-Host ('Already running: ' + $Prior.url)
        Write-Host 'Connection settings: secrets/gpu-backend.env'
        exit 0
    }
    throw 'Previous server session is incomplete. Stop it with scripts/stop_gpu_server.ps1 first.'
}
$Api = $null
$Proxy = $null
try {
    $Api = Start-Process -FilePath $Python -ArgumentList @('scripts/serve_site.py','--backend-only','--port',"$Port") -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $RunRoot 'api.out.log') -RedirectStandardError (Join-Path $RunRoot 'api.err.log')
    $Ready = $false
    for ($Attempt = 0; $Attempt -lt 120; $Attempt++) {
        if ($Api.HasExited) { throw 'GPU API exited. Inspect tmp/gpu-server/api.err.log.' }
        try {
            $Health = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2
            if ($Health.model_ready -and $Health.semantic_mode -eq 'qwen' -and $Health.semantic_model_sha256) { $Ready = $true; break }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    if (-not $Ready) { throw 'Qwen did not become ready; no public tunnel opened.' }
    $Proxy = Start-Process -FilePath $Tunnel -ArgumentList @('tunnel','--no-autoupdate','--protocol','http2','--url',"http://127.0.0.1:$Port") -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $RunRoot 'tunnel.out.log') -RedirectStandardError (Join-Path $RunRoot 'tunnel.err.log')
    $PublicUrl = $null
    for ($Attempt = 0; $Attempt -lt 90; $Attempt++) {
        if ($Proxy.HasExited) { throw 'Tunnel exited; inspect tmp/gpu-server/tunnel.err.log.' }
        $Log = Get-Content -LiteralPath (Join-Path $RunRoot 'tunnel.err.log') -Raw -ErrorAction SilentlyContinue
        if ($Log -match 'https://[a-z0-9-]+\.trycloudflare\.com') { $PublicUrl = $Matches[0]; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $PublicUrl) { throw 'Tunnel did not provide an address.' }
    $ApiKey = if ($env:FORTEBANK_API_KEY) { $env:FORTEBANK_API_KEY } else { (Get-Content -LiteralPath (Join-Path $SecretsRoot 'site.key') -Raw).Trim() }
    [IO.File]::WriteAllText((Join-Path $SecretsRoot 'gpu-backend.env'), "QAITU_GPU_API_URL=$PublicUrl`nQAITU_GPU_API_KEY=$ApiKey`n")
    @{url=$PublicUrl; port=$Port; api_pid=$Api.Id; api_started=$Api.StartTime.ToUniversalTime().ToString('o'); tunnel_pid=$Proxy.Id; tunnel_started=$Proxy.StartTime.ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath $SessionPath -Encoding utf8
    Write-Host "GPU backend: $PublicUrl"
    Write-Host 'Private connection settings: secrets/gpu-backend.env (for the website server, not browsers).'
    Write-Host 'Keep this PC powered on and awake. The temporary address changes after a tunnel restart.'
} catch {
    if ($Proxy -and -not $Proxy.HasExited) { Stop-Process -InputObject $Proxy }
    if ($Api -and -not $Api.HasExited) {
        # Windows virtual environments may spawn a child Python interpreter.
        Get-CimInstance Win32_Process -Filter "ParentProcessId=$($Api.Id)" | Where-Object { $_.Name -eq 'python.exe' } | ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
        Stop-Process -InputObject $Api
    }
    throw
}
