# GPU server on the participant computer

Double-click `START_GPU_SERVER.cmd` on the GPU computer. It reuses the existing
Python environment and Qwen weights, starts the ONNX + semantic Qwen API on
loopback port 8001 and opens an HTTPS Cloudflare tunnel. No UI, model or secret
is installed on the jury's computer. Only the website server calls this API.

Private connection settings are written to `secrets/gpu-backend.env`:
`QAITU_GPU_API_URL` is the HTTPS base URL; `QAITU_GPU_API_KEY` is the value for
the `X-API-Key` header. These are handoff settings, not an automatically enabled
remote-runtime configuration in the teamlead website. Its server must forward
uploads to `POST /api/v1/analyze`, retain the returned `job_id` and poll
`GET /api/v1/jobs/{job_id}`. Keep its existing browser sessions and per-user job
ownership. Never send the GPU API key to browser JavaScript.

`GET /health` reports `semantic_mode=qwen`, the base model SHA and server-side
compute. No browser GPU is used. The trained binary adapter is not loaded for
semantic dossier generation. Public results include descriptions and JPEGs;
this endpoint does not implement the website's internal progress/rule-review
routes. Those remain the responsibility of the website server.

One video runs at a time; concurrent submissions return 429. Analysis failures
return `failed`. The backend root does not serve the older website. Existing
public website URLs are not changed by this launcher. No remote host was
reconfigured by creating the tunnel.

Keep this computer awake and connected to power and internet. The launcher
runs services in the background; use `scripts/stop_gpu_server.ps1` to stop the
managed processes. It is not installed as a Windows boot service. Quick Tunnel
addresses are temporary and change on restart; update the website server's
settings after that. A persistent address requires a named tunnel/domain.
Traffic passes through Cloudflare; AI inference remains on this computer.

Reference: https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/
