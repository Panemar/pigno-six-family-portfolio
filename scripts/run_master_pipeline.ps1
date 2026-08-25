param([switch]$AuditOnly)
$ErrorActionPreference = 'Stop'
if (-not $AuditOnly) { throw 'La reejecución completa requiere un nuevo run_id y copia de trabajo; no sobrescriba la evidencia congelada.' }
$python = 'C:\Users\yunim\Documents\BRIDGE\pigno_dynamic_vscode_pipeline_v1_2\.venv\Scripts\python.exe'
& $python (Join-Path $PSScriptRoot '87_verify_frozen_final_package.py')
if ($LASTEXITCODE -ne 0) { throw 'La verificación read-only del paquete falló.' }
