$ErrorActionPreference = 'Stop'
$dashboardDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$port = 4174
$codexPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'

if (Get-Command py -ErrorAction SilentlyContinue) {
    Write-Host "MORE-HD Training Observatory (MADBase) tersedia di http://localhost:$port" -ForegroundColor Cyan
    Set-Location -LiteralPath $dashboardDir
    & py -m http.server $port
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "MORE-HD Training Observatory (MADBase) tersedia di http://localhost:$port" -ForegroundColor Cyan
    Set-Location -LiteralPath $dashboardDir
    & python -m http.server $port
} elseif (Test-Path -LiteralPath $codexPython) {
    Write-Host "MORE-HD Training Observatory (MADBase) tersedia di http://localhost:$port" -ForegroundColor Cyan
    Set-Location -LiteralPath $dashboardDir
    & $codexPython -m http.server $port
} else {
    throw 'Python tidak ditemukan. Instal Python 3 atau jalankan server statis pilihan Anda pada folder dashboard.'
}
