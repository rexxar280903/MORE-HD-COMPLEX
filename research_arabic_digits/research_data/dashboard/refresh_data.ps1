$ErrorActionPreference = 'Stop'
$dashboardDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$builder = Join-Path $dashboardDir 'build_data.py'
$codexPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -X utf8 $builder
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python -X utf8 $builder
} elseif (Test-Path -LiteralPath $codexPython) {
    & $codexPython -X utf8 $builder
} else {
    throw 'Python tidak ditemukan. Instal Python 3 untuk memperbarui data dashboard.'
}
