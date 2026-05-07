param([string]$Script = "simulations\mfm_vortex.mx3")
$ErrorActionPreference = "Stop"
$exe = Get-Command mumax3 -ErrorAction SilentlyContinue
if (-not $exe) {
    Write-Error "mumax3 не найден в PATH. Установите MuMax3 и добавьте каталог с mumax3.exe в PATH."
}
& mumax3 $Script
