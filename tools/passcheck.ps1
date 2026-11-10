# After a build/pass: stop the memwatch logger, report leftover python/ffmpeg processes, current GPU state, and the
# peaks from the memwatch CSV (peak dedicated VRAM, peak shared = spill, lowest free RAM).
# Usage: powershell -File tools/passcheck.ps1 <memwatch.csv>
param([string]$csv)
$stray = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'memwatch\.ps1' })
foreach ($p in $stray) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
$left = @(Get-Process python, ffmpeg -ErrorAction SilentlyContinue).Count
$gpu = (& nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader,nounits) -split ',\s*'
$d = 0; $s = 0; $r = 999
if (Test-Path $csv) {
    foreach ($row in (Import-Csv $csv)) {
        if ([double]$row.dedicated_mb -gt $d) { $d = [double]$row.dedicated_mb }
        if ([double]$row.shared_mb -gt $s) { $s = [double]$row.shared_mb }
        if ([double]$row.ram_free_gb -lt $r) { $r = [double]$row.ram_free_gb }
    }
}
"strays stopped: $(($stray | ForEach-Object { $_.ProcessId }) -join ' '); python/ffmpeg left: $left; GPU $($gpu[0]) MiB, $($gpu[1]) C; peak VRAM $d MB, peak shared $s MB, min free RAM $r GB"
