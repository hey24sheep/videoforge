# Logs GPU memory (dedicated + shared, per process) and free RAM to a CSV while a build runs.
# Usage: powershell -File tools/memwatch.ps1 <out.csv> [max_minutes=60] [interval_s=2]
# Shared usage that rises during a GPU pass = VRAM spilled into system RAM (the user's "never spill" rule).
param([string]$csv, [int]$minutes = 60, [double]$interval = 2)
"time,pid,dedicated_mb,shared_mb,ram_free_gb" | Out-File -FilePath $csv -Encoding utf8
$end = (Get-Date).AddMinutes($minutes)
while ((Get-Date) -lt $end) {
    $t = Get-Date -Format 'HH:mm:ss'
    $free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
    $rows = @{}
    try {
        $c = Get-Counter '\GPU Process Memory(*)\Dedicated Usage', '\GPU Process Memory(*)\Shared Usage' -ErrorAction Stop
        foreach ($s in $c.CounterSamples) {
            if ($s.InstanceName -match 'pid_(\d+)') {
                $procId = $Matches[1]
                if (-not $rows.ContainsKey($procId)) { $rows[$procId] = @(0, 0) }
                if ($s.Path -match 'dedicated') { $rows[$procId][0] += $s.CookedValue } else { $rows[$procId][1] += $s.CookedValue }
            }
        }
    } catch { }
    $py = @(Get-Process python, ffmpeg -ErrorAction SilentlyContinue | ForEach-Object { "$($_.Id)" })
    $hit = $false
    foreach ($k in $rows.Keys) {
        if ($py -contains $k) {
            "$t,$k,$([math]::Round($rows[$k][0] / 1MB)),$([math]::Round($rows[$k][1] / 1MB)),$free" | Out-File -FilePath $csv -Append -Encoding utf8
            $hit = $true
        }
    }
    if (-not $hit) { "$t,-,0,0,$free" | Out-File -FilePath $csv -Append -Encoding utf8 }
    Start-Sleep -Seconds $interval
}
