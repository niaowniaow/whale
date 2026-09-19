# Download the complete Syzygy 3-4-5 set (~0.92 GB, 290 tables + checksum)
# into <repo>/tables/ (sequential single connection, polite to the mirror).
# Usage from repo root: powershell -File tools/dl_syzygy.ps1
# Re-runnable: skips files that already exist; verify with checksum.md5.
$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
$base = "http://tablebase.sesse.net/syzygy/3-4-5"
$dest = Join-Path $root "tables"
$index = Join-Path $root "sesse_index.html"
$log = Join-Path $root "dl_syzygy.log"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

function Log($msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    $line | Tee-Object -FilePath $log -Append | Out-Null
    Write-Host $line
}

if (-not (Test-Path $index)) {
    Log "fetching directory index..."
    & curl.exe -sS -f --retry 3 "$base/" -o $index
    if ($LASTEXITCODE -ne 0) { Log "failed to fetch index"; exit 1 }
}
$files = @("checksum.md5") + (
    (Select-String -Path $index -Pattern 'href="([^"]+\.rtb[WZwz])"' -AllMatches).Matches |
    ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
)
Log "total files: $($files.Count)"
$done = 0; $failed = @()
foreach ($f in $files) {
    $out = Join-Path $dest $f
    if ((Test-Path $out) -and ((Get-Item $out).Length -gt 0)) {
        $done++
        continue
    }
    & curl.exe -sS -f -C - --retry 3 --retry-delay 5 --max-time 600 "$base/$f" -o $out
    if ($LASTEXITCODE -eq 0 -and (Test-Path $out) -and ((Get-Item $out).Length -gt 0)) {
        $done++
    } else {
        $failed += $f
        Log "FAILED: $f"
    }
    if ($done % 25 -eq 0) { Log "progress: $done/$($files.Count)" }
    Start-Sleep -Milliseconds 300
}
Log "done: $done/$($files.Count) failed: $($failed.Count)"
if ($failed.Count -gt 0) { $failed | Out-File (Join-Path $root "dl_syzygy.failed.txt") }
