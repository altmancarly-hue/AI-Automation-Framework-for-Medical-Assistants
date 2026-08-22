<#
.SYNOPSIS
    Verifies that last night's backup actually happened and is plausibly usable.

.DESCRIPTION
    "The backup is running" is a belief, not a fact, and it is the belief that
    ends practices. This checks four things a backup product's own green tick
    does not always check:

      1. A NEW backup file exists since the cutoff. Not "a file exists" -- a
         file NEWER than the last run should have produced.
      2. It is not suspiciously small. A backup that shrank by more than a
         threshold against the previous one is the classic silent failure: the
         job ran, the source path was wrong, and it dutifully backed up an
         empty folder every night for four months.
      3. The count of restore points is what the retention policy says. Silent
         pruning is how a practice discovers it has one night of history.
      4. The Windows event logs have no backup errors since the cutoff. Several
         logs are checked, not just Windows Server Backup: a practice running
         Veeam, Datto, Acronis or a Synology appliance logs elsewhere, and an
         earlier version queried one log, found nothing, and printed a
         reassuring "0" for a check it had not performed. If none of the known
         logs exists on this machine, it says so instead of reporting zero.

    IT DOES NOT TEST A RESTORE, and no script can claim a backup is good
    without one. Restore testing is a calendar item with a human on it. This
    tells you the far more common thing: that the job did not run at all.

.PARAMETER BackupPath
    Folder holding the backup files or restore points.

.PARAMETER Pattern
    File pattern to count. Default * (every file). Note that "*.*" -- the
    intuitive spelling -- excludes extensionless files.

.PARAMETER EventLogs
    Which event logs to check for backup errors. Defaults to the common ones.

.PARAMETER MaxAgeHours
    How recent the newest backup must be. Default 26 -- a little over a day, so
    a nightly job that slipped an hour does not raise a false alarm.

.PARAMETER MinRestorePoints
    How many files the retention policy should be keeping. Default 7.

.PARAMETER ShrinkPercent
    Alarm if the newest backup is smaller than the previous one by more than
    this percentage. Default 40.

.EXAMPLE
    .\Test-BackupHealth.ps1 -BackupPath "E:\Backups\EHR" -MinRestorePoints 14

.EXAMPLE
    .\Test-BackupHealth.ps1 -BackupPath "\\NAS\backups" -Pattern "*.vbk" -Write
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BackupPath,
    [string]$Pattern = "*",
    [string[]]$EventLogs = @(
        'Microsoft-Windows-Backup', 'Veeam Agent', 'Veeam Backup', 'Application'
    ),
    [string]$EventProviderFilter = 'backup|veeam|datto|acronis|shadow|vss',
    [int]$MaxAgeHours = 26,
    [int]$MinRestorePoints = 7,
    [int]$ShrinkPercent = 40,
    [string]$OutDir = ".\out",
    [switch]$Write
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$problems = New-Object System.Collections.Generic.List[string]
$facts = [ordered]@{}

if (-not (Test-Path -LiteralPath $BackupPath)) {
    Write-Host "REFUSED: $BackupPath does not exist or is not reachable from this machine." -ForegroundColor Red
    Write-Host "         An unreachable backup target is itself the finding."
    exit 2
}

$files = @(Get-ChildItem -LiteralPath $BackupPath -Filter $Pattern -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending)

$facts['backup path'] = $BackupPath
$facts['files matching pattern'] = $files.Count

if ($files.Count -eq 0) {
    $problems.Add("no files matching '$Pattern' in $BackupPath -- the job has never written here, or it writes somewhere else")
}
else {
    $newest = $files[0]
    $ageHours = [math]::Round(((Get-Date) - $newest.LastWriteTime).TotalHours, 1)
    $facts['newest backup'] = $newest.Name
    $facts['newest written'] = $newest.LastWriteTime.ToString('yyyy-MM-dd HH:mm')
    $facts['age (hours)'] = $ageHours
    $facts['newest size (MB)'] = [math]::Round($newest.Length / 1MB, 1)

    if ($ageHours -gt $MaxAgeHours) {
        $problems.Add("newest backup is $ageHours hours old; expected within $MaxAgeHours")
    }
    if ($files.Count -lt $MinRestorePoints) {
        if ($files.Count -eq 1) {
            # A product that overwrites one full file legitimately has one.
            $problems.Add("only 1 file matches '$Pattern'. If this product overwrites a single full backup, pass -MinRestorePoints 1 and note that the size-comparison check cannot run.")
        }
        else {
            $problems.Add("$($files.Count) restore point(s) retained; policy expects at least $MinRestorePoints")
        }
    }
    if ($files.Count -ge 2) {
        $previous = $files[1]
        $facts['previous size (MB)'] = [math]::Round($previous.Length / 1MB, 1)
        if ($previous.Length -gt 0) {
            $change = [math]::Round((($newest.Length - $previous.Length) / $previous.Length) * 100, 1)
            $facts['size change (%)'] = $change
            if ($change -lt (-1 * $ShrinkPercent)) {
                $problems.Add("newest backup is $([math]::Abs($change))% smaller than the previous one -- check the source path before trusting it")
            }
        }
    }
    if ($newest.Length -eq 0) {
        $problems.Add("newest backup is zero bytes")
    }
}

# Event logs, best effort across the products a small practice actually runs.
$since = (Get-Date).AddHours(-1 * $MaxAgeHours)
$checked = New-Object System.Collections.Generic.List[string]
$found = New-Object System.Collections.Generic.List[object]
foreach ($log in $EventLogs) {
    try {
        $events = @(Get-WinEvent -FilterHashtable @{
            LogName = $log; StartTime = $since; Level = 1, 2
        } -ErrorAction Stop)
        $checked.Add($log)
        foreach ($event in $events) {
            # The Application log carries everything; keep only what looks like
            # a backup product.
            if ($log -eq 'Application' -and $event.ProviderName -notmatch $EventProviderFilter) {
                continue
            }
            $found.Add($event)
        }
    }
    catch {
        # Log absent, or no matching events. Both are normal.
        if ($_.Exception.Message -notmatch 'No events were found') { continue }
        $checked.Add($log)
    }
}
if ($checked.Count -eq 0) {
    $facts['event logs checked'] = 'NONE of the known backup logs exist on this machine'
    $problems.Add("no backup event log found on this machine (looked for: $($EventLogs -join ', ')) -- the event-log check did NOT run, and a zero here would have been misleading")
}
else {
    $facts['event logs checked'] = $checked -join ', '
    $facts['backup errors in event log'] = $found.Count
    foreach ($event in $found | Select-Object -First 5) {
        $problems.Add("event log: $($event.TimeCreated.ToString('yyyy-MM-dd HH:mm')) $($event.Message -split "`n" | Select-Object -First 1)")
    }
}

Write-Host ""
Write-Host ("=" * 78)
Write-Host "Backup health check  ($(Get-Date -Format 'yyyy-MM-dd HH:mm'))"
Write-Host ("=" * 78)
Write-Host ""
foreach ($key in $facts.Keys) {
    Write-Host ("  {0,-38} {1}" -f $key, $facts[$key])
}
Write-Host ""
if ($problems.Count -eq 0) {
    Write-Host "  No problems found." -ForegroundColor Green
}
else {
    foreach ($problem in $problems) { Write-Host "  ! $problem" -ForegroundColor Yellow }
}
Write-Host ""
Write-Host "  This does NOT prove a restore works. Nothing but a test restore does."
Write-Host "  Put one on the calendar; this tells you the job ran."
Write-Host ""

if ($Write) {
    if (-not (Test-Path -LiteralPath $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
    $stamp = Get-Date -Format 'yyyyMMdd'
    $path = Join-Path $OutDir "${stamp}_backup_health.txt"
    $lines = @("Backup health check $(Get-Date -Format 'yyyy-MM-dd HH:mm')", "")
    foreach ($key in $facts.Keys) { $lines += ("{0,-38} {1}" -f $key, $facts[$key]) }
    $lines += ""
    $lines += $problems
    $lines | Set-Content -Path $path -Encoding UTF8
    Write-Host "wrote $path"
}

if ($problems.Count -gt 0) { exit 1 } else { exit 0 }
