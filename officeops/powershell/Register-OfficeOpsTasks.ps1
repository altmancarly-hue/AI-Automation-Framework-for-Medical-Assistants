<#
.SYNOPSIS
    Registers the officeops daily jobs in Windows Task Scheduler.

.DESCRIPTION
    A script nobody runs is a script that does not exist. This registers the
    handful that belong on a timer, at the times they are actually useful:

      * confirm-list at 15:00 -- late enough that the day's changes are in,
        early enough that somebody can still make the calls.
      * fridge-log at 07:30 and 17:00 -- the two times CDC asks for a reading,
        so a missed reading is caught the same day rather than at month end.
      * vaccine-inventory and credential-tracker on Mondays.
      * backup health and the PHI share audit at 08:00, daily and weekly.

    REQUIRES ELEVATION. Register-ScheduledTask needs an administrative token,
    and the tasks are registered with -LogonType S4U so they run on a locked or
    logged-out workstation -- without a principal they only run while that user
    is signed in, so the 07:30 fridge job would not have run at all.

    -WhatIf is the recommended first run and lists exactly what would be
    registered. ConfirmImpact is High, so an unqualified run prompts.

    ON ALERTING: Windows Task Scheduler has no conditional-on-exit-code action,
    and its "Send an e-mail" action was removed after Windows 7. Each job is
    therefore registered to run a small wrapper (see -WrapperPath) that inspects
    %ERRORLEVEL% and calls whatever the practice uses to send mail. Registering
    the bare command and telling people to "attach a send-email action on
    non-zero exit" describes a feature that does not exist.

.PARAMETER RepoRoot
    Path to the nsp repo (the folder containing the officeops package).

.PARAMETER ExportDir
    Where the practice's nightly exports land.

.PARAMETER Python
    Python executable. Default "python".

.EXAMPLE
    .\Register-OfficeOpsTasks.ps1 -RepoRoot C:\nsp -ExportDir D:\Exports -WhatIf
#>
#Requires -RunAsAdministrator
[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'High')]
param(
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)][string]$ExportDir,
    # "python" on Windows is frequently the Microsoft Store stub. Resolve it
    # before registering six jobs that all fail the same way at 07:30.
    [string]$Python = "python.exe",
    [string]$OutDir = "C:\OfficeOps\out",
    [string]$WrapperPath = "",
    [string]$RunAsUser = "$env:USERDOMAIN\$env:USERNAME"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$resolved = Get-Command $Python -ErrorAction SilentlyContinue
if (-not $resolved) {
    Write-Host "REFUSED: '$Python' is not on PATH for this account." -ForegroundColor Red
    Write-Host "         Pass -Python with a full path, e.g. C:\Python311\python.exe"
    exit 2
}
$Python = $resolved.Source
if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot 'officeops'))) {
    Write-Host "REFUSED: $RepoRoot does not contain an 'officeops' folder." -ForegroundColor Red
    exit 2
}

$jobs = @(
    @{ Name = 'OfficeOps-ConfirmList'; Time = '15:00'; Days = 'Daily'
       Args = "-m officeops confirm-list `"$ExportDir\schedule.csv`" --write --out `"$OutDir`"" }
    @{ Name = 'OfficeOps-FridgeLog-AM'; Time = '07:30'; Days = 'Daily'
       Args = "-m officeops fridge-log `"$ExportDir\fridge_log.csv`" --write --out `"$OutDir`"" }
    @{ Name = 'OfficeOps-FridgeLog-PM'; Time = '17:00'; Days = 'Daily'
       Args = "-m officeops fridge-log `"$ExportDir\fridge_log.csv`" --write --out `"$OutDir`"" }
    @{ Name = 'OfficeOps-VaccineInventory'; Time = '08:00'; Days = 'Monday'
       Args = "-m officeops vaccine-inventory `"$ExportDir\vaccine_inventory.csv`" --write --out `"$OutDir`"" }
    @{ Name = 'OfficeOps-Credentials'; Time = '08:15'; Days = 'Monday'
       Args = "-m officeops credential-tracker `"$ExportDir\credentials.csv`" --write --out `"$OutDir`"" }
    @{ Name = 'OfficeOps-LabFollowup'; Time = '08:30'; Days = 'Daily'
       Args = "-m officeops lab-followup `"$ExportDir\orders.csv`" --write --out `"$OutDir`"" }
)

# The two PowerShell jobs the help text advertises. Registered here rather than
# described and omitted -- a practice that runs this believed backup monitoring
# was scheduled, and it was not.
$psJobs = @(
    @{ Name = 'OfficeOps-BackupHealth'; Time = '08:00'; Days = 'Daily'
       Script = 'Test-BackupHealth.ps1'
       Args = "-BackupPath `"$ExportDir`" -Write -OutDir `"$OutDir`"" }
    @{ Name = 'OfficeOps-PhiShareAudit'; Time = '08:45'; Days = 'Monday'
       Script = 'Invoke-PhiShareAudit.ps1'
       Args = "-Path `"$ExportDir`" -Write -OutDir `"$OutDir`"" }
)

$failed = 0
$registered = 0

function Register-Job {
    param($Name, $Time, $Days, $Execute, $Arguments, $Description)

    $action = New-ScheduledTaskAction -Execute $Execute -Argument $Arguments -WorkingDirectory $RepoRoot
    $trigger = if ($Days -eq 'Daily') {
        New-ScheduledTaskTrigger -Daily -At $Time
    }
    else {
        New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Days -At $Time
    }
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries
    # S4U: runs whether or not the account is signed in, without storing a
    # password. Without a principal the job only runs while that user is logged
    # on, which is not true of a front-desk workstation at 07:30.
    $principal = New-ScheduledTaskPrincipal -UserId $RunAsUser -LogonType S4U -RunLevel Highest

    if (-not $PSCmdlet.ShouldProcess($Name, "register scheduled task at $Time ($Days)")) {
        return
    }
    try {
        Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger `
            -Settings $settings -Principal $principal -Description $Description -Force | Out-Null
        Write-Host "registered $Name at $Time ($Days)"
        $script:registered++
    }
    catch {
        Write-Host "FAILED to register ${Name}: $($_.Exception.Message)" -ForegroundColor Red
        $script:failed++
    }
}

foreach ($job in $jobs) {
    $execute = $Python
    $arguments = $job.Args
    if ($WrapperPath) {
        # The wrapper inspects %ERRORLEVEL% and mails on non-zero. See the help.
        $execute = $WrapperPath
        $arguments = "`"$Python`" $($job.Args)"
    }
    Register-Job -Name $job.Name -Time $job.Time -Days $job.Days `
        -Execute $execute -Arguments $arguments -Description "officeops: $($job.Args)"
}

foreach ($job in $psJobs) {
    $script = Join-Path $RepoRoot "officeops\powershell\$($job.Script)"
    if (-not (Test-Path -LiteralPath $script)) {
        Write-Host "SKIPPED $($job.Name): $script not found" -ForegroundColor Yellow
        $failed++
        continue
    }
    Register-Job -Name $job.Name -Time $job.Time -Days $job.Days `
        -Execute "powershell.exe" `
        -Arguments "-NoProfile -ExecutionPolicy Bypass -File `"$script`" $($job.Args)" `
        -Description "officeops: $($job.Script)"
}

Write-Host ""
Write-Host "registered: $registered   failed: $failed"
Write-Host ""
Write-Host "Every job exits 0 clean, 1 with findings, 2 when the input could not be read."
Write-Host "Task Scheduler cannot act on an exit code and its send-email action was"
Write-Host "removed after Windows 7, so pass -WrapperPath pointing at a .cmd like:"
Write-Host ""
Write-Host "    @echo off"
Write-Host "    %*"
Write-Host "    if errorlevel 1 powershell -NoProfile -File C:\OfficeOps\Send-Report.ps1 -Code %ERRORLEVEL%"
Write-Host ""
if ($failed -gt 0) { exit 1 } else { exit 0 }
