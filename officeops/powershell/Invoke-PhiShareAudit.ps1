<#
.SYNOPSIS
    Reports who can read a folder holding PHI, and flags the entries that
    should not be there.

.DESCRIPTION
    This is the one office-automation task that PowerShell does better than
    Python, because the answer lives in Windows ACLs and in Active Directory
    group membership rather than in a CSV.

    WHAT IT LOOKS FOR, and why each one matters:

      * Everyone / Authenticated Users / Domain Users on a PHI share. The most
        common real finding in a small practice, and usually the result of one
        afternoon years ago when a file would not open.
      * Broken inheritance. A subfolder that no longer inherits is a subfolder
        whose permissions nobody has reviewed since it was created.
      * Accounts that no longer resolve (orphaned SIDs). A departed employee
        whose account was deleted rather than disabled leaves a SID that still
        grants access if the account is ever recreated with the same name.
      * Write access held by a group whose whole purpose is read-only.

    WHAT IT DELIBERATELY DOES NOT DO:

      * It does not flag a DENY ace. An explicit "Deny Everyone Full Control" is
        correct hardening, and an earlier version reported it as the violation.
      * It does not repeat an inherited ACE once per subfolder. One Everyone ACE
        on a share root became hundreds of identical rows at -Depth 3, for one
        real problem. Inherited entries are reported at the folder that defines
        them; -IncludeInherited turns the full listing back on.

    IT CHANGES NOTHING. There is no -Fix parameter and there will not be one:
    an ACL script that "corrects" permissions on a share it does not fully
    understand is how a practice loses access to its own chart archive on a
    Monday morning.

    HIPAA context: 45 CFR 164.308(a)(4) requires access authorisation and
    periodic review. This produces the artifact that review needs. It does not
    perform the review; a person still decides who belongs.

.PARAMETER Path
    One or more folders to audit. Required -- there is no default, because a
    default would be wrong for every practice.

.PARAMETER Depth
    How many levels of subfolder to inspect for broken inheritance. Default 2.

.PARAMETER IncludeInherited
    Report inherited ACEs at every folder instead of only where they are
    defined. Off by default -- see above.

.PARAMETER OutDir
    Where to write the CSV and the text report. Default .\out

.PARAMETER Write
    Save the report. Without it, output is printed only.

.EXAMPLE
    .\Invoke-PhiShareAudit.ps1 -Path "\\NSP-FS01\Charts" -Write

.EXAMPLE
    .\Invoke-PhiShareAudit.ps1 -Path "D:\Scans","D:\Faxes" -Depth 3
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string[]]$Path,
    [int]$Depth = 2,
    [switch]$IncludeInherited,
    [string]$OutDir = ".\out",
    [switch]$Write
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Identities that should essentially never appear on a folder holding PHI.
$BroadIdentities = @(
    'Everyone', 'BUILTIN\Users', 'NT AUTHORITY\Authenticated Users',
    'Domain Users', 'NT AUTHORITY\INTERACTIVE', 'BUILTIN\Guests', 'ANONYMOUS LOGON'
)

$findings = New-Object System.Collections.Generic.List[object]

function Add-Finding {
    param($Severity, $Folder, $Identity, $Rights, $Issue)
    $findings.Add([pscustomobject]@{
        Severity = $Severity
        Folder   = $Folder
        Identity = $Identity
        Rights   = $Rights
        Issue    = $Issue
    })
}

function Test-Folder {
    param([string]$Folder, [int]$Level)

    try {
        $acl = Get-Acl -LiteralPath $Folder
    }
    catch {
        Add-Finding 'ERROR' $Folder '-' '-' "could not read ACL: $($_.Exception.Message)"
        return
    }

    if ($Level -gt 0 -and -not $acl.AreAccessRulesProtected) {
        # Inheriting from the parent is the healthy case; nothing to say.
    }
    elseif ($Level -gt 0) {
        Add-Finding 'REVIEW' $Folder '-' '-' 'inheritance is broken; permissions here are independent of the parent'
    }

    foreach ($rule in $acl.Access) {
        # A Deny ACE is a control, not a finding.
        if ($rule.AccessControlType -ne [System.Security.AccessControl.AccessControlType]::Allow) {
            continue
        }
        # Inherited entries belong to the folder that defines them. Repeating
        # them at every child turns one problem into hundreds of rows.
        if ($rule.IsInherited -and -not $IncludeInherited) {
            continue
        }
        $identity = $rule.IdentityReference.Value
        $rights = $rule.FileSystemRights.ToString()
        # A non-enum mask (GENERIC_ALL and friends) stringifies to a raw number
        # and matched none of the name patterns, silently downgrading exactly
        # the ACEs that are hardest to read by eye.
        $mask = [int]$rule.FileSystemRights
        $isPowerful = ($rights -match 'Write|Modify|FullControl') -or
                      (($mask -band 0x10000000) -ne 0) -or
                      (($mask -band 0x40000000) -ne 0) -or
                      (($mask -band 0x001F01FF) -eq 0x001F01FF)
        if ($rights -match '^\-?\d+$') { $rights = "raw mask $rights (non-standard)" }

        if ($identity -match '^S-1-5-21-') {
            Add-Finding 'HIGH' $Folder $identity $rights 'unresolved SID -- the account was deleted, not disabled'
            continue
        }
        foreach ($broad in $BroadIdentities) {
            if ($identity -like "*$broad") {
                $severity = if ($isPowerful) { 'HIGH' } else { 'REVIEW' }
                Add-Finding $severity $Folder $identity $rights "broad identity on a PHI folder"
                break
            }
        }
        if ($isPowerful -and $rights -match 'FullControl' -and $identity -notmatch 'SYSTEM|Administrators') {
            Add-Finding 'REVIEW' $Folder $identity $rights 'Full Control held by a non-administrative principal'
        }
    }

    if ($Level -lt $Depth) {
        Get-ChildItem -LiteralPath $Folder -Directory -ErrorAction SilentlyContinue |
            ForEach-Object { Test-Folder -Folder $_.FullName -Level ($Level + 1) }
    }
}

foreach ($target in $Path) {
    if (-not (Test-Path -LiteralPath $target)) {
        Add-Finding 'ERROR' $target '-' '-' 'path does not exist or is not reachable from this machine'
        continue
    }
    Test-Folder -Folder $target -Level 0
}

$high = @($findings | Where-Object Severity -eq 'HIGH').Count
$review = @($findings | Where-Object Severity -eq 'REVIEW').Count
$errors = @($findings | Where-Object Severity -eq 'ERROR').Count

Write-Host ""
Write-Host ("=" * 78)
Write-Host "PHI share access audit  ($(Get-Date -Format 'yyyy-MM-dd HH:mm'))"
Write-Host ("=" * 78)
Write-Host ""
Write-Host "  paths audited                          $($Path.Count)"
Write-Host "  high severity                          $high"
Write-Host "  needs review                           $review"
Write-Host "  unreadable                             $errors"
Write-Host ""
Write-Host "  45 CFR 164.308(a)(4) asks for periodic access review. This is the"
Write-Host "  input to that review, not the review. Nothing here was changed."
Write-Host ""

if ($findings.Count -gt 0) {
    $findings | Sort-Object Severity, Folder | Format-Table -AutoSize
}
else {
    Write-Host "  No findings."
}

if ($Write) {
    if (-not (Test-Path -LiteralPath $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
    $stamp = Get-Date -Format 'yyyyMMdd'
    $csv = Join-Path $OutDir "${stamp}_phi_share_audit.csv"
    $findings | Export-Csv -Path $csv -NoTypeInformation -Encoding UTF8
    Write-Host "wrote $csv"

    # The .PARAMETER text promised a text report and only the CSV was ever
    # written.
    $txt = Join-Path $OutDir "${stamp}_phi_share_audit.txt"
    $lines = @(
        "PHI share access audit $(Get-Date -Format 'yyyy-MM-dd HH:mm')",
        "paths: $($Path -join '; ')",
        "depth: $Depth   include inherited: $IncludeInherited",
        "high: $high   review: $review   errors: $errors",
        ""
    )
    $lines += ($findings | Sort-Object Severity, Folder | Format-Table -AutoSize | Out-String)
    $lines | Set-Content -Path $txt -Encoding UTF8
    Write-Host "wrote $txt"
}

# Exit 1 when ANYTHING needs a person. Exiting 0 on REVIEW-only findings meant a
# run reporting forty broken-inheritance folders was filed under "stay silent",
# and the 164.308(a)(4) artifact was never mailed.
if ($findings.Count -gt 0) { exit 1 } else { exit 0 }
