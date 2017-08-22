<#
.SYNOPSIS
Retrieves and extracts the GTK lirary from the "http://win32builder.gnome.org/" page.
.EXAMPLE
Get-GTKPlus
#>

# 3.0?
# Import-Module "PS-Zip.psm1" -Force
# Import-Module "E:\python-pcl\pkg-config\PS-Zip.psm1"

# current officially supported version
$gtkVersion = "3.10.4"
$gtkDate = "20131202"

# FUll versioned pack file name to download

# determine 32 or 64bit OS?
# https://social.technet.microsoft.com/Forums/office/en-US/5dfeb3ab-6265-40cd-a4ac-05428b9db5c3/determine-32-or-64bit-os?forum=winserverpowershell
# https://sqlpowershell.wordpress.com/2014/01/06/powershell-find-os-architecture-32-bit-or-64-bit-of-local-or-remote-machines-using-powershell/
if ([System.IntPtr]::Size -eq 4)
{
    # 32 bit logic here
    # Write "32-bit OS"
    $os_bit = "win32"
}
else
{
    # 64 bit logic here
    # Write "64-bit OS"
    $os_bit = "win64"
}

$zipFilePath = "gtk+-bundle_$gtkVersion-" + "$gtkDate" + "_$os_bit.zip"

# base URL to download the pack file from
$SourceURLBase = "http://win32builder.gnome.org/$zipFilePath"

# download the pack and extract the files into the curent directory 
# How to get the current directory of the cmdlet being executed
# http://stackoverflow.com/questions/8343767/how-to-get-the-current-directory-of-the-cmdlet-being-executed
$dstPath = (Get-Item -Path ".\" -Verbose).FullName
$dstFile = $zipFilePath

# Version Check
# PowerShell Version 2.0
# 1.0 Blank
# $PSVersionTable

# Download gtk
# Write $SourceURLBase
# PowerShell Version 3.0
# Invoke-WebRequest -UseBasicParsing -Uri $packSourceURLBase | Expand-Stream -Destination $dstPath
# 2.0
$cli = New-Object System.Net.WebClient
$cli.DownloadFile($SourceURLBase, (Join-Path $dstPath $dstFile))

# Extract zip File
# 3.0
# New-ZipExtract -source $zipFilePath -destination $dstPath -force -verbose
# 2.0
$shell = New-Object -ComObject shell.application
$zip = $shell.NameSpace((Join-Path $dstPath $dstFile))
$dest = $shell.NameSpace((Split-Path (Join-Path $dstPath $dstFile) -Parent))
$dest.CopyHere($zip.Items()) 

# Copy binary
Copy-Item $dstPath/bin/* $dstPath
