<#
.SYNOPSIS
Retrieves and extracts the GTK lirary from the "http://ftp.gnome.org/" page.
.EXAMPLE
Get-GTKPlus
#>

# pkg-config downloads
# https://stackoverflow.com/questions/1710922/how-to-install-pkg-config-in-windows/25605631

# determine 32 or 64bit OS?
# https://social.technet.microsoft.com/Forums/office/en-US/5dfeb3ab-6265-40cd-a4ac-05428b9db5c3/determine-32-or-64bit-os?forum=winserverpowershell
# https://sqlpowershell.wordpress.com/2014/01/06/powershell-find-os-architecture-32-bit-or-64-bit-of-local-or-remote-machines-using-powershell/
if ([System.IntPtr]::Size -eq 4)
{
    # 32 bit logic here
    # Write "32-bit OS"
    $os_bit = "win32"
    $glibversion = "2.28"
    $zipFilePath1 = "pkg-config_0.26-1_win32.zip"
    $zipFilePath2 = "glib_2.28.8-1_win32.zip"
    $zipFilePath3 = "gettext-runtime_0.18.1.1-2_win32.zip"
}
else
{
    # 64 bit logic here
    # Write "64-bit OS"
    $os_bit = "win64"
    $glibversion = "2.26"
    $zipFilePath1 = "pkg-config_0.23-2_win64.zip"
    $zipFilePath2 = "glib_2.26.1-1_win64.zip"
    $zipFilePath3 = "gettext-runtime_0.18.1.1-2_win64.zip"
}

# $zipFilePath = "gtk+-bundle_$gtkVersion-" + "$gtkDate" + "_$os_bit.zip"
# base URL to download the pack file from
# 404
# $SourceURLBase = "http://win32builder.gnome.org/$zipFilePath"
$SourceURLBase1 = "http://ftp.gnome.org/pub/gnome/binaries/$os_bit/dependencies/$zipFilePath1"
$SourceURLBase2 = "http://ftp.gnome.org/pub/gnome/binaries/$os_bit/glib/$glibversion/$zipFilePath2"
$SourceURLBase3 = "http://ftp.gnome.org/pub/gnome/binaries/$os_bit/dependencies/$zipFilePath3"

# download the pack and extract the files into the curent directory 
# How to get the current directory of the cmdlet being executed
# http://stackoverflow.com/questions/8343767/how-to-get-the-current-directory-of-the-cmdlet-being-executed
$dstPath = (Get-Item -Path ".\" -Verbose).FullName
# $dstFile = $zipFilePath
$dstFile1 = $zipFilePath1
$dstFile2 = $zipFilePath2
$dstFile3 = $zipFilePath3

# Version Check
# PowerShell Version 2.0
# 1.0 Blank
# $PSVersionTable

# Download gtk
switch($PSVersionTable.PSVersion.Major)
{
    2
    {
        # 2.0(Windows 10 Error [not use .NetFramework 2.0/3.5])
        # use .Net Framework setting(JP)
        # https://qiita.com/miyamiya/items/95745587ced2c02a1966
        $cli = New-Object System.Net.WebClient
        $cli.DownloadFile($SourceURLBase1, (Join-Path $dstPath $dstFile1))
        $cli.DownloadFile($SourceURLBase2, (Join-Path $dstPath $dstFile2))
        $cli.DownloadFile($SourceURLBase3, (Join-Path $dstPath $dstFile3))
        $shell = New-Object -ComObject shell.application
        $zip1 = $shell.NameSpace((Join-Path $dstPath $dstFile1))
        $dest1 = $shell.NameSpace((Split-Path (Join-Path $dstPath $dstFile1) -Parent))
        $dest1.CopyHere($zip1.Items())
        $zip2 = $shell.NameSpace((Join-Path $dstPath $dstFile2))
        $dest2 = $shell.NameSpace((Split-Path (Join-Path $dstPath $dstFile2) -Parent))
        $dest2.CopyHere($zip2.Items())
        $zip3 = $shell.NameSpace((Join-Path $dstPath $dstFile3))
        $dest3 = $shell.NameSpace((Split-Path (Join-Path $dstPath $dstFile3) -Parent))
        $dest3.CopyHere($zip3.Items())
    }
    default
    {
        # PowerShell Version 3.0-
        Invoke-WebRequest -Uri $SourceURLBase1 -OutFile (Join-Path $dstPath $dstFile1)
        Invoke-WebRequest -Uri $SourceURLBase2 -OutFile (Join-Path $dstPath $dstFile2)
        Invoke-WebRequest -Uri $SourceURLBase3 -OutFile (Join-Path $dstPath $dstFile3)
        # Extract zip File
        # 3.0-
        # New-ZipExtract -source $zipFilePath1 -destination $dstPath -force -verbose
        # New-ZipExtract -source $zipFilePath2 -destination $dstPath -force -verbose
        # New-ZipExtract -source $zipFilePath3 -destination $dstPath -force -verbose
        # 5.0-
        Expand-Archive -Force -Path $zipFilePath1 -DestinationPath $dstPath
        Expand-Archive -Force -Path $zipFilePath2 -DestinationPath $dstPath
        Expand-Archive -Force -Path $zipFilePath3 -DestinationPath $dstPath
    }
}

# Copy binary
Copy-Item $dstPath/bin/* $dstPath
