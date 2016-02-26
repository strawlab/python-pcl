# Sample script to install Python and pip under Windows
# Authors: Olivier Grisel, Jonathan Helmus, Kyle Kastner, and Alex Willmer
# License: CC0 1.0 Universal: http://creativecommons.org/publicdomain/zero/1.0/

$BASE_PCL_URL = "http://jaist.dl.sourceforge.net/project/pointclouds/"

$PYTHON_PRERELEASE_REGEX = @"
(?x)
(?<major>\d+)
\.
(?<minor>\d+)
\.
(?<micro>\d+)
(?<prerelease>[a-z]{1,2}\d+)
"@


function Download ($filename, $url) {
    $webclient = New-Object System.Net.WebClient

    $basedir = $pwd.Path + "\"
    $filepath = $basedir + $filename
    if (Test-Path $filename) {
        Write-Host "Reusing" $filepath
        return $filepath
    }

    # Download and retry up to 3 times in case of network transient errors.
    Write-Host "Downloading" $filename "from" $url
    $retry_attempts = 2
    for ($i = 0; $i -lt $retry_attempts; $i++) {
        try {
            $webclient.DownloadFile($url, $filepath)
            break
        }
        Catch [Exception]{
            Start-Sleep 1
        }
    }
    if (Test-Path $filepath) {
        Write-Host "File saved at" $filepath
    } else {
        # Retry once to get the error message if any at the last try
        $webclient.DownloadFile($url, $filepath)
    }
    return $filepath
}


function ParsePythonVersion ($python_version) {
    if ($python_version -match $PYTHON_PRERELEASE_REGEX) {
        return ([int]$matches.major, [int]$matches.minor, [int]$matches.micro,
                $matches.prerelease)
    }
    $version_obj = [version]$python_version
    return ($version_obj.major, $version_obj.minor, $version_obj.build, "")
}

function InstallPCLEXE ($exepath, $pcl_home, $install_log) {
    $install_args = "/quiet InstallAllUsers=1 TargetDir=$pcl_home"
    RunCommand $exepath $install_args
}

function InstallPCLMSI ($msipath, $pcl_home, $install_log) {
    $install_args = "/qn /log $install_log /i $msipath TARGETDIR=$pcl_home"
    $uninstall_args = "/qn /x $msipath"
    RunCommand "msiexec.exe" $install_args
    if (-not(Test-Path $python_home)) {
        Write-Host "Python seems to be installed else-where, reinstalling."
        RunCommand "msiexec.exe" $uninstall_args
        RunCommand "msiexec.exe" $install_args
    }
}

function RunCommand ($command, $command_args) {
    Write-Host $command $command_args
    Start-Process -FilePath $command -ArgumentList $command_args -Wait -Passthru
}

function DownloadPCL ($pcl_version, $platform_suffix) {
    # $major, $minor, $micro, $prerelease = ParsePythonVersion $pcl_version
    $major, $minor, $micro = ParsePythonVersion $pcl_version

    if ($major -le 1 -and $minor -eq 6) 
    {
        # $url = http://jaist.dl.sourceforge.net/project/pointclouds/1.6.0/PCL-1.6.0-AllInOne-msvc2010-win64.exe
        $dir = "$major.$minor.$micro"
        $msvcver = "msvc2010"
        $url = $dir"/PCL-"$dir"-AllInOne-msvc2010-"$platform_suffix".exe
    }
    else if (($major -le 1 -and $minor -eq 7 -and )
    {
        # $url = http://jaist.dl.sourceforge.net/project/pointclouds/1.6.0/PCL-1.6.0-AllInOne-msvc2015-win64.exe
        $dir = "$major.$minor.$micro"
        $msvcver = "msvc2015"
    } else {
        $dir = "$major.$minor.$micro"
        $msvcver = "msvc2015"
    }

    $filename = "PCL-$dir-AllInOne-$msvcver-$platform_suffix.exe"
    $url = "$BASE_PCL_URL$dir/PCL-$dir-AllInOne-$msvcver-$platform_suffix.exe"
    # ’u‚«Š·‚¦—\’è
    $filepath = Download $filename $url
    return $filepath
}

function InstallPCL ($pcl_version, $architecture, $pcl_home) {
    $pcl_path = $python_home + "\Scripts\pip.exe"
    
    if ($architecture -eq "32") {
        $platform_suffix = "win32"
    } else {
        $platform_suffix = "win64"
    }

    $installer_path = DownloadPCL $pcl_version $platform_suffix
    $installer_ext = [System.IO.Path]::GetExtension($installer_path)
    Write-Host "Installing $installer_path to $python_home"
    if ($installer_ext -eq '.msi') {
        InstallPCLMSI $installer_path $python_home $install_log
    } else {
        InstallPCLEXE $installer_path $python_home $install_log
    }
    if (Test-Path $python_home) {
        Write-Host "Python $python_version ($architecture) installation complete"
    } else {
        Write-Host "Failed to install Python in $python_home"
        Get-Content -Path $install_log
        Exit 1
    }

}

$pcl_version = "1.6.0"

function main () {
    # InstallPython $env:PYTHON_VERSION $env:PYTHON_ARCH $env:PYTHON
    # InstallPip $env:PYTHON
    # DownloadPCL $pcl_version, $platform_suffix
    InstallPCL $pcl_version, $platform_suffix
}

main
