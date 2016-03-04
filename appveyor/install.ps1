# Sample script to install PointCloudLibrary and pip under Windows
# Authors: Olivier Grisel, Jonathan Helmus, Kyle Kastner, and Alex Willmer
# License: CC0 1.0 Universal: http://creativecommons.org/publicdomain/zero/1.0/

$BASE_PCL_URL = "http://jaist.dl.sourceforge.net/project/pointclouds/"
$BASE_NUMPY_WHL_URL = "http://www.lfd.uci.edu/~gohlke/pythonlibs/tugyrhqo/"

$PYTHON_PRERELEASE_REGEX = @"
(?x)
(?<major>\d+)
\.
(?<minor>\d+)
\.
(?<micro>\d+)
(?<prerelease>[a-z]{1,2}\d+)
"@

$PCL_PRERELEASE_REGEX = @"
(?x)
(?<major>\d+)
\.
(?<minor>\d+)
\.
(?<micro>\d+)
"@


function Download ($filename, $url) {
    $webclient = New-Object System.Net.WebClient

    $basedir = $pwd.Path + "\"
    $filepath = $basedir + $filename
    if (Test-Path $filename) 
    {
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
    
    if (Test-Path $filepath) 
    {
        Write-Host "File saved at" $filepath
    }
    else 
    {
        # Retry once to get the error message if any at the last try
        $webclient.DownloadFile($url, $filepath)
    }
    return $filepath
}


function ParsePythonVersion ($python_version) 
{
    if ($python_version -match $PYTHON_PRERELEASE_REGEX) 
    {
        return ([int]$matches.major, [int]$matches.minor, [int]$matches.micro, $matches.prerelease)
    }
    
    $version_obj = [version]$python_version
    return ($version_obj.major, $version_obj.minor, $version_obj.build, "")
}

function ParsePCLVersion ($pcl_version) 
{
    if ($python_version -match $PCL_PRERELEASE_REGEX) 
    {
        return ([int]$matches.major, [int]$matches.minor, [int]$matches.micro)
    }
    
    # Convert NG
    $version_obj = [version]$pcl_version
    return ($version_obj.major, $version_obj.minor, $version_obj.micro, "")
}


function InstallPCLEXE ($exepath, $pcl_home, $install_log)
{
    $install_args = "/quiet InstallAllUsers=1 TargetDir=$pcl_home"
    RunCommand $exepath $install_args
}

function InstallPCLMSI ($msipath, $pcl_home, $install_log)
{
    $install_args = "/qn /log $install_log /i $msipath TARGETDIR=$pcl_home"
    $uninstall_args = "/qn /x $msipath"
    RunCommand "msiexec.exe" $install_args
    if (-not(Test-Path $python_home)) 
    {
        Write-Host "PointCloudLibrary seems to be installed else-where, reinstalling."
        RunCommand "msiexec.exe" $uninstall_args
        RunCommand "msiexec.exe" $install_args
    }
}

function RunCommand ($command, $command_args) 
{
    Write-Host $command $command_args
    Start-Process -FilePath $command -ArgumentList $command_args -Wait -Passthru
}

function DownloadPCL ($pcl_version, $platform_suffix) 
{
    # $major, $minor, $micro, $prerelease = ParsePCLVersion $pcl_version
    $major, $minor, $micro = ParsePCLVersion $pcl_version

    if ($major -le 1 -and $minor -eq 6) 
    {
        # $url = http://jaist.dl.sourceforge.net/project/pointclouds/1.6.0/PCL-1.6.0-AllInOne-msvc2010-win64.exe
        $dir = "$major.$minor.$micro"
        $msvcver = "msvc2010"
        $url = "$dir/PCL-$dir-AllInOne-msvc2010-$platform_suffix.exe"
    }
    elseif ($major -le 1 -and $minor -eq 7)
    {
        # $url = http://jaist.dl.sourceforge.net/project/pointclouds/1.6.0/PCL-1.6.0-AllInOne-msvc2015-win64.exe
        $dir = "$major.$minor.$micro"
        $msvcver = "msvc2015"
    }
    elseif ($major -le 1 -and $minor -eq 8)
    {
        # $url = http://jaist.dl.sourceforge.net/project/pointclouds/1.6.0/PCL-1.6.0-AllInOne-msvc2015-win64.exe
        $dir = "$major.$minor.$micro"
        $msvcver = "msvc2015"
    } 
    else 
    {
        $dir = "$major.$minor.$micro"
        $msvcver = "msvc2015"
    }

    $filename = "PCL-$dir-AllInOne-$msvcver-$platform_suffix.exe"
    $url = "$BASE_PCL_URL$dir/PCL-$dir-AllInOne-$msvcver-$platform_suffix.exe"
    # ’u‚«Š·‚¦—\’è
    $filepath = Download $filename $url
    return $filepath
}

function ParsePythonVersion ($python_version) 
{
    if ($python_version -match $PYTHON_PRERELEASE_REGEX) 
    {
        return ([int]$matches.major, [int]$matches.minor, [int]$matches.micro, $matches.prerelease)
    }
    
    # Convert NG
    $version_obj = [version]$python_version
    return ($version_obj.major, $version_obj.minor, $version_obj.build, "")
}

function InstallNumpy ($python_version, $architecture, $python_home) 
{
    $major, $minor, $micro, $prerelease = ParsePythonVersion $python_version

    $cp_ver = "cp$major$minor"

    if ($architecture -eq "32") {
        $platform_suffix = "32"
    } else {
        $platform_suffix = "win_amd64"
    }

    $mathLib = "mkl"
    if ($mathLib -eq "mkl")
    {
        $cp_last_ver = "$cp_verm"
    }
    else
    {
        $cp_last_ver = "none"
    }
    
    $numpy_ver = "1.10.4"

    Write-Host "Installing Python" $python_version "for" $architecture "bit architecture to" $python_home
    # if (Test-Path $python_home) 
    # {
    #     Write-Host $python_home "already exists, skipping."
    #     return $false
    # }

    # http://www.lfd.uci.edu/~gohlke/pythonlibs/tugyrhqo/numpy-1.10.4+mkl-cp27-cp27m-win32.whl
    # http://www.lfd.uci.edu/~gohlke/pythonlibs/tugyrhqo/numpy-1.10.4+mkl-cp27-cp27m-win_amd64.whl
    # http://www.lfd.uci.edu/~gohlke/pythonlibs/tugyrhqo/numpy-1.10.4+mkl-cp34-cp34m-win32.whl
    # http://www.lfd.uci.edu/~gohlke/pythonlibs/tugyrhqo/numpy-1.10.4+mkl-cp34-cp34m-win_amd64.whl
    # http://www.lfd.uci.edu/~gohlke/pythonlibs/tugyrhqo/numpy-1.10.4+mkl-cp35-cp35m-win32.whl
    # http://www.lfd.uci.edu/~gohlke/pythonlibs/tugyrhqo/numpy-1.10.4+mkl-cp35-cp35m-win_amd64.whl
    # numpy-1.10.4+vanilla-cp27-none-win32.whl
    # numpy-1.10.4+vanilla-cp27-none-win_amd64.whl
    # numpy-1.10.4+vanilla-cp34-none-win32.whl
    # numpy-1.10.4+vanilla-cp34-none-win_amd64.whl
    # numpy-1.10.4+vanilla-cp35-none-win32.whl
    # numpy-1.10.4+vanilla-cp35-none-win_amd64.whl
    # numpy-1.11.0rc1+mkl-cp27-cp27m-win32.whl
    # numpy-1.11.0rc1+mkl-cp27-cp27m-win_amd64.whl
    # numpy-1.11.0rc1+mkl-cp34-cp34m-win32.whl
    # numpy-1.11.0rc1+mkl-cp34-cp34m-win_amd64.whl
    # numpy-1.11.0rc1+mkl-cp35-cp35m-win32.whl
    # numpy-1.11.0rc1+mkl-cp35-cp35m-win_amd64.whl
    $filename = "numpy-$numpy_ver+$mathLib-$cp_ver-$cp_last_ver-$platform_suffix.whl"
    $url = "$BASE_NUMPY_WHL_URLnumpy-$numpy_ver+$mathLib-$cp_ver-$cp_last_ver-$platform_suffix.whl"
    # replace another function
    $filepath = Download $filename $url
    return $filepath
}

function InstallPCL ($pcl_version, $architecture, $pcl_home) 
{
    # $pcl_path = $python_home + "\Scripts\pip.exe"
    
    if ($architecture -eq "32")
    {
        $platform_suffix = "win32"
    }
    else
    {
        $platform_suffix = "win64"
    }
    
    $installer_path = DownloadPCL $pcl_version $platform_suffix
    $installer_ext = [System.IO.Path]::GetExtension($installer_path)
    Write-Host "Installing $installer_path to $pcl_home"
    $install_log = $pcl_home + ".log"
    if ($installer_ext -eq '.msi')
    {
        InstallPCLMSI $installer_path $python_home $install_log
    }
    else
    {
        InstallPCLEXE $installer_path $python_home $install_log
    }
    
    if (Test-Path $python_home) 
    {
        Write-Host "Python $python_version ($architecture) installation complete"
    }
    else 
    {
        Write-Host "Failed to install Python in $python_home"
        Get-Content -Path $install_log
        Exit 1
    }
}

# http://www.lfd.uci.edu/~gohlke/pythonlibs/

$pcl_version = "1.6.0"

function main () {
    # InstallPython $env:PYTHON_VERSION $env:PYTHON_ARCH $env:PYTHON
    # http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
    InstallNumpy $env:PYTHON_VERSION $env:PYTHON_ARCH $env:PYTHON
    InstallPCL $pcl_version, $env:PYTHON_ARCH, "C:\project"
}

main
