# Sample script to install PointCloudLibrary and pip under Windows
# Authors: Olivier Grisel, Jonathan Helmus, Kyle Kastner, and Alex Willmer
# License: CC0 1.0 Universal: http://creativecommons.org/publicdomain/zero/1.0/

$BASE_PCL16_URL = "http://jaist.dl.sourceforge.net/project/pointclouds/"
$BASE_PCL_URL = "https://github.com/PointCloudLibrary/pcl/releases/download"

$PYTHON_PRERELEASE_REGEX = @"
(?x)
(?<major>\d+)
\.
(?<minor>\d+)
\.
(?<micro>\d+)
"@


$PCL_PRERELEASE_REGEX = @"
(?x)
(?<major>\d+)
\.
(?<minor>\d+)
\.
(?<micro>\d+)
"@


function Download ($filename, $url) 
{
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
        Write-Host $(Get-ChildItem $filepath).Length
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
    if ($pcl_version -match $PCL_PRERELEASE_REGEX) 
    {
        return ([int]$matches.major, [int]$matches.minor, [int]$matches.micro)
        # return ([int]$matches.major, [int]$matches.minor, [int]$matches.micro, $matches.prerelease)
    }
    
    # Convert NG
    $version_obj = [version]$pcl_version
    return ($version_obj.major, $version_obj.minor, $version_obj.build, "")
}


function InstallPCLEXE ($exepath, $pcl_home, $install_log)
{
    # http://www.ibm.com/support/knowledgecenter/SS2RWS_2.1.0/com.ibm.zsecure.doc_2.1/visual_client/responseexamples.html?lang=ja
    $install_args = "/S /v/qn /v/norestart"
    RunCommand "schtasks" "/create /tn pclinstall /RL HIGHEST /tr `"$exepath $install_args`" /sc once /st 23:59"
    # Check TaskList
    RunCommand "schtasks" "/query /v"
    RunCommand "sleep" "10"
    RunCommand "schtasks" "/run /tn pclinstall"
    RunCommand "sleep" "600"
    RunCommand "schtasks" "/delete /tn pclinstall /f"
}

function InstallPCLMSI ($msipath, $pcl_home, $install_log)
{
    $install_args = "/qn /log $install_log /i $msipath TARGETDIR=$pcl_home"
    $uninstall_args = "/qn /x $msipath"
    RunCommand "msiexec.exe" $install_args
    if (-not(Test-Path $pcl_home)) 
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

function DownloadPCL ($pcl_version, $platform_suffix, $msvc_version) 
{
    # $major, $minor, $micro, $prerelease = ParsePCLVersion $pcl_version
    $major, $minor, $micro = ParsePCLVersion $pcl_version

    if ($major -le 1 -and $minor -eq 6) 
    {
        # $url = http://jaist.dl.sourceforge.net/project/pointclouds/1.6.0/PCL-1.6.0-AllInOne-msvc2010-win64.exe
        # $dir = "$major.$minor.$micro"
        $dir = "$major.$minor.0"
        $msvcver = "msvc2010"
        
        $filename = "PCL-" + "$dir" + "-AllInOne-" + "$msvcver" + "-" + "$platform_suffix.exe"
        $url = "$BASE_PCL16_URL" + "$dir" + "/PCL-" + "$dir" + "-AllInOne-" + "$msvcver" + "-" + "$platform_suffix.exe"
    }
    elseif ($major -le 1 -and $minor -eq 8)
    {
        # 1.8.0 NG
        # $dir = "$major.$minor.$micro"
        # fix 1.8.1
        $dir = "$major.$minor.1"
        
        # $msvcver = "msvc2015"
        # 2015 or 2017
        $msvcver = "msvc" + "$msvc_version"
        
        $filename = "PCL-" + "$dir" + "-AllInOne-" + "$msvcver" + "-" + "$platform_suffix.exe"
        
        $url = "$BASE_PCL_URL" + "/pcl-" + "$dir" + "/" + "$filename"
    }
    elseif ($major -le 1 -and $minor -eq 9)
    {
        $dir = "$major.$minor.micro"
        # 2015? or 2017?
        $msvcver = "msvc" + "$msvc_version"
        
        $msvcver = "msvc" + "$msvc_version"
        
        $filename = "PCL-" + "$dir" + "-AllInOne-" + "$msvcver" + "-" + "$platform_suffix.exe"
        
        $url = "$BASE_PCL_URL" + "/pcl-" + "$dir" + "/" + "$filename"
    }
    else
    {
        $dir = "$major.$minor.$micro"
        $msvcver = "msvc2015"
    }

    # (plan modified function)
    $filepath = Download $filename $url
    return $filepath
}

function InstallPCL ($pcl_version, $architecture, $pcl_home, $build_worker_image) 
{
    if ($build_worker_image -eq "Visual Studio 2015")
    {
        $msvc_version = "2015"
    }
    elseif ($build_worker_image -eq "Visual Studio 2017")
    {
        $msvc_version = "2017"
    }
    else
    {
        $msvc_version = "2015"
    }

    if ($architecture -eq "32")
    {
        $platform_suffix = "win32"
    }
    else
    {
        $platform_suffix = "win64"
    }
    
    $installer_path = DownloadPCL $pcl_version $platform_suffix $msvc_version
    $installer_ext = [System.IO.Path]::GetExtension($installer_path)
    Write-Host "Installing $installer_path to $pcl_home"
    $install_log = $pcl_home + "install.log"
    if ($installer_ext -eq '.msi')
    {
        InstallPCLMSI $installer_path $pcl_home $install_log
    }
    else
    {
        InstallPCLEXE $installer_path $pcl_home $install_log
    }
    
    if (Test-Path $pcl_home) 
    {
        Write-Host "PointCloudLibrary $pcl_version ($architecture) installation complete"
    }
    else 
    {
        Write-Host "Failed to install PointCloudLibrary in $pcl_home"
        # Get-Content -Path $install_log
        # Exit 1
    }
    
    # use 1.6 only
    CopyPCLHeader ($pcl_version, $pcl_home)
}

function CopyPCLHeader ($pcl_version, $pcl_home) 
{
    $major, $minor, $micro = ParsePCLVersion $pcl_version

    if ($major -le 1 -and $minor -eq 6) 
    {
        # - copy .\\appveyor\\bfgs.h "%PCL_ROOT%\include\pcl-%PCL_VERSION%\pcl\registration\bfgs.h"
        # - copy .\\appveyor\\eigen.h "%PCL_ROOT%\include\pcl-%PCL_VERSION%\pcl\registration\eigen.h"
        $current_dir =  [System.IO.Directory]::GetCurrentDirectory()
        Copy-Item $current_dir\appveyor\bfgs.h $pcl_home\include\pcl-1.6\pcl\registration
        Copy-Item $current_dir\appveyor\eigen.h $pcl_home\include\pcl-1.6\pcl\registration
    }
}

function InstallOpenNI ($openni_version, $architecture, $pcl_home, $openni_root)
{
    if ($architecture -eq "32")
    {
        $platform_suffix = "win32"
    }
    else
    {
        $platform_suffix = "win64"
    }
    

    $installer_filename = "OpenNI-" + "$platform_suffix" + "-" + "$openni_version" + "-Dev.msi"
    $installer_path = $pcl_home + "\3rdParty\OpenNI\" + $installer_filename
        
    $current_dir =  [System.IO.Directory]::GetCurrentDirectory()
    Copy-Item $installer_path $current_dir

    Write-Host "Installing $installer_filename to $openni_root"
    $install_log = $openni_root + "\install.log"
    if ($installer_ext -eq '.msi')
    {
        InstallOpenNIMSI $installer_filename $openni_root $install_log
    }
    else
    {
        InstallOpenNIEXE $installer_filename $openni_root $install_log
    }
    
    if (Test-Path $openni_root) 
    {
        Write-Host "OpenNI $openni_version ($architecture) installation complete"
    }
    else 
    {
        Write-Host "Failed to install OpenNI in $openni_root"
        # Get-Content -Path $install_log
        # Exit 1
    }
}


function InstallOpenNIMSI ($msipath, $openni_home, $install_log)
{
    $install_args = "/qn /norestart /lv install.log"
    $uninstall_args = "$msipath /qn /x"

    RunCommand "schtasks" "/create /tn openni_install /RL HIGHEST /tr `"msiexec.exe /i $msipath $install_args`" /sc once /st 23:59"
    RunCommand "sleep" "10"
    RunCommand "schtasks" "/run /tn openni_install"
    RunCommand "sleep" "90"
    RunCommand "schtasks" "/delete /tn openni_install /f"
}

function InstallOpenNIEXE ($exepath, $openni_home, $install_log)
{
    $install_args = "/S /v/qn /v/norestart"
    RunCommand "schtasks" "/create /tn openni_install /RL HIGHEST /tr `"$exepath $install_args`" /sc once /st 23:59"
    RunCommand "sleep" "10"
    RunCommand "schtasks" "/run /tn openni_install"
    RunCommand "sleep" "90"
    RunCommand "schtasks" "/delete /tn openni_install /f"
}

function InstallOpenNI2 ($openni_version, $architecture, $pcl_home, $openni2_root)
{
    if ($architecture -eq "32")
    {
        $platform_suffix = "win32"
    }
    else
    {
        $platform_suffix = "x64"
    }
    
    $installer_filename = "OpenNI-Windows-" + "$platform_suffix" + "-" + "$openni_version" + ".msi"
    # $installer_path = $pcl_home + "\3rdParty\OpenNI2\OpenNI-Windows-" + "$platform_suffix" + "-" + "$openni_version" + ".msi"
    $installer_path = $pcl_home + "\3rdParty\OpenNI2\" + $installer_filename
    # Copy-Item $installer_path $installer_filename
    $current_dir =  [System.IO.Directory]::GetCurrentDirectory()
    Copy-Item $installer_path $current_dir
    
    $installer_ext = [System.IO.Path]::GetExtension($installer_path)
    Write-Host "Installing $installer_path"
    $install_log = $pcl_home + "\install.log"
    if ($installer_ext -eq '.msi')
    {
        # InstallOpenNI2MSI $installer_path $install_log
        InstallOpenNI2MSI $installer_filename $install_log
    }
    else
    {
        InstallOpenNI2EXE $installer_path $install_log
    }
    
    if (Test-Path $openni2_root) 
    {
        Write-Host "OpenNI2 $openni_version ($architecture) installation complete"
    }
    else 
    {
        Write-Host "Failed to install OpenNI2 in $openni2_root"
        # Exit 1
    }
}


function InstallOpenNI2MSI ($msipath, $install_log)
{
    # # http://www.ibm.com/support/knowledgecenter/SS2RWS_2.1.0/com.ibm.zsecure.doc_2.1/visual_client/responseexamples.html?lang=ja
    # $install_args = "/qn /norestart"
    # optput log
    # $install_args = "/qn /norestart /lv $install_log"
    $install_args = "/qn /norestart /lv install.log"
    $uninstall_args = "$msipath /qn /x"

    RunCommand "schtasks" "/create /tn openni_install /RL HIGHEST /tr `"msiexec.exe /i $msipath $install_args`" /sc once /st 23:59"
    # NG
    # RunCommand "schtasks" "/create /tn openni_install /RL HIGHEST /tr `"msiexec.exe /i `"$msipath`" $install_args`" /sc once /st 23:59"
    # NG
    # RunCommand "schtasks" "/create /tn openni_install /rl HIGHEST /tr `"$msipath $install_args`" /sc once /st 23:59"
    RunCommand "sleep" "10"
    RunCommand "schtasks" "/run /tn openni_install"
    RunCommand "sleep" "180"
    RunCommand "schtasks" "/delete /tn openni_install /f"
}

function InstallOpenNI2EXE ($exepath, $install_log)
{
    # http://www.ibm.com/support/knowledgecenter/SS2RWS_2.1.0/com.ibm.zsecure.doc_2.1/visual_client/responseexamples.html?lang=ja
    $install_args = "/S /v/qn /v/norestart"
    
    # RunCommand schtasks /create /tn openni_install /RL HIGHEST /tr $exepath /S /v/norestart /v/qn /sc once /st 23:59
    # RunCommand schtasks /run /tn openni_install
    # RunCommand schtasks /delete /tn openni_install /f
    # RunCommand sleep 90
    RunCommand "schtasks" "/create /tn openni_install /RL HIGHEST /tr `"$exepath $install_args`" /sc once /st 23:59"
    RunCommand "sleep" "10"
    RunCommand "schtasks" "/run /tn openni_install"
    RunCommand "sleep" "90"
    RunCommand "schtasks" "/delete /tn openni_install /f"
}

function main () 
{
    # http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
    InstallPCL $env:PCL_VERSION $env:PYTHON_ARCH $env:PCL_ROOT $env:APPVEYOR_BUILD_WORKER_IMAGE
    $major, $minor, $micro = ParsePCLVersion $env:PCL_VERSION
    if ($major -le 1 -and $minor -eq 6) 
    {
        InstallOpenNI $env:OPENNI_VERSION $env:PYTHON_ARCH $env:PCL_ROOT $env:OPENNI_ROOT
    }
    else
    {
        InstallOpenNI2 $env:OPENNI_VERSION $env:PYTHON_ARCH $env:PCL_ROOT $env:OPENNI_ROOT
    }
}

main
