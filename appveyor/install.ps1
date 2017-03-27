# Sample script to install PointCloudLibrary and pip under Windows
# Authors: Olivier Grisel, Jonathan Helmus, Kyle Kastner, and Alex Willmer
# License: CC0 1.0 Universal: http://creativecommons.org/publicdomain/zero/1.0/

$BASE_PCL_URL = "http://jaist.dl.sourceforge.net/project/pointclouds/"
$BASE_NUMPY_WHL_URL = "http://www.lfd.uci.edu/~gohlke/pythonlibs/"
$NUMPY_DOWNLOAD_URL = "djcobkfp"

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
(?<prerelease>[a-z]{1,2}\d+)
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
        # return ([int]$matches.major, [int]$matches.minor, [int]$matches.micro)
        return ([int]$matches.major, [int]$matches.minor, [int]$matches.micro, $matches.prerelease)
    }
    
    # Convert NG
    $version_obj = [version]$pcl_version
    return ($version_obj.major, $version_obj.minor, $version_obj.build, "")
}


function InstallPCLEXE ($exepath, $pcl_home, $install_log)
{
    # old
    # $install_args = "/quiet InstallAllUsers=1 TargetDir=$pcl_home"
    # RunCommand $exepath $install_args
    
    # http://www.ibm.com/support/knowledgecenter/SS2RWS_2.1.0/com.ibm.zsecure.doc_2.1/visual_client/responseexamples.html?lang=ja
    $install_args = "/S /v/qn /v/norestart"
    # RunCommand schtasks /create /tn pclinstall /RL HIGHEST /tr $exepath /S /v/norestart /v/qn /sc once /st 23:59
    # RunCommand schtasks /run /tn pclinstall
    # RunCommand schtasks /delete /tn pclinstall /f
    # RunCommand sleep 600
    RunCommand "schtasks" "/create /tn pclinstall /RL HIGHEST /tr `"$exepath $install_args`" /sc once /st 23:59"
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

function DownloadPCL ($pcl_version, $platform_suffix) 
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
        $url = "$BASE_PCL_URL" + "$dir" + "/PCL-" + "$dir" + "-AllInOne-" + "$msvcver" + "-" + "$platform_suffix.exe"
    }
    elseif ($major -le 1 -and $minor -eq 7)
    {
        # $url = "https://onedrive.live.com/redir?resid=EC9EBB2646FF189A!51249&authkey=!ABJC39YpCnE4So8&ithint=file%2cexe"
        # $url = "https://onedrive.live.com/redir?resid=EC9EBB2646FF189A!51248&authkey=!AOPBX-WypndUncw&ithint=file%2cexe"
        # $dir = "$major.$minor.$micro"
        $dir = "$major.$minor.2"
        $msvcver = "msvc2015"
        
        if ($platform_suffix -eq "win32") 
        {
            # NG : バイナリの Download まではいかない
            # $url = "https://onedrive.live.com/redir?resid=EC9EBB2646FF189A!51249&authkey=!ABJC39YpCnE4So8&ithint=file%2cexe"
            # 直接リンクを設定しても一定時間超えるとNG
            # Note : 外部サービスに exeを置いて Download する?(チェック用に)
            $url = "https://6gjmvg-ch3302.files.1drv.com/y3mINszazs-p2iitS2ZEzuvPRAqgqppQIAvM17aMB-S2y4iQ_jTyq0HVmZYBXZ1FAAl9VHzYCoQ6g_fRWaZMkA8AgvrfDvK5AZLy0s-guH9DPAYAlaTJo9-Hnr1xOeeA2t6u0uw9SHvO8CSZWnS5VwC-g/PCL-1.7.2-AllInOne-msvc2015-win32.exe?download&psid=1"
        }
        else 
        {
            # NG : バイナリの Download まではいかない
            # $url = "https://onedrive.live.com/redir?resid=EC9EBB2646FF189A!51248&authkey=!AOPBX-WypndUncw&ithint=file%2cexe"
            # 直接リンクを設定しても一定時間超えるとNG
            # Note : 外部サービスに exeを置いて Download する?(チェック用に)
            $url = "https://6gjnvg-ch3302.files.1drv.com/y3mY7SSNjvbvx74d4JIAMAob0A87UF5PNGI6sqJJVR7_QQ453NyhBlvXDpjHW49fHJl4D6nCKJ8CWHS7J_D734Mr2zQS32uT7kUqn6vE0cbNj_ISISKJZ28CPvOpbgKfRSMvCqrpQAXR3yBddzAY3kvSg/PCL-1.7.2-AllInOne-msvc2015-win64.exe?download&psid=1"
        }
        
        $filename = "PCL-" + "$dir" + "-AllInOne-" + "$msvcver" + "-" + "$platform_suffix.exe"
    }
    elseif ($major -le 1 -and $minor -eq 8)
    {
        # $url = "https://onedrive.live.com/hogehoge"
        # $dir = "$major.$minor.$micro"
        $dir = "$major.$minor.0"
        $msvcver = "msvc2015"
        
        if ($platform_suffix -eq "win32") 
        {
            # NG : バイナリの Download まではいかない
            # $url = "https://onedrive.live.com/redir?resid=EC9EBB2646FF189A!51249&authkey=!ABJC39YpCnE4So8&ithint=file%2cexe"
            # 直接リンクを設定しても一定時間超えるとNG
            # Note : 外部サービスにインストーラを置いて Download する?(チェック用に)
            $url = "https://6gjgdw-ch3302.files.1drv.com/y3m4fhIqdKOvjS4UfVakYyeZML0quorktQkKIWELuJCtxWh1jVFvzB0XcH5NlSsHrWVsgVlTMvbZkDyGv6FWix_pu24ORHI065W224v-gtAZmUrjBdbnda8AR1D-ehuHMWj-bhY2SHQmgo9LUXuAAcvOw/PCL-1.8.0-AllInOne-msvc2015-win32.exe?download&psid=1"
        }
        else 
        {
            # NG : バイナリの Download まではいかない
            # $url = "https://onedrive.live.com/?authkey=%21AINUVKSzTRdWdS4&id=EC9EBB2646FF189A%2151744&cid=EC9EBB2646FF189A"
            # 直接リンクを設定しても一定時間超えるとNG
            # Note : 外部サービスにインストーラを置いて Download する?(チェック用に)
            $url = "https://6gi8dw-ch3302.files.1drv.com/y3mbO30Vy-o-tHwr_TFCfdEtcGx2zOi4X-S74Oun4BqvPejbSd8arDm-32QXDkram4hLMCXEunFnANWw2NMCcG4BYtBqZOrkJ2EsAZ4_lH4U08ne8v5U_nWLJAj2anBNjZF59NER9m5D03wax0BNZHaZwmEk3TkpyIhIUjGQoFlIdI/PCL-1.8.0-AllInOne-msvc2015-win64.exe?download&psid=1"
        }
        
        $filename = "PCL-" + "$dir" + "-AllInOne-" + "$msvcver" + "-" + "$platform_suffix.exe"
    }
    else
    {
        $dir = "$major.$minor.$micro"
        $msvcver = "msvc2015"
    }

    # $filename = "PCL-" + "$dir" + "-AllInOne-" + "$msvcver" + "-" + "$platform_suffix.exe"
    # $url = "$BASE_PCL_URL" + "$dir" + "/PCL-" + "$dir" + "-AllInOne-" + "$msvcver" + "-" + "$platform_suffix.exe"

    # (plan modified function)
    $filepath = Download $filename $url
    return $filepath
}

function DownloadGTKPlus ($gtk_version, $platform_suffix) 
{
    $filename = "gtk+-bundle_3.10.4-20131202_" + "$platform_suffix.zip"
    $url = "http://win32builder.gnome.org/" + "gtk+-bundle_3.10.4-20131202_" + "$platform_suffix.zip"

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
        $platform_suffix = "win32"
    } else {
        $platform_suffix = "win_amd64"
    }

    $mathLib = "mkl"
    if ($mathLib -eq "mkl")
    {
        $cp_last_ver = $cp_ver + "m"
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
    $filename = "numpy-" + "$numpy_ver" + "+" + "$mathLib" + "-" + "$cp_ver" + "-" + "$cp_last_ver" +"-" + "$platform_suffix.whl"
    $url = "$BASE_NUMPY_WHL_URL" + "$NUMPY_DOWNLOAD_URL" + "/numpy-" + "$numpy_ver" + "+" + "$mathLib" + "-" + "$cp_ver" + "-" + "$cp_last_ver" + "-" + "$platform_suffix.whl"
    # replace another function
    $filepath = Download $filename $url
    return $filepath
}

function InstallPCL ($pcl_version, $architecture, $pcl_home) 
{
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
}

# function DownloadOpenNI ($openni_version, $platform_suffix) 
# {
#     # $major, $minor, $micro, $prerelease = ParsePCLVersion $openni_version
#     $major, $minor, $micro = ParsePCLVersion $openni_version
# 
#     if ($major -eq 1 -and $minor -eq 5 and $minor -eq 7)
#     {
#         if ($platform_suffix -eq "win32") 
#         {
#             $url = "https://6qjmvg.bn1304.livefilestore.com/y3mi0KDkRbdIuOAXcGR3CNpXQZpmrSWltVYkTeL2qYl3Ag0fmxgTgtlxqOziG_Y55DoM7I8bLuVPxMYiZ94vCEZxBgQzCpWFaGS61rB9iP1trpLEStK8OH8VQ_v7HVrLQaQE2UgpunA3tZGEcxclHvD6g/OpenNI-Win32-1.5.7.10-Dev.zip?download&psid=1"
#         }
#         else
#         {
#             $url = "https://6qjmvg.bn1303.livefilestore.com/y3mEZfnt7ecywLJeAFqeZC0CdqIegwWqae5CHypCheKcyQv00BB4qMGhUW03FAJlubPymQz1hFHKLgRdE-2TO8b6VAZ4s_pe-FL-FY6I2RqCi8vcwOvEx1REMcZo_8Iz_bxNwEREtNH9M5TX8uo1yl9ZA/OpenNI-Win64-1.5.7.10-Dev.zip?download&psid=1"
#         }
#         
#         # OpenNI-Win32-1.5.7.10-Dev.zip
#         # OpenNI-Win64-1.5.7.10-Dev.zip
#         $filename = "OpenNI-" + "$platform_suffix" + "$major" + "." + "$minor" + "." + "$micro" + "." + "$msvcver" + "-" + "Dev.exe"
#     }
#     elseif ($major -eq 2 -and $minor -eq 8)
#     {
#         # $url = "https://onedrive.live.com/hogehoge"
#         # $dir = "$major.$minor.$micro"
#         $dir = "$major.$minor.0"
#         $msvcver = "msvc2015"
#     }
#     else
#     {
#         $dir = "$major.$minor.$micro"
#     }
# 
#     # $filename = "PCL-" + "$dir" + "-AllInOne-" + "$msvcver" + "-" + "$platform_suffix.exe"
#     # $url = "$BASE_PCL_URL" + "$dir" + "/PCL-" + "$dir" + "-AllInOne-" + "$msvcver" + "-" + "$platform_suffix.exe"
# 
#     # (plan modified function)
#     $filepath = Download $filename $url
#     return $filepath
# }

function InstallOpenNI ($pcl_home, $openni_version, $architecture, $openni_home) 
{
    if ($architecture -eq "32")
    {
        $platform_suffix = "win32"
    }
    else
    {
        $platform_suffix = "win64"
    }
    
    $installer_path = $pcl_home + "\3rdParty\OpenNI\OpenNI-" + "$platform_suffix" + "-" + "$openni_version" + "-Dev.msi"
    $installer_ext = [System.IO.Path]::GetExtension($installer_path)
    Write-Host "Installing $installer_path to $openni_home"
    $install_log = $openni_home + "\install.log"
    if ($installer_ext -eq '.msi')
    {
        InstallOpenNIMSI $installer_path $openni_home $install_log
    }
    else
    {
        InstallOpenNIEXE $installer_path $openni_home $install_log
    }
    
    if (Test-Path $openni_home) 
    {
        Write-Host "OpenNI $openni_version ($architecture) installation complete"
    }
    else 
    {
        Write-Host "Failed to install OpenNI in $openni_home"
        # Get-Content -Path $install_log
        # Exit 1
    }
}


function InstallOpenNIMSI ($msipath, $openni_home, $install_log)
{
    # # http://www.ibm.com/support/knowledgecenter/SS2RWS_2.1.0/com.ibm.zsecure.doc_2.1/visual_client/responseexamples.html?lang=ja
    # $install_args = "/S /v/qn /v/norestart"
    # # RunCommand schtasks /create /tn openni_install /RL HIGHEST /tr $exepath /S /v/norestart /v/qn /sc once /st 23:59
    # # RunCommand schtasks /run /tn openni_install
    # # RunCommand schtasks /delete /tn openni_install /f
    # # RunCommand sleep 90
    # RunCommand "schtasks" "/create /tn openni_install /RL HIGHEST /tr `"$exepath $install_args`" /sc once /st 23:59"
    # RunCommand "sleep" "10"
    # RunCommand "schtasks" "/run /tn openni_install"
    # RunCommand "sleep" "90"
    # RunCommand "schtasks" "/delete /tn openni_install /f"

    # $install_args = "/qn /log $install_log /i $msipath TARGETDIR=$openni_home"
    # $install_args = "/qn /i `\`"$msipath`\`""
    # $uninstall_args = "/qn /x `\`"$msipath`\`""
    # $install_args = "/qn /i `"$msipath`""
    # $uninstall_args = "/qn /x `"$msipath`""
    $install_args = "/qn /norestart"
    $uninstall_args = "$msipath /qn /x"

    # RunCommand "msiexec.exe" $install_args
    # task use
    # RunCommand "schtasks" "/create /tn openni_install /RL HIGHEST /tr `"msiexec.exe $install_args`" /sc once /st 23:59"
    RunCommand "schtasks" "/create /tn openni_install /RL HIGHEST /tr `"msiexec.exe /i $msipath $install_args`" /sc once /st 23:59"
    RunCommand "sleep" "10"
    RunCommand "schtasks" "/run /tn openni_install"
    RunCommand "sleep" "90"
    RunCommand "schtasks" "/delete /tn openni_install /f"
    # if (-not(Test-Path $openni_home)) 
    # {
    #     Write-Host "OpenNI seems to be installed else-where, reinstalling."
    #     # RunCommand "msiexec.exe" $uninstall_args
    #     RunCommand "schtasks" "/create /tn openni_install /RL HIGHEST /tr `"$install_args`" /sc once /st 23:59"
    #     RunCommand "sleep" "10"
    #     RunCommand "schtasks" "/run /tn openni_install"
    #     RunCommand "sleep" "90"
    #     RunCommand "schtasks" "/delete /tn openni_install /f"
    # 
    #     # RunCommand "msiexec.exe" $install_args
    #     RunCommand "schtasks" "/create /tn openni_install /RL HIGHEST /tr `"$install_args`" /sc once /st 23:59"
    #     RunCommand "sleep" "10"
    #     RunCommand "schtasks" "/run /tn openni_install"
    #     RunCommand "sleep" "90"
    #     RunCommand "schtasks" "/delete /tn openni_install /f"
    # }
}

function InstallOpenNIEXE ($exepath, $openni_home, $install_log)
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
    # InstallPython $env:PYTHON_VERSION $env:PYTHON_ARCH $env:PYTHON
    # http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
    # InstallNumpy $env:PYTHON_VERSION $env:PYTHON_ARCH $env:PYTHON
    InstallPCL $env:PCL_VERSION $env:PYTHON_ARCH $env:PCL_ROOT
    # InstallOpenNI $env:PCL_ROOT_83 $env:OPENNI_VERSION $env:PYTHON_ARCH $env:OPENNI_ROOT
}

main
