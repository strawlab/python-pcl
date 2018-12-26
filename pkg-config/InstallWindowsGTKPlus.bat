@echo off

REM HINT=http://qiita.com/usagi/items/2623145f22faf54b99e0

cd /d%~dp0
:checkMandatoryLevel
for /f "tokens=1 delims=," %%i in ('whoami /groups /FO CSV /NH') do (
    if "%%~i"=="BUILTIN\Administrators" set ADMIN=yes
    if "%%~i"=="Mandatory Label\High Mandatory Level" set ELEVATED=yes
)

if "%ADMIN%" neq "yes" (
    echo This file needs to be executed with administrator authority{not Administrators Group}
   if "%1" neq "/R" goto runas
   goto exit1
)
if "%ELEVATED%" neq "yes" (
    echo This file needs to be executed with administrator authority{The process has not been promoted}
   if "%1" neq "/R" goto runas
   goto exit1
)


:admins
    REM Install GTK+
    REM powershell -NoProfile -ExecutionPolicy Unrestricted  .\Install-GTKPlus.ps1
    REM powershell -v 2 -NoProfile -ExecutionPolicy Unrestricted  .\Install-GTKPlus.ps1
    REM powershell -v 3 -NoProfile -ExecutionPolicy Unrestricted  .\Install-GTKPlus.ps1
    REM default use upper ver 5(use zip archive package)
    powershell -v 5 -NoProfile -ExecutionPolicy Unrestricted  .\Install-GTKPlus.ps1

    goto exit1

:runas
    REM Re-run as administrator
    powershell -Command Start-Process -Verb runas "%0" -ArgumentList "/R" 

:exit1