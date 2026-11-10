@echo off
rem videoforge launcher: `vf <command> ...` from anywhere (render python + this repo on the path)
setlocal
set "VF_ROOT=%~dp0"
set "PYTHONPATH=%VF_ROOT%;%PYTHONPATH%"
set "PYTHONIOENCODING=utf-8"
pushd "%VF_ROOT%"
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m vf %*
set RC=%ERRORLEVEL%
popd
exit /b %RC%
