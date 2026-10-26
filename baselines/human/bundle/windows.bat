@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Starts the study on Windows. The first start fetches uv and the console's packages into app\uv
rem (uv keeps the Python it fetches in the user's own folder); later starts reuse them offline.
rem No parenthesised blocks: a folder name with brackets or an apostrophe cannot break the parse.
cd /d "%~dp0"
if not exist "kit\console\index.html" goto inzip
set "KIT=%~dp0kit"
set "UV=%~dp0app\uv\uv.exe"
set "REQ=%~dp0app\requirements.txt"
set "PYTHONUTF8=1"
rem The x64 Python on Windows on Arm too: every pinned package has x64 wheels.
set "UV_PYTHON_ARCH=x86_64"
set "UV_CACHE_DIR=%~dp0app\uv\cache"
set "UV_INSTALL_DIR=%~dp0app\uv"
set "UV_NO_MODIFY_PATH=1"
if exist "app\uv\ready" goto start
echo First start: downloading what the study needs. This takes several minutes and needs the internet.
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/0.13.5/install.ps1 | iex"
if not exist "app\uv\uv.exe" goto failed
"%UV%" run --managed-python --no-project --python 3.12 --with-requirements "%REQ%" python -c pass || goto failed
type nul > "app\uv\ready"
:start
"%UV%" run --offline --managed-python --no-project --quiet --python 3.12 --with-requirements "%REQ%" python -c pass || goto offline
cd /d "%~dp0app"
"%UV%" run --offline --managed-python --no-project --quiet --python 3.12 --with-requirements "%REQ%" python -m baselines.human.study_server --kit "%KIT%"
pause
exit /b
:offline
del "app\uv\ready"
echo The study stopped. Close this window and start the study again; that start needs the internet.
pause
exit /b 1
:failed
echo Setup did not finish. Check the internet connection, close this window and start the study again.
pause
exit /b 1
:inzip
echo The study is still inside the zip. Close this window, right-click the zip, choose Extract All,
echo then start the study from the extracted folder.
pause
exit /b 1
