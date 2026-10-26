@echo off
rem Starts the study on Windows. The first start fetches uv (which brings its own Python)
rem into this folder and the console's packages into uv's cache; later starts reuse them offline.
rem No parenthesised blocks: a folder name with brackets or an apostrophe cannot break the parse.
cd /d "%~dp0"
if not exist "kit\console\index.html" goto inzip
set "PYTHONPATH=%~dp0app"
set "PYTHONUTF8=1"
rem The x64 Python on Windows on Arm too: every pinned package has x64 wheels.
set "UV_PYTHON_ARCH=x86_64"
set "UV_INSTALL_DIR=%~dp0app\uv"
set "UV_NO_MODIFY_PATH=1"
if exist "app\uv\ready" goto start
echo First start: downloading what the study needs. This takes several minutes and needs the internet.
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/0.13.5/install.ps1 | iex"
if not exist "app\uv\uv.exe" goto failed
"app\uv\uv.exe" run --no-project --python 3.12 --with-requirements "app\requirements.txt" python -c pass || goto failed
type nul > "app\uv\ready"
:start
echo Starting the study. Your browser will open. Keep this window open until you finish.
"app\uv\uv.exe" run --offline --no-project --quiet --python 3.12 --with-requirements "app\requirements.txt" python -m baselines.human.study_server --kit "%~dp0kit"
pause
exit /b
:failed
echo Setup did not finish. Check the internet connection, close this window and start the study again.
pause
exit /b 1
:inzip
echo The study is still inside the zip. Close this window, right-click the zip, choose Extract All,
echo then start the study from the extracted folder.
pause
exit /b 1
