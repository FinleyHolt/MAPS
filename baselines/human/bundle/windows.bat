@echo off
rem Starts the study on Windows. The first start fetches uv (which brings its own Python)
rem into this folder and the console's packages into uv's cache; later starts reuse them.
cd /d "%~dp0"
if not exist "app\uv\uv.exe" (
  echo First start: setting up. This takes a few minutes and needs the internet.
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR='%~dp0app\uv'; $env:UV_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/install.ps1 | iex"
)
set "PYTHONPATH=%~dp0app"
echo Starting the study. Your browser will open. Keep this window open until you finish.
"app\uv\uv.exe" run --no-project --quiet --python 3.12 --with-requirements "app\requirements.txt" python -m baselines.human.study_server --kit "%~dp0kit"
pause
