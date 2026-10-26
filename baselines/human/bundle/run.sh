#!/bin/sh
# Starts the study on macOS and Linux. The first start fetches uv (which brings its own Python)
# into this folder and the console's packages into uv's cache; later starts reuse them.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
UV="$HERE/app/uv/uv"
if [ ! -x "$UV" ]; then
  echo "First start: setting up. This takes a few minutes and needs the internet."
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HERE/app/uv" UV_NO_MODIFY_PATH=1 sh
fi
export PYTHONPATH="$HERE/app"
echo "Starting the study. Your browser will open. Keep this window open until you finish."
exec "$UV" run --no-project --quiet --python 3.12 --with-requirements "$HERE/app/requirements.txt" \
  python -m baselines.human.study_server --kit "$HERE/kit"
