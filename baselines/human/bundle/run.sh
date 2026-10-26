#!/bin/sh
# Starts the study on macOS and Linux. The first start fetches uv (which brings its own Python)
# into this folder and the console's packages into uv's cache; later starts reuse them offline.
cd "$(dirname "$0")/.." || exit 1
HERE="$(pwd)"
UV="$HERE/app/uv/uv"
REQ="$HERE/app/requirements.txt"
export PYTHONPATH="$HERE/app" PYTHONUTF8=1
if [ ! -f app/uv/ready ]; then
  echo "First start: downloading what the study needs. This takes several minutes and needs the internet."
  curl -LsSf https://astral.sh/uv/0.13.5/install.sh | env UV_INSTALL_DIR="$HERE/app/uv" UV_NO_MODIFY_PATH=1 sh
  if ! "$UV" run --no-project --python 3.12 --with-requirements "$REQ" python -c pass; then
    echo "Setup did not finish. Check the internet connection, close this window and start the study again."
    exit 1
  fi
  touch app/uv/ready
fi
echo "Starting the study. Your browser will open. Keep this window open until you finish."
exec "$UV" run --offline --no-project --quiet --python 3.12 --with-requirements "$REQ" \
  python -m baselines.human.study_server --kit "$HERE/kit"
