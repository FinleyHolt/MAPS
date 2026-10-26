#!/bin/sh
# Starts the study on macOS and Linux. The first start fetches uv (which brings its own Python)
# and the console's packages into app/uv; later starts reuse them offline.
cd "$(dirname "$0")/.." || exit 1
HERE="$(pwd)"
UV="$HERE/app/uv/uv"
REQ="$HERE/app/requirements.txt"
export PYTHONUTF8=1 UV_CACHE_DIR="$HERE/app/uv/cache"
if [ ! -f app/uv/ready ]; then
  echo "First start: downloading what the study needs. This takes several minutes and needs the internet."
  curl -LsSf https://astral.sh/uv/0.13.5/install.sh |
    env UV_INSTALL_DIR="$HERE/app/uv" UV_NO_MODIFY_PATH=1 UV_PRINT_QUIET=1 sh
  if ! "$UV" run --managed-python --no-project --python 3.12 --with-requirements "$REQ" python -c pass; then
    echo "Setup did not finish. Check the internet connection, close this window and start the study again."
    exit 1
  fi
  touch app/uv/ready
fi
if ! "$UV" run --offline --managed-python --no-project --quiet --python 3.12 --with-requirements "$REQ" python -c pass; then
  rm -f app/uv/ready
  echo "The study stopped. Close this window and start the study again; that start needs the internet."
  exit 1
fi
cd app || exit 1
exec "$UV" run --offline --managed-python --no-project --quiet --python 3.12 --with-requirements "$REQ" \
  python -m baselines.human.study_server --kit "$HERE/kit"
