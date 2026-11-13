#!/bin/bash
# The README Quickstart, steps 2 to 4: the release's example runs open in the console.
set -euo pipefail
# PORT, when set, moves the console off the README's 8788.
d=$(mktemp -d)
curl -fsSL -o "$d/runs.zip" https://github.com/FinleyHolt/MAPS/releases/latest/download/MAPS-example-runs.zip
unzip -q "$d/runs.zip" -d "$d"
folder=$(dirname "$(find "$d" -name basemap.jpg | head -1)")
SEARCH_CALCITY_TILES=$folder/bake python -m station.serve --bench calcity \
    --out "$folder/runs" --basemap "$folder/basemap.jpg" ${PORT:+--port $PORT} > "$d/station.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null || true' EXIT
for _ in $(seq 60); do
  if curl -fs -o "$d/index.html" http://127.0.0.1:${PORT:-8788}/ && grep -qi '<html' "$d/index.html"; then
    n=$(curl -fs "http://127.0.0.1:${PORT:-8788}/missions/runs" | python -c "import json,sys; print(len(json.load(sys.stdin)['runs']))")
    echo "console answers with $n runs"
    [ "$n" = 10 ] && exit 0
    break
  fi
  kill -0 $pid 2>/dev/null || break
  sleep 1
done
cat "$d/station.log"
exit 1
