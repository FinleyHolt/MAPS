#!/bin/sh
# The Mac's double-click launcher; app/run.sh does the work, on Linux too.
exec /bin/sh "$(dirname "$0")/app/run.sh"
