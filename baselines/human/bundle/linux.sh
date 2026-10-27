#!/bin/sh
# The Linux launcher: run it from a terminal in the study folder (sh "Start study (Linux).sh"); app/run.sh does the work.
exec /bin/sh "$(dirname "$0")/app/run.sh"
