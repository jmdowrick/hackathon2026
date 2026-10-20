#!/usr/bin/env bash
# Start (or RESUME) the gramicidin run, detached, on a remote workstation.
#
#   PYTHON=/path/to/python PLATFORM=CUDA ./launch_ga.sh --ns 100 [--device 0]
#
# Survives the ssh session ending. Re-running the same command after an
# interruption resumes from the last checkpoint; run_ga.py's lock refuses a
# second copy while one is still running.
set -uo pipefail                 # not -e: report failures, don't vanish
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
PLATFORM="${PLATFORM:-CUDA}"

# Fail NOW, in the terminal, if OpenMM or the platform is missing -- not
# hours later in a log nobody is watching.
"$PY" - "$PLATFORM" <<'PY' || { echo "FATAL: no OpenMM with platform $PLATFORM for $PY"; exit 1; }
import sys, openmm
from openmm import Platform
names = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
print("openmm", openmm.__version__, "platforms", names, flush=True)
sys.exit(0 if sys.argv[1] in names else 1)
PY

if command -v setsid >/dev/null 2>&1; then DETACH="setsid nohup"; else DETACH="nohup"; fi
$DETACH "$PY" run_ga.py --dir . --platform "$PLATFORM" "$@" >> run.log 2>&1 < /dev/null &
echo $! > run.pid
sleep 5
if kill -0 "$(cat run.pid)" 2>/dev/null; then
    echo "running, pid $(cat run.pid). Follow with:  tail -f $(pwd)/run.log"
else
    echo "FATAL: run exited within 5 s:"; tail -20 run.log; exit 1
fi
