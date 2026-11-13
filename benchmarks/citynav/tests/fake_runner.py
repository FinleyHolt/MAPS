"""A stand-in attempt child for the driver tests; the episode id's map names its behaviour."""

import json
import os
import signal
import sys
import time
from pathlib import Path

argv = sys.argv
eid = argv[argv.index("--episode-id") + 1]
kind = eid.split("::")[0]
if kind == "nodir":
    sys.exit(2)
d = Path(argv[argv.index("--attempt-root") + 1]) / f"maps_20260101_000000_citynav_{kind}"
d.mkdir(parents=True)
print(d, flush=True)
row: dict = {"episode_id": eid, "status": "ok"}
if kind == "ok":
    row["metrics"] = {"SR": 1, "NE": 3.0}
    row["deadline"] = float(os.environ["CITYNAV_EPISODE_DEADLINE_EPOCH"])
elif kind == "miss":
    row["error"] = "RuntimeError: boom"
elif kind == "hog":
    # One block past the driver's memory cap, flown by the mission's interpreter.
    from maps.library.interpreter import MissionPython
    from maps.mission.tests.fakes import quiet_env

    out = MissionPython(quiet_env()).run_block("x = [0] * (2 * 10**8)\nprint(len(x))")
    row["error"] = out.get("error")
elif kind == "infra":
    row.update(status="infra", error="SamServerOutage: down")
elif kind in ("sleep", "reasoner", "sam"):
    time.sleep(60)
elif kind == "kill":
    os.kill(os.getpid(), signal.SIGKILL)
elif kind == "norow":
    sys.exit(0)
elif kind == "exit":
    sys.exit(3)
(d / "episode.json").write_text(json.dumps(row))
