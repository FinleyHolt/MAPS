"""Parse each exported Cal City instruction into CityNav's processed form.

Every released checkpoint reads ``data/cityrefer/processed_descriptions.json``,
one ``{target, landmarks, surroundings}`` per description: MGP captions its
GSAM target and surroundings maps with them, and all three draw the named
``landmarks`` on their landmark map. CityNav ships that file, produced by its
authors' LLM parse; Cal City needs its own. This asks an OpenAI-compatible
chat endpoint, once per exported block, and keeps only landmark names the
block actually holds: ``LandmarkMap`` snaps every query to the nearest
catalog name, so an invented name would silently become a real one.

Resumable: blocks already in the output are skipped, and the file is
rewritten after every block.

Environment:

  ``RELEASED_PARSE_URL``      base URL, for example ``http://127.0.0.1:8001/v1``
  ``RELEASED_PARSE_MODEL``    served model name (default: the first of /models)
  ``RELEASED_PARSE_API_KEY``  bearer token, when the endpoint wants one

Usage::

    python -m baselines.released.describe_instructions --export DIR
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

from baselines.released.calcity_export import TARGET_OBJECT_ID

PROMPT = """\
An aerial navigation instruction describes one target to find.

Instruction: {description}

Named places on the map: {landmarks}

Return one JSON object with three keys:
  "target": a short noun phrase naming the target object itself.
  "landmarks": the named places the instruction refers to, copied exactly
    from the list above; [] when it refers to none of them.
  "surroundings": short noun phrases for unnamed objects the instruction
    places near the target.
Return only the JSON object."""


def parse_reply(text: str, block_landmarks: list[str]) -> dict:
    """The reply's JSON object, held to the processed-description schema.

    Landmarks outside the block's own names are dropped rather than snapped.
    Raises ``ValueError`` on a reply with no usable object.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    raw = json.loads(match.group(0))
    target = str(raw.get("target") or "").strip()
    if not target:
        raise ValueError(f"reply names no target: {raw!r}")
    known = set(block_landmarks)
    return {
        "target": target,
        "landmarks": [n for n in raw.get("landmarks") or [] if n in known],
        "surroundings": [str(s).strip() for s in raw.get("surroundings") or [] if str(s).strip()],
    }


def _post(url: str, payload: dict | None, api_key: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def describe(export_dir: Path, base_url: str, model: str | None, api_key: str | None) -> Path:
    manifest = json.loads((export_dir / "manifest.json").read_text())
    descriptions = {}
    for shard in sorted((export_dir / "shards").iterdir()):
        for row in json.loads((shard / "data/citynav/citynav_test_unseen.json").read_text()):
            descriptions[f"calcity_block_{row['block']}"] = row["descriptions"][0]
    out = export_dir / "processed_descriptions.json"
    done = json.loads(out.read_text()) if out.exists() else {}
    base = base_url.rstrip("/")
    model = model or _post(f"{base}/models", None, api_key)["data"][0]["id"]
    for block, meta in manifest["blocks"].items():
        if block in done:
            continue
        prompt = PROMPT.format(
            description=descriptions[block],
            landmarks="; ".join(meta["landmarks"]) or "(none)",
        )
        reply = _post(f"{base}/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 512,
        }, api_key)
        parsed = parse_reply(reply["choices"][0]["message"]["content"], meta["landmarks"])
        done[block] = {str(TARGET_OBJECT_ID): [parsed]}
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(done))
        tmp.replace(out)
    missing = set(manifest["blocks"]) - set(done)
    if missing:
        raise SystemExit(f"{len(missing)} blocks unparsed")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", type=Path, required=True,
                    help="the calcity_export output directory")
    args = ap.parse_args(argv)
    url = os.environ.get("RELEASED_PARSE_URL")
    if not url:
        raise SystemExit("set RELEASED_PARSE_URL to an OpenAI-compatible /v1 base URL")
    out = describe(args.export, url, os.environ.get("RELEASED_PARSE_MODEL"),
                   os.environ.get("RELEASED_PARSE_API_KEY"))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
