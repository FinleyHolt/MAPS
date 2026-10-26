"""The participant's zip: the launchers keep their execute bit, Windows files carry CRLF, the
code rides without the bench's corpus, and a kit's results never ship."""

import zipfile

from baselines.human.study_package import build_bundle


def test_bundle(tmp_path) -> None:
    kit = tmp_path / "kit"
    (kit / "plan").mkdir(parents=True)
    (kit / "plan" / "plan.json").write_text("{}")
    (kit / "results" / "p01").mkdir(parents=True)
    (kit / "results" / "p01" / "answers.jsonl").write_text("{}\n")
    build_bundle(kit, tmp_path / "b.zip")
    z = zipfile.ZipFile(tmp_path / "b.zip")
    names = set(z.namelist())
    mode = lambda n: z.getinfo(n).external_attr >> 16
    assert mode("MAPS-study/Start study (Mac).command") & 0o111 and mode("MAPS-study/app/run.sh") & 0o111
    assert b"\r\n" in z.read("MAPS-study/Start study (Windows).bat")
    assert "MAPS-study/app/baselines/human/study_server.py" in names and "MAPS-study/kit/plan/plan.json" in names
    assert not any("/tests/" in n or "/calcity/data/" in n or "/results/" in n for n in names)
