"""The null reference the ledger prints beside every pairing.

A p-value read alone invites a story. These pin what the tool must and must
not call noise, because getting that wrong is worse than not showing it: a
"noise floor" that quietly includes real A/Bs would talk someone out of a
genuine effect.

Measured on the real ledger while this was written: identical code, seed
apart, produced a +9.5 pt delta at p=0.007 — more extreme than a result being
treated as suspicious in the same session.
"""

from __future__ import annotations

from benchmarks.citynav.analysis import episode_ledger


# --- the null reference ------------------------------------------------


def _ledger_with(tmp_path, cols):
    """cols: {column: (provenance, {episode: sr})}."""
    import json

    p = tmp_path / "ledger.jsonl"
    with p.open("w") as fh:
        for name, (prov, out) in cols.items():
            fh.write(json.dumps({"kind": "column", "column": name,
                                 "provenance": prov}) + "\n")
            for e, sr in out.items():
                fh.write(json.dumps({"column": name, "kind": "cell",
                                     "episode_id": e, "sr": sr}) + "\n")
    return p


def _prov(**over):
    base = {"split": "train_seen", "served_model": "m", "look_tool": True,
            "config_sha256": "cfg", "git_commit": "sha",
            "search_env_flags": {"SEARCH_CITYNAV_TILES": "/t/citynav_2m_osmv3"}}
    base.update(over)
    return base


def test_a_null_pairing_needs_more_than_a_shared_commit(tmp_path):
    """Two columns can share a sha and still be different treatments — the
    oracle base/gt arms do exactly that. Anything the provenance records as
    an ``arm`` difference disqualifies the pair, or the 'noise' floor would
    quietly include real A/Bs."""
    eps = {f"e{i}": i % 2 for i in range(60)}
    led = _ledger_with(tmp_path, {
        "sha:a_k1": (_prov(), eps),
        "sha:a_k2": (_prov(), eps),
        "sha:b_k1": (_prov(config_sha256="other"), eps),   # arm differs
    })
    rows = episode_ledger.null_reference(led)
    pairs = {tuple(sorted((r["a"], r["b"]))) for r in rows}
    assert ("sha:a_k1", "sha:a_k2") in pairs
    assert not any("sha:b_k1" in p for p in pairs), "an arm difference is not null"


def test_a_null_row_carries_its_discordance(tmp_path):
    """The flip counts size the instrument: SD(delta) = sqrt(disc)/n, so a row
    without them can only be read against someone's memory of the noise. They
    were ``None`` on every row for as long as they existed — ``ab_stats`` keys
    them ``wins_fail_to_success`` / ``losses_success_to_fail``, never
    ``mcnemar['wins']`` — and nothing downstream noticed, because nothing
    downstream read them."""
    a = {f"e{i}": int(i < 30) for i in range(60)}
    b = {f"e{i}": int(i < 20 or i >= 50) for i in range(60)}  # 10 losses, 10 wins
    led = _ledger_with(tmp_path, {"sha:a_k1": (_prov(), a), "sha:a_k2": (_prov(), b)})
    row = episode_ledger.null_reference(led)[0]
    assert (row["wins"], row["losses"]) == (10, 10)
    assert "disc=  20" in episode_ledger.render_null([row])


def test_seed_only_pairs_are_marked_apart_from_the_rest(tmp_path):
    """Provenance cannot see every treatment (mtp0 vs mtp3 shares everything
    it records), so the name carries the last distinction and the report has
    to separate the two classes rather than average them."""
    eps = {f"e{i}": i % 2 for i in range(60)}
    led = _ledger_with(tmp_path, {
        "sha:oraclec_base_k2": (_prov(), eps),
        "sha:oraclec_base_k3": (_prov(), eps),
        "sha:sxs_mtp0_750": (_prov(), eps),
        "sha:sxs_mtp3_750": (_prov(), eps),
    })
    rows = episode_ledger.null_reference(led)
    marked = {tuple(sorted((r["a"], r["b"]))): r["replicate"] for r in rows}
    assert marked[("sha:oraclec_base_k2", "sha:oraclec_base_k3")] is True
    assert marked[("sha:sxs_mtp0_750", "sha:sxs_mtp3_750")] is False
    text = episode_ledger.render_null(rows)
    assert "seed-only pairs" in text and "*" in text


def test_an_empty_null_says_so_instead_of_reassuring(tmp_path):
    """A ledger with no replicate pair cannot show the scale. Printing
    nothing would read as 'no noise here', which is the opposite of true."""
    led = _ledger_with(tmp_path, {"sha:only": (_prov(), {"e1": 1})})
    text = episode_ledger.render_null(episode_ledger.null_reference(led))
    assert "cannot show you" in text and "suspicion" in text


def test_pair_narrows_to_a_listed_subset_after_the_identity_checks(tmp_path):
    """A subset read (the flown share of a block) is the same pairing over
    fewer cells: the 90% overlap refusal is about column identity and runs on
    the full intersection, and a list matching nothing is a refusal, not a
    zero."""
    cells_a = {f"e{i}": int(i % 2 == 0) for i in range(40)}
    cells_b = {f"e{i}": int(i % 3 == 0) for i in range(40)}
    p = _ledger_with(tmp_path, {"x:a": (_prov(), cells_a),
                                "y:b": (_prov(git_commit="y"), cells_b)})
    whole = episode_ledger.pair(p, "x:a", "y:b")
    assert whole["subset"] is None and whole["shared"] == 40
    listed = {f"e{i}" for i in range(10)} | {"e999"}
    part = episode_ledger.pair(p, "x:a", "y:b", episodes=listed)
    assert part["shared"] == 10
    assert part["subset"] == {"listed": 11, "matched": 10, "of": 40}
    import pytest
    with pytest.raises(SystemExit, match="none of the 1 listed"):
        episode_ledger.pair(p, "x:a", "y:b", episodes={"e999"})
