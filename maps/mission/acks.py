"""What a verb answers a library call with: its ack and the notes joined onto
it, and an error served through the escalating repeat breaker."""

from __future__ import annotations

import re

from maps.query.result_tables import MODEL_UNSERVED_COLUMNS
from maps.withhold import withheld


def _summarise_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _live_counts(c_n: int, a_n: int, sep: str = ",") -> str:
    """The census's fid counts; the prior term only when the prior is served."""
    return f"{c_n} detection fid(s)" + (
        "" if withheld("prior") else f"{sep} {a_n} prior map row fid(s)")


def _add_note(d: dict, text: str, sep: str = " ", *,
              before: bool = False) -> None:
    """Join ``text`` onto ``d["note"]``, after it (or before it)."""
    old = d.get("note")
    d["note"] = ((f"{text}{sep}{old}" if before else f"{old}{sep}{text}")
                 if old else text)


class _Acks:
    """A verb's ack: ``{key: ack}``, and an error through the escalating
    breaker that tells a repeated same-class error from a first one."""

    def _loop_breaker_facts(self, n: int) -> str:
        c_n, a_n, cols, more = self._census(8)
        parts = [
            f" This error class has now occurred {n}x.",
            f" Live surface: {_live_counts(c_n, a_n, ';')}.",
        ]
        if cols:
            parts.append(
                f" Columns computed on rows so far: {cols}{more}.")
        return "".join(parts)

    def _err_class_sig(self, err: str) -> str:
        # Same-class signature: line prefix, quoted tokens (a run of them
        # counts as one) and numbers stripped.
        s = re.sub(r"^line \d+: ", "", str(err))
        s = re.sub(r"'[^']*'(, '[^']*')*", "'_'", s)
        s = re.sub(r"\d+", "_", s)
        return s[:100]

    def _census(self, cap: int) -> tuple[int, int, list[str], str]:
        """Live counts: detections, priors, the first ``cap`` computed
        columns and the " (+N more)" past them."""
        c_n = len(self.ns.get("cands") or [])
        a_n = len(self.ns.get("anchors") or [])
        cols = sorted({k for row in self.row_latest.values()
                       for k, v in row.items()
                       if k not in MODEL_UNSERVED_COLUMNS
                       and v not in (None, "")})
        more = f" (+{len(cols) - cap} more)" if len(cols) > cap else ""
        return c_n, a_n, cols[:cap], more

    def _repeated_rejection_state(self, err: str, n: int, consec: int) -> str:
        """The 3rd-consecutive-rejection fact, served instead of the remedy
        and census again: the counts, the rejection's first clause and the
        live state."""
        first = str(err).split(". ", 1)[0].strip().rstrip(".")
        if not first.endswith(("?", "!")):
            first += "."
        c_n, a_n, cols, more = self._census(12)
        return (
            f"rejected again -- the same error class, {consec}x "
            f"consecutively ({n}x): {first} The full "
            f"rejection text was served at its 1st occurrence and the "
            f"live-surface census at its 2nd (both above in this "
            f"conversation, unchanged). The executor is deterministic on "
            f"the same surface: re-issuing the same call unchanged "
            f"returns exactly this rejection. Live now: "
            f"{_live_counts(c_n, a_n)}"
            + (f"; columns computed on rows: {cols}{more}." if cols else "."))

    def _loop_broken_error(self, err: str, sig_of: str | None = None) -> str:
        """A repeated same-class error, escalating: verbatim first, then with
        the live census, then (3rd consecutive on) the repetition fact
        alone. A shape error gets only its count. A successful call resets
        the consecutive streak, never the class count."""
        sig = self._err_class_sig(err if sig_of is None else sig_of)
        n = self._err_class_counts[sig] = self._err_class_counts.get(sig, 0) + 1
        if self._err_sig == sig:
            self._err_n += 1
        else:
            self._err_sig, self._err_n = sig, 1
        consec = self._err_n
        if n < 2:
            return err
        # The escalation is a new sentence; a caret line is no sentence.
        if not err.endswith((".", "!", "?", "^")):
            err += "."
        # A syntax error's repeat serves only its count.
        if "syntax error" in err:
            return err + f" This error class has now occurred {n}x."
        if consec >= 3:
            return self._repeated_rejection_state(err, n, consec)
        return err + self._loop_breaker_facts(n)

    def _plan_err(self, ack: dict, msg: str) -> dict:
        """A plan error, through the escalating loop breaker."""
        return {**ack, "error": self._loop_broken_error(str(msg))}

    def _acked(self, key: str, ack: dict) -> dict:
        """A verb's ``{key: ack}``; an ack carrying an error is a plan error.
        (A look or a map that errors delivered nothing.)"""
        if ack.get("error"):
            return self._plan_err({key: ack}, str(ack["error"]))
        return {key: ack}
