"""What a verb answers a library call with: its ack and the notes joined onto
it, and an error served through the escalating repeat breaker."""

from __future__ import annotations

import re


def summarise_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def live_counts(c_n: int, a_n: int, sep: str = ",") -> str:
    """The census's fid counts."""
    return f"{c_n} detection fid(s){sep} {a_n} prior map row fid(s)"


def add_note(d: dict, text: str, sep: str = " ", *, before: bool = False) -> None:
    """Join ``text`` onto ``d["note"]``, after it (or before it)."""
    old = d.get("note")
    d["note"] = (f"{text}{sep}{old}" if before else f"{old}{sep}{text}") if old else text


def error_class_signature(err: str) -> str:
    """An error's same-class signature: line prefix, quoted tokens (a run of
    them counts as one) and numbers stripped."""
    s = re.sub(r"^line \d+: ", "", str(err))
    s = re.sub(r"'[^']*'(, '[^']*')*", "'_'", s)
    s = re.sub(r"\d+", "_", s)
    return s[:100]


class Rejections:
    """A verb's refusals, through the escalating breaker that tells a
    repeated same-class error from a first one. ``surface`` is read for the
    live census a repeat serves."""

    def __init__(self, surface):
        self.surface = surface
        self.new_turn()

    def new_turn(self) -> None:
        """An operator turn counts its error classes afresh."""
        self._class_counts: dict[str, int] = {}
        self._streak_sig, self._streak_n = None, 0

    def end_streak(self) -> None:
        """A successful call ends the consecutive streak, never the class
        count: the next same-class error serves its full text again."""
        self._streak_sig, self._streak_n = None, 0

    def _loop_breaker_facts(self, n: int) -> str:
        c_n, a_n, cols, more = self.surface.census(8)
        parts = [
            f" This error class has now occurred {n}x.",
            f" Live surface: {live_counts(c_n, a_n, ';')}.",
        ]
        if cols:
            parts.append(f" Columns computed on rows so far: {cols}{more}.")
        return "".join(parts)

    def _repeated_rejection_state(self, err: str, n: int, consec: int) -> str:
        """The 3rd-consecutive-rejection fact, served instead of the remedy
        and census again: the counts, the rejection's first clause and the
        live state."""
        first = str(err).split(". ", 1)[0].strip().rstrip(".")
        if not first.endswith(("?", "!")):
            first += "."
        c_n, a_n, cols, more = self.surface.census(12)
        return (
            f"rejected again -- the same error class, {consec}x "
            f"consecutively ({n}x): {first} The full "
            f"rejection text was served at its 1st occurrence and the "
            f"live-surface census at its 2nd (both above in this "
            f"conversation, unchanged). The executor is deterministic on "
            f"the same surface: re-issuing the same call unchanged "
            f"returns exactly this rejection. Live now: "
            f"{live_counts(c_n, a_n)}"
            + (f"; columns computed on rows: {cols}{more}." if cols else ".")
        )

    def escalated(self, err: str, sig_of: str | None = None) -> str:
        """A repeated same-class error, escalating: verbatim first, then with
        the live census, then (3rd consecutive on) the repetition fact
        alone. A shape error gets only its count."""
        sig = error_class_signature(err if sig_of is None else sig_of)
        n = self._class_counts[sig] = self._class_counts.get(sig, 0) + 1
        if self._streak_sig == sig:
            self._streak_n += 1
        else:
            self._streak_sig, self._streak_n = sig, 1
        consec = self._streak_n
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

    def error(self, ack: dict, msg: str) -> dict:
        """``ack`` with a plan error, through the escalating loop breaker."""
        return {**ack, "error": self.escalated(str(msg))}

    def acked(self, key: str, ack: dict) -> dict:
        """A verb's ``{key: ack}``; an ack carrying an error is a plan error.
        (A look or a map that errors delivered nothing.)"""
        if ack.get("error"):
            return self.error({key: ack}, str(ack["error"]))
        return {key: ack}
