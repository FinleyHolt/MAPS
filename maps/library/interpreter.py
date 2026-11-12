"""The interpreter for the mission's ``run`` channel: one persistent namespace
per mission, in which every block the model sends is vetted (``block_scan``),
compiled and executed under the liveness ceilings (``block_ceilings``), and
the receipt each block folds into.

Every mission call (`find`, `go_to`, `look_for`, ...) is a real function in
that namespace (``calls``), so the model's Python behaves as Python and an
error costs only the line it is on."""

from __future__ import annotations

import ast
import difflib
import re
import time

from maps.library.block_ceilings import (
    block_line,
    charging_lines,
    code_objects,
    outer_block_line,
)
from maps.library.block_scan import (
    BLOCK_FILENAME,
    LIBRARY_CALLS,
    METHOD_FOR_FREE_SPELLING,
    SAFE_BUILTINS,
    SAFE_MODULES,
    safe_import,
    scan_block,
    shadowing_remedy,
    show_bare_expressions,
)
from maps.library.call_arguments import MissionEnv, Names, _Stop, no_such_fid
from maps.library.calls import Library
from maps.library.op_compiler import CodeError, fid_like
from maps.library.rows import Row, Rows, as_called

# Liveness ceiling on one block's own lines (harness frames are untraced).
# A block that runs 100k model-written lines is a runaway loop, not a plan.
TRACE_LINE_CEILING = 100_000

# Wall-clock ceiling on ONE block's execution: it catches a BOUNDED loop
# whose calls are each slow (`for d in rows: find(near=d, dwithin=40)` over
# the live world state), since the runner thread is the one executing the
# block. Past it the block stops at its line with a served error.
TRACE_DEADLINE_S = 90.0

# The printed channel is capped the way a notebook caps cell output, so a
# loop that prints a city-wide result cannot flood the context.
PRINT_LINE_CAP = 60
PRINT_CHAR_CAP = 8_000


class MissionNamespace(dict):
    """The exec globals (CPython honours ``__missing__`` on a dict-subclass
    globals): an undefined name that spells a served fid is a :class:`Row` of
    the live row, a prior_/detected_ fid with no row is refused, any other
    fid spelling is the string; a name this mission unbound says why, and a
    near-miss gets a did-you-mean."""

    def __init__(self, base: dict):
        super().__init__(base)
        #: The mission's library, read by the rows a name resolves to (and,
        #: under this name, by ``rows._Cuts._mission``); set once it is built.
        self.mp: Library | None = None
        #: name -> why this mission removed it.
        self.unbound: dict = {}

    def __missing__(self, k):
        mp = self.mp
        if fid_like(k):
            row = mp.env.row_of_fid(k)
            if isinstance(row, dict):
                return Row(row, mp=mp)
            if k.startswith(("prior_", "detected_")):
                raise no_such_fid(k)
            return k
        # A plain KeyError is what lets LOAD_GLOBAL fall through to
        # __builtins__.
        if k in (dict.get(self, "__builtins__") or {}):
            raise KeyError(k)
        if k in self.unbound:
            raise CodeError(f"{k} is unbound: {self.unbound[k]}")
        # A near-miss of a LIVE fid names the correction (detect_59 ->
        # detected_59), but ONLY when the prefix is a near-miss of a fid
        # family: a `corner_5` typo falls through to the bound handles.
        m = re.fullmatch(r"([A-Za-z]+)_(\d+)", k)
        if m:
            for pref in ("detected", "prior"):
                if not difflib.get_close_matches(m.group(1), [pref], n=1, cutoff=0.75):
                    continue
                fix = f"{pref}_{m.group(2)}"
                if fix != k and mp.env.row_of_fid(fix) is not None:
                    raise CodeError(f"there is no {k} on this surface -- did you mean {fix}?")
        # A free spelling of a method: the name stays undefined and the
        # refusal names the method.
        near = difflib.get_close_matches(k, tuple(METHOD_FOR_FREE_SPELLING), n=1, cutoff=0.8)
        if k in METHOD_FOR_FREE_SPELLING or near:
            k = k if k in METHOD_FOR_FREE_SPELLING else near[0]
            raise CodeError(
                f"there is no {k}() here -- did you mean "
                f"{METHOD_FOR_FREE_SPELLING[k]}? A cut and a sort are methods on the "
                "rows they act on."
            )
        # The suggestion pool: the mission calls and the model's own bound
        # handles, never the module names.
        known = [
            n
            for n in self
            if isinstance(n, str) and not n.startswith("__") and n not in SAFE_MODULES
        ]
        close = difflib.get_close_matches(k, LIBRARY_CALLS + tuple(known), n=2, cutoff=0.6)
        if close:
            raise CodeError(
                f"there is no {k} here -- did you mean " + " or ".join(dict.fromkeys(close)) + "?"
            )
        raise KeyError(k)


class MissionPython:
    """The persistent interpreter for one mission's run channel: the
    namespace the model's blocks run in, the library its calls are
    (:class:`Library`) and the receipt each block returns."""

    def __init__(self, env: MissionEnv):
        self.env = env
        self.ns = MissionNamespace({"__builtins__": dict(SAFE_BUILTINS)})
        self.names = Names(self.ns)
        self.library = self.ns.mp = Library(env, self.names)
        self.ns["__builtins__"]["print"] = self._print
        self.ns["__builtins__"]["__import__"] = safe_import
        self.ns["__show__"] = self._show
        self.ns.update(SAFE_MODULES)
        # The mission calls are builtins, as len is: a name the model binds
        # over one shadows it until ``del`` gives the call back.
        for name in LIBRARY_CALLS:
            self.ns["__builtins__"][name] = as_called(getattr(self.library, name), name)
        self._picked: dict = {}  # fids the model's Row names hold
        # Every block's code objects: a def from an earlier block is still
        # the model's own lines.
        self._codes: list = []
        # per-block state
        self._block_codes: set = set()
        self._printed = ""
        self._lines = 0
        self._deadline = 0.0

    def _print(self, *parts, sep=" ", end="\n", flush=False):
        # print()'s own signature; the block's output is one text, cut into
        # lines when the block ends.
        self._printed += (" " if sep is None else sep).join(str(p)[:2_000] for p in parts) + (
            "\n" if end is None else end
        )

    def _show(self, value, label: str | None):
        if value is None:
            return
        library = self.library
        if (
            isinstance(value, Rows)
            and not value._spec
            and library.unrendered.get(value._ref) == self.env.version()
        ):
            return
        if isinstance(value, Rows):
            # Key the envelope by the MODEL's spelling ("r1[0:5]"), not the
            # executor's ref the window compiled onto (the window shows it
            # under a label; a nameless set is served whole all the same).
            library.shown_whole.add(library._window(value._ref, value._spec, label, value._ref))
        elif isinstance(value, Row) and value._src and not re.fullmatch(r"\w+", label):
            # Render the ROW: a one-row window keeps the ordering
            # attribute a bare dict loses, so the model sees the row it
            # asked for rather than a census of one.
            ref, i = value._src
            spec = f"[{i}:{i + 1}]"
            library._window(ref, spec, label, ref)
        elif isinstance(value, Row):
            library.out.setdefault("shown", {})[label] = value._bind
        else:
            self._printed += (
                f"{label} = {value!r}" if label != repr(value) else repr(value)
            ) + "\n"

    def run_block(self, code: str) -> dict:
        out: dict = {}
        self.library.begin_block(out)
        self.names.begin_block(code)
        self._printed = ""
        self._lines = 0
        try:
            tree = show_bare_expressions(scan_block(code, bound=self.ns))
            compiled = compile(tree, BLOCK_FILENAME, "exec")
        except CodeError as e:
            return self.env.say_err(out, f"nothing in this block ran -- {e}")
        codes = code_objects(compiled)
        self._block_codes = set(codes)  # the block the model just sent
        self.names.running(self._block_codes)
        self._codes.extend(codes)

        self._deadline = time.monotonic() + TRACE_DEADLINE_S
        try:
            with charging_lines(self):
                exec(compiled, self.ns)
        except Exception as e:
            self._unbind_failed_assign(tree, outer_block_line())
            if not isinstance(e, _Stop):  # a _Stop's refusal is in the receipt
                line = block_line(self._block_codes)
                what = str(e) if isinstance(e, CodeError) else f"{type(e).__name__}: {e}"
                if isinstance(e, NameError) and e.name in LIBRARY_CALLS:
                    # a call name resolves through the builtins, so only a
                    # del of it misses
                    what = (
                        f"nothing is bound over {e.name} to del; "
                        f"{e.name}() is the mission call itself and stays."
                    )
                elif isinstance(e, TypeError) and str(e).endswith("is not callable"):
                    shadowed = next(
                        (
                            n.func.id
                            for n in ast.walk(tree)
                            if isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Name)
                            and n.lineno <= line <= n.end_lineno
                            and n.func.id in LIBRARY_CALLS
                            and n.func.id in self.ns
                        ),
                        None,
                    )
                    if shadowed:
                        what += f": {shadowing_remedy(shadowed)}"
                out.update(self.env.say_err({}, f"line {line}: {what}"))
        return self._finish(out)

    def _tick(self) -> None:
        """One line of the block's OWN code ran: the liveness ceilings."""
        self._lines += 1
        if self._lines > TRACE_LINE_CEILING:
            raise CodeError(
                f"the block ran past {TRACE_LINE_CEILING:,} of its "
                "own lines -- an unbounded loop. Simplify the block."
            )
        if time.monotonic() > self._deadline:
            raise CodeError(
                f"the block ran past {TRACE_DEADLINE_S:.0f} s -- a slow or "
                "large query, most likely a find/near loop over the whole "
                "map. Scope it tighter (fewer rows, a smaller dwithin)"
                " or split it across turns."
            )

    def _unbind_failed_assign(self, tree: ast.Module, line: int) -> None:
        """A rebind that failed UNBINDS the name: Python leaves the target
        holding its OLD value when the right-hand side raises, and a stale
        pick that stays bound is a wrong commit."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and node.lineno <= line <= node.end_lineno:
                for t in node.targets:
                    # ast.walk reaches Names inside tuple/list unpacking
                    # too -- `a, b = <raises>` must unbind BOTH.
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            self.ns.pop(n.id, None)
                            self.ns.unbound[n.id] = (
                                "its assignment failed, so the stale "
                                "value was removed -- re-derive it"
                            )

    def _finish(self, out: dict) -> dict:
        if self._printed:
            lines = self._printed.splitlines()
            if len(lines) > PRINT_LINE_CAP:
                kept = (
                    lines[: PRINT_LINE_CAP - 8]
                    + [
                        f"... [{len(lines) - PRINT_LINE_CAP + 3} printed "
                        "lines suppressed -- slice or aggregate instead of "
                        "printing whole sets]"
                    ]
                    + lines[-5:]
                )
                lines = kept
            text = "\n".join(lines)
            if len(text) > PRINT_CHAR_CAP:
                text = (
                    text[:PRINT_CHAR_CAP] + f"\n... [printed output clipped at "
                    f"{PRINT_CHAR_CAP:,} chars]"
                )
            out["printed"] = text
        # Named picks: the fid a model-named Row holds, receipted once.
        picked = {}
        for k, v in self.ns.items():
            if isinstance(v, Row) and not k.startswith("__"):
                if self._picked.get(k) != v.fid and v.fid:
                    pk = {"fid": str(v.fid)}
                    if v.name:
                        pk["name"] = str(v.name)
                    picked[k] = pk
                self._picked[k] = v.fid
        if picked:
            out["picked"] = picked
        # Serve the model's own names over the executor's refs.
        mine = dict(self.names.shown)
        for k, v in self.ns.items():
            if isinstance(v, Rows) and not v._spec and not k.startswith("__"):
                mine[v._ref] = k
        res = out.get("result")
        if isinstance(res, dict) and isinstance(res.get("results"), dict):
            results = dict(res["results"])
            # ONE BLOCK, ONE TABLE: the block's LAST result and everything
            # the model explicitly showed keep their rows; every other
            # envelope folds to its hint under the model's name for it.
            keys = list(results)
            answer = keys[-1] if keys else None
            answer_env = results.get(answer)
            said = set(
                str(answer_env.get("hint") or "").splitlines()
                if isinstance(answer_env, dict)
                else ()
            )
            folded = []
            for k in keys[:-1]:
                if k in self.library.shown_whole:
                    continue
                env = results.pop(k)
                hint = env.get("hint") if isinstance(env, dict) else None
                # Under a name the model can read, else unprefixed; a line
                # the answer already says is dropped, but a named result
                # keeps its own status line.
                name = mine.get(k) or self.names.labels.get(k)
                keep = [
                    ln
                    for i, ln in enumerate(str(hint or "").splitlines())
                    if (i == 0 and name) or ln not in said
                ]
                if keep:
                    hint = "\n".join(keep)
                    folded.append(f"{name}: {hint}" if name else hint)
            if folded and isinstance(results.get(answer), dict):
                a = dict(results[answer])
                a["hint"] = "\n".join(([a["hint"]] if a.get("hint") else []) + folded)
                results[answer] = a
            # A key is the set's spelling: an unbound find its call unless
            # another key already says it, a nameless set its mission number.
            said = {mine[k] for k in results if k in mine}
            keys = []
            for k in results:
                name = mine.get(k) or self.names.labels.get(k)
                if name is None or (k not in mine and name in said):
                    name = self.names.unnamed_label(k)
                said.add(name)
                keys.append(name)
            # The legend rides the receipt's first table only.
            envs, legend = list(results.values()), False
            for i, env in enumerate(envs):
                if isinstance(env, dict) and "legend" in env:
                    if legend:
                        envs[i] = {k: v for k, v in env.items() if k != "legend"}
                    legend = True
            out["result"] = {**res, "results": dict(zip(keys, envs))}
        names = sorted(
            k for k, v in self.ns.items() if isinstance(v, (Rows, Row)) and not k.startswith("__")
        )
        if names:
            out["handles"] = names
        return out
