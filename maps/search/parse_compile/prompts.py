"""The served surface: ONE tool, ``run``, whose argument is a block of
ordinary Python.

The bench episode is ONE mission conversation. Each call carries one block;
``python_surface`` executes it in a namespace that persists for the whole
mission, and ``code_surface`` compiles the library calls it made onto the
existing op DAG (find / compare / filter_items / score_by, the flight and
perception verbs). The mission ends on the model's plain no-tool-call
reply, and the last flight is the answer.

``CODE_DOC`` is the served description -- the language legend -- and
``CODE_SYSTEM_PROMPT`` the mission map beside it. The worked example is
asserted at import against the real parser, because the model copies its
shape and an example the surface would refuse teaches a form that cannot
run.

NO DECODER MASK. Python is not EBNF-able here: ``strict: true`` on the
one-string envelope fixes the tool name and the argument shape, and the
parser's own refusals -- ordinary tool results the model can act on -- do
the rest.

``tool_choice`` is ALWAYS ``"auto"`` -- never required/named -- because auto
IS the abstain path: the plain reply that ends the mission has to stay
emittable on every turn.
"""

from __future__ import annotations

import textwrap

from maps.search.withhold import WITHHOLDS, withheld

# ── THE served surface ───────────────────────────────────────────────────────
# The op DAG in Python's shape: one tool, one string argument.
CODE_DOC = (
    "Write ordinary Python. A name you bind stays bound for the whole"
    " mission. Lines run in order; an error stops the block at its line"
    " and everything before it ran and stands. A bare handle on its own"
    " line SHOWS the rows, r1[0] one row, r1[0:20] a window; print() and"
    " any bare value come back in the receipt. The block's LAST result is"
    " served in full -- earlier assignments come back as a one-line"
    " summary, so show a set you want to read.\n"
    # A BLOCK IS SEVERAL LINES, and the shape of the ONE worked example is
    # the shape that comes back. The say legend's single example is a
    # flight onto a junction and the model copies BOTH halves of it; one
    # legend line carrying a crossing moved crossing-as-anchor 9 -> 24
    # (2.7x, p=0.0015). Every entry below is a one-line menu item, and the
    # first live episode on this surface wrote one call per block, five
    # blocks, find and fly only.
    #
    # The example carries the WHOLE chain the surface exists for, ending on
    # the one that reaches a part of a thing by measuring rather than by
    # estimating: watch a class, narrow to the thing, COVER it so the
    # segmenter puts a row on the part that matters, then fly to that row.
    #
    # IT PREVIOUSLY ENDED THE NARROWING ON `h = r2[0]` -- rank 0 by
    # DISTANCE, taken straight after asking the picture which one had the
    # pool. The model copied both halves: `look(r[a:b], "<question>")` in
    # 89 of 500 episodes and a colour ordering in 2. So the example spent
    # its ordering on proximity, spent its discrimination on a question
    # whose answer it then discarded, and taught that the answer is
    # whatever sorted first.
    #
    # ONE BLOCK NOW CARRIES THE WHOLE CHAIN. Lines execute eagerly in
    # source order -- a find AFTER an explore reads what the pass just
    # detected -- so the worked example is the full arc: watch, cover,
    # query what the pass found, rank on every criterion the cue names,
    # look, fly. That eager order is the execution model; the example is
    # how it is taught.
    "\nA block is any number of lines, run in order. This is one block:\n"
    "  look_for(\"pool\", \"house\")\n"
    "  explore(\"Mill Lane\")\n"
    "  pools = find(\"pool\")\n"
    "  r1 = find(\"house\", nearest=pools, color=\"white\")\n"
    "  look(r1[0:20])\n"
    "  go_to(r1[0])\n"
    "\nEach line runs the moment it is reached -- the find after the"
    " explore reads the rows that pass just detected.\n"
    "\nTHE MAP -- free, moves nothing\n"
    # The link that makes look_for read as load-bearing rather than
    # optional. find WORKS with an empty watchlist -- it returns priors --
    # so nothing in a find receipt says the segmenter is off, and the
    # first live episode read `find` succeeding as perception working.
    "  find reads ONE set of rows: the PRIOR map, plus whatever flights"
    " have detected. Nothing is detected until look_for names a class, so"
    " before that the set is the prior map alone.\n"
    "  r1 = find(\"house\")                 rows of that name or kind\n"
    "  r1 = find(\"Icaarus Court\")         a NAMED place: the name IS the"
    " query, on its own. There is no name= or query= to put it in.\n"
    "  r1 = find(\"house\", nearest=<place>)   ORDERS by distance to it and DROPS NOTHING -- every row of the class is still there, best first. To keep only the close ones say near= with dwithin=.\n"
    "  r1 = find(\"house\", nearest=[<place>, <place>])   ORDERS by BOTH"
    " distances together -- how you say a spot between two things, or a"
    " corner. When the description names two places, naming both here"
    " ranks far better than picking one and flying to where they meet.\n"
    "  r1 = find(\"house\", within=<place>)  KEEPS only what TOUCHES it --"
    " distance 0, \"on\" or \"at\", never \"nearby\"\n"
    "  r1 = find(\"house\", color=\"white\")   ORDERS by how well each row's"
    " OWN seen colour matches that word, best first, and drops none. It is"
    " the one criterion here that is not geometry, so it is the only way to"
    " rank on what a thing LOOKS like without flying to each one. Only a"
    " row something has SEEN carries a colour; a row nothing has looked at"
    " is UNKNOWN, and unknown ranks between the two -- above a row seen to"
    " be the wrong colour, below one seen to be right. Name it in the same"
    " find as the places and they rank together:"
    " find(\"house\", nearest=<place>, color=\"red\").\n"
    "  r1 = find(\"house\", east_of=<place>, west_of=<place>)   KEEPS what is"
    " on that side of each place, NEAREST first. north_of= and south_of="
    " too, one keyword per place, as many as the description names. A row"
    " the map places on the other side is dropped; a row it cannot place"
    " either way is KEPT and its column reads blank, so a blank means"
    " unknown and never \"no\". If nothing surveyed is on that side it says"
    " so -- the map holds no answer there and only looking will.\n"
    "  r2 = intersection(\"A Road\", \"B St\")  where two roads cross;"
    " SEVERAL is normal, so index the one you mean\n"
    "  A CORNER is two of those cuts against the SAME place:"
    " corner = intersection(\"86th Street\", \"Jacaranda Avenue\") then"
    " find(\"house\", north_of=corner, east_of=corner) is the house on the"
    " NORTHEAST corner of them. Any diagonal is its two halves that way,"
    " off a crossing, a road's end, or any place at all.\n"
    "  r3 = ends_of(<a road row>)         where a road ENDS -- one row per"
    " terminus, so a cul-de-sac or \"the end of\" a street is a thing you"
    " can go to. Reads a road; a footprint has no ends.\n"
    "  A CUT and a SORT are methods on the rows themselves, and they"
    " CHAIN.\n"
    "  r4 = r1.filter(within=<place>)      keep what TOUCHES it\n"
    "  r4 = r1.filter(near=<place>, dwithin=25)   keep what is within that"
    " many metres and DROP the rest. within= is contact; this is a stated"
    " GAP, which is how you say beside, outside, in the yard of -- a thing"
    " parked outside a house touches nothing, so containment cannot say"
    " it.\n"
    "  r3 = r1.order(nearest=<place>)      put them in ONE order, nearest"
    " first; r1.order(color=\"red\") orders by colour instead. ONE key: a"
    " table sorts by one column, and asking for two ranks whatever is"
    " middling at both above the thing that is right.\n"
    "  r1.order(direction=\"north\")   the northernmost FIRST. A direction"
    " ORDERS globally -- it is a fact about the map, not about any place,"
    " so it takes no anchor. Relating to a place is the other question and"
    " it is a CUT: east_of= keeps what lies that way of it, facing= keeps"
    " what faces it that way. Eight words here too.\n"
    "  FILTERS COMPOSE, ORDER DOES NOT. A cue names several things at"
    " once; each one is its own cut, and the last thing left is what you"
    " sort by. A cut you did not write is a thing you have not measured,"
    " never a thing that happens to be true.\n"
    "  A SET in near=, within= or nearest= is its UNION, and the distance"
    " is the gap to its NEAREST member -- so a whole CLASS is one place to"
    " measure from. r1.filter(near=find(\"house\"), dwithin=25) keeps"
    " every car outside SOME house without your having to pick the house"
    " first, and picking one house instead answers a different question:"
    " the nearest car to THAT house is not the car the cue described.\n"
    "  r4 = r1.filter(west_of=<place>)   keep what lies that way of it; north_of= south_of= east_of= too, and they CHAIN. A row the map cannot place either way is KEPT with a blank column -- unmeasured is never refuted.\n"
    "  Relating rows to a place also stamps WHICH WAY EACH ONE FACES it: \"facing <place>\" reads a compass word -- north, southeast -- for the direction from that row to that place. A thing faces what it fronts, so a house whose front is on the road reads facing <road>. It rides every near= and nearest= you already write; read it like any column, r[\"facing Poppy Boulevard\"].\n"
    "  r5 = r1.filter(facing=<place>, direction=\"south\")   keep the"
    " south-facing ones -- the cut on that same column. direction= takes"
    " north south east west, and a DIRECTION COMPOSES: a thing facing"
    " southwest faces south AND west, so direction=\"south\" keeps it. For"
    " the diagonal itself chain the two -- .filter(facing=<place>,"
    " direction=\"south\").filter(facing=<place>, direction=\"west\")."
    " A row TOUCHING the place has no facing word and is KEPT --"
    " unmeasured is never refuted.\n"
    "  find takes the cut too: find(<thing>, near=<place>, dwithin=25)"
    " KEEPS what is that close and DROPS the rest. near= always CUTS and always needs its gap; nearest= always ORDERS and never drops -- the same split .filter() and .order() use.\n"
    "  distance(a, b) is metres between two FOOTPRINTS -- the same"
    " quantity the \"distance to <ref>\" column carries -- and"
    " a.distance(b) is that measurement as a method. A set counts"
    " as ONE place, so it is the gap to its nearest member.\n"
    "  Either takes a plain function instead, over rows you already hold:"
    " r1.filter(lambda r: r.distance(houses) <= 25) and"
    " r1.order(lambda r: r.distance(poppy)) are the same two operations.\n"
    "  A row's xy is [east, north] in metres: +x runs EAST and +y runs NORTH, so a larger y is further north. Compass words follow it -- 0 = north, 90 = east.\n"
    "  r5 = r3[0]   r6 = r3[0:20]          ONE row / the first 20, a set\n"
    "  r1.exact / r1.tagged / r1.fuzzy    the find's match TIERS as"
    " sets: the rows that ARE the queried thing, the tag-carriers, the"
    " word-sharers. Live views -- explore(r1.exact), look(r1.tagged),"
    " anything a set can.\n"
    "  more_of(r1)                         the next block of a long result\n"
    "\nSENSING -- free, moves nothing\n"
    "  look_for(\"pool\", \"house\")   what the segmenter looks for on"
    " EVERY later flight. NOTHING is detected until you say this. Up to"
    " four classes, each naming a THING, never a colour.\n"
    "  ARM EVERY CLASS THE CUE NAMES, in one call, before you fly --"
    " the thing you are looking for AND everything it is described"
    " relative to. \"a red car outside a house\" is TWO classes: the car"
    " is what you want and the house is what makes it the right one, and"
    " a house nothing segmented is a row that does not exist to measure"
    " against. The classes cost nothing to add: the sensor runs once per"
    " picture and every class you named rides that same pass. The cost is"
    " all on the other side -- a class you left out is missing from every"
    " flight of the mission, and a relation you cannot measure is one you"
    " will end up guessing.\n"
    "  look_for(\"pool\", at=<place>)   the same, plus extra"
    " segmentation there.\n"
    "  stand(at=<place>, facing=\"north\")   a ground viewpoint; facing"
    " takes a compass word or a place. Its sides ride every find row;"
    " crops and maps face its way.\n"
    "  look(r1[0])       a CLOSE look at one row -- the sensor's own"
    " pixels, continuous: shape, colour, condition, what stands beside it."
    " A row's columns carry only what OSM recorded; anything else about a"
    " thing is only in the pixels. THE PIXELS ARE THE SENSOR'S COVERAGE:"
    " a row on ground no flight has covered has none yet -- explore"
    " covers the ground, then look.\n"
    "  look(r1, marks=<place>)  one crop PER ROW, in the set's order, each"
    " badged with the row it shows; marks= draws what ELSE bears on the"
    " choice -- a named road writes its NAME on the picture, so a"
    " candidate standing on the wrong street shows it. A long set is"
    " served from the front, so slice the part you mean.\n"
    "  show_map(around=<place>)   the world-state map centred there: what"
    " is KNOWN, drawn as a schematic, not what is there.\n"
    "\nWHAT YOU SEE BECOMES ROWS -- the distinction the map cannot make,"
    " asked as a query instead of as a question\n"
    # THE REPLACEMENT FOR THE LOOKING QUESTION. `look(rows, "which of these
    # has a pool behind it?")` is gone: it answered into the picture and
    # bound nothing, so what it resolved had nowhere to land, and the
    # measured behaviour was to ask it (89 of 500 episodes) and then index
    # rank 0 by distance anyway. A set slot folds to its NEAREST member, so
    # the same question is a sort over rows the segmenter produced.
    "  A thing the map does not carry -- a pool, solar panels, a shed, a"
    " parked car -- is not a row until look_for names it and a flight goes"
    " over it. Then it IS one, and it sorts and cuts like any other:\n"
    "    look_for(\"pool\")\n"
    "    houses = find(\"house\", nearest=\"Glade Avenue\")\n"
    "    explore(houses[0])\n"
    "    pools = find(\"pool\")\n"
    "    answer = find(\"house\", nearest=pools)   the house nearest a pool"
    " ranks first\n"
    "  A SET in nearest= / near= / within= is ONE place: the gap to its"
    " NEAREST member, one column, not one column per row. So nearest=pools"
    " ranks every house by its own closest pool, and"
    " houses.filter(within=pools) keeps only the houses that hold one.\n"
    "  The same goes for a cue WORD no column carries -- \"small\","
    " \"organized\", a shape: it is only in the pixels. look_for the"
    " classes the cue names AND what stands on the thing -- a lot is"
    " found by its cars, a helipad by its H, and what stands on a thing"
    " is often the stronger detection -- explore the named place so the"
    " candidates become rows, then look at them and choose by what the"
    " crops show.\n"
    "\nMOVING -- the only two calls that move\n"
    "  go_to(r5)         FLIES to that thing. Your answer is wherever you"
    " flew LAST.\n"
    "  explore(r5)       a covering pass over the WHOLE of it, one stop at"
    " a time, segmenting your watched classes at every stop. Hands back a"
    " handle over the rows it found, and SHOWS ITS CATCH: one crop per"
    " row it added, in pass order. Ends on its last stop.\n"
    "  explore(r6)       a SET is one place to cover: ONE pass over the"
    " union of its rows, chained end to end."
    " explore(\"Willow Avenue\") covers EVERY segment of that name -- the"
    " name is the whole street, not its first fragment.\n"
    "  BOTH TAKE A THING ON THE MAP, never a coordinate. To reach a PART"
    " of something -- the west side of a store, the bulb at the end of a"
    " court -- look_for the class you want, explore the thing so the"
    " segmenter puts rows on it, then go_to the row you mean. That is a"
    " measured position; a coordinate you read off a picture is a guess.\n"
    "\nTYPES\n"
    "  a handle holds a SET; r1[0] is a ROW; a quoted name is a PLACE;"
    " [x, y] is a POINT -- points RELATE and LOOK, they do not move you.\n"
    "  a ROW is its served columns, readable as data: r.xy, r[\"class\"],"
    " r.get(\"color\"), r.keys(); a spaced column goes through"
    " r[\"distance to ...\"]; print(r) shows them. Two sets add into a"
    " plain list of rows to loop over.\n"
    "  go_to takes ONE THING; explore takes ONE THING or ONE SET -- a set"
    " is one place to cover. Every RELATING slot -- near= nearest= within="
    " east_of= west_of= north_of= south_of= -- takes one place, a point"
    " included, or a set of them, which it relates to as ONE place.\n"
    "  show_map(around=) and stand(at=, facing=) take ONE thing: a"
    " viewpoint stands somewhere and looks at something. Index a set"
    " first.\n"
    "  A fid or a point binds: h = \"prior_3123\" is a thing you can move"
    " to; corner = [3411, 1468] is a place to look and relate near, and"
    " binding it does not make it movable.\n"
    "\nORDINARY PYTHON RUNS HERE\n"
    "  Variables, arithmetic, f-strings, if / for / while, len(),"
    " sorted(), print(). No underscore names. go_to and"
    " explore cannot go inside a loop, because a loop would move to every"
    " row of the set -- loop to LOOK, then move once to the place you"
    " chose, or explore(the set) to cover it all in one pass.\n")

# THE WORKED BLOCK IS SERVED CODE, so the surface's own scan is what says
# whether it is right. A legend that teaches a form the surface refuses is
# the failure this arm was built after: 30 blocks in 22% of episodes were
# the model writing sentences the prompt taught and the parser would not
# take. Read back out of the legend rather than restated here -- a second
# copy is a second thing to go stale.
import ast as _ast

from maps.search.parse_compile.python_surface import (
    scan_block as _scan_block)

_WORKED = "\n".join(
    ln[2:] for ln in CODE_DOC.partition("This is one block:\n")[2]
    .partition("\nEach line runs")[0].splitlines() if ln.startswith("  "))
assert _WORKED.count("\n") >= 5, _WORKED       # the scan found the block
_tree = _scan_block(_WORKED)                   # the surface accepts it
_calls = [n.func.id for n in _ast.walk(_tree)
          if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)]
# and it exercises the chain it exists to teach: watch, COVER, retrieve
# inside the pass, rank, look, fly to what the pass put there.
assert {"look_for", "find", "look",
        "explore", "go_to"} <= set(_calls), _calls
# ORDERING IS PINNED ON THE ARGS, not on the token `order`: the example
# RANKS ON EVERY CRITERION THE CUE NAMED, appearance included -- that is
# the thing that went extinct (13 colour-ordering heads over 2 of 500
# episodes, against 440 over 175 of 300 on the sentence language) and the
# worked block is the lever on it.
_find_kwargs = {k.arg for n in _ast.walk(_tree)
                if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)
                and n.func.id == "find" for k in n.keywords}
assert "color" in _find_kwargs, _find_kwargs
# nearest= is the ORDERING spelling (near= now cuts and needs dwithin=),
# so the spatial criterion the worked block must carry is either.
assert _find_kwargs & {"nearest", "near", "within", "east_of", "west_of",
                       "north_of", "south_of"}, _find_kwargs
del _WORKED, _tree, _calls, _find_kwargs

# NO `strict`. It asked the SERVE to compile a decode-time constraint out of
# the schema below, and that schema is `{code: string}` -- every token sequence
# satisfies it, so the only thing a tag could ever hold was the tool name and
# the JSON envelope. The envelope is the part models never get wrong.
#
# MEASURED, on the arms that were GENUINELY unenforced -- gemma-4 E2B on a
# stock venv (381 calls), Muse Glimmer, whose parser compiles nothing (2,305),
# and this flag's own removal arm (1,266). 3,952 calls across a 130x parameter
# range, of which ONE left the served surface.
#
# It is worth naming, because it is the whole of what this flag bought:
#
#     show_map(around="prior_150")   <- called as a TOOL
#     {"error": "unknown tool 'show_map'; available: run"}
#     run(code="for q in [...]: r = find(q) ...")   <- recovered, next turn
#
# show_map is a code-surface FUNCTION; the model emitted it one level up.
# The harness named the error, the model fixed it in one turn, and the
# episode scored SR=1 in BOTH arms. One recoverable turn per ~4,000 calls,
# against roughly 20 turns spent of a 100-turn cap.
#
# Paired on identical val_seen episodes, arms differing only by this flag:
# 36/48 against 34/48 on the shared set, 3 wins against 5 losses, exact
# McNemar p = 0.73. Stopped at 48 of 64 shared episodes once the answer
# stopped moving, so it is a partial read and says so -- and it is quoted
# over the earlier 29/29 tie because that tie was the FAST half of the set,
# where short episodes skew easy.
#
# The flag was also never uniform: it constrained qwen and inkling and did
# nothing at all for gemma or muse, so it was never the fleet-wide guarantee
# it reads as. benchmarks/citynav/analysis/tool_call_audit.py catches this
# class after a run instead -- it is what found the show_map call.
#
# It was not even uniform. sglang's qwen parser built a tag from this flag and
# its inkling parser builds one either way off the auto path, so the fleet has
# been running constrained and unconstrained side by side to identical
# conformance. benchmarks/citynav/analysis/tool_call_audit.py measures the
# property after a run instead, on real traffic, at zero serving cost.
CODE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "run",
        "description": CODE_DOC,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["code"],
            "properties": {"code": {"type": "string"}},
        },
    },
}

CODE_TOOLS: list = [CODE_TOOL]


# ── the system prompt ────────────────────────────────────────────────────────
# The mission's load-bearing facts, in this surface's notation: the last
# flight is the answer, naming commits nothing, the answer is a point, a
# picture is north-up and its caption carries the world rectangle, a name is
# not a place, a colour orders and never drops. The from-looking
# prefix becomes stand().
#
# It exists because serving the sentence prompt beside a code tool served
# two contradictory surfaces at once: 30 blocks in 22% of episodes were
# the model writing sentences the prompt taught and the parser refused,
# and every tool result footer told it to call look_for("<class>") on an
# arm where no such call existed.
CODE_SYSTEM_PROMPT = """\
=== MISSION MAP ===
You are the reasoner for a drone search instruction, in ONE persistent
conversation. You write ordinary Python: one block per call, its lines
run in order, and an error stops the block at its line -- everything
before it ran and stands. Names persist for the whole mission.
- go_to(...) FLIES NOW: one aerial stop, flown the moment the block
  lands, and it LEAVES you there. A place flies to the middle of it.
  The flight folds that stop's perception into the world-state: SAM
  segments the classes you have looked for, and a flight with none
  looked for adds no rows. It perceives ONE nadir view square centred on
  its stop; nothing outside it is seen. Before the first flight the
  graph holds only PRIOR map features.
- explore(...) flies a covering PASS over the whole of a place, stop by
  stop, segmenting your watched classes at each. It hands back a handle
  over the rows it found and shows its CATCH -- one crop per row it
  added. It ENDS on its last stop. A SET is one place: explore(rows)
  covers the union in one pass, and explore of a NAME covers every
  segment bearing it.
- BOTH MOVING CALLS TAKE A THING ON THE MAP -- a row, a fid, a name --
  and never a coordinate. Points RELATE and LOOK (near=, within=,
  show_map(around=), stand(at=)); they do not move you.
- stand(at=..., facing=...) adopts a ground viewpoint without flying:
  rows carry its sides and what renders faces its way. look_for(...)
  directs the sensors. Neither moves the drone or ends anything.
- Your mission ends when you reply in plain text with no tool call. The
  LAST flight's target IS your answer, a scouting point included;
  naming the object in your reply commits nothing, only flying to it
  does. A reply before any flight answers nothing. When you know which
  object the instruction means -- and a cue word no column carries is
  known only from a look, never from rank -- go_to() its row, then
  reply.
- Your answer is the ROW of your last flight, and it is only as good as
  that row's fit to the thing you were asked for. If you can see the
  thing, fly to the row that IS it -- so the way to answer with a PART of
  something is to make the segmenter put a row on that part and go_to it.
- A later operator message is an UPDATE to the same search.
- A SCORED sort is evidence of the SORTED column -- a distance, a colour
  -- never that its head is the thing. An attribute in NO column is
  settled by look(), not by rank.

=== WORLD CONVENTIONS ===
- EVERY PICTURE IS NORTH-UP UNLESS ITS CAPTION SAYS OTHERWISE, and its
  caption carries the world rectangle it covers: [x left..right, y
  bottom..top]. The TOP edge is the LARGER y, so y DECREASES as you look
  down the image; x increases to the right. Reading a position off a
  picture says WHERE TO LOOK -- find(..., nearest=[x, y]),
  show_map(around=[x, y]) -- and never where the answer is: a picture
  gives your estimate of a position, a row gives a measurement of one,
  and only rows can be moved to.
- One node set, provenance-tagged: prior (map features) | runtime
  (in-flight detections). A prior carries raw OSM tags, and a name only
  where OSM has one; a detection's class is the detector's measurement.
  Per-flight perception colour-enriches in-view priors and folds new
  detections in.
- WHAT NO COLUMN CARRIES IS IN THE PIXELS. A cue word no tag and no
  column spells -- a shape, a condition, what stands beside a thing --
  is read from a crop and nowhere else. Crops are earned by flying: a
  look at ground no pass has covered has nothing to show yet.
- Rows display when a real ordering exists; an UNORDERED result displays
  census + len only. Every result stays live behind its handle.
- A NAME is not a PLACE: a name match says the NAME matched, never WHICH
  instance the cue means.
- A SPATIAL relation is what drops rows; a colour ORDERS and never drops.
- look_for(...) takes plain visual classes -- look_for("pool", "house")
  -- no proper nouns, no colour qualifier.
"""


# ── bench-only withholds (maps.search.withhold) ──────────────────────────────
# Each withhold's edits to the flown text, applied at call time. An edit is
# ("sub", old, new): replace text written once; ("drop", opening): remove the
# one entry that opens so; ("cut", start, end): remove from start up to end.
# In the description an entry is a line; in the system turn it is a bullet,
# matched on its words joined by single spaces and re-wrapped at 72 columns
# when an edit changes it. With nothing withheld the literals are served as
# written.
_DOC_EDITS: dict[str, list[tuple[str, ...]]] = {
    "look_for": [
        ("sub", '  look_for("pool", "house")\n  explore("Mill Lane")\n',
         '  explore("Mill Lane")\n'),
        ("cut", "\nEach line runs the moment it is reached", "\nTHE MAP"),
        ("drop", "  find reads ONE set of rows"),
        ("drop", '  look_for("pool", "house")   what the segmenter'),
        ("drop", "  ARM EVERY CLASS THE CUE NAMES"),
        ("drop", '  look_for("pool", at=<place>)'),
        ("cut", "WHAT YOU SEE BECOMES ROWS", "MOVING -- "),
        ("sub", "one stop at a time, segmenting your watched classes at every"
                " stop.", "one stop at a time."),
        ("sub", " To reach a PART of something -- the west side of a store,"
                " the bulb at the end of a court -- look_for the class you"
                " want, explore the thing so the segmenter puts rows on it,"
                " then go_to the row you mean. That is a measured position; a"
                " coordinate you read off a picture is a guess.", ""),
    ],
    "look": [
        ("sub", "  look(r1[0:20])\n  go_to(r1[0])\n", "  go_to(r1[0])\n"),
        ("sub", "explore(r1.exact), look(r1.tagged), anything a set can.",
         "explore(r1.exact), anything a set can."),
        ("drop", "  look(r1[0])       a CLOSE look"),
        ("drop", "  look(r1, marks=<place>)"),
        ("sub", " look_for the classes the cue names AND what stands on the"
                " thing -- a lot is found by its cars, a helipad by its H, and"
                " what stands on a thing is often the stronger detection --"
                " explore the named place so the candidates become rows, then"
                " look at them and choose by what the crops show.", ""),
    ],
}
_TURN_EDITS: dict[str, list[tuple[str, ...]]] = {
    "look_for": [
        ("sub", " The flight folds that stop's perception into the"
                " world-state: SAM segments the classes you have looked for,"
                " and a flight with none looked for adds no rows.", ""),
        ("sub", "stop by stop, segmenting your watched classes at each.",
         "stop by stop."),
        ("sub", " look_for(...) directs the sensors. Neither moves the drone"
                " or ends anything.", ""),
        ("sub", " If you can see the thing, fly to the row that IS it -- so"
                " the way to answer with a PART of something is to make the"
                " segmenter put a row on that part and go_to it.", ""),
        ("drop", "- One node set, provenance-tagged"),
        ("drop", "- look_for(...) takes plain visual classes"),
    ],
    "look": [
        ("sub", " -- and a cue word no column carries is known only from a"
                " look, never from rank --", ","),
        ("sub", " An attribute in NO column is settled by look(), not by"
                " rank.", ""),
        ("drop", "- WHAT NO COLUMN CARRIES IS IN THE PIXELS"),
    ],
    "mission_map": [
        ("cut", "=== MISSION MAP ===", "=== WORLD CONVENTIONS ==="),
    ],
}


def _units(text: str, turn: bool) -> list[str]:
    """The text as entries that join back to it: a turn bullet with its
    continuation lines, otherwise one line each."""
    out: list[str] = []
    for ln in text.splitlines(keepends=True):
        if turn and out and ln.startswith("  ") and out[-1].startswith("- "):
            out[-1] += ln
        else:
            out.append(ln)
    return out


def _flat(unit: str) -> str:
    return " ".join(unit.split())


def _edit(text: str, edit: tuple[str, ...], turn: bool) -> str | None:
    """``text`` with ``edit`` applied, or None when it names no text there."""
    kind, *args = edit
    if kind == "cut":
        i, j = text.find(args[0]), text.find(args[1])
        return text[:i] + text[j:] if 0 <= i < j else None
    units = _units(text, turn)
    for k, u in enumerate(units):
        if kind == "drop" and u.startswith(args[0]):
            return "".join(units[:k] + units[k + 1:])
        if kind == "sub" and turn and u.startswith("- ") and args[0] in _flat(u):
            flat = _flat(u).replace(args[0], args[1])
            units[k] = textwrap.fill(
                flat[2:], 72, initial_indent="- ", subsequent_indent="  ",
                break_on_hyphens=False, break_long_words=False) + "\n"
            return "".join(units)
    if kind == "sub" and not turn and text.count(args[0]) == 1:
        return text.replace(args[0], args[1])
    return None


def _served(text: str, edits: dict, turn: bool) -> str:
    for name in WITHHOLDS:
        if name in edits and withheld(name):
            for e in edits[name]:
                # An earlier withhold may already have taken this text.
                text = _edit(text, e, turn) or text
    return text


def served_doc() -> str:
    """The ``run`` description this run serves."""
    return _served(CODE_DOC, _DOC_EDITS, turn=False)


def served_tools() -> list:
    """The tools this run serves: ``CODE_TOOLS`` over ``served_doc()``."""
    doc = served_doc()
    if doc == CODE_DOC:
        return CODE_TOOLS
    return [{**CODE_TOOL, "function": {**CODE_TOOL["function"], "description": doc}}]


def served_system_prompt() -> str:
    """The mission prompt this run serves (``system_turn`` is
    ``reasoner_agent.served_system_text``'s)."""
    return _served(CODE_SYSTEM_PROMPT, _TURN_EDITS, turn=True)


# Every edit names text the flown literals carry once, so an edit to a
# literal that strands one fails here instead of serving the literal.
for _edits, _text, _turn in ((_DOC_EDITS, CODE_DOC, False),
                             (_TURN_EDITS, CODE_SYSTEM_PROMPT, True)):
    for _name, _list in _edits.items():
        assert _name in WITHHOLDS, _name
        for _e in _list:
            assert _edit(_text, _e, _turn) not in (None, _text), (_name, _e)
            if _e[0] == "drop":
                _n = sum(u.startswith(_e[1]) for u in _units(_text, _turn))
            elif _e[0] == "sub" and _turn:
                _n = sum(_e[1] in _flat(u) for u in _units(_text, _turn))
            else:
                _n = 1 if _e[0] == "cut" else _text.count(_e[1])
            assert _n == 1, (_name, _e)
del _edits, _text, _turn, _name, _list, _e, _n
