"""The served text: the ``run`` tool's description (``RUN_TOOL_DOC``), the
system prompt (``SYSTEM_PROMPT``) and the tool schema (``SERVED_TOOLS``)."""

from __future__ import annotations

RUN_TOOL_DOC = (
    "Write ordinary Python. A name you bind stays bound for the whole"
    " mission. Lines run in order; an error stops the block at its line"
    " and everything before it ran and stands. A bare handle on its own"
    " line SHOWS the rows, r1[0] one row, r1[0:20] a window; print() and"
    " any bare value come back in the receipt. The block's LAST result is"
    " served in full -- earlier assignments come back as a one-line"
    " summary, so show a set you want to read.\n"
    "\nA block is any number of lines, run in order. This is one block:\n"
    '  look_for("pool", "house")\n'
    '  explore("Mill Lane")\n'
    '  pools = find("pool")\n'
    '  r1 = find("house", nearest=pools, color="white")\n'
    "  look(r1[0:20])\n"
    "  go_to(r1[0])\n"
    "\nEach line runs the moment it is reached -- the find after the"
    " explore reads the rows that pass just detected.\n"
    "\nTHE MAP -- free, moves nothing\n"
    "  find reads ONE set of rows: the PRIOR map, plus whatever flights"
    " have detected. Nothing is detected until look_for names a class, so"
    " before that the set is the prior map alone.\n"
    '  r1 = find("house")                 rows of that name or kind\n'
    '  r1 = find("Icaarus Court")         a NAMED place: the name IS the'
    " query, on its own. There is no name= or query= to put it in.\n"
    '  r1 = find("house", nearest=<place>)   ORDERS by distance to it and DROPS NOTHING -- every row of the class is still there, best first. To keep only the close ones say near= with dwithin=.\n'
    '  r1 = find("house", nearest=[<place>, <place>])   ORDERS by BOTH'
    " distances together -- how you say a spot between two things, or a"
    " corner. When the description names two places, naming both here"
    " ranks far better than picking one and flying to where they meet.\n"
    '  r1 = find("house", within=<place>)  KEEPS only what TOUCHES it --'
    ' distance 0, "on" or "at", never "nearby"\n'
    '  r1 = find("house", color="white")   ORDERS by how well each row\'s'
    " OWN seen colour matches that word, best first, and drops none. It is"
    " the one criterion here that is not geometry, so it is the only way to"
    " rank on what a thing LOOKS like without flying to each one. Only a"
    " row something has SEEN carries a colour; a row nothing has looked at"
    " is UNKNOWN, and unknown ranks between the two -- above a row seen to"
    " be the wrong colour, below one seen to be right. Name it in the same"
    " find as the places and they rank together:"
    ' find("house", nearest=<place>, color="red").\n'
    '  r1 = find("house", east_of=<place>, west_of=<place>)   KEEPS what is'
    " on that side of each place, NEAREST first. north_of= and south_of="
    " too, one keyword per place, as many as the description names. A row"
    " the map places on the other side is dropped; a row it cannot place"
    " either way is KEPT and its column reads blank, so a blank means"
    ' unknown and never "no". If nothing surveyed is on that side it says'
    " so -- the map holds no answer there and only looking will.\n"
    '  r2 = intersection("A Road", "B St")  where two roads cross;'
    " SEVERAL is normal, so index the one you mean\n"
    "  A CORNER is two of those cuts against the SAME place:"
    ' corner = intersection("86th Street", "Jacaranda Avenue") then'
    ' find("house", north_of=corner, east_of=corner) is the house on the'
    " NORTHEAST corner of them. Any diagonal is its two halves that way,"
    " off a crossing, a road's end, or any place at all.\n"
    "  r3 = ends_of(<a road row>)         where a road ENDS -- one row per"
    ' terminus, so a cul-de-sac or "the end of" a street is a thing you'
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
    ' first; r1.order(color="red") orders by colour instead. ONE key: a'
    " table sorts by one column, and asking for two ranks whatever is"
    " middling at both above the thing that is right.\n"
    '  r1.order(direction="north")   the northernmost FIRST. A direction'
    " ORDERS globally -- it is a fact about the map, not about any place,"
    " so it takes no anchor. Relating to a place is the other question and"
    " it is a CUT: east_of= keeps what lies that way of it, facing= keeps"
    " what faces it that way.\n"
    "  FILTERS COMPOSE, ORDER DOES NOT. A cue names several things at"
    " once; each one is its own cut, and the last thing left is what you"
    " sort by. A cut you did not write is a thing you have not measured,"
    " never a thing that happens to be true.\n"
    "  A SET in near=, within= or nearest= is its UNION, and the distance"
    " is the gap to its NEAREST member -- so a whole CLASS is one place to"
    ' measure from. r1.filter(near=find("house"), dwithin=25) keeps'
    " every car outside SOME house without your having to pick the house"
    " first, and picking one house instead answers a different question:"
    " the nearest car to THAT house is not the car the cue described.\n"
    "  r4 = r1.filter(west_of=<place>)   keep what lies that way of it; north_of= south_of= east_of= too, and they CHAIN. A row the map cannot place either way is KEPT with a blank column -- unmeasured is never refuted.\n"
    '  Relating rows to a place also stamps WHICH WAY EACH ONE FACES it: "facing <place>" reads a compass word -- north, southeast -- for the direction from that row to that place. A thing faces what it fronts, so a house whose front is on the road reads facing <road>. It rides every near= and nearest= you already write; read it like any column, r["facing Poppy Boulevard"].\n'
    '  r5 = r1.filter(facing=<place>, direction="south")   keep the'
    " south-facing ones -- the cut on that same column. direction= takes"
    " north south east west, and a DIRECTION COMPOSES: a thing facing"
    ' southwest faces south AND west, so direction="south" keeps it. For'
    " the diagonal itself chain the two -- .filter(facing=<place>,"
    ' direction="south").filter(facing=<place>, direction="west").'
    " A row TOUCHING the place has no facing word and is KEPT --"
    " unmeasured is never refuted.\n"
    "  find takes the cut too: find(<thing>, near=<place>, dwithin=25)"
    " KEEPS what is that close and DROPS the rest. near= always CUTS and always needs its gap; nearest= always ORDERS and never drops -- the same split .filter() and .order() use.\n"
    "  distance(a, b) is metres between two FOOTPRINTS -- the same"
    ' quantity the "distance to <ref>" column carries -- and'
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
    '  look_for("pool", "house")   what the segmenter looks for on'
    " EVERY later flight. NOTHING is detected until you say this. Up to"
    " four classes, each naming a THING, never a colour.\n"
    "  ARM EVERY CLASS THE CUE NAMES, in one call, before you fly --"
    " the thing you are looking for AND everything it is described"
    ' relative to. "a red car outside a house" is TWO classes: the car'
    " is what you want and the house is what makes it the right one, and"
    " a house nothing segmented is a row that does not exist to measure"
    " against. The classes cost nothing to add: the sensor runs once per"
    " picture and every class you named rides that same pass. The cost is"
    " all on the other side -- a class you left out is missing from every"
    " flight of the mission, and a relation you cannot measure is one you"
    " will end up guessing.\n"
    '  look_for("pool", at=<place>)   the same, plus extra'
    " segmentation there.\n"
    '  stand(at=<place>, facing="north")   a ground viewpoint; facing'
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
    "  A thing the map does not carry -- a pool, solar panels, a shed, a"
    " parked car -- is not a row until look_for names it and a flight goes"
    " over it. Then it IS one, and it sorts and cuts like any other:\n"
    '    look_for("pool")\n'
    '    houses = find("house", nearest="Glade Avenue")\n'
    "    explore(houses[0])\n"
    '    pools = find("pool")\n'
    '    answer = find("house", nearest=pools)   the house nearest a pool'
    " ranks first\n"
    "  A SET in nearest= / near= / within= is ONE place: the gap to its"
    " NEAREST member, one column, not one column per row. So nearest=pools"
    " ranks every house by its own closest pool, and"
    " houses.filter(within=pools) keeps only the houses that hold one.\n"
    '  The same goes for a cue WORD no column carries -- "small",'
    ' "organized", a shape: it is only in the pixels. look_for the'
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
    ' explore("Willow Avenue") covers EVERY segment of that name -- the'
    " name is the whole street, not its first fragment.\n"
    "  BOTH TAKE A THING ON THE MAP, never a coordinate. To reach a PART"
    " of something -- the west side of a store, the bulb at the end of a"
    " court -- look_for the class you want, explore the thing so the"
    " segmenter puts rows on it, then go_to the row you mean. That is a"
    " measured position; a coordinate you read off a picture is a guess.\n"
    "\nTYPES\n"
    "  a handle holds a SET; r1[0] is a ROW; a quoted name is a PLACE;"
    " [x, y] is a POINT -- points RELATE and LOOK, they do not move you.\n"
    '  a ROW is its served columns, readable as data: r.xy, r["class"],'
    ' r.get("color"), r.keys(); a spaced column goes through'
    ' r["distance to ..."]; print(r) shows them. Two sets add into a'
    " plain list of rows to loop over.\n"
    "  go_to takes ONE THING; explore takes ONE THING or ONE SET -- a set"
    " is one place to cover. Every RELATING slot -- near= nearest= within="
    " east_of= west_of= north_of= south_of= -- takes one place, a point"
    " included, or a set of them, which it relates to as ONE place.\n"
    "  show_map(around=) and stand(at=, facing=) take ONE thing: a"
    " viewpoint stands somewhere and looks at something. Index a set"
    " first.\n"
    '  A fid or a point binds: h = "prior_3123" is a thing you can move'
    " to; corner = [3411, 1468] is a place to look and relate near, and"
    " binding it does not make it movable.\n"
    "\nORDINARY PYTHON RUNS HERE\n"
    "  Variables, arithmetic, f-strings, if / for / while, len(),"
    " sorted(), print(). No underscore names. go_to and"
    " explore cannot go inside a loop, because a loop would move to every"
    " row of the set -- loop to LOOK, then move once to the place you"
    " chose, or explore(the set) to cover it all in one pass.\n"
)

# The system prompt, in the surface's own notation (another notation would
# teach forms the parser refuses).
SYSTEM_PROMPT = """\
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
- Rows display when a result has an order -- a find's match ranking,
  tied or not, or a sort -- and a cut keeps its input's. An UNORDERED
  result (the rows a flight added) displays census + len only. Every
  result stays live behind its handle.
- A NAME is not a PLACE: a name match says the NAME matched, never WHICH
  instance the cue means.
- A SPATIAL relation is what drops rows; a colour ORDERS and never drops.
- look_for(...) takes plain visual classes -- look_for("pool", "house")
  -- no proper nouns, no colour qualifier.
"""


#: The one tool a run serves, ``run``: a block of Python in ``code``.
SERVED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run",
            "description": RUN_TOOL_DOC,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["code"],
                "properties": {"code": {"type": "string"}},
            },
        },
    }
]


def served_system_prompt(sensor_description: str = "") -> str:
    """The one system message a run serves: the system prompt, then the
    rig's sensor block and what the segmenter can see (the chat template
    takes one system turn)."""
    if not sensor_description.strip():
        return SYSTEM_PROMPT
    return (
        SYSTEM_PROMPT
        + "\n\nSensor rig (active for this run):\n"
        + sensor_description.strip()
        + "\n\nA concept this rig cannot see produces an empty SAM response."
        " SAM covers un-cataloged / movable objects; static map features"
        " are already in the graph as prior footprints with geometry and"
        " raw OSM tags."
        " SAM segments roads as merged blobs and over-splits buildings."
    )
