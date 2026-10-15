"""ViSA's two model calls, as prose and strict JSON schemas.

The paper gives no prompt text, only the roles: the Visual Prompt Generator
and the three-stage Verification Module (literal attribute matching, spatial
topology verification by mark id, geographic boundary validation against the
landmark prior K_prior), with a natural-language guidance signal back to
perception when the scene is inconclusive: the paper's example rejects a
candidate, "sends a closed-loop guidance signal back to the VPG ... and
triggers a move primitive", and the next waypoint's perception "adjusts its
attention based on the feedback". The decision is one of the paper's task
primitives (stop, move, ascend, descend) with the optional signal beside it.
"""

from __future__ import annotations

from baselines.seam import STR, STRS, obj

SYSTEM = (
    "You are the reasoning module of a UAV searching a city for one described "
    "target from a nadir (straight-down, north-up) aerial view. Think in short "
    "drafts, then answer with the JSON the schema asks for and nothing else."
)

ANALYSIS_PROMPT = """Instruction: "{instruction}"

Analyse this navigation instruction for a nadir aerial search.
- target: the one object to fly to (its phrase, its object class as one or two
  words, its visible attributes such as colour, size, roof, orientation).
- landmarks: every named place the instruction refers to (streets, buildings,
  parks, businesses), verbatim.
- surroundings: the other object classes the instruction mentions around the
  target (cars, trees, buildings, roads, lots).
- concepts: the short noun phrases an open-vocabulary detector should segment
  in a nadir view to find candidates, each as the instruction phrases it with
  its attributes: the attributed target phrase, each surrounding object as
  phrased, and the landmark nouns (e.g. "red car", "white car", "parking lot",
  "tram depot"). Never a bare class on its own. Two to twelve phrases."""

ANALYSIS_SCHEMA = obj({
    "target": obj({"phrase": STR, "class": STR, "attributes": STRS}),
    "landmarks": STRS, "surroundings": STRS, "concepts": STRS,
})

VERIFY_PROMPT = """Instruction: "{instruction}"
Target sought: {target_phrase} (class: {target_class}; attributes: {target_attributes}).

The image is the current nadir view, north up, {view_side_m:.0f} m on a side,
altitude {altitude_m:.0f} m. Candidate objects carry numbered marks; the legend:
{legend}

Landmark prior (K_prior), relative to the centre of this view:
{landmarks}
{guidance_block}
Perform the three-stage verification, in order, referring to candidates ONLY
by their mark numbers:
1. Literal attribute matching: which marks match the target's visible
   attributes (class, colour, size, roof, orientation)? List them as
   candidate_marks. Judge only what is directly visible; a mark whose
   attributes the view cannot confirm is pending, not rejected: list it in
   pending_marks.
2. Spatial topology verification: for the candidate marks, do their positions
   relative to the OTHER marks match the instruction's spatial words (behind,
   in front of, next to, between, across from, in the middle row ...)? In this
   north-up view "behind" and "in front of" follow the described object's
   facing when it is visible, otherwise north/south. List the marks whose
   topology is consistent.
3. Geographic boundary validation: do the consistent marks lie where the
   landmark prior says the target should be (on the named street, inside the
   named lot, beside the named building)? List the marks that pass.

Then decide one task primitive:
- "stop": exactly one mark passes all three stages; set target_mark to it.
- "move": no mark is confirmed here (rejected, or pending on evidence this
  view cannot give); the UAV continues to the next waypoint. When the scene
  told you something about the target, write guidance for the next
  perception round in the form "Search for a red car positioned behind [3],
  not in front", and list in guidance_concepts the short noun phrases the
  detector should look for there; otherwise leave both empty.
- "descend": the view is too coarse to read the attributes; "ascend": the
  named landmarks are outside this view and a wider one would place them.
  Guidance may accompany either.
target_mark is 0 unless the verdict is "stop"."""

_MARKS = {"type": "array", "items": {"type": "integer"}}
VERIFY_SCHEMA = obj({
    "attribute_match": obj({"candidate_marks": _MARKS, "pending_marks": _MARKS, "reasoning": STR}),
    "spatial_topology": obj({"consistent_marks": _MARKS, "reasoning": STR}),
    "geographic_boundary": obj({"valid_marks": _MARKS, "reasoning": STR}),
    "verdict": {"type": "string", "enum": ["stop", "move", "ascend", "descend"]},
    "target_mark": {"type": "integer"},
    "guidance": STR,
    "guidance_concepts": STRS,
})


def guidance_block(guidance: str | None) -> str:
    if not guidance:
        return ""
    return f"\nGuidance from the previous verification round: \"{guidance}\"\n"
