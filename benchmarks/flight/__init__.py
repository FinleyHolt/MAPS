"""The flight harness: one mission flown on an aircraft, held open across operator turns.

``mission`` is the method half of the station that flew the C-12 over
California City: the bench's engine with its ``fly`` wrapped so a flight is a
leg the aircraft flies and an observation the recording holds. ``live_source``
turns the aircraft's recorded frames into that observation. ``vehicle`` is the
boundary to the aircraft, and ``twin`` is one implementation of it that flies
legs kinematically over a supplied ortho.
"""
