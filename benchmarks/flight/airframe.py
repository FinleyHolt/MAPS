"""The C-12's geometry as the flight harness reads it."""

#: Antenna-to-camera arm ``(fwd, right, down)`` in body metres, from the
#: airframe stations: GPS antenna at WL 124.4, camera at WL 77.26. The
#: horizontal terms are zero; that offset lands in the boresight solve's GNSS bias.
C12_LEVER_M = (0.0, 0.0, (124.4 - 77.26) * 0.0254)

#: The C-12's camera, whose horizontal field of view is the width OSR reads.
C12_SENSOR = "oryx_orx32_25mm_nadir"

#: 160 knots, the speed the twin flies.
C12_SPEED_MPS = 160.0 * 0.514444
