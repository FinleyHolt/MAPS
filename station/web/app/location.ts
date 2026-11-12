/** Where one of the station's maps draws from: `GET location?map=`.
 *
 * The location is not compiled into the bundle: the station holding the
 * mission also holds the imagery.
 */

export type DisplayLocation = {
  name: string;
  label: string;
  /** west, south, east, north, in degrees. */
  bounds: [number, number, number, number];
  center: { lat: number; lon: number };
  /** A tile URL template, same-origin, e.g. "tiles/{z}/{x}/{y}.png". */
  tiles: string;
  minzoom: number;
  maxzoom: number;
  attribution: string;
};

/** A map's location, or null when the station serves no imagery of it.
 *
 * Null is a normal answer, not a failure: the map still shows the mission
 * over an empty paper background.
 */
export async function fetchLocation(map: string | null): Promise<DisplayLocation | null> {
  try {
    const response = await fetch(map ? `location?map=${encodeURIComponent(map)}` : "location", { cache: "no-store" });
    if (!response.ok) return null;
    const value = (await response.json()) as DisplayLocation;
    return Array.isArray(value?.bounds) && value.bounds.length === 4 ? value : null;
  } catch {
    return null;
  }
}
