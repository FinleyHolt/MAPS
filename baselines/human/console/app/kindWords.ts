/** The street-map search's kinds of place (studyMap.tsx): everyday words for what the prior
 * names otherwise, a lake is water, a court a sports field. */
export const KIND_WORDS: Record<string, string[]> = {
  lake: ["water"], pond: ["water"], water: ["water"],
  baseball: ["sports"], field: ["sports", "grass"], court: ["sports field"],
  pool: ["pool"], parking: ["parking"], church: ["worship", "church"], school: ["school"],
};

/** Whether a prior feature of ``kind`` answers the search ``q``. */
export const kindMatches = (kind: string, q: string) => (KIND_WORDS[q] ?? [q]).some((w) => kind.toLowerCase().includes(w));
