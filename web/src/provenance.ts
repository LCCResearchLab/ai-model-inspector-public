/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * A research collection is one dataset inside Books.
 * Adding a later book means adding another collection here. It does not rename the instrument.
 */
export type ResearchCollection = {
  id: string;
  label: string;
  researchContext: string;
  measurement: string;
};

export const RESEARCH_COLLECTIONS: ResearchCollection[] = [
  {
    id: "freshman",
    label: "Freshman Research",
    researchContext: "Book Three — From Interpretation to Measurement",
    measurement: "AI Model Inspector — Weight Magnitudes and Directions",
  },
  {
    id: "qwen",
    label: "Qwen Research",
    researchContext: "No book chapter is recorded for this adapter.",
    measurement: "AI Model Inspector — Checkpoint inspection, stored weights, and effective-update magnitude",
  },
];

export function collectionById(id: string): ResearchCollection | undefined {
  return RESEARCH_COLLECTIONS.find((collection) => collection.id === id);
}

/** The study name before an em dash, with the comma written as a separator. */
export function studyLine(experiment: string): string {
  const mark = " — ";
  const splitAt = experiment.indexOf(mark);
  const study = splitAt < 0 ? experiment : experiment.slice(0, splitAt);
  return study.replace(/, /g, " · ");
}

/** The stored experiment string stays intact. The title after the dash can be emphasized. */
export function experimentTitle(experiment: string): string | null {
  const mark = " — ";
  const splitAt = experiment.indexOf(mark);
  if (splitAt < 0) return null;
  return experiment.slice(splitAt + mark.length);
}
