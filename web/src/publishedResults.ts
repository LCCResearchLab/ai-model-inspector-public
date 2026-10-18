/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * Published Books results. These values are copied from the research record.
 * This file does not calculate a measurement.
 */

export const RECORDED_RESULTS_LABEL = "Recorded Research Results";

export const CHECKPOINT_MISSING =
  "The checkpoint file is not on this computer. The recorded research results remain available. Live inspection needs the original checkpoint.";

export const BASE_RECORDED_NOTE =
  "No LoRA configuration, effective-update magnitude, or direction comparison is recorded for the starting model.";

export const F6_DIRECTION_NOTE =
  "No direction comparison is recorded for F6. The comparisons below are the recorded Chapter 5 results among F5-1, F5-2, and F5-3.";

export const QWEN_RECORDED_NOTE = "No recorded research results are included for this adapter.";

export const BOOK_PAGES = [
  {
    id: "one",
    title: "Book One",
    hasRecordedMeasurements: false,
    note: "No measurement record for Book One is included in this distribution.",
  },
  {
    id: "two",
    title: "Book Two",
    hasRecordedMeasurements: false,
    note: "No separate measurement record for Book Two is included. Chapter 5 and Chapter 6 name the lessons. Those lessons are measured in the Book Three record.",
  },
  {
    id: "three",
    title: "Book Three — From Interpretation to Measurement",
    hasRecordedMeasurements: true,
    note: "Freshman Research is the published record for this book.",
  },
] as const;

export type PublishedSubject = {
  id: string;
  weight_sha256: string;
  rank: number;
  lora_alpha: number;
  scale: number;
  pair_count: number;
  effective_update_norm: number;
  relative_to_base: number;
};

export type PublishedCosine = {
  left: string;
  right: string;
  cosine: number;
  mean_pair_cosine: number;
};

export type PublishedResearch = {
  label: string;
  record: string;
  measured_on: string;
  subjects: PublishedSubject[];
  chapter_five_cosines: PublishedCosine[];
};

export type RecordedLine = {
  label: string;
  value: string;
  testId: string;
};

export function formatRecorded(value: number): string {
  return JSON.stringify(value);
}

export function recordedMeasurementLines(subject: PublishedSubject): RecordedLine[] {
  return [
    { label: "LoRA rank", value: formatRecorded(subject.rank), testId: "rank" },
    { label: "Alpha", value: formatRecorded(subject.lora_alpha), testId: "alpha" },
    { label: "Scaling", value: formatRecorded(subject.scale), testId: "scaling" },
    { label: "LoRA A/B pairs", value: formatRecorded(subject.pair_count), testId: "pairs" },
    { label: "Effective-update magnitude (L2)", value: formatRecorded(subject.effective_update_norm), testId: "magnitude" },
    { label: "Relative to matching BASE projections", value: formatRecorded(subject.relative_to_base), testId: "relative" },
    { label: "Weight SHA-256", value: subject.weight_sha256, testId: "sha256" },
  ];
}

export function recordedPairLabel(row: PublishedCosine): string {
  return `${row.left} compared with ${row.right}`;
}

export function recordedDirectionView(subjectId: string, cosines: PublishedCosine[]): { rows: PublishedCosine[]; note: string | null } {
  if (subjectId === "BASE") return { rows: [], note: null };
  if (subjectId === "F6") return { rows: cosines, note: F6_DIRECTION_NOTE };
  return { rows: cosines, note: null };
}

export function booksWithoutMeasurements(): typeof BOOK_PAGES[number][] {
  return BOOK_PAGES.filter((book) => !book.hasRecordedMeasurements);
}
