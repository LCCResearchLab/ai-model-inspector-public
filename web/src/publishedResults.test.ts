/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * Recorded Books results stay readable when the checkpoint files are absent.
 * These tests do not calculate a measurement.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  BOOK_PAGES,
  QWEN_RECORDED_NOTE,
  booksWithoutMeasurements,
  formatRecorded,
  recordedDirectionView,
  recordedMeasurementLines,
  recordedPairLabel,
  type PublishedResearch,
} from "./publishedResults.ts";

const recordPath = new URL("../../inspector-engine/research/freshman_validation.json", import.meta.url);
const recordText = readFileSync(recordPath, "utf8");
const record = JSON.parse(recordText) as {
  record: string;
  measured_on: string;
  subjects: PublishedResearch["subjects"];
  chapter_five_cosines: PublishedResearch["chapter_five_cosines"];
};

const published: PublishedResearch = {
  label: "Recorded Research Results",
  record: record.record,
  measured_on: record.measured_on,
  subjects: record.subjects,
  chapter_five_cosines: record.chapter_five_cosines,
};

test("F5-1 recorded results keep the published digits without a checkpoint", () => {
  const subject = published.subjects.find((item) => item.id === "F5-1");
  assert.ok(subject);
  const lines = recordedMeasurementLines(subject);
  const value = (testId: string) => lines.find((line) => line.testId === testId)?.value;
  assert.equal(value("rank"), "16");
  assert.equal(value("alpha"), "32");
  assert.equal(value("scaling"), "2");
  assert.equal(value("pairs"), "96");
  assert.equal(value("magnitude"), "1.155947");
  assert.equal(value("relative"), "0.000497");
  assert.equal(value("sha256"), "001ff8f0f87e37cb7e5302d34ddd30e89972908e979d20f57f28f66b6e5308c2");
  for (const line of lines) assert.ok(recordText.includes(line.value));
  const directions = recordedDirectionView("F5-1", published.chapter_five_cosines);
  assert.equal(directions.note, null);
  assert.deepEqual(
    directions.rows.map((row) => [recordedPairLabel(row), formatRecorded(row.cosine), formatRecorded(row.mean_pair_cosine)]),
    [
      ["F5-1 compared with F5-2", "0.00820179", "0.00817211"],
      ["F5-1 compared with F5-3", "0.00877463", "0.00875743"],
      ["F5-2 compared with F5-3", "0.00931618", "0.0093219"],
    ],
  );
});

test("every recorded field matches the validation file", () => {
  for (const subject of published.subjects) {
    const lines = recordedMeasurementLines(subject);
    const byId = Object.fromEntries(lines.map((line) => [line.testId, line.value]));
    assert.equal(byId.rank, JSON.stringify(subject.rank));
    assert.equal(byId.alpha, JSON.stringify(subject.lora_alpha));
    assert.equal(byId.scaling, JSON.stringify(subject.scale));
    assert.equal(byId.pairs, JSON.stringify(subject.pair_count));
    assert.equal(byId.magnitude, JSON.stringify(subject.effective_update_norm));
    assert.equal(byId.relative, JSON.stringify(subject.relative_to_base));
    assert.equal(byId.sha256, subject.weight_sha256);
    for (const line of lines) assert.ok(recordText.includes(line.value), line.value);
  }
  assert.deepEqual(
    published.subjects.map((subject) => [subject.id, JSON.stringify(subject.relative_to_base)]),
    [
      ["F5-1", "0.000497"],
      ["F5-2", "0.000498"],
      ["F5-3", "0.000496"],
      ["F6", "0.004285"],
    ],
  );
});

test("every recorded freshman subject stays in the published record", () => {
  assert.deepEqual(published.subjects.map((item) => item.id), ["F5-1", "F5-2", "F5-3", "F6"]);
  const magnitudes = published.subjects.map((item) => recordedMeasurementLines(item).find((line) => line.testId === "magnitude")?.value);
  assert.deepEqual(magnitudes, ["1.155947", "1.159056", "1.154512", "9.970915"]);
  const f6 = recordedDirectionView("F6", published.chapter_five_cosines);
  assert.match(f6.note || "", /No direction comparison is recorded for F6/);
  assert.equal(f6.rows.length, 3);
  assert.equal(recordedDirectionView("BASE", published.chapter_five_cosines).rows.length, 0);
});

test("books one and two do not invent measurements", () => {
  const omitted = booksWithoutMeasurements();
  assert.deepEqual(omitted.map((book) => book.id), ["one", "two"]);
  for (const book of omitted) {
    assert.equal(book.hasRecordedMeasurements, false);
    assert.doesNotMatch(book.note, /1\.155947|0\.00820179|rank 16|alpha 32/i);
  }
  assert.equal(BOOK_PAGES[2].hasRecordedMeasurements, true);
  assert.doesNotMatch(QWEN_RECORDED_NOTE, /1\.155947|rank 16/);
});

test("the books screen keeps recorded results outside the checkpoint check", () => {
  const book = readFileSync(new URL("./book.tsx", import.meta.url), "utf8");
  const publishedSource = readFileSync(new URL("./publishedResults.ts", import.meta.url), "utf8");
  assert.match(publishedSource, /Recorded Research Results/);
  assert.match(book, /RECORDED_RESULTS_LABEL/);
  assert.match(book, /Inspect Local Checkpoint/);
  assert.match(book, /CHECKPOINT_MISSING/);
  const workspace = book.slice(book.indexOf("function FreshmanWorkspace"));
  assert.match(workspace, /<RecordedResearchResults subject=\{subject\} published=\{published\} \/>/);
  const recordedAt = workspace.indexOf("<RecordedResearchResults");
  const checkpointAt = workspace.indexOf('tab === "checkpoint"');
  assert.ok(recordedAt >= 0 && checkpointAt > recordedAt);
  assert.match(book, /Open →/);
  assert.doesNotMatch(book, /disabled=\{!subject\.inspectable\}/);
  const recordedBody = book.slice(book.indexOf("function RecordedSubjectBody"), book.indexOf("function RecordedDirectionBlock"));
  assert.match(recordedBody, /MATCHING_BASE_PROJECTIONS_HELP/);
  assert.match(book, /recordedPairLabel/);
  assert.match(book, /Freshman pair/);
  assert.match(book, /Each row compares the recorded effective-update direction/);
});
