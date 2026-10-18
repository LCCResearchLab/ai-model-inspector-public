/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * The percentage is finished work. A single request with no count stays indeterminate.
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import { downloadProgress, weightStepLabel, workPercent } from "./workProgress.ts";

const order = ["F5-1", "F5-2", "F5-3", "F6"];

test("zero finished weight steps is 0 percent, and each finished step adds 20", () => {
  assert.equal(workPercent(0, 5), 0);
  assert.equal(workPercent(1, 5), 20);
  assert.equal(workPercent(2, 5), 40);
  assert.equal(workPercent(3, 5), 60);
  assert.equal(workPercent(4, 5), 80);
  assert.equal(workPercent(5, 5), 100);
  assert.equal(workPercent(undefined, 5), null);
  assert.equal(workPercent(1, undefined), null);
});

test("1 of 5 names the stored-weight step while it is running", () => {
  assert.equal(weightStepLabel(0, "F5-1", order), "1 of 5. Reading stored weights for F5-1.");
  assert.equal(weightStepLabel(1, "F5-1", order), "2 of 5. Measuring the effective update for F5-1.");
  assert.equal(weightStepLabel(2, "F5-1", order), "3 of 5. Measuring the effective update for F5-2.");
  assert.equal(weightStepLabel(3, "F5-1", order), "4 of 5. Measuring the effective update for F5-3.");
  assert.equal(weightStepLabel(4, "F5-1", order), "5 of 5. Measuring the effective update for F6.");
  assert.equal(workPercent(0, 5), 0);
  assert.equal(workPercent(1, 5), 20);
  assert.equal(workPercent(4, 5), 80);
});

test("a download uses byte counts when the server provides a total", () => {
  const counted = downloadProgress({ message: "Saving the adapter.", bytes_got: 50, bytes_total: 200, percent: 99 });
  assert.equal(counted.completed, 50);
  assert.equal(counted.total, 200);
  assert.equal(workPercent(counted.completed, counted.total), 25);
  assert.match(counted.label, /50 B of 200 B/);
  const unknown = downloadProgress({ message: "Saving the adapter.", bytes_got: 10, bytes_total: 0, percent: 100 });
  assert.equal(unknown.completed, undefined);
  assert.equal(unknown.total, undefined);
  assert.equal(workPercent(unknown.completed, unknown.total), null);
});
