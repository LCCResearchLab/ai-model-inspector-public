/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * Session cache for checkpoint measurements. A later visit to the same file
 * shows the saved result instead of repeating the calculation.
 */
import type { CheckpointReport, EffectiveUpdateReport, WeightStatisticsReport } from "./inspectorClient";

export type CachedInspection = {
  report?: CheckpointReport;
  stats?: WeightStatisticsReport;
  update?: EffectiveUpdateReport;
};

const cache = new Map<string, CachedInspection>();

export function cachedInspection(path: string): CachedInspection | null {
  return cache.get(path) ?? null;
}

export function rememberInspection(path: string, patch: CachedInspection): CachedInspection {
  const next = { ...(cache.get(path) ?? {}), ...patch };
  cache.set(path, next);
  return next;
}
