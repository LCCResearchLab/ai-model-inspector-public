/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * Shared text-size preference for the local console.
 * The stored value changes type size only. It does not change a measurement.
 */
export const TEXT_ZOOM_KEY = "aimi-text-zoom";
export const TEXT_ZOOM_MIN = 0.8;
export const TEXT_ZOOM_MAX = 1.5;
export const TEXT_ZOOM_STEP = 0.1;
export const TEXT_ZOOM_DEFAULT = 1;

export type TextZoomStorage = {
  getItem(key: string): string | null;
};

export function clampTextZoom(value: number): number {
  if (!Number.isFinite(value)) return TEXT_ZOOM_DEFAULT;
  const percent = Math.round(value * 100);
  const clamped = Math.min(TEXT_ZOOM_MAX * 100, Math.max(TEXT_ZOOM_MIN * 100, percent));
  return (Math.round(clamped / 10) * 10) / 100;
}

export function readTextZoom(storage: TextZoomStorage | null): number {
  if (!storage) return TEXT_ZOOM_DEFAULT;
  const raw = storage.getItem(TEXT_ZOOM_KEY);
  if (raw == null || raw.trim() === "") return TEXT_ZOOM_DEFAULT;
  return clampTextZoom(Number(raw));
}

export function stepTextZoom(current: number, direction: -1 | 1): number {
  return clampTextZoom(clampTextZoom(current) + direction * TEXT_ZOOM_STEP);
}
