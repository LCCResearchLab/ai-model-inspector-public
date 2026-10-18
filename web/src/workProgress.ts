/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * Progress counts finished operations. A missing count stays indeterminate.
 * Zero finished operations is 0%, which is a count, not an estimated time.
 */

export function workPercent(completed: number | undefined, total: number | undefined): number | null {
  if (completed == null || total == null || !(total > 0) || !Number.isFinite(completed)) return null;
  return Math.max(0, Math.min(100, Math.round((completed / total) * 100)));
}

/** "k of n" is the operation now running. The caller’s percentage is the number already finished. */
export function weightStepLabel(completed: number, subjectId: string, updateOrder: string[]): string {
  const total = 1 + updateOrder.length;
  const running = Math.min(completed + 1, total);
  if (completed <= 0) return `${running} of ${total}. Reading stored weights for ${subjectId}.`;
  const updateIndex = Math.min(completed, updateOrder.length) - 1;
  return `${running} of ${total}. Measuring the effective update for ${updateOrder[updateIndex]}.`;
}

export function formatByteCount(value: number): string {
  if (!Number.isFinite(value) || value < 0) return "unknown";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  const shown = unit === 0 ? String(Math.round(size)) : String(Number(size.toPrecision(3)));
  return `${shown} ${units[unit]}`;
}

/** Byte counts from the server override any summary percentage. Without a total, the bar stays indeterminate. */
export function downloadProgress(job: {
  message: string;
  bytes_got: number;
  bytes_total: number;
  percent: number;
}): { label: string; completed?: number; total?: number } {
  const message = job.message || "Downloading the selected files.";
  if (!(job.bytes_total > 0) || !Number.isFinite(job.bytes_got)) return { label: message };
  return {
    label: `${message} ${formatByteCount(job.bytes_got)} of ${formatByteCount(job.bytes_total)}.`,
    completed: job.bytes_got,
    total: job.bytes_total,
  };
}
