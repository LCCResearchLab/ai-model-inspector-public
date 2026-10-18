/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * Visible progress while a local read or measurement is still running.
 * Pass a completed count to show that fraction. Omit it when the engine has no count.
 */
import { workPercent } from "./workProgress";

export function WorkProgress({
  label,
  completed,
  total,
}: {
  label: string;
  completed?: number;
  total?: number;
}) {
  const percent = workPercent(completed, total);
  return (
    <div className="work">
      <p>
        {label}
        {percent == null ? "" : ` ${percent}%`}
      </p>
      <div
        className={percent == null ? "bar indeterminate" : "bar"}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={percent == null ? undefined : 100}
        aria-valuenow={percent ?? undefined}
        aria-label={label}
      >
        <span style={percent == null ? undefined : { width: `${percent}%` }} />
      </div>
    </div>
  );
}
