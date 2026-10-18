/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * Shared checkpoint panel for Books and for a file selected in Model Library.
 * The numbers come from the Inspector API. This file does not calculate them.
 */
import { useEffect, useState, type ReactNode } from "react";
import { cachedInspection, rememberInspection } from "./inspectionCache";
import {
  effectiveUpdateAvailability,
  fetchEffectiveUpdate,
  fetchWeightStatistics,
  fileKindLabel,
  formatMeasure,
  inspectCheckpoint,
  type CheckpointReport,
  type EffectiveUpdateReport,
  type WeightStatisticsReport,
} from "./inspectorClient";
import { WorkProgress } from "./progress";
import { ReportView } from "./report";

export function CheckpointInspector({
  apiBase,
  path,
  title,
  provenance,
  measurements = true,
  onMeasured,
  onReport,
}: {
  apiBase: string;
  path: string;
  title: string;
  provenance?: ReactNode;
  measurements?: boolean;
  onMeasured?: (result: { stats: WeightStatisticsReport | null; update: EffectiveUpdateReport | null }) => void;
  onReport?: (report: CheckpointReport) => void;
}) {
  const [report, setReport] = useState<CheckpointReport | null>(null);
  const [stats, setStats] = useState<WeightStatisticsReport | null>(null);
  const [update, setUpdate] = useState<EffectiveUpdateReport | null>(null);
  const [error, setError] = useState("");
  const [work, setWork] = useState("");

  useEffect(() => {
    const saved = cachedInspection(path);
    setReport(saved?.report ?? null);
    setStats(saved?.stats ?? null);
    setUpdate(saved?.update ?? null);
    setError("");
    if (saved?.report) onReport?.(saved.report);
    onMeasured?.({ stats: saved?.stats ?? null, update: saved?.update ?? null });
    if (saved?.report || !path) return;
    let cancelled = false;
    setWork("Reading the checkpoint on this computer.");
    void inspectCheckpoint(apiBase, path)
      .then((result) => {
        if (cancelled) return;
        if (result.ok) {
          setReport(result);
          rememberInspection(path, { report: result });
          onReport?.(result);
          setError("");
        } else setError(result.message);
      })
      .catch((caught) => {
        if (!cancelled) setError(caught instanceof Error ? caught.message : "The checkpoint inspection did not finish.");
      })
      .finally(() => {
        if (!cancelled) setWork("");
      });
    return () => {
      cancelled = true;
    };
  }, [apiBase, path]);

  const availability = report ? effectiveUpdateAvailability(report) : null;
  const safetensors = path.toLowerCase().endsWith(".safetensors");

  async function measureWeights() {
    setWork("Reading stored weights.");
    setError("");
    try {
      const result = await fetchWeightStatistics(apiBase, path);
      if (result.ok) {
        setStats(result);
        const saved = rememberInspection(path, { stats: result });
        onMeasured?.({ stats: result, update: saved.update ?? update });
      } else {
        setStats(null);
        setError(result.message);
      }
    } catch (caught) {
      setStats(null);
      setError(caught instanceof Error ? caught.message : "The stored-weight measurement did not finish.");
    } finally {
      setWork("");
    }
  }

  async function measureUpdate() {
    setWork("Measuring the effective update.");
    setError("");
    try {
      const result = await fetchEffectiveUpdate(apiBase, path);
      if (result.ok) {
        setUpdate(result);
        const saved = rememberInspection(path, { update: result });
        onMeasured?.({ stats: saved.stats ?? stats, update: result });
      } else {
        setUpdate(null);
        setError(result.message);
      }
    } catch (caught) {
      setUpdate(null);
      setError(caught instanceof Error ? caught.message : "The effective-update measurement did not finish.");
    } finally {
      setWork("");
    }
  }

  return (
    <section className="checkpoint-inspector">
      <h2>Checkpoint Inspector</h2>
      <p className="meta">{title}</p>
      {provenance}
      {report ? (
        <>
          <p className="meta">Checkpoint type {fileKindLabel(report)}</p>
          <p className="meta">Tensors {report.tensor_count.toLocaleString()}</p>
          <p className="hash">SHA-256 {report.checkpoint.sha256}</p>
        </>
      ) : null}
      {measurements ? (
        <>
          <div className="row">
            <button type="button" className="quiet" disabled={work !== "" || !safetensors} onClick={() => void measureWeights()}>
              {stats ? "Measure stored weights again" : "Stored weights"}
            </button>
            <button type="button" className="quiet" disabled={work !== "" || !availability?.enabled} onClick={() => void measureUpdate()}>
              {update ? "Measure effective update again" : "Effective update"}
            </button>
          </div>
          <p className="note">{safetensors ? "Stored weights can be read from this Safetensors file." : "Stored weights reads a Safetensors file."}</p>
          <p className="note">
            {availability
              ? availability.reason
              : "Effective update stays off until the file is identified as a LoRA adapter with its configuration."}
          </p>
        </>
      ) : (
        <p className="note">Stored-weight statistics and an effective LoRA update apply to a LoRA adapter. This file is the starting checkpoint those adapters are compared with.</p>
      )}
      {work ? <WorkProgress label={work} /> : null}
      {error ? <p className="error">{error}</p> : null}
      {stats ? (
        <section className="card">
          <h2>Stored weights</h2>
          {stats.checkpoint ? <p className="meta">Fingerprint {stats.checkpoint.sha256}</p> : null}
          <p className="limits">{stats.limits}</p>
          <table>
            <thead>
              <tr>
                <th>Group</th>
                <th>Tensors</th>
                <th>L2</th>
              </tr>
            </thead>
            <tbody>
              {stats.groups.map((group) => (
                <tr key={group.role}>
                  <td>{group.role}</td>
                  <td>{group.tensor_count}</td>
                  <td>{formatMeasure(group.l2_norm)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : null}
      {update ? (
        <section className="card">
          <h2>Effective update</h2>
          <p className="limits">{update.limits}</p>
          <p className="meta" data-testid="pair-count">LoRA A/B pairs {update.pair_count}</p>
          <p className="meta" data-testid="rank">
            Rank {update.configuration_agreement.r}. {update.sources?.r}
          </p>
          <p className="meta" data-testid="alpha">
            Alpha {update.configuration_agreement.lora_alpha}. {update.sources?.lora_alpha}
          </p>
          <p className="meta" data-testid="scale">
            Scale {formatMeasure(update.configuration_agreement.scale)}. {update.sources?.scale}
          </p>
          <p className="meta" data-testid="effective-norm">Effective update norm {formatMeasure(update.total_frobenius_norm)}</p>
          {update.base_checkpoint?.path_display ? (
            <p className="meta" data-testid="base-checkpoint">Base checkpoint {update.base_checkpoint.path_display}</p>
          ) : null}
          {update.relative_measured && update.total_relative_frobenius != null ? (
            <p className="meta" data-testid="relative-magnitude">
              Relative magnitude {formatMeasure(update.total_relative_frobenius)}. {update.relative_note}
            </p>
          ) : (
            <p className="meta" data-testid="relative-magnitude">{update.relative_note || "No relative magnitude was calculated."}</p>
          )}
        </section>
      ) : null}
      {report ? <ReportView report={report} /> : null}
    </section>
  );
}
