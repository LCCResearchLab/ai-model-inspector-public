/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * Renders a checkpoint report returned by the Inspector API.
 */
import {
  METADATA_TYPE_LABEL,
  OBSERVED_LABEL,
  adapterAbsentNote,
  adapterApplies,
  fileKindLabel,
  observedLayerFacts,
  type CheckpointReport,
} from "./inspectorClient";

function shown(value: unknown): string {
  if (value == null || value === "") return "unknown";
  if (Array.isArray(value)) return value.length ? value.map(String).join(", ") : "unknown";
  return String(value);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = units[0];
  for (let i = 0; i < units.length; i += 1) {
    unit = units[i];
    if (value < 1024 || i === units.length - 1) break;
    value /= 1024;
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)}\u00a0${unit}`;
}

export function ProjectPath({ label, display }: { label?: string | null; display?: string | null }) {
  if (!display) return <p className="meta">No file path is recorded.</p>;
  return (
    <>
      <p className="meta">{label || "Project-relative path"}</p>
      <p className="path">{display}</p>
    </>
  );
}

export function ReportView({ report }: { report: CheckpointReport }) {
  const arch = report.declared_architecture;
  const adapter = report.declared_adapter;
  const observed = report.observed_checkpoint;
  const layers = observedLayerFacts(observed.layer_indices);
  const applies = adapterApplies(report);
  return (
    <div className="report-grid">
      <section className="card">
        <h2>File</h2>
        <p className="meta">{report.checkpoint.name}</p>
        <p className="meta">{fileKindLabel(report)}</p>
        <p className="meta">{formatBytes(report.checkpoint.size_bytes)}</p>
        <p className="hash" data-testid="checkpoint-sha256">
          SHA-256 {report.checkpoint.sha256}
        </p>
        <details className="technical">
          <summary>Technical details</summary>
          <ProjectPath label={report.checkpoint.path_label} display={report.checkpoint.path_display} />
          <p className="hash">{report.checkpoint.sha256}</p>
          <p className="meta">
            Safetensors metadata classification: {METADATA_TYPE_LABEL[report.checkpoint.checkpoint_type]}
          </p>
          <p className="meta">Metadata basis: {report.checkpoint.checkpoint_type_basis}</p>
        </details>
      </section>
      <section className="card">
        <h2>Observed checkpoint</h2>
        <p className="meta">{OBSERVED_LABEL[observed.classification]}</p>
        <p className="meta">{observed.classification_basis}</p>
        <p className="meta">Distinct layers in this file {layers.distinct}</p>
        <p className="meta">Highest layer index in this file {layers.highest == null ? "none" : layers.highest}</p>
        <p className="meta">Modules {observed.adapter_modules.length ? observed.adapter_modules.join(", ") : "none listed"}</p>
        <p className="meta" data-testid="tensor-count">
          Stored tensors {report.tensor_count}
        </p>
        <p className="meta">Stored parameters {report.total_stored_parameters.toLocaleString()}</p>
        <p className="limits">{report.parameter_scope}</p>
      </section>
      <section className="card">
        <h2>Declared architecture</h2>
        {report.validation.warnings.some((warning) => warning.code === "base_architecture_outside_roots") ? (
          <p className="note">These fields are unknown because the corresponding base-model configuration was not inspected.</p>
        ) : null}
        <p className="meta">Model type {shown(arch.model_type)}</p>
        <p className="meta">Architecture {shown(arch.architecture_name)}</p>
        <p className="meta">Hidden size {shown(arch.hidden_size)}</p>
        <p className="meta">Declared layers {shown(arch.num_hidden_layers)}</p>
        <p className="meta">Attention heads {shown(arch.num_attention_heads)}</p>
        <p className="limits">{arch.limits}</p>
      </section>
      <section className="card">
        <h2>Adapter</h2>
        {applies ? (
          <>
            <p className="meta">PEFT type {shown(adapter.peft_type)}</p>
            <p className="meta">Rank {shown(adapter.r)}</p>
            <p className="meta">Alpha {shown(adapter.lora_alpha)}, copied from adapter_config.json. It is not measured from the tensor values.</p>
            <p className="meta">Target modules {shown(adapter.target_modules)}</p>
            <p className="meta">
              Base model{" "}
              {report.validation.warnings.some((warning) => warning.code === "base_architecture_outside_roots")
                ? "named in adapter_config.json. That file is outside the Inspector directories and was not read."
                : shown(adapter.base_model_name_or_path)}
            </p>
            <p className="limits">{adapter.limits}</p>
          </>
        ) : (
          <p className="meta">{adapterAbsentNote(report)}</p>
        )}
      </section>
      <section className="card span">
        <h2>Tensors</h2>
        <div className="scroll">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Shape</th>
                <th>Dtype</th>
                <th>Parameters</th>
              </tr>
            </thead>
            <tbody>
              {report.tensors.map((tensor) => (
                <tr key={tensor.name}>
                  <td className="path">{tensor.name}</td>
                  <td>{tensor.shape.join(" × ")}</td>
                  <td>{tensor.dtype}</td>
                  <td>{tensor.parameter_count.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="card span">
        <h2>Validation</h2>
        {report.validation.warnings.length === 0 ? <p>No validation warnings.</p> : null}
        {report.validation.warnings.map((warning) => (
          <p key={warning.code} className="meta">
            {warning.code}: {warning.message}
          </p>
        ))}
        <p className="limits">{report.validation.limits}</p>
      </section>
    </div>
  );
}
