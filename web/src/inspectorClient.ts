/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * Client for the local Inspector API.
 * The web console calls these functions. Scientific measurements stay in the Python engine.
 */
export const INSPECTOR_PORT = "8810";
export const DEFAULT_API_BASE = `http://127.0.0.1:${INSPECTOR_PORT}`;

/** The packaged page uses the port it was opened on. The Vite dev server stays on 5173 and calls 8810. */
export function packagedApiBase(): string {
  if (typeof window === "undefined") return DEFAULT_API_BASE;
  const host = window.location.hostname;
  const local = host === "127.0.0.1" || host === "localhost";
  if (local && window.location.port && window.location.port !== "5173") return window.location.origin;
  return DEFAULT_API_BASE;
}
export const CONTROLLER_PORT = 8811;
export const CONTROLLER_BASE = `http://127.0.0.1:${CONTROLLER_PORT}`;

export type ControllerProcessState = "running" | "stopped" | "foreign" | "unresponsive" | "error" | "unknown";

export type InspectorHealth = {
  ok?: boolean;
  service?: string;
  version?: string;
  port?: number;
};

function trimBase(url: string): string {
  return (url || "").trim().replace(/\/+$/, "");
}

function isLoopbackHost(host: string): boolean {
  const h = (host || "").replace(/^\[|\]$/g, "").trim().toLowerCase();
  return h === "localhost" || h === "127.0.0.1" || h === "10.0.2.2" || h === "::1";
}

export function parseApiAddress(apiBase: string): { host: string; port: number } | null {
  const base = trimBase(apiBase);
  if (!base) return null;
  try {
    const url = new URL(base.includes("://") ? base : `http://${base}`);
    const port = url.port ? Number(url.port) : url.protocol === "https:" ? 443 : 80;
    if (!url.hostname || !Number.isInteger(port) || port < 1 || port > 65535) return null;
    return { host: url.hostname.replace(/^\[|\]$/g, ""), port };
  } catch {
    return null;
  }
}

export function isLocalApiHost(host: string): boolean {
  return isLoopbackHost(host);
}

export function subjectLine(subject: { seed: number | null; base_model: string; experiment: string }): string {
  if (subject.seed == null) return `Original ${subject.base_model}`;
  const chapter = subject.experiment.includes("Chapter 6") ? "Chapter 6" : "Chapter 5";
  return `${chapter}, seed ${subject.seed}`;
}

/** Display rounding only. The measured value stays the one returned by the API. */
export function formatMeasure(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "unknown";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 0.001 || abs >= 100000)) return value.toExponential(3);
  return String(Number(value.toPrecision(4)));
}

export const MATCHING_BASE_PROJECTIONS_LABEL = "Relative to matching BASE projections";

export const MATCHING_BASE_PROJECTIONS_HELP =
  "The denominator is the combined Frobenius norm of the 96 BASE attention projection matrices corresponding to the LoRA updates: Q, K, V, and O across 24 transformer layers. It is not the norm of the entire BASE model.";

export const SHARED_LAYER_SCALE_TITLE = "Layer comparison — shared scale";

export const UPDATE_DIRECTION_TITLE = "Direction of the effective update";

export const UPDATE_DIRECTION_HELP =
  "Cosine similarity compares the direction of the effective update, scale times B times A. The overall cosine treats every corresponding matrix entry as one coordinate, so a larger update contributes more. The mean pair cosine gives each layer and projection an equal vote. A value of 1 is the same direction, 0 means the updates are orthogonal, and -1 is the opposite direction. The comparison does not use the size of the update by itself. It is not a measurement of acquired knowledge.";

/** Largest per-layer effective-update norm across every subject in the comparison. */
export function sharedLayerMaximum(groups: number[][]): number {
  let maximum = 0;
  for (const group of groups) {
    for (const value of group) {
      if (Number.isFinite(value) && value > maximum) maximum = value;
    }
  }
  return maximum;
}

/** Bar width for one value against one maximum. A subject-only maximum is a different call. */
export function barPercent(value: number, maximum: number): number {
  if (!(maximum > 0) || !Number.isFinite(value) || value <= 0) return 0;
  return (value / maximum) * 100;
}

export type TensorRecord = {
  name: string;
  shape: number[];
  dtype: string;
  safetensors_dtype: string;
  parameter_count: number;
};

export type DeclaredArchitecture = {
  available: boolean;
  source: string;
  path: string | null;
  origin: string;
  model_type: string | number | null;
  architecture_name: string | string[] | null;
  num_hidden_layers: number | null;
  hidden_size: number | null;
  num_attention_heads: number | null;
  vocab_size: number | null;
  unknown_fields: string[];
  other_properties: Record<string, string | number | boolean | Array<string | number>>;
  limits: string;
};

export type DeclaredAdapter = {
  available: boolean;
  source: string;
  path: string | null;
  peft_type: string | null;
  fine_tune_type: string | null;
  base_model_name_or_path: string | null;
  base_model_path: string | null;
  base_model_path_key: string | null;
  r: number | null;
  r_key: string | null;
  lora_alpha: number | null;
  lora_dropout: number | null;
  target_modules: string | string[] | null;
  adapter_num_layers: number | null;
  task_type: string | null;
  bias: string | null;
  unknown_fields: string[];
  other_properties: Record<string, string | number | boolean | Array<string | number>>;
  limits: string;
};

export type ObservedCheckpoint = {
  classification: "full_model" | "adapter" | "unknown_subset" | "unknown";
  classification_basis: string;
  layer_indices: number[];
  adapter_modules: string[];
};

export type ValidationWarning = {
  code: string;
  message: string;
};

export type CheckpointReport = {
  ok: true;
  engine_version: string;
  inspected_at: string;
  checkpoint: {
    name: string;
    path: string;
    path_label?: string | null;
    path_display?: string | null;
    path_scope?: "project" | "external" | "unavailable";
    format: string;
    size_bytes: number;
    sha256: string;
    checkpoint_type: "base" | "lora_adapter" | "unknown";
    checkpoint_type_basis: string;
  };
  configuration: {
    source: "safetensors_metadata" | "unavailable";
    metadata: Record<string, string> | null;
    limits: string;
  };
  declared_architecture: DeclaredArchitecture;
  declared_adapter: DeclaredAdapter;
  referenced_base_architecture: DeclaredArchitecture | null;
  observed_checkpoint: ObservedCheckpoint;
  validation: {
    warnings: ValidationWarning[];
    limits: string;
  };
  tensors: TensorRecord[];
  tensor_count: number;
  total_stored_parameters: number;
  parameter_scope: string;
  research?: ResearchFingerprint;
};

export type CheckpointError = {
  ok: false;
  error: string;
  message: string;
};

export const METADATA_TYPE_LABEL = {
  base: "Base checkpoint",
  lora_adapter: "LoRA adapter",
  unknown: "Unknown",
} as const;

/** Count of distinct layer indices in one file, and the highest index among them. */
export function observedLayerFacts(indices: readonly number[]): { distinct: number; highest: number | null } {
  if (indices.length === 0) return { distinct: 0, highest: null };
  return { distinct: indices.length, highest: Math.max(...indices) };
}

export const OBSERVED_LABEL = {
  full_model: "Full-model weights",
  adapter: "Adapter weights",
  unknown_subset: "Unknown subset",
  unknown: "Unknown",
} as const;

type CheckpointKind = {
  checkpoint: { checkpoint_type: CheckpointReport["checkpoint"]["checkpoint_type"] };
  declared_adapter: { available: boolean };
  observed_checkpoint: { classification: ObservedCheckpoint["classification"] };
};

/** File label. A confirmed full-model inventory is shown even when safetensors metadata has no type. */
export function fileKindLabel(report: CheckpointKind): string {
  if (report.observed_checkpoint.classification === "full_model") return "Full-model checkpoint";
  if (report.checkpoint.checkpoint_type === "lora_adapter") return "LoRA adapter";
  if (report.observed_checkpoint.classification === "adapter") return "Adapter weights";
  if (report.checkpoint.checkpoint_type === "base") return "Base checkpoint";
  if (report.observed_checkpoint.classification === "unknown_subset") return "Unknown subset";
  return "Unknown";
}

export function adapterApplies(report: CheckpointKind): boolean {
  return (
    report.declared_adapter.available ||
    report.observed_checkpoint.classification === "adapter" ||
    report.checkpoint.checkpoint_type === "lora_adapter"
  );
}

export function adapterAbsentNote(report: CheckpointKind): string {
  if (report.observed_checkpoint.classification === "full_model") return "Not applicable — full-model checkpoint.";
  return "Not applicable — this checkpoint is not an adapter.";
}

/** Effective update is a LoRA measurement. A base shard does not receive one. */
export function effectiveUpdateAvailability(report: {
  checkpoint: { checkpoint_type: string };
  declared_adapter: {
    available: boolean;
    peft_type: string | null;
    r: number | null;
    lora_alpha: number | null;
    base_model_name_or_path: string | null;
    base_model_path: string | null;
  };
  observed_checkpoint: { classification: string };
}): { enabled: boolean; reason: string } {
  const adapter = report.declared_adapter;
  const lora = (adapter.peft_type || "").toUpperCase() === "LORA";
  const rank = typeof adapter.r === "number" && adapter.r > 0;
  const alpha = typeof adapter.lora_alpha === "number";
  const base = Boolean(adapter.base_model_name_or_path || adapter.base_model_path);
  if (adapter.available && lora && rank && alpha && base) {
    return { enabled: true, reason: "A LoRA adapter and its configuration are beside this file." };
  }
  const baseFile = report.checkpoint.checkpoint_type === "base" || report.observed_checkpoint.classification === "full_model";
  if (baseFile || !adapter.available) {
    return {
      enabled: false,
      reason: "Effective update applies to a LoRA adapter with its configuration and base-model relationship. This file is not that adapter.",
    };
  }
  if (!lora) {
    return { enabled: false, reason: "Effective update is measured for a LoRA adapter (peft_type LORA)." };
  }
  if (!base) {
    return { enabled: false, reason: "Effective update needs the adapter's base-model relationship in adapter_config.json." };
  }
  return { enabled: false, reason: "Effective update needs a LoRA rank and alpha in adapter_config.json." };
}

export function downloadedFilesLabel(model: {
  checkpoint_status?: string | null;
  weight_shards_present?: number | null;
  files: { name: string }[];
}): string {
  const weights = model.files.filter((file) => file.name.toLowerCase().endsWith(".safetensors"));
  if (model.checkpoint_status === "complete") {
    return weights.length > 1 ? `${weights.length} shards (complete)` : "Complete checkpoint";
  }
  if (model.checkpoint_status === "partial") {
    const count = model.weight_shards_present ?? weights.length;
    return count === 1 ? "1 shard (partial)" : `${count} shards (partial)`;
  }
  return weights.length === 1 ? "1 file" : `${weights.length} files`;
}

export async function inspectCheckpoint(
  apiBase: string,
  path: string,
): Promise<CheckpointReport | CheckpointError> {
  const base = trimBase(apiBase);
  try {
    const res = await fetch(`${base}/inspect/checkpoint`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const data = (await res.json().catch(() => ({}))) as CheckpointReport | CheckpointError;
    if (!res.ok || data.ok !== true) {
      const err = data as CheckpointError;
      return {
        ok: false,
        error: err.error || "request_failed",
        message: err.message || `Inspection failed (HTTP ${res.status}).`,
      };
    }
    return data;
  } catch (error) {
    const message = error instanceof Error ? error.message : "The Inspector API did not respond.";
    return { ok: false, error: "unreachable", message };
  }
}

export type HubFile = {
  name: string;
  size_bytes: number | null;
  size_human: string;
  compatible: boolean;
  local_bytes?: number;
};

/** Matches STORAGE_MARGIN in the download check. The library keeps a little extra room to finish a file. */
export const STORAGE_MARGIN = 1.1;

export type LibraryStorage = {
  free_bytes: number;
  free_label: string;
};

/** Bytes still to be written. A file already stored at the expected size needs nothing. */
export function bytesStillNeeded(
  names: string[],
  files: { name: string; size_bytes: number | null; local_bytes?: number }[],
): number | null {
  const byName = new Map(files.map((file) => [file.name, file]));
  let needed = 0;
  for (const name of names) {
    const file = byName.get(name);
    const size = file?.size_bytes;
    if (size == null || size < 0) return null;
    if ((file?.local_bytes ?? 0) === size) continue;
    needed += size;
  }
  return needed;
}

/** True when the volume can hold this download, including the reserve the server requires. */
export function enoughFreeSpace(needed: number | null, freeBytes: number | null | undefined): boolean {
  if (needed == null || freeBytes == null) return true;
  if (needed <= 0) return true;
  return freeBytes >= Math.trunc(needed * STORAGE_MARGIN);
}

export function downloadSpaceRequired(needed: number): number {
  return Math.trunc(needed * STORAGE_MARGIN);
}

export type HfAuthStatus = "none" | "configured" | "invalid";

export function hfAuthLabel(status: HfAuthStatus | string | undefined): "No token" | "Configured" | "Invalid" {
  if (status === "configured") return "Configured";
  if (status === "invalid") return "Invalid";
  return "No token";
}

/** Compact status for the Hugging Face section. A rejected token stays stored and is not described as connected. */
export function hfSectionAuth(status: HfAuthStatus | string | undefined): {
  access: "Public access" | "Connected" | "Token rejected";
  action: "Configure Token" | "Manage Token";
} {
  if (status === "configured" || status === "validated") return { access: "Connected", action: "Manage Token" };
  if (status === "invalid") return { access: "Token rejected", action: "Manage Token" };
  return { access: "Public access", action: "Configure Token" };
}

/** A cloud key is connected only after that provider accepts it. A saved key is not shared across providers. */
export function cloudSectionAuth(status: string | undefined): {
  access: "Not connected" | "Saved, not tested" | "Connected" | "Key rejected";
  action: "Configure Key" | "Manage Key";
  models: boolean;
} {
  if (status === "validated") return { access: "Connected", action: "Manage Key", models: true };
  if (status === "invalid") return { access: "Key rejected", action: "Manage Key", models: false };
  if (status === "configured") return { access: "Saved, not tested", action: "Manage Key", models: false };
  return { access: "Not connected", action: "Configure Key", models: false };
}

/** A missing or rejected token blocks only a private or gated download. */
export function gatedDownloadAllowed(status: HfAuthStatus | string | undefined, restricted: boolean): boolean {
  return !restricted || status === "configured";
}

export function authFailureNeedsToken(code: string | undefined): boolean {
  return code === "hub_unauthorized" || code === "hub_token_rejected" || code === "hub_access_denied";
}

export type CheckpointPlan = {
  identifiable: boolean;
  reason: string | null;
  shards: string[];
  metadata: string[];
  files: string[];
  shard_count: number;
  total_bytes: number | null;
  total_label: string | null;
};

export function formatDownloadSize(size: number): string {
  let value = size;
  for (const unit of ["B", "KB", "MB", "GB", "TB"]) {
    if (value < 1024 || unit === "TB") {
      if (unit === "B") return `${Math.trunc(value)} B`;
      return `${value.toFixed(1)} ${unit}`;
    }
    value /= 1024;
  }
  return `${size} B`;
}

const WEIGHT_FORMATS: { suffix: string; label: string }[] = [
  { suffix: ".gguf", label: "GGUF" },
  { suffix: ".litertlm", label: "LiteRT-LM" },
  { suffix: ".safetensors", label: "Safetensors" },
  { suffix: ".onnx", label: "ONNX" },
  { suffix: ".bin", label: "PyTorch" },
  { suffix: ".pt", label: "PyTorch" },
  { suffix: ".pth", label: "PyTorch" },
];

export const UNSUPPORTED_CHECKPOINT = "Checkpoint: Unsupported format — no inspectable Safetensors files.";

export function hasInspectableSafetensors(files: { name: string }[]): boolean {
  return files.some((file) => file.name.toLowerCase().endsWith(".safetensors"));
}

/** Weight formats present in a file list, in a stable order. */
export function availableFormats(files: { name: string }[]): string[] {
  const found: string[] = [];
  for (const format of WEIGHT_FORMATS) {
    if (files.some((file) => file.name.toLowerCase().endsWith(format.suffix)) && !found.includes(format.label)) {
      found.push(format.label);
    }
  }
  return found;
}

/** Credential state on a model page. Repository visibility is a separate fact. */
export function modelAuthenticationLabel(status: HfAuthStatus | string | undefined): "Connected" | "Not configured" | "Token rejected" {
  if (status === "configured" || status === "validated") return "Connected";
  if (status === "invalid") return "Token rejected";
  return "Not configured";
}

/** Describe the current selection. Two shards of a larger checkpoint stay partial. */
export function downloadSummary(
  plan: CheckpointPlan | null | undefined,
  selected: string[],
  files: { name: string; size_bytes: number | null }[],
): { selectedLine: string; sizeLine: string; checkpointLine: string } {
  const sizes = new Map(files.map((file) => [file.name, file.size_bytes]));
  const unknown = selected.some((name) => {
    const size = sizes.get(name);
    return size == null || size < 0;
  });
  const selectedBytes = selected.reduce((sum, name) => sum + (sizes.get(name) ?? 0), 0);
  const sizeLine =
    selected.length === 0
      ? "Download size: —"
      : unknown
        ? "Download size: Unknown"
        : `Download size: Approximately ${formatDownloadSize(selectedBytes)}`;
  const shards = plan?.identifiable ? plan.shards : [];
  if (!hasInspectableSafetensors(files)) {
    const count = selected.length;
    return {
      selectedLine: `Selected: ${count} ${count === 1 ? "file" : "files"}`,
      sizeLine,
      checkpointLine: UNSUPPORTED_CHECKPOINT,
    };
  }
  if (shards.length === 0) {
    const count = selected.length;
    return {
      selectedLine: `Selected: ${count} ${count === 1 ? "file" : "files"}`,
      sizeLine,
      checkpointLine: "Checkpoint: Partial — individual shard inspection only",
    };
  }
  const shardNames = new Set(shards);
  const selectedShards = selected.filter((name) => shardNames.has(name)).length;
  const metadata = plan?.metadata ?? [];
  const complete =
    shards.every((name) => selected.includes(name)) && metadata.every((name) => selected.includes(name));
  return {
    selectedLine: `Selected: ${selectedShards} of ${shards.length} weight shards`,
    sizeLine,
    checkpointLine: complete
      ? "Checkpoint: Complete — all required weight shards and configuration metadata"
      : "Checkpoint: Partial — individual shard inspection only",
  };
}

export type ModelCard = {
  ok: true;
  repo: string;
  revision: string;
  card_url: string;
  hf_token_configured: boolean;
  hf_auth?: HfAuthStatus;
  hf_auth_label?: string;
  auth_note?: string;
  private: boolean;
  gated: boolean;
  description: string | null;
  license: string | null;
  pipeline: string | null;
  library: string | null;
  model_type: string | null;
  architectures: string[] | null;
  base_model: string | null;
  tags: string[];
  files: HubFile[];
  checkpoint?: CheckpointPlan;
  storage?: LibraryStorage;
  provider?: string;
  access?: string;
  download_status?: string;
  capabilities?: InspectionCapabilities;
  limits: string;
};

export type InspectionCapabilities = {
  model_information: boolean;
  architecture_details: boolean;
  tensor_inspection: boolean;
  weight_magnitude: boolean;
  weight_direction: boolean;
  behavioral_evaluation: boolean;
};

export type LocalModelFile = {
  name: string;
  size_bytes: number | null;
  size_human: string | null;
  path: string;
  path_label?: string | null;
  path_display?: string | null;
  path_scope?: "project" | "external" | "unavailable";
  inspectable: boolean;
};

export type LocalModel = {
  title?: string;
  repo: string;
  revision: string;
  card_url: string;
  downloaded_at: string;
  bundled?: boolean;
  provider?: string;
  access?: string;
  download_status?: string;
  capabilities?: InspectionCapabilities;
  files: LocalModelFile[];
  checkpoint_status?: string | null;
  checkpoint_label?: string | null;
  integrity_status?: string | null;
  integrity_label?: string | null;
  inspection_status?: string | null;
  inspection_label?: string | null;
  weight_shards_present?: number | null;
  weight_shards_required?: number | null;
  storage_bytes?: number | null;
  storage_label?: string | null;
};

/** Research packages are opened from Books, not listed as general local imports. */
const RESEARCH_LIBRARY_REPOS = new Set(["book-three/freshman", "book-three/qwen"]);

export function libraryBundledModels(models: LocalModel[]): LocalModel[] {
  return models.filter((model) => !RESEARCH_LIBRARY_REPOS.has(model.repo));
}

export type DownloadJob = {
  ok: true;
  id: string;
  status: string;
  percent: number;
  bytes_got: number;
  bytes_total: number;
  message: string;
  error: string | null;
  error_code?: string | null;
  repo: string;
  revision: string;
  manifest: { files: LocalModelFile[] } | null;
};

async function readJson(res: Response): Promise<Record<string, unknown>> {
  const data = (await res.json().catch(() => ({}))) as Record<string, unknown>;
  return data;
}

function asError(data: Record<string, unknown>, status: number): CheckpointError {
  return {
    ok: false,
    error: typeof data.error === "string" ? data.error : "request_failed",
    message: typeof data.message === "string" ? data.message : `Request failed (HTTP ${status}).`,
  };
}

export type ModelSearchHit = {
  repo: string;
  pipeline: string | null;
  task_label: string | null;
  library: string | null;
  model_type: string | null;
  architectures: string[] | null;
  architecture: string | null;
  official: boolean;
  parameters: number | null;
  parameters_label: string | null;
  downloads: number | null;
  downloads_label: string | null;
  license: string | null;
  private: boolean;
  gated: boolean;
  access: "public" | "private" | "gated" | null;
};

const REPO_PART = /^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94})$/;
export const SEARCH_TEXT_LIMIT = 200;
export const SEARCH_EMPTY_RESULTS = "No matching models found.";
export const SEARCH_TOO_LONG = "Search is too long. Enter a shorter model name or repository ID.";
export const SEARCH_INVALID_URL = "Enter a valid Hugging Face repository URL.";

export type DiscoverSearchInput =
  | { kind: "empty" }
  | { kind: "repository"; repo: string }
  | { kind: "search"; term: string }
  | { kind: "invalid"; message: string };

function looksLikeRepositoryUrl(text: string): boolean {
  return text.includes("://") || /^(?:www\.)?(?:huggingface\.co|hf\.co)\//i.test(text);
}

/** Classify Discover input before any request. A model name stays a catalog search. */
export function discoverSearchInput(text: string): DiscoverSearchInput {
  const raw = text.trim();
  if (!raw) return { kind: "empty" };
  if (looksLikeRepositoryUrl(raw)) {
    const repo = repositoryReference(raw);
    if (repo) return { kind: "repository", repo };
    return { kind: "invalid", message: SEARCH_INVALID_URL };
  }
  if (raw.length > SEARCH_TEXT_LIMIT) return { kind: "invalid", message: SEARCH_TOO_LONG };
  const collapsed = raw.replace(/[\n\r\0]+/g, " ").replace(/[ \t]{2,}/g, " ").trim();
  if (!collapsed) return { kind: "empty" };
  if (collapsed.length > SEARCH_TEXT_LIMIT) return { kind: "invalid", message: SEARCH_TOO_LONG };
  const repo = repositoryReference(collapsed);
  if (repo) return { kind: "repository", repo };
  return { kind: "search", term: collapsed };
}

/** Owner/name from a repository id or Hugging Face URL. A model name such as Qwen stays a search. */
export function repositoryReference(text: string): string | null {
  const raw = text.trim();
  if (!raw || raw.length > 2000 || /[\n\r\0]/.test(raw)) return null;
  let repo = raw.split("?")[0].replace(/^\/+|\/+$/g, "");
  if (raw.includes("://") || raw.startsWith("huggingface.co/") || raw.startsWith("hf.co/")) {
    const withScheme = raw.includes("://") ? raw : `https://${raw}`;
    let parsed: URL;
    try {
      parsed = new URL(withScheme);
    } catch {
      return null;
    }
    const host = parsed.hostname.toLowerCase().replace(/^www\./, "");
    if (host !== "huggingface.co" && host !== "hf.co") return null;
    const parts = parsed.pathname.split("/").filter(Boolean).map((part) => decodeURIComponent(part));
    if (parts.length < 2) return null;
    if (parts.length > 2 && !(parts.length >= 4 && (parts[2] === "tree" || parts[2] === "blob" || parts[2] === "resolve"))) {
      return null;
    }
    repo = `${parts[0]}/${parts[1]}`;
  }
  const pieces = repo.split("/");
  if (pieces.length !== 2 || !REPO_PART.test(pieces[0]) || !REPO_PART.test(pieces[1])) return null;
  return `${pieces[0]}/${pieces[1]}`;
}

export type SearchSort = "relevance" | "downloads" | "updated";
export type SearchTask = "all" | "text" | "image" | "other";
export type SearchScope = "public" | "private";

export type ModelSearch = {
  ok: true;
  query: string;
  scope?: SearchScope;
  results: ModelSearchHit[];
  next_cursor: string | null;
  direct_repo?: string | null;
  permission_note?: string | null;
  hf_auth?: HfAuthStatus;
  hf_auth_label?: string;
  hf_validated?: boolean;
};

export function shownOrUnknown(value: string | null | undefined): string {
  return value && value.trim() ? value : "Unknown";
}

const TASK_LABELS: Record<string, string> = {
  "text-generation": "Text Generation",
  "text2text-generation": "Text-to-Text",
  conversational: "Conversational",
  "text-to-image": "Text-to-Image",
  "image-to-image": "Image-to-Image",
  "unconditional-image-generation": "Image Generation",
  "image-to-text": "Image-to-Text",
  "image-text-to-text": "Image-Text-to-Text",
  "text-to-speech": "Text-to-Speech",
  "text-to-audio": "Text-to-Audio",
  "text-to-video": "Text-to-Video",
  "image-to-video": "Image-to-Video",
  "automatic-speech-recognition": "Speech Recognition",
  "token-classification": "Token Classification",
  "question-answering": "Question Answering",
  "fill-mask": "Fill-Mask",
  "feature-extraction": "Feature Extraction",
  "sentence-similarity": "Sentence Similarity",
  translation: "Translation",
  summarization: "Summarization",
  "zero-shot-classification": "Zero-Shot Classification",
};

export function taskLabel(pipeline: string | null | undefined): string {
  const key = pipeline?.trim() ?? "";
  if (!key) return "";
  if (TASK_LABELS[key]) return TASK_LABELS[key];
  let label = "";
  for (const piece of key.split("-")) {
    if (!piece) continue;
    const word = piece.toLowerCase() === "to" ? "to" : piece.slice(0, 1).toUpperCase() + piece.slice(1);
    if (word === "to" && label) label += "-to-";
    else if (label.endsWith("-to-")) label += word;
    else if (label) label += ` ${word}`;
    else label += word;
  }
  return label;
}

export function architectureLabel(modelType: string | null | undefined, architectures?: string[] | null): string {
  const cleaned = modelType?.trim() ?? "";
  const folded = cleaned.toLowerCase().replaceAll("_", "-");
  if (cleaned && !TASK_LABELS[folded]) return cleaned.slice(0, 1).toUpperCase() + cleaned.slice(1);
  if (architectures && architectures.length > 0) return architectures.join(", ");
  return "";
}

export async function searchModels(
  apiBase: string,
  query: string,
  cursor?: string,
  sort?: SearchSort,
  task?: SearchTask,
  scope?: SearchScope,
): Promise<ModelSearch | CheckpointError> {
  const base = trimBase(apiBase);
  const params = new URLSearchParams({ q: query });
  if (cursor) params.set("cursor", cursor);
  if (sort) params.set("sort", sort);
  if (task) params.set("task", task);
  if (scope) params.set("scope", scope);
  try {
    const res = await fetch(`${base}/models/search?${params.toString()}`);
    const data = await readJson(res);
    if (!res.ok || data.ok !== true) return asError(data, res.status);
    return data as ModelSearch;
  } catch (error) {
    const message = error instanceof Error ? error.message : "The Inspector API did not respond.";
    return { ok: false, error: "unreachable", message };
  }
}

export async function discoverModel(apiBase: string, query: string): Promise<ModelCard | CheckpointError> {
  const base = trimBase(apiBase);
  try {
    const res = await fetch(`${base}/models/discover?q=${encodeURIComponent(query)}`);
    const data = await readJson(res);
    if (!res.ok || data.ok !== true) return asError(data, res.status);
    return data as ModelCard;
  } catch (error) {
    const message = error instanceof Error ? error.message : "The Inspector API did not respond.";
    return { ok: false, error: "unreachable", message };
  }
}

export type CredentialProvider = {
  ok: true;
  id: string;
  label: string;
  available: boolean;
  status: "configured" | "validated" | "not_configured" | "invalid";
  status_label: string;
  source: "environment" | "keychain" | null;
  saved: boolean;
  message?: string;
};

export type CredentialList = {
  ok: true;
  precedence: string;
  providers: CredentialProvider[];
};

export async function listCredentials(
  apiBase: string,
  controllerToken: string,
): Promise<CredentialList | CheckpointError> {
  return credentialRequest<CredentialList>(apiBase, "/credentials", controllerToken);
}

export async function saveCredential(
  apiBase: string,
  controllerToken: string,
  token: string,
): Promise<CredentialProvider | CheckpointError> {
  return credentialRequest<CredentialProvider>(apiBase, "/credentials/huggingface", controllerToken, {
    method: "POST",
    body: JSON.stringify({ token }),
  });
}

export async function testCredential(
  apiBase: string,
  controllerToken: string,
): Promise<CredentialProvider | CheckpointError> {
  return credentialRequest<CredentialProvider>(apiBase, "/credentials/huggingface/test", controllerToken, { method: "POST" });
}

export async function removeCredential(
  apiBase: string,
  controllerToken: string,
): Promise<CredentialProvider | CheckpointError> {
  return credentialRequest<CredentialProvider>(apiBase, "/credentials/huggingface/remove", controllerToken, { method: "POST" });
}

export type CloudFact = {
  name: string;
  value: string;
};

export type CloudModelRow = {
  id: string;
  display_name: string | null;
  detail: string | null;
  facts?: CloudFact[];
};

export type SourcedValue = {
  field: string;
  value: string;
  source: string;
};

const PROVIDER_API = "Provider API";
const NOT_REPORTED = "Not reported by this provider's model list.";
const NOT_ACCESSIBLE = "Not accessible through this provider's API.";
const IDENTIFIER_SOURCE = "Inferred from the model identifier. This is not an architecture measurement.";

export function modelIdentifierGroup(id: string): string {
  const name = id.toLowerCase();
  if (name.startsWith("gpt-image") || name.startsWith("chatgpt-image") || name.startsWith("dall-e")) return "Image";
  if (name.startsWith("sora-")) return "Video";
  if (name.startsWith("text-embedding") || name.includes("embedding")) return "Embeddings";
  if (name.startsWith("omni-moderation")) return "Moderation";
  if (
    name.startsWith("tts-") ||
    name.startsWith("whisper") ||
    name.includes("transcribe") ||
    name.includes("-tts") ||
    name.startsWith("gpt-audio") ||
    name.startsWith("gpt-realtime") ||
    name.startsWith("gpt-live")
  ) {
    return "Speech";
  }
  if (/^o[134]($|[^0-9])/.test(name)) return "o-series";
  if (name.startsWith("gpt-")) return "GPT";
  if (name.startsWith("claude")) return "Claude";
  if (name.startsWith("gemini")) return "Gemini";
  if (name.startsWith("babbage") || name.startsWith("davinci")) return "Legacy completion";
  if (name.startsWith("chat-")) return "Chat alias";
  return "Not classified";
}

function factValue(model: CloudModelRow, name: string): string | null {
  return model.facts?.find((fact) => fact.name === name)?.value ?? null;
}

function reportedOr(model: CloudModelRow, factName: string, absentField: string): SourcedValue {
  const value = factValue(model, factName);
  if (value) return { field: factName, value, source: PROVIDER_API };
  return { field: absentField, value: NOT_REPORTED, source: PROVIDER_API };
}

export function cloudModelDetails(providerName: string, model: CloudModelRow): {
  identity: SourcedValue[];
  capabilities: SourcedValue[];
  internal: SourcedValue[];
} {
  const identity: SourcedValue[] = [
    { field: "Model ID", value: model.id, source: PROVIDER_API },
    { field: "Provider", value: providerName, source: "The provider selected for this list." },
  ];
  if (model.display_name) identity.push({ field: "Display name", value: model.display_name, source: PROVIDER_API });
  for (const name of ["Owner", "Created", "Shutdown date", "Type"]) {
    const value = factValue(model, name);
    if (value) identity.push({ field: name, value, source: PROVIDER_API });
  }
  identity.push({
    field: "Model family",
    value: modelIdentifierGroup(model.id),
    source: IDENTIFIER_SOURCE,
  });
  return {
    identity,
    capabilities: [
      reportedOr(model, "Input token limit", "Context window"),
      reportedOr(model, "Output token limit", "Maximum output tokens"),
      { field: "Input and output modalities", value: NOT_REPORTED, source: PROVIDER_API },
      reportedOr(model, "Supported methods", "Supported capabilities"),
      { field: "Tool calling", value: NOT_REPORTED, source: PROVIDER_API },
      { field: "Structured output", value: NOT_REPORTED, source: PROVIDER_API },
    ],
    internal: [
      { field: "Weights and tensors", value: `Not accessible through the ${providerName} API.`, source: PROVIDER_API },
      { field: "Tensor shapes", value: NOT_ACCESSIBLE, source: PROVIDER_API },
      { field: "Tensor values", value: NOT_ACCESSIBLE, source: PROVIDER_API },
      { field: "Weight magnitudes", value: NOT_ACCESSIBLE, source: PROVIDER_API },
      { field: "Parameter count", value: NOT_REPORTED, source: PROVIDER_API },
      { field: "Layer count", value: NOT_REPORTED, source: PROVIDER_API },
      { field: "Architecture", value: NOT_REPORTED, source: PROVIDER_API },
    ],
  };
}

export type CloudCatalog = {
  ok: true;
  id: string;
  label: string;
  models: CloudModelRow[];
  truncated: boolean;
  note: string;
};

const CLOUD_PROVIDER_IDS = ["openai", "anthropic", "google"] as const;

function cloudCredentialPath(provider: string, suffix = ""): string | null {
  if (!CLOUD_PROVIDER_IDS.includes(provider as (typeof CLOUD_PROVIDER_IDS)[number])) return null;
  return `/credentials/${provider}${suffix}`;
}

export async function saveCloudCredential(
  apiBase: string,
  controllerToken: string,
  provider: string,
  token: string,
): Promise<CredentialProvider | CheckpointError> {
  const path = cloudCredentialPath(provider);
  if (!path) return { ok: false, error: "unknown_provider", message: "That provider is not part of Cloud Models." };
  return credentialRequest<CredentialProvider>(apiBase, path, controllerToken, {
    method: "POST",
    body: JSON.stringify({ token }),
  });
}

export async function testCloudCredential(
  apiBase: string,
  controllerToken: string,
  provider: string,
): Promise<CredentialProvider | CheckpointError> {
  const path = cloudCredentialPath(provider, "/test");
  if (!path) return { ok: false, error: "unknown_provider", message: "That provider is not part of Cloud Models." };
  return credentialRequest<CredentialProvider>(apiBase, path, controllerToken, { method: "POST" });
}

export async function removeCloudCredential(
  apiBase: string,
  controllerToken: string,
  provider: string,
): Promise<CredentialProvider | CheckpointError> {
  const path = cloudCredentialPath(provider, "/remove");
  if (!path) return { ok: false, error: "unknown_provider", message: "That provider is not part of Cloud Models." };
  return credentialRequest<CredentialProvider>(apiBase, path, controllerToken, { method: "POST" });
}

export async function listCloudModels(
  apiBase: string,
  controllerToken: string,
  provider: string,
): Promise<CloudCatalog | CheckpointError> {
  const path = cloudCredentialPath(provider, "/models");
  if (!path) return { ok: false, error: "unknown_provider", message: "That provider is not part of Cloud Models." };
  return credentialRequest<CloudCatalog>(apiBase, path, controllerToken);
}

async function credentialRequest<T extends CredentialList | CredentialProvider | CloudCatalog>(
  apiBase: string,
  path: string,
  controllerToken: string,
  init?: RequestInit,
): Promise<T | CheckpointError> {
  const base = trimBase(apiBase);
  try {
    const res = await fetch(`${base}${path}`, {
      ...init,
      cache: "no-store",
      headers: {
        Authorization: `Bearer ${controllerToken}`,
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
      },
    });
    const data = await readJson(res);
    if (!res.ok || data.ok !== true) return asError(data, res.status);
    return data as T;
  } catch (error) {
    const message = error instanceof Error ? error.message : "The Inspector API did not respond.";
    return { ok: false, error: "unreachable", message };
  }
}

export async function listLocalModels(
  apiBase: string,
): Promise<
  | { ok: true; models: LocalModel[]; bundled: LocalModel[]; hf_token_configured: boolean; hf_auth?: HfAuthStatus; hf_auth_label?: string; hf_validated?: boolean }
  | CheckpointError
> {
  const base = trimBase(apiBase);
  try {
    const res = await fetch(`${base}/models/local`);
    const data = await readJson(res);
    if (!res.ok || data.ok !== true) return asError(data, res.status);
    return data as { ok: true; models: LocalModel[]; bundled: LocalModel[]; hf_token_configured: boolean; hf_auth?: HfAuthStatus; hf_auth_label?: string; hf_validated?: boolean };
  } catch (error) {
    const message = error instanceof Error ? error.message : "The Inspector API did not respond.";
    return { ok: false, error: "unreachable", message };
  }
}

export async function startModelDownload(
  apiBase: string,
  repo: string,
  files: string[],
  revision: string,
  mode: "selected" | "complete" = "selected",
): Promise<DownloadJob | CheckpointError> {
  const base = trimBase(apiBase);
  try {
    const res = await fetch(`${base}/models/download`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo, files, revision, mode }),
    });
    const data = await readJson(res);
    if (!res.ok || data.ok !== true) return asError(data, res.status);
    return data as DownloadJob;
  } catch (error) {
    const message = error instanceof Error ? error.message : "The Inspector API did not respond.";
    return { ok: false, error: "unreachable", message };
  }
}

export async function deleteDownloaded(
  apiBase: string,
  repo: string,
  scope: "file" | "model",
  file?: string,
): Promise<{ ok: true; freed_bytes: number; freed_label: string | null; removed: string[] } | CheckpointError> {
  const base = trimBase(apiBase);
  try {
    const res = await fetch(`${base}/models/library/delete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo, scope, file: file ?? null }),
    });
    const data = await readJson(res);
    if (!res.ok || data.ok !== true) return asError(data, res.status);
    return data as { ok: true; freed_bytes: number; freed_label: string | null; removed: string[] };
  } catch (error) {
    const message = error instanceof Error ? error.message : "The Inspector API did not respond.";
    return { ok: false, error: "unreachable", message };
  }
}

export async function modelDownloadStatus(apiBase: string, jobId: string): Promise<DownloadJob | CheckpointError> {
  const base = trimBase(apiBase);
  try {
    const res = await fetch(`${base}/models/jobs/${encodeURIComponent(jobId)}`);
    const data = await readJson(res);
    if (!res.ok || data.ok !== true) return asError(data, res.status);
    return data as DownloadJob;
  } catch (error) {
    const message = error instanceof Error ? error.message : "The Inspector API did not respond.";
    return { ok: false, error: "unreachable", message };
  }
}

export async function cancelModelDownload(apiBase: string, jobId: string): Promise<DownloadJob | CheckpointError> {
  const base = trimBase(apiBase);
  try {
    const res = await fetch(`${base}/models/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
    const data = await readJson(res);
    if (!res.ok || data.ok !== true) return asError(data, res.status);
    return data as DownloadJob;
  } catch (error) {
    const message = error instanceof Error ? error.message : "The Inspector API did not respond.";
    return { ok: false, error: "unreachable", message };
  }
}

export type ResearchSubject = {
  id: string;
  experiment: string;
  experiment_role: string;
  unit_id: string | null;
  seed: number | null;
  epoch: number | null;
  base_model: string;
  artifact_type: string;
  artifact_label: string;
  full_student_checkpoint: string;
  full_student_label: string;
  relative_path: string;
  location: string | null;
  location_label?: string | null;
  location_display?: string | null;
  location_scope?: "project" | "external" | "unavailable";
  file_name: string | null;
  size_bytes: number | null;
  size_human: string | null;
  expected_sha256: string | null;
  observed_sha256: string | null;
  hash_status: "verified_match" | "mismatch" | "unreferenced" | "missing" | "not_configured";
  inspectable: boolean;
  status: string;
  provenance?: {
    weight: { identity: string; expected_sha256: string | null; observed_sha256: string | null; status: string; note: string };
    original_configuration: { identity: string; sha256: string | null; note: string } | null;
    packaged_configuration: { identity: string; sha256: string | null; matches_original: boolean | null; note: string } | null;
    base_weights: { identity: string; status: string; expected_sha256: string | null; observed_sha256: string | null; location: string | null; note: string } | null;
  };
};

export type ResearchFingerprint = {
  subject_id: string;
  hash_status: string;
  expected_sha256: string | null;
  observed_sha256: string | null;
  artifact_type: string;
  artifact_label: string;
  full_student_checkpoint: string;
};

export async function listFreshmanResearch(apiBase: string): Promise<{ ok: true; limits: string; subjects: ResearchSubject[] } | CheckpointError> {
  const base = trimBase(apiBase);
  try {
    const res = await fetch(`${base}/research/book-three/freshman`);
    const data = await readJson(res);
    if (!res.ok || data.ok !== true) return asError(data, res.status);
    return data as { ok: true; limits: string; subjects: ResearchSubject[] };
  } catch (error) {
    const message = error instanceof Error ? error.message : "The Inspector API did not respond.";
    return { ok: false, error: "unreachable", message };
  }
}


export type WeightGroup = {
  role: string;
  tensor_count: number;
  parameter_count: number;
  finite_count: number;
  minimum: number | null;
  maximum: number | null;
  mean: number | null;
  standard_deviation: number | null;
  l1_norm: number | null;
  l2_norm: number | null;
  zero_count: number;
  nonzero_count: number;
  nan_count: number;
  inf_count: number;
};

export type MeasuredCheckpoint = {
  name: string;
  path: string;
  path_label?: string | null;
  path_display?: string | null;
  path_scope?: "project" | "external" | "unavailable";
  sha256: string;
};

export type MeasurementResearch = {
  subject_id: string;
  hash_status?: string;
  artifact_type?: string;
  full_student_checkpoint?: string;
  provenance?: ResearchSubject["provenance"];
};

export type WeightStatisticsReport = {
  ok: true;
  instrument: string;
  limits: string;
  groups: WeightGroup[];
  checkpoint?: MeasuredCheckpoint;
  research?: MeasurementResearch;
};

export type EffectiveGroup = {
  label: string | number;
  pair_count: number;
  frobenius_norm: number;
};

export type AlignmentSlice = {
  label: string | number;
  pair_count: number;
  cosine: number | null;
};

export type AlignmentComparison = {
  left: string;
  right: string;
  cosine: number | null;
  mean_pair_cosine: number | null;
  pair_count: number;
  by_projection: AlignmentSlice[];
  by_layer: AlignmentSlice[];
};

export type AlignmentBaseline = {
  seed: number;
  draws: number;
  entry_count: number;
  dense_isotropic_scale: number | null;
  random_factor_median_absolute: number | null;
  random_factor_maximum_absolute: number | null;
  limits: string;
};

export type UpdateAlignmentReport = {
  ok: true;
  instrument: string;
  limits: string;
  subjects: { subject_id: string | null; name: string | null; total_frobenius_norm: number }[];
  comparisons: AlignmentComparison[];
  baseline: AlignmentBaseline;
};

export type EffectiveUpdateReport = {
  ok: true;
  instrument: string;
  limits: string;
  configuration_agreement: {
    agreed: boolean;
    scale: number;
    scale_rule: string;
    r: number;
    lora_alpha: number | null;
    differing_fields: string[];
    configuration_file_count?: number;
  };
  pair_count: number;
  total_frobenius_norm: number;
  total_base_frobenius_norm: number | null;
  total_relative_frobenius: number | null;
  relative_measured?: boolean;
  relative_note?: string;
  base_checkpoint?: { name: string; path: string; path_label?: string; path_display?: string | null } | null;
  sources?: { lora_alpha: string; r: string; scale: string };
  by_projection: EffectiveGroup[];
  by_layer: EffectiveGroup[];
  research?: MeasurementResearch;
};

async function postInspect<T>(apiBase: string, action: string, path: string): Promise<T | CheckpointError> {
  const base = trimBase(apiBase);
  try {
    const res = await fetch(`${base}/inspect/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const data = (await res.json().catch(() => ({}))) as T | CheckpointError;
    if (!res.ok || (data as { ok?: boolean }).ok !== true) {
      const err = data as CheckpointError;
      return { ok: false, error: err.error || "request_failed", message: err.message || `Measurement failed (HTTP ${res.status}).` };
    }
    return data;
  } catch (error) {
    const message = error instanceof Error ? error.message : "The Inspector API did not respond.";
    return { ok: false, error: "unreachable", message };
  }
}

/** Stored-weight statistics for any checkpoint path. */
export function fetchWeightStatistics(apiBase: string, path: string): Promise<WeightStatisticsReport | CheckpointError> {
  return postInspect(apiBase, "weight-statistics", path);
}

/** Effective LoRA update for any checkpoint path. The engine applies the same instrument to every file. */
export function fetchEffectiveUpdate(apiBase: string, path: string): Promise<EffectiveUpdateReport | CheckpointError> {
  return postInspect(apiBase, "effective-update", path);
}

/** Direction of effective updates. The engine reads the checkpoints; the page does not send matrices. */
export async function fetchUpdateAlignment(apiBase: string, paths: string[]): Promise<UpdateAlignmentReport | CheckpointError> {
  const base = trimBase(apiBase);
  try {
    const res = await fetch(`${base}/inspect/update-alignment`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paths }),
    });
    const data = (await res.json().catch(() => ({}))) as UpdateAlignmentReport | CheckpointError;
    if (!res.ok || data.ok !== true) {
      const err = data as CheckpointError;
      return { ok: false, error: err.error || "request_failed", message: err.message || `Measurement failed (HTTP ${res.status}).` };
    }
    return data;
  } catch (error) {
    const message = error instanceof Error ? error.message : "The Inspector API did not respond.";
    return { ok: false, error: "unreachable", message };
  }
}

export async function probeController(timeoutMs = 1500): Promise<boolean> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${CONTROLLER_BASE}/health`, { signal: ctrl.signal });
    const data = (await res.json().catch(() => ({}))) as { ok?: boolean; service?: string };
    return Boolean(res.ok && data.ok === true && data.service === "ai-model-inspector-controller");
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

export async function controllerAction(
  action: "status" | "start" | "stop",
  port: number,
  token: string,
  timeoutMs: number,
): Promise<{ reachable: boolean; authorized: boolean; body: Record<string, unknown> | null }> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${CONTROLLER_BASE}${action === "status" ? `/v1/status?port=${port}` : `/v1/${action}`}`, {
      method: action === "status" ? "GET" : "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        ...(action === "status" ? {} : { "Content-Type": "application/json" }),
      },
      body: action === "status" ? undefined : JSON.stringify({ port }),
      signal: ctrl.signal,
    });
    const body = (await res.json().catch(() => null)) as Record<string, unknown> | null;
    return { reachable: true, authorized: res.status !== 401, body };
  } catch {
    return { reachable: false, authorized: false, body: null };
  } finally {
    clearTimeout(timer);
  }
}

const PROCESS_STATES = new Set<ControllerProcessState>(["running", "stopped", "foreign", "unresponsive", "error", "unknown"]);

/** Ask the local controller whether the Inspector process is running. Both consoles use this. */
export async function readControllerProcess(
  port: number,
  token: string | null,
): Promise<{
  controllerReachable: boolean;
  credentialPresent: boolean;
  authorized: boolean;
  processState: ControllerProcessState | null;
}> {
  const reachable = await probeController();
  const credentialPresent = Boolean(token);
  if (!reachable || !token) {
    return { controllerReachable: reachable, credentialPresent, authorized: false, processState: null };
  }
  const result = await controllerAction("status", port, token, 8000);
  const state = result.body?.state;
  let processState: ControllerProcessState | null = null;
  if (typeof state === "string" && PROCESS_STATES.has(state as ControllerProcessState)) {
    processState = state as ControllerProcessState;
  } else if (result.reachable && result.authorized) {
    processState = "unknown";
  }
  return {
    controllerReachable: result.reachable,
    credentialPresent,
    authorized: result.authorized,
    processState,
  };
}

export async function probeHealth(apiBase: string, timeoutMs = 1500): Promise<InspectorHealth | null> {
  const base = trimBase(apiBase);
  if (!base) return null;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${base}/health`, { signal: ctrl.signal });
    const data = (await res.json().catch(() => ({}))) as InspectorHealth;
    if (!res.ok || data.ok !== true) return null;
    return data;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}
