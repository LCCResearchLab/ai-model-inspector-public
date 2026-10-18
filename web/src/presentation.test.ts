/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * Presentation checks for the effective-weight milestone.
 * These tests do not recompute the matrix measurements.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  authFailureNeedsToken,
  bytesStillNeeded,
  availableFormats,
  downloadSummary,
  hasInspectableSafetensors,
  modelAuthenticationLabel,
  UNSUPPORTED_CHECKPOINT,
  downloadedFilesLabel,
  enoughFreeSpace,
  effectiveUpdateAvailability,
  gatedDownloadAllowed,
  cloudModelDetails,
  cloudSectionAuth,
  hfAuthLabel,
  modelIdentifierGroup,
  hfSectionAuth,
  libraryBundledModels,
  architectureLabel,
  discoverSearchInput,
  SEARCH_EMPTY_RESULTS,
  SEARCH_INVALID_URL,
  SEARCH_TOO_LONG,
  observedLayerFacts,
  repositoryReference,
  shownOrUnknown,
  taskLabel,
  type LocalModel,
} from "./inspectorClient.ts";
import { experimentTitle, RESEARCH_COLLECTIONS } from "./provenance.ts";
import { cloudModelHash, modelHash, parseRoute } from "./route.ts";
import { clampTextZoom, readTextZoom, stepTextZoom, TEXT_ZOOM_DEFAULT, TEXT_ZOOM_KEY, TEXT_ZOOM_MAX, TEXT_ZOOM_MIN } from "./textZoom.ts";
import {
  MATCHING_BASE_PROJECTIONS_HELP,
  MATCHING_BASE_PROJECTIONS_LABEL,
  SHARED_LAYER_SCALE_TITLE,
  UPDATE_DIRECTION_HELP,
  UPDATE_DIRECTION_TITLE,
  barPercent,
  sharedLayerMaximum,
} from "./inspectorClient.ts";

const chapterFive = [0.2346, 0.2371, 0.2337];
const chapterSix = [1.373, 1.435, 3.24];

test("shared layer scale uses one maximum for every subject", () => {
  const shared = sharedLayerMaximum([chapterFive, chapterSix]);
  assert.equal(shared, 3.24);
  assert.equal(barPercent(3.24, shared), 100);
  const subjectOnly = sharedLayerMaximum([chapterFive]);
  assert.equal(barPercent(0.2371, subjectOnly), 100);
  assert.ok(barPercent(0.2371, shared) < 10);
  assert.notEqual(barPercent(0.2371, shared), barPercent(0.2371, subjectOnly));
});

test("a zero or missing maximum does not invent a full bar", () => {
  assert.equal(barPercent(1.156, 0), 0);
  assert.equal(sharedLayerMaximum([[], []]), 0);
});

test("the BASE denominator label names the 96 matching projections", () => {
  assert.equal(MATCHING_BASE_PROJECTIONS_LABEL, "Relative to matching BASE projections");
  assert.match(MATCHING_BASE_PROJECTIONS_HELP, /96 BASE attention projection matrices/);
  assert.match(MATCHING_BASE_PROJECTIONS_HELP, /Q, K, V, and O across 24 transformer layers/);
  assert.match(MATCHING_BASE_PROJECTIONS_HELP, /not the norm of the entire BASE model/);
  assert.equal(SHARED_LAYER_SCALE_TITLE, "Layer comparison — shared scale");
});

test("a book experiment keeps its origin separate from the measurement", () => {
  assert.equal(experimentTitle("Book Two, Chapter 5 — After the Lesson Is Gone"), "After the Lesson Is Gone");
  assert.equal(experimentTitle("Starting model"), null);
  assert.equal(RESEARCH_COLLECTIONS[0].researchContext, "Book Three — From Interpretation to Measurement");
  assert.match(RESEARCH_COLLECTIONS[0].measurement, /AI Model Inspector/);
  const qwen = RESEARCH_COLLECTIONS.find((collection) => collection.id === "qwen");
  assert.equal(qwen?.label, "Qwen Research");
  assert.equal(qwen?.researchContext, "No book chapter is recorded for this adapter.");
  assert.doesNotMatch(qwen?.researchContext || "", /Chapter/);
});

test("research packages stay out of Import Local", () => {
  const sample = (repo: string) => ({ repo, revision: "", card_url: "", downloaded_at: "", files: [] }) as LocalModel;
  const kept = libraryBundledModels([sample("book-three/freshman"), sample("book-three/qwen"), sample("org/toy")]);
  assert.deepEqual(kept.map((model) => model.repo), ["org/toy"]);
});

test("authentication blocks only the operations that need a credential", () => {
  assert.equal(hfAuthLabel(undefined), "No token");
  assert.equal(hfAuthLabel("not_configured"), "No token");
  assert.equal(hfAuthLabel("configured"), "Configured");
  assert.equal(hfAuthLabel("invalid"), "Invalid");
  assert.deepEqual(hfSectionAuth(undefined), { access: "Public access", action: "Configure Token" });
  assert.deepEqual(hfSectionAuth("none"), { access: "Public access", action: "Configure Token" });
  assert.deepEqual(hfSectionAuth("configured"), { access: "Connected", action: "Manage Token" });
  assert.deepEqual(hfSectionAuth("validated"), { access: "Connected", action: "Manage Token" });
  assert.deepEqual(hfSectionAuth("invalid"), { access: "Token rejected", action: "Manage Token" });
  assert.deepEqual(cloudSectionAuth(undefined), { access: "Not connected", action: "Configure Key", models: false });
  assert.deepEqual(cloudSectionAuth("configured"), { access: "Saved, not tested", action: "Manage Key", models: false });
  assert.deepEqual(cloudSectionAuth("validated"), { access: "Connected", action: "Manage Key", models: true });
  assert.deepEqual(cloudSectionAuth("invalid"), { access: "Key rejected", action: "Manage Key", models: false });
  assert.equal(gatedDownloadAllowed("none", false), true);
  assert.equal(gatedDownloadAllowed("invalid", false), true);
  assert.equal(gatedDownloadAllowed("none", true), false);
  assert.equal(gatedDownloadAllowed("invalid", true), false);
  assert.equal(gatedDownloadAllowed("configured", true), true);
  assert.equal(authFailureNeedsToken("hub_unauthorized"), true);
  assert.equal(authFailureNeedsToken("hub_token_rejected"), true);
  assert.equal(authFailureNeedsToken("hub_access_denied"), true);
  assert.equal(authFailureNeedsToken("missing_file"), false);
  const library = readFileSync(new URL("./library.tsx", import.meta.url), "utf8");
  assert.match(library, /disabled=\{busy \|\| typedSearch\.kind === "empty" \|\| typedSearch\.kind === "invalid"\}/);
  assert.match(library, /Public models stay available/);
  assert.match(library, /Search by model name, repository ID, or URL/);
  assert.match(library, /Examples: Qwen, Qwen\/Qwen3-Coder-30B-A3B-Instruct/);
  assert.match(library, /discoverSearchInput\(query\)/);
  assert.match(library, /SEARCH_EMPTY_RESULTS/);
  assert.match(library, /className="private-label">Private/);
  assert.doesNotMatch(library, /Public Models/);
  assert.doesNotMatch(library, /My Private Models/);
  assert.match(library, /Open repository/);
  assert.match(library, /Load More/);
  assert.match(library, /Most Downloads/);
  assert.match(library, /Recently Updated/);
  assert.match(library, /Text Generation/);
  assert.match(library, /Image Generation/);
  assert.match(library, /title=\{hit\.repo\}/);
  assert.match(library, /className="repo-id"/);
  assert.match(library, /← Back to Search Results/);
  assert.match(library, /GGUF files are not currently supported for tensor inspection/);
  assert.match(library, /Model Overview/);
  assert.match(library, /Download Selection/);
  assert.match(library, /Download Complete Model/);
  assert.match(library, /disabled=\{busy \|\| selected\.length > 0 \|\| !downloadAllowed \|\| !card\.checkpoint\?\.identifiable \|\| !completeFits\}/);
  assert.match(library, /disabled=\{busy \|\| selected\.length === 0 \|\| !downloadAllowed \|\| !selectedFits\}/);
  assert.match(library, /Free space:/);
  assert.match(library, /Clear file selections to download the complete model\./);
  assert.match(library, /Download Selected Files/);
  const stored = [
    { name: "model.safetensors", size_bytes: 1000, local_bytes: 1000 },
    { name: "config.json", size_bytes: 12, local_bytes: 0 },
  ];
  assert.equal(bytesStillNeeded(["model.safetensors"], stored), 0);
  assert.equal(bytesStillNeeded(["model.safetensors", "config.json"], stored), 12);
  assert.equal(bytesStillNeeded(["model.safetensors"], [{ name: "model.safetensors", size_bytes: 1000, local_bytes: 999 }]), 1000);
  assert.equal(enoughFreeSpace(0, 0), true);
  assert.equal(enoughFreeSpace(1000, 1099), false);
  assert.equal(enoughFreeSpace(1000, 1100), true);
  assert.equal(enoughFreeSpace(null, 0), true);
  const shards = Array.from({ length: 16 }, (_, index) => `model-${String(index + 1).padStart(5, "0")}-of-00016.safetensors`);
  const plan = {
    identifiable: true,
    reason: null,
    shards,
    metadata: ["config.json", "model.safetensors.index.json"],
    files: [...shards, "config.json", "model.safetensors.index.json"],
    shard_count: 16,
    total_bytes: null,
    total_label: null,
  };
  const files = plan.files.map((name) => ({ name, size_bytes: name.endsWith(".safetensors") ? 2_516_582_400 : 1000 }));
  const partial = downloadSummary(plan, [shards[4], shards[15]], files);
  assert.equal(partial.selectedLine, "Selected: 2 of 16 weight shards");
  assert.equal(partial.sizeLine, "Download size: Approximately 4.7 GB");
  assert.equal(partial.checkpointLine, "Checkpoint: Partial — individual shard inspection only");
  const complete = downloadSummary(plan, plan.files, files);
  assert.equal(complete.checkpointLine, "Checkpoint: Complete — all required weight shards and configuration metadata");
  const unsupported = downloadSummary(
    { identifiable: false, reason: "no index", shards: [], metadata: [], files: [], shard_count: 0, total_bytes: null, total_label: null },
    [],
    [
      { name: "gemma.gguf", size_bytes: 1 },
      { name: "gemma.litertlm", size_bytes: 1 },
    ],
  );
  assert.equal(unsupported.checkpointLine, UNSUPPORTED_CHECKPOINT);
  assert.equal(hasInspectableSafetensors([{ name: "gemma.gguf" }]), false);
  assert.deepEqual(availableFormats([{ name: "gemma.gguf" }, { name: "gemma.litertlm" }, { name: "README.md" }]), ["GGUF", "LiteRT-LM"]);
  assert.equal(modelAuthenticationLabel("configured"), "Connected");
  assert.equal(modelAuthenticationLabel("validated"), "Connected");
  assert.equal(modelAuthenticationLabel("none"), "Not configured");
  assert.equal(modelAuthenticationLabel("invalid"), "Token rejected");
  assert.match(library, /Supported files/);
  assert.match(library, /Available formats/);
  assert.match(library, /Inspection Not supported/);
  assert.match(library, /Download Disabled/);
  assert.match(library, /Authentication \{modelAuthenticationLabel/);
  assert.equal(
    downloadedFilesLabel({
      checkpoint_status: "partial",
      weight_shards_present: 1,
      files: [{ name: "model-00016-of-00016.safetensors" }],
    }),
    "1 shard (partial)",
  );
  assert.equal(
    downloadedFilesLabel({
      checkpoint_status: "complete",
      weight_shards_present: 1,
      files: [{ name: "model.safetensors" }, { name: "config.json" }],
    }),
    "Complete checkpoint",
  );
  const shard = {
    checkpoint: { checkpoint_type: "unknown" },
    declared_adapter: {
      available: false,
      peft_type: null,
      r: null,
      lora_alpha: null,
      base_model_name_or_path: null,
      base_model_path: null,
    },
    observed_checkpoint: { classification: "unknown" },
  };
  assert.equal(effectiveUpdateAvailability(shard).enabled, false);
  const lora = {
    checkpoint: { checkpoint_type: "lora_adapter" },
    declared_adapter: {
      available: true,
      peft_type: "LORA",
      r: 16,
      lora_alpha: 32,
      base_model_name_or_path: "org/base",
      base_model_path: null,
    },
    observed_checkpoint: { classification: "adapter" },
  };
  assert.equal(effectiveUpdateAvailability(lora).enabled, true);
  assert.match(library, /Delete Entire Model/);
  assert.match(library, /Complete model analysis is unavailable until all required shards are present/);
  assert.match(library, /The Hugging Face repository is not changed/);
  assert.match(library, /Integrity not verified — no published checksum was available/);
  assert.match(library, /Inspectable — the checkpoint was parsed and its tensors can be examined/);
  assert.equal(shownOrUnknown(null), "Unknown");
  assert.equal(shownOrUnknown("qwen2"), "qwen2");
  assert.equal(taskLabel("text-generation"), "Text Generation");
  assert.equal(taskLabel("text-to-image"), "Text-to-Image");
  assert.equal(taskLabel("image-text-to-text"), "Image-Text-to-Text");
  assert.equal(taskLabel(null), "");
  assert.equal(architectureLabel("qwen2"), "Qwen2");
  assert.equal(architectureLabel("qwen3"), "Qwen3");
  assert.equal(architectureLabel("text-to-image"), "");
  assert.equal(architectureLabel(null, ["Qwen2ForCausalLM"]), "Qwen2ForCausalLM");
  assert.equal(architectureLabel(null), "");
  assert.match(library, /View Models/);
  assert.match(library, /catalog\.models\.length === 1 \? "model" : "models"/);
  assert.match(library, /localeCompare\(b\.id, "en"\)/);
  assert.doesNotMatch(library, /No inference request is sent, and this list does not include internal weights/);
  assert.match(library, /does not download or inspect proprietary model weights/);
  assert.match(library, /About Cloud Inspection/);
  assert.match(library, /Generally unavailable/);
  assert.doesNotMatch(library, /Inspect Capabilities/);
  assert.doesNotMatch(library, /disabled=\{!auth/);
});

test("credentials stay inside Hugging Face and Cloud Models", () => {
  const library = readFileSync(new URL("./library.tsx", import.meta.url), "utf8");
  const route = readFileSync(new URL("./route.ts", import.meta.url), "utf8");
  assert.match(library, /Get HF Token/);
  assert.match(library, /label: "Discover"/);
  assert.match(library, /label: "Downloaded"/);
  assert.match(library, /label: "Import Local"/);
  assert.doesNotMatch(library, /label: "HF Token"/);
  assert.match(library, /hfSectionAuth/);
  assert.match(library, /Authentication: \{shown\.access\}/);
  assert.match(library, /shown\.access === "Connected" \? <span className="hf-ok"> ✓<\/span>/);
  assert.match(library, /className="row search-actions"/);
  assert.match(library, /\{shown\.action\}/);
  assert.match(library, /Configure Token/);
  assert.match(library, /Manage Token/);
  assert.match(library, /https:\/\/huggingface\.co\/settings\/tokens/);
  assert.match(readFileSync(new URL("./inspectorClient.ts", import.meta.url), "utf8"), /Configure Key/);
  assert.match(library, /Manage Key/);
  assert.match(library, /Show key/);
  assert.match(library, /Hide key/);
  assert.match(library, /Show key reveals only the text in this box/);
  assert.match(library, /type=\{reveal \? "text" : "password"\}/);
  assert.match(library, /View Models/);
  assert.match(library, /cloudSectionAuth/);
  assert.match(library, /className="providers cloud"/);
  assert.match(library, /shown\.access === "Connected" \? <span className="connected-mark"> ✓<\/span>/);
  assert.match(library, /Not supported yet/);
  assert.match(library, /Get API Key/);
  assert.match(library, /https:\/\/platform\.openai\.com\/api-keys/);
  assert.match(library, /https:\/\/console\.anthropic\.com\//);
  assert.match(library, /https:\/\/aistudio\.google\.com\/apikey/);
  assert.match(library, /Google Gemini/);
  assert.match(library, /Get Key/);
  assert.doesNotMatch(library, />\s*API Keys\s*</);
  assert.match(library, /#\/library\/huggingface\/token/);
  assert.doesNotMatch(library, /#\/library\/keys/);
  assert.match(route, /parts\[2\] === "token"/);
  assert.match(route, /section: "token"/);
});

test("the weights screen uses the shared scale and the denominator wording", () => {
  const web = readFileSync(new URL("./book.tsx", import.meta.url), "utf8");
  assert.match(web, /SHARED_LAYER_SCALE_TITLE/);
  assert.match(web, /MATCHING_BASE_PROJECTIONS_LABEL/);
  assert.match(web, /MATCHING_BASE_PROJECTIONS_HELP/);
  assert.match(web, /sharedLayerMaximum/);
  assert.match(web, /Books &amp; Research/);
  assert.match(web, /Inspect →/);
  assert.match(web, /Overview/);
  assert.match(web, /Freshman Research/);
  assert.match(web, /Qwen Research/);
  assert.match(web, /Research Findings/);
  assert.match(readFileSync(new URL("./library.tsx", import.meta.url), "utf8"), /Configure Token/);
  assert.doesNotMatch(web, /<h1>Book Three<\/h1>/);
  assert.match(web, /UPDATE_DIRECTION_TITLE/);
  assert.match(web, /UPDATE_DIRECTION_HELP/);
  assert.equal(UPDATE_DIRECTION_TITLE, "Direction of the effective update");
  assert.match(UPDATE_DIRECTION_HELP, /orthogonal/);
  assert.match(UPDATE_DIRECTION_HELP, /not a measurement of acquired knowledge/);
  assert.doesNotMatch(web, /Relative to BASE/);
});

test("a single high layer index is not shown as that many layers", () => {
  assert.deepEqual(observedLayerFacts([47]), { distinct: 1, highest: 47 });
  assert.deepEqual(observedLayerFacts([0, 1, 2, 3, 4]), { distinct: 5, highest: 4 });
  assert.deepEqual(observedLayerFacts([]), { distinct: 0, highest: null });
  const report = readFileSync(new URL("./report.tsx", import.meta.url), "utf8");
  assert.match(report, /Distinct layers in this file/);
  assert.match(report, /Highest layer index in this file/);
  assert.match(report, /Declared layers/);
  assert.doesNotMatch(report, /Layers \{observed\.layer_indices/);
});

test("checkpoint screens label a path from the project root", () => {
  const report = readFileSync(new URL("./report.tsx", import.meta.url), "utf8");
  const book = readFileSync(new URL("./book.tsx", import.meta.url), "utf8");
  assert.match(report, /Project-relative path/);
  assert.match(report, /function ProjectPath/);
  assert.match(book, /<ProjectPath/);
  assert.doesNotMatch(report, /<p className="path">\{report\.checkpoint\.path\}/);
  assert.doesNotMatch(book, /<p className="path">\{subject\.location\}/);
  assert.doesNotMatch(book, /measured\?\.path \|\| subject\.location/);
});

test("a repository id or URL opens that repository and a model name stays a search", () => {
  assert.equal(repositoryReference("Qwen"), null);
  assert.equal(repositoryReference("Llama"), null);
  assert.equal(repositoryReference("SmolLM"), null);
  assert.equal(repositoryReference("Qwen/Qwen3-Coder-30B-A3B-Instruct"), "Qwen/Qwen3-Coder-30B-A3B-Instruct");
  assert.equal(
    repositoryReference("https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct"),
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
  );
  assert.equal(
    repositoryReference("https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/tree/main"),
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
  );
  assert.equal(repositoryReference("https://example.com/Qwen/Qwen3-Coder-30B-A3B-Instruct"), null);
  assert.equal(repositoryReference("https://www.huggingface.co/owner/private-model"), "owner/private-model");
  assert.deepEqual(discoverSearchInput("Qwen"), { kind: "search", term: "Qwen" });
  assert.deepEqual(discoverSearchInput("   "), { kind: "empty" });
  assert.deepEqual(discoverSearchInput("owner/private-model"), { kind: "repository", repo: "owner/private-model" });
  assert.deepEqual(discoverSearchInput("https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/tree/main"), {
    kind: "repository",
    repo: "Qwen/Qwen3-Coder-30B-A3B-Instruct",
  });
  assert.deepEqual(discoverSearchInput("https://example.com/owner/model"), { kind: "invalid", message: SEARCH_INVALID_URL });
  assert.deepEqual(discoverSearchInput("https://huggingface.co/only-owner"), { kind: "invalid", message: SEARCH_INVALID_URL });
  assert.deepEqual(discoverSearchInput(`${"Approved. ".repeat(40)}Qwen`), { kind: "invalid", message: SEARCH_TOO_LONG });
  assert.equal(discoverSearchInput("a".repeat(200)).kind, "search");
  assert.equal(SEARCH_EMPTY_RESULTS, "No matching models found.");
  assert.equal(repositoryReference("owner/private-model"), "owner/private-model");
  assert.equal(modelHash(repositoryReference("owner/private-model") || ""), "#/library/huggingface/model/owner/private-model");
});

test("model details keep the repository id in the address", () => {
  const previous = globalThis.window;
  const hash = modelHash("Qwen/Qwen3-0.6B");
  assert.equal(hash, "#/library/huggingface/model/Qwen/Qwen3-0.6B");
  globalThis.window = { location: { hash } } as Window & typeof globalThis;
  try {
    assert.deepEqual(parseRoute(), { view: "library", section: "discover", repo: "Qwen/Qwen3-0.6B" });
    globalThis.window = { location: { hash: "#/library/huggingface" } } as Window & typeof globalThis;
    assert.deepEqual(parseRoute(), { view: "library", section: "discover" });
    globalThis.window = { location: { hash: "#/library/cloud/models/openai" } } as Window & typeof globalThis;
    assert.deepEqual(parseRoute(), { view: "library", section: "models", cloudProvider: "openai" });
    globalThis.window = { location: { hash: "#/library/cloud/models/openai/gpt-5-mini" } } as Window & typeof globalThis;
    assert.deepEqual(parseRoute(), { view: "library", section: "models", cloudProvider: "openai", cloudModel: "gpt-5-mini" });
    assert.equal(cloudModelHash("openai", "gpt-5-mini"), "#/library/cloud/models/openai/gpt-5-mini");
    assert.equal(modelIdentifierGroup("gpt-5-mini"), "GPT");
    assert.equal(modelIdentifierGroup("o1-pro"), "o-series");
    assert.equal(modelIdentifierGroup("gpt-4o-mini-tts"), "Speech");
    assert.equal(modelIdentifierGroup("claude-test"), "Claude");
    assert.equal(modelIdentifierGroup("gemini-test"), "Gemini");
    const openai = cloudModelDetails("OpenAI", {
      id: "gpt-5-mini",
      display_name: null,
      detail: "Owned by system",
      facts: [
        { name: "Owner", value: "system" },
        { name: "Created", value: "2025-08-07 00:00:00 UTC" },
      ],
    });
    assert.equal(openai.identity.find((row) => row.field === "Owner")?.source, "Provider API");
    assert.equal(openai.identity.find((row) => row.field === "Model family")?.value, "GPT");
    assert.match(openai.identity.find((row) => row.field === "Model family")?.source || "", /not an architecture measurement/);
    assert.equal(openai.capabilities.find((row) => row.field === "Context window")?.value, "Not reported by this provider's model list.");
    assert.equal(openai.internal.find((row) => row.field === "Parameter count")?.value, "Not reported by this provider's model list.");
    assert.match(openai.internal.find((row) => row.field === "Weights and tensors")?.value || "", /Not accessible through the OpenAI API/);
    const gemini = cloudModelDetails("Google Gemini", {
      id: "gemini-test",
      display_name: "Gemini Test",
      detail: null,
      facts: [
        { name: "Input token limit", value: "1048576" },
        { name: "Supported methods", value: "generateContent" },
      ],
    });
    assert.equal(gemini.capabilities.find((row) => row.field === "Input token limit")?.source, "Provider API");
    assert.equal(gemini.capabilities.find((row) => row.field === "Supported methods")?.value, "generateContent");
    const library = readFileSync(new URL("./library.tsx", import.meta.url), "utf8");
    assert.match(library, /View Details/);
    assert.doesNotMatch(library, /128000|parameter count of|billion parameters/i);
    globalThis.window = { location: { hash: "#/library/cloud/providers/google" } } as Window & typeof globalThis;
    assert.deepEqual(parseRoute(), { view: "library", section: "providers", cloudProvider: "google" });
    globalThis.window = { location: { hash: "#/about" } } as Window & typeof globalThis;
    assert.deepEqual(parseRoute(), { view: "about" });
    const app = readFileSync(new URL("./App.tsx", import.meta.url), "utf8");
    const css = readFileSync(new URL("./styles.css", import.meta.url), "utf8");
    assert.match(app, /About LCC Research Lab/);
    assert.match(app, /Adventures in AI/);
    assert.match(app, /https:\/\/lccresearchlab\.com/);
    assert.match(app, /mailto:lccresearchlab@gmail\.com/);
    assert.match(app, /https:\/\/github\.com\/LCCResearchLab/);
    assert.match(app, /health\?\.version/);
    assert.doesNotMatch(app, /Why We Work With AI|why-we-work-with-ai|\.mp3|<audio|team-song/);
    assert.doesNotMatch(css, /team-song|why-we-work-with-ai/);
    assert.match(css, /\.about-page \{[^}]*max-width: 800px/);
  } finally {
    globalThis.window = previous;
  }
});

test("text size stays between 80 and 150 percent", () => {
  assert.equal(TEXT_ZOOM_MIN, 0.8);
  assert.equal(TEXT_ZOOM_MAX, 1.5);
  assert.equal(clampTextZoom(0.79), 0.8);
  assert.equal(clampTextZoom(1.51), 1.5);
  assert.equal(clampTextZoom(Number.NaN), TEXT_ZOOM_DEFAULT);
  assert.equal(stepTextZoom(1, 1), 1.1);
  assert.equal(stepTextZoom(1.5, 1), 1.5);
  assert.equal(stepTextZoom(0.8, -1), 0.8);
  let size = 1;
  for (let step = 0; step < 8; step += 1) size = stepTextZoom(size, 1);
  assert.equal(size, 1.5);
  const stored = new Map<string, string>();
  const storage = {
    getItem: (key: string) => stored.get(key) ?? null,
    setItem: (key: string, value: string) => {
      stored.set(key, value);
    },
  };
  assert.equal(readTextZoom(storage), TEXT_ZOOM_DEFAULT);
  storage.setItem(TEXT_ZOOM_KEY, "1.2");
  assert.equal(readTextZoom(storage), 1.2);
  storage.setItem(TEXT_ZOOM_KEY, "nope");
  assert.equal(readTextZoom(storage), TEXT_ZOOM_DEFAULT);
  const app = readFileSync(new URL("./App.tsx", import.meta.url), "utf8");
  const css = readFileSync(new URL("./styles.css", import.meta.url), "utf8");
  assert.match(app, /Decrease text size/);
  assert.match(app, /Increase text size/);
  assert.match(app, /Reset text size/);
  assert.match(app, /TEXT_ZOOM_KEY/);
  assert.match(css, /font-size: calc\(16px \* var\(--text-zoom\)\)/);
  assert.match(css, /height: 100dvh/);
  assert.match(css, /min-width: 960px/);
  assert.match(css, /overflow-x: auto/);
  assert.doesNotMatch(css, /@media/);
  assert.match(css, /main \{[^}]*overflow-y: auto/);
  assert.match(app, /mainRef\.current\?\.scrollTo\(0, 0\)/);
  assert.match(css, /\.bar \{\s*height: 10px;/);
  assert.doesNotMatch(css, /\.bar \{[^}]*var\(--text-zoom\)/);
});

test("books and library files share one checkpoint inspector", () => {
  const panel = readFileSync(new URL("./checkpointInspector.tsx", import.meta.url), "utf8");
  const book = readFileSync(new URL("./book.tsx", import.meta.url), "utf8");
  const library = readFileSync(new URL("./library.tsx", import.meta.url), "utf8");
  assert.match(panel, /Checkpoint Inspector/);
  assert.match(panel, /ReportView/);
  assert.match(panel, /Stored weights/);
  assert.match(panel, /Effective update/);
  assert.match(book, /CheckpointInspector/);
  assert.match(library, /CheckpointInspector/);
  assert.match(book, /Compare directions/);
});
