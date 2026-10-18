/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * Model Library. Hugging Face is a repository. Cloud Models is a remote access method.
 */
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { loadControllerCredential } from "./credential";
import {
  cancelModelDownload,
  deleteDownloaded,
  discoverModel,
  downloadedFilesLabel,
  authFailureNeedsToken,
  bytesStillNeeded,
  downloadSpaceRequired,
  enoughFreeSpace,
  formatDownloadSize,
  availableFormats,
  downloadSummary,
  hasInspectableSafetensors,
  modelAuthenticationLabel,
  gatedDownloadAllowed,
  cloudModelDetails,
  cloudSectionAuth,
  hfSectionAuth,
  listCloudModels,
  libraryBundledModels,
  listCredentials,
  listLocalModels,
  modelDownloadStatus,
  removeCloudCredential,
  removeCredential,
  saveCloudCredential,
  saveCredential,
  discoverSearchInput,
  SEARCH_EMPTY_RESULTS,
  searchModels,
  architectureLabel,
  shownOrUnknown,
  startModelDownload,
  taskLabel,
  testCloudCredential,
  testCredential,
  type ModelSearchHit,
  type SearchSort,
  type SearchTask,
  type HfAuthStatus,
  type CloudCatalog,
  type SourcedValue,
  type CredentialProvider,
  type DownloadJob,
  type LocalModel,
  type ModelCard,
} from "./inspectorClient";
import { WorkProgress } from "./progress";
import { downloadProgress } from "./workProgress";
import { cloudModelHash, go, modelHash, type LibrarySection } from "./route";
import { CheckpointInspector } from "./checkpointInspector";

export function LibraryConsole({
  apiBase,
  section,
  path,
  repo,
  cloudProvider,
  cloudModel,
}: {
  apiBase: string;
  section: LibrarySection;
  path?: string;
  repo?: string;
  cloudProvider?: string;
  cloudModel?: string;
}) {
  return <LibraryHome apiBase={apiBase} section={section} path={path} repo={repo} cloudProvider={cloudProvider} cloudModel={cloudModel} />;
}

function LibraryHome({
  apiBase,
  section,
  path,
  repo,
  cloudProvider,
  cloudModel,
}: {
  apiBase: string;
  section: LibrarySection;
  path?: string;
  repo?: string;
  cloudProvider?: string;
  cloudModel?: string;
}) {
  const [bundled, setBundled] = useState<LocalModel[]>([]);
  const [models, setModels] = useState<LocalModel[]>([]);
  const [auth, setAuth] = useState<HfAuthStatus>("none");
  const [directRepo, setDirectRepo] = useState<string | null>(null);
  const [permissionNote, setPermissionNote] = useState("");
  const [accessMessage, setAccessMessage] = useState("");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<SearchSort>("relevance");
  const [task, setTask] = useState<SearchTask>("all");
  const [results, setResults] = useState<ModelSearchHit[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [searched, setSearched] = useState(false);
  const [card, setCard] = useState<ModelCard | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [job, setJob] = useState<DownloadJob | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState("");
  const searchScroll = useRef(0);
  const openedFromSearch = useRef(false);
  const cardRef = useRef<ModelCard | null>(null);

  async function refresh() {
    const result = await listLocalModels(apiBase);
    if (!result.ok) {
      setError(result.message);
      return;
    }
    setError("");
    setBundled(result.bundled);
    setModels(result.models);
    setAuth(result.hf_auth === "configured" || result.hf_auth === "invalid" ? result.hf_auth : "none");
  }

  useEffect(() => {
    refresh();
  }, [apiBase]);

  useEffect(() => {
    if (!job || job.status !== "running") return;
    const timer = window.setInterval(async () => {
      const next = await modelDownloadStatus(apiBase, job.id);
      if (!next.ok) {
        setError(next.message);
        return;
      }
      setJob(next);
      if (next.error_code && authFailureNeedsToken(next.error_code)) {
        setAccessMessage(next.error || next.message);
        if (next.error_code === "hub_token_rejected") setAuth("invalid");
      }
      if (next.status === "done") refresh();
    }, 1000);
    return () => window.clearInterval(timer);
  }, [apiBase, job?.id, job?.status]);

  function applyAuth(status: string | undefined) {
    if (status === "configured" || status === "invalid") setAuth(status);
    else if (status === "none") setAuth("none");
  }

  async function search(cursor?: string, nextSort = sort, nextTask = task) {
    const decision = discoverSearchInput(query);
    if (decision.kind === "empty") return;
    if (!cursor && decision.kind === "repository") {
      setError("");
      setAccessMessage("");
      openModel(decision.repo);
      return;
    }
    if (!cursor && decision.kind === "invalid") {
      setError(decision.message);
      setAccessMessage("");
      setResults([]);
      setSearched(false);
      setNextCursor(null);
      setDirectRepo(null);
      setPermissionNote("");
      return;
    }
    if (decision.kind !== "search") return;
    const term = decision.term;
    setBusy(true);
    setProgress(cursor ? "Loading more models." : "Searching Hugging Face.");
    setError("");
    setAccessMessage("");
    if (!cursor) {
      setCard(null);
      setSelected([]);
      setResults([]);
      setNextCursor(null);
      setSearched(false);
      setDirectRepo(null);
      setPermissionNote("");
    }
    try {
      const result = await searchModels(apiBase, term, cursor, nextSort, nextTask);
      if (!result.ok) {
        if (authFailureNeedsToken(result.error)) {
          setAccessMessage(result.message);
          if (result.error === "hub_token_rejected") setAuth("invalid");
        } else setError(result.message);
        return;
      }
      setSearched(true);
      setResults((current) => (cursor ? [...current, ...result.results.filter((hit) => !current.some((row) => row.repo === hit.repo))] : result.results));
      setNextCursor(result.next_cursor);
      setDirectRepo(result.direct_repo && !result.results.some((hit) => hit.repo === result.direct_repo) ? result.direct_repo : null);
      setPermissionNote(result.permission_note || "");
      applyAuth(result.hf_auth);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The model search did not finish.");
    } finally {
      setBusy(false);
      setProgress("");
    }
  }

  useEffect(() => {
    cardRef.current = card;
  }, [card]);

  useEffect(() => {
    const previous = window.history.scrollRestoration;
    window.history.scrollRestoration = "manual";
    return () => {
      window.history.scrollRestoration = previous;
    };
  }, []);

  function openModel(repoName: string) {
    searchScroll.current = window.scrollY;
    openedFromSearch.current = true;
    go(modelHash(repoName));
  }

  function backToSearch() {
    if (openedFromSearch.current) {
      window.history.back();
      return;
    }
    go("#/library/huggingface");
  }

  useEffect(() => {
    if (!repo || cardRef.current?.repo === repo) {
      if (!repo) {
        setBusy(false);
        setProgress("");
      }
      return;
    }
    let cancelled = false;
    setBusy(true);
    setProgress("Opening the model card.");
    setSelected([]);
    setError("");
    setAccessMessage("");
    void (async () => {
      try {
        const result = await discoverModel(apiBase, repo);
        if (cancelled) return;
        if (!result.ok) {
          if (authFailureNeedsToken(result.error)) {
            setAccessMessage(result.message);
            if (result.error === "hub_token_rejected") setAuth("invalid");
          } else setError(result.message);
          setCard(null);
          return;
        }
        setCard(result);
        applyAuth(result.hf_auth);
      } catch (caught) {
        if (!cancelled) setError(caught instanceof Error ? caught.message : "The model lookup did not finish.");
      } finally {
        if (!cancelled) {
          setBusy(false);
          setProgress("");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [apiBase, repo]);

  useLayoutEffect(() => {
    if (section !== "discover" || path) return;
    if (repo) {
      window.scrollTo(0, 0);
      return;
    }
    window.scrollTo(0, searchScroll.current);
  }, [repo, section, path]);

  async function download(mode: "selected" | "complete") {
    if (!card) return;
    const files = mode === "complete" ? card.checkpoint?.files ?? [] : selected;
    if (mode === "complete" && (!card.checkpoint?.identifiable || files.length === 0)) return;
    if (mode === "selected" && files.length === 0) return;
    if (!enoughFreeSpace(bytesStillNeeded(files, card.files), card.storage?.free_bytes)) {
      setError("There is not enough free space for this download. Nothing was downloaded.");
      return;
    }
    if (mode === "complete") setSelected(files);
    setBusy(true);
    setError("");
    setAccessMessage("");
    try {
      const result = await startModelDownload(apiBase, card.repo, files, card.revision, mode);
      if (!result.ok) {
        if (authFailureNeedsToken(result.error)) {
          setAccessMessage(result.message);
          if (result.error === "hub_token_rejected") setAuth("invalid");
        } else setError(result.message);
        return;
      }
      setJob(result);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The download did not start.");
    } finally {
      setBusy(false);
    }
  }

  const typedSearch = discoverSearchInput(query);
  const shown = hfSectionAuth(auth);
  const downloaded = models;
  const packaged = libraryBundledModels(bundled);
  const huggingface = path != null || section === "discover" || section === "downloaded" || section === "import" || section === "token";
  return (
    <>
      <h1>Model Library</h1>
      <p className="lede">Books keeps the research experiments. Hugging Face supplies accessible weights. Cloud Models lists account models and does not run inference.</p>
      <LibraryNav section={section} />
      {huggingface ? <HuggingFaceNav section={path ? null : section} /> : null}
      {section === "providers" || section === "models" || section === "capabilities" ? <CloudNav section={section} /> : null}
      {error && !(section === "discover") ? <p className="error">{error}</p> : null}
      {path ? <FileInspection apiBase={apiBase} path={path} models={models} onInspected={() => void refresh()} /> : null}
      {!path && section === "downloaded" ? (
        <section className="card">
          <h2>Downloaded</h2>
          <p className="note">These files were downloaded from Hugging Face into the Inspector library. Inspection reads them on this computer and does not require an HF Token.</p>
          <DownloadedTable
            models={downloaded}
            apiBase={apiBase}
            activeRepo={job?.status === "running" ? job.repo : ""}
            onChanged={() => void refresh()}
          />
        </section>
      ) : null}
      {!path && section === "import" ? <ImportLocal models={packaged} /> : null}
      {!path && (section === "providers" || section === "models" || section === "capabilities") ? (
        <CloudSection apiBase={apiBase} section={section} cloudProvider={cloudProvider} cloudModel={cloudModel} />
      ) : null}
      {!path && section === "token" ? <HuggingFaceToken apiBase={apiBase} onChanged={() => void refresh()} /> : null}
      {!path && section === "discover" && repo ? (
        <ModelDetails
          repo={repo}
          card={card && card.repo === repo ? card : null}
          auth={auth}
          selected={selected}
          setSelected={setSelected}
          busy={busy}
          progress={progress}
          accessMessage={accessMessage}
          job={job}
          error={error}
          onBack={backToSearch}
          onDownload={download}
          onCancel={async () => {
            if (!job) return;
            const next = await cancelModelDownload(apiBase, job.id);
            if (next.ok) setJob(next);
            else setError(next.message);
          }}
        />
      ) : null}
      {!path && section === "discover" && !repo ? (
      <section className="card">
        <h2>Discover</h2>
        <p className="limits">Search does not download a model. View opens the file list. A download starts only after files are selected. A saved HF Token also includes private repositories that token can access. Public models stay available when no token is configured or the saved token is invalid.</p>
        {accessMessage ? <AuthAccess message={accessMessage} action={hfSectionAuth(auth).action} /> : null}
        <div className="search-bar">
          <label htmlFor="model-search">Search Hugging Face Models</label>
          <div className="row search-actions">
            <input
              id="model-search"
              type="text"
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                setError("");
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter") void search();
              }}
              placeholder="Search by model name, repository ID, or URL"
            />
            <button type="button" disabled={busy || typedSearch.kind === "empty" || typedSearch.kind === "invalid"} onClick={() => void search()}>
              {busy ? "Searching…" : "Search"}
            </button>
            <button type="button" onClick={() => go("#/library/huggingface/token")}>
              {shown.action}
            </button>
          </div>
          <p className="search-auth">
            Authentication: {shown.access}
            {shown.access === "Connected" ? <span className="hf-ok"> ✓</span> : null}
          </p>
          {typedSearch.kind === "invalid" ? <p className="error">{typedSearch.message}</p> : error ? <p className="error">{error}</p> : null}
          <p className="note">Examples: Qwen, Qwen/Qwen3-Coder-30B-A3B-Instruct</p>
          <div className="row">
            <label htmlFor="search-sort">Sort</label>
            <select
              id="search-sort"
              value={sort}
              disabled={busy}
              onChange={(event) => {
                const next = event.target.value as SearchSort;
                setSort(next);
                if (discoverSearchInput(query).kind === "search") void search(undefined, next, task);
              }}
            >
              <option value="relevance">Relevance</option>
              <option value="downloads">Most Downloads</option>
              <option value="updated">Recently Updated</option>
            </select>
            <label htmlFor="search-task">Model type</label>
            <select
              id="search-task"
              value={task}
              disabled={busy}
              onChange={(event) => {
                const next = event.target.value as SearchTask;
                setTask(next);
                if (discoverSearchInput(query).kind === "search") void search(undefined, sort, next);
              }}
            >
              <option value="all">All</option>
              <option value="text">Text Generation</option>
              <option value="image">Image Generation</option>
              <option value="other">Other</option>
            </select>
          </div>
          <p className="note">
            A repository whose publisher matches the search name is listed before community models. Community models stay in the results. Private repositories this token can access are included when a token is saved.
          </p>
        </div>
        {progress ? <WorkProgress label={progress} /> : null}
        {permissionNote ? <p className="note">{permissionNote}</p> : null}
        {directRepo ? (
          <p className="note">
            {directRepo} is not in these results.{" "}
            <button type="button" onClick={() => openModel(directRepo)}>
              Open repository
            </button>
          </p>
        ) : null}
        {searched && results.length === 0 && !busy ? (
          <p className="note">{SEARCH_EMPTY_RESULTS}</p>
        ) : null}
        {results.length > 0 ? (
          <>
            <div className="results-scroll">
              <table className="results">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Task</th>
                    <th>Architecture</th>
                    <th>Parameters</th>
                    <th>Downloads</th>
                    <th>License</th>
                    <th>Access</th>
                    <th className="view"><span className="visually-hidden">View</span></th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((hit) => (
                    <tr key={hit.repo}>
                      <td className="repo">
                        <span className="repo-line">
                          <span className="repo-id" title={hit.repo}>{hit.repo}</span>
                          {hit.official ? <span className="official">Official</span> : null}
                          {hit.private ? <span className="private-label">Private</span> : null}
                        </span>
                      </td>
                      <td>{shownOrUnknown(hit.task_label || taskLabel(hit.pipeline))}</td>
                      <td>{shownOrUnknown(hit.architecture)}</td>
                      <td>{shownOrUnknown(hit.parameters_label)}</td>
                      <td>{shownOrUnknown(hit.downloads_label)}</td>
                      <td>{shownOrUnknown(hit.license)}</td>
                      <td>{hit.access === "public" ? "Public" : hit.access === "private" ? "Private" : hit.access === "gated" ? "Gated" : "Unknown"}</td>
                      <td className="view">
                        <button type="button" disabled={busy} onClick={() => openModel(hit.repo)}>
                          View
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="note">Unknown access means Hugging Face did not report whether the repository is public, gated, or private.</p>
            {nextCursor ? (
              <div className="row">
                <button type="button" disabled={busy} onClick={() => void search(nextCursor)}>
                  Load More
                </button>
              </div>
            ) : null}
          </>
        ) : null}
        {job ? <DownloadStatus job={job} auth={auth} onCancel={async () => {
          const next = await cancelModelDownload(apiBase, job.id);
          if (next.ok) setJob(next);
          else setError(next.message);
        }} /> : null}
      </section>
      ) : null}
    </>
  );
}

function ModelDetails({
  repo,
  card,
  auth,
  selected,
  setSelected,
  busy,
  progress,
  accessMessage,
  job,
  error,
  onBack,
  onDownload,
  onCancel,
}: {
  repo: string;
  card: ModelCard | null;
  auth: HfAuthStatus;
  selected: string[];
  setSelected: (value: string[] | ((current: string[]) => string[])) => void;
  busy: boolean;
  progress: string;
  accessMessage: string;
  job: DownloadJob | null;
  error: string;
  onBack: () => void;
  onDownload: (mode: "selected" | "complete") => void;
  onCancel: () => void;
}) {
  const hasGguf = card?.files.some((file) => file.name.toLowerCase().endsWith(".gguf")) ?? false;
  const hasLitert = card?.files.some((file) => file.name.toLowerCase().endsWith(".litertlm")) ?? false;
  const inspectable = card ? hasInspectableSafetensors(card.files) : false;
  const supportedCount = card ? card.files.filter((file) => file.compatible).length : 0;
  const formats = card ? availableFormats(card.files) : [];
  const summary = card ? downloadSummary(card.checkpoint, selected, card.files) : null;
  const downloadAllowed = card ? gatedDownloadAllowed(auth, card.private || card.gated) : false;
  const freeBytes = card?.storage?.free_bytes;
  const completeNeeded = card?.checkpoint?.identifiable ? bytesStillNeeded(card.checkpoint.files, card.files) : null;
  const selectedNeeded = card ? bytesStillNeeded(selected, card.files) : null;
  const completeFits = enoughFreeSpace(completeNeeded, freeBytes);
  const selectedFits = enoughFreeSpace(selectedNeeded, freeBytes);
  const access = card ? (card.private ? "Private" : card.gated ? "Gated" : "Public") : "";
  return (
    <section className="card">
      <button type="button" className="back" onClick={onBack}>
        ← Back to Search Results
      </button>
      <h2 className="model-title">{repo}</h2>
      {progress ? <WorkProgress label={progress} /> : null}
      {accessMessage ? <AuthAccess message={accessMessage} action={hfSectionAuth(auth).action} /> : null}
      {error ? <p className="error">{error}</p> : null}
      {card ? (
        <>
          <h3>Model Overview</h3>
          <p className="meta">{plainDescription(card.description)}</p>
          <p className="meta">Revision {card.revision}</p>
          {card.auth_note && !(card.private || card.gated) ? <p className="note">{card.auth_note}</p> : null}
          <p className="limits">{card.limits}</p>
          <h3>Architecture and Metadata</h3>
          <p className="meta">Task {shownOrUnknown(taskLabel(card.pipeline))}</p>
          <p className="meta">Architecture {shownOrUnknown(architectureLabel(card.model_type, card.architectures))}</p>
          <p className="meta">License {card.license || "Unknown"}</p>
          <p className="meta">Library {card.library || "Unknown"}</p>
          <p className="meta">Access {access}</p>
          <p className="meta">Authentication {modelAuthenticationLabel(card.hf_auth || auth)}</p>
          <CapabilityList capabilities={card.capabilities} />
          <h3>Available Files</h3>
          {hasGguf ? (
            <p className="note">GGUF files are not currently supported for tensor inspection. AI Model Inspector supports compatible Safetensors checkpoints.</p>
          ) : null}
          {hasLitert ? (
            <p className="note">LiteRT-LM files are not currently supported for tensor inspection. AI Model Inspector supports compatible Safetensors checkpoints.</p>
          ) : null}
          {!inspectable ? (
            <>
              <p className="meta">Supported files {supportedCount}</p>
              <p className="meta">Available formats {formats.length > 0 ? formats.join(", ") : "None"}</p>
              <p className="meta">Inspection Not supported</p>
              <p className="meta">Download Disabled</p>
            </>
          ) : null}
          <div className="results-scroll">
            <table>
              <thead>
                <tr>
                  <th></th>
                  <th>File</th>
                  <th>Size</th>
                  <th>Compatible</th>
                </tr>
              </thead>
              <tbody>
                {card.files.map((file) => (
                  <tr key={file.name} className={file.compatible ? "compatible" : "unsupported"}>
                    <td>
                      <input
                        type="checkbox"
                        disabled={!file.compatible}
                        checked={selected.includes(file.name)}
                        aria-label={`Select ${file.name}`}
                        onChange={(event) => {
                          setSelected((current) =>
                            event.target.checked ? [...current, file.name] : current.filter((name) => name !== file.name),
                          );
                        }}
                      />
                    </td>
                    <td className="path">{file.name}</td>
                    <td>{file.size_human}</td>
                    <td>{file.compatible ? "yes" : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <h3>Download Selection</h3>
          <p className="note">Nothing is downloaded until you choose one of the actions below.</p>
          {summary ? (
            <>
              <p className="meta">{summary.selectedLine}</p>
              <p className="meta">{summary.sizeLine}</p>
              <p className="meta">{summary.checkpointLine}</p>
            </>
          ) : null}
          {card.checkpoint?.identifiable ? (
            <p className="note">
              Complete model: {card.checkpoint.shard_count} weight {card.checkpoint.shard_count === 1 ? "shard" : "shards"}
              {card.checkpoint.total_label ? `, approximately ${card.checkpoint.total_label}` : ""}.
            </p>
          ) : inspectable ? (
            <p className="note">{card.checkpoint?.reason || "A complete checkpoint cannot be identified from the files in this repository."}</p>
          ) : null}
          <p className="meta">Free space: {card.storage?.free_label ?? "Unknown"}</p>
          {completeNeeded != null && !completeFits ? (
            <p className="note">
              The complete model does not fit. It needs approximately {formatDownloadSize(downloadSpaceRequired(completeNeeded))} of free space.
            </p>
          ) : null}
          {selected.length > 0 && selectedNeeded != null && !selectedFits ? (
            <p className="note">
              The selected files do not fit. They need approximately {formatDownloadSize(downloadSpaceRequired(selectedNeeded))} of free space.
            </p>
          ) : null}
          {inspectable && (card.private || card.gated) ? (
            downloadAllowed ? (
              <p className="note">Download uses the stored HF Token. Access to this repository still has to be granted on the Hub.</p>
            ) : (
              <AuthAccess
                message="Download is disabled for this private or gated repository because there is no valid HF Token. Public models can still be searched and downloaded."
                action={hfSectionAuth(auth).action}
              />
            )
          ) : null}
          {selected.length > 0 ? <p className="note">Clear file selections to download the complete model.</p> : null}
          <div className="row">
            <button
              type="button"
              disabled={busy || selected.length > 0 || !downloadAllowed || !card.checkpoint?.identifiable || !completeFits}
              title={
                selected.length > 0
                  ? "Clear file selections to download the complete model."
                  : !completeFits
                    ? "There is not enough free space for the complete model."
                    : undefined
              }
              onClick={() => onDownload("complete")}
            >
              Download Complete Model
            </button>
            <button
              type="button"
              disabled={busy || selected.length === 0 || !downloadAllowed || !selectedFits}
              title={selected.length > 0 && !selectedFits ? "There is not enough free space for the selected files." : undefined}
              onClick={() => onDownload("selected")}
            >
              Download Selected Files
            </button>
          </div>
          {job ? <DownloadStatus job={job} auth={auth} onCancel={onCancel} /> : null}
        </>
      ) : null}
    </section>
  );
}

function plainDescription(value: string | null): string {
  if (!value || !value.trim()) return "No description on the card.";
  const text = value
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, "\"")
    .replace(/\s+/g, " ")
    .trim();
  return text || "No description on the card.";
}

function DownloadStatus({ job, auth, onCancel }: { job: DownloadJob; auth: HfAuthStatus; onCancel: () => void }) {
  return (
    <div>
      <h3>Download</h3>
      {job.status === "running" ? (
        <WorkProgress {...downloadProgress(job)} />
      ) : (
        <p className="meta">
          {job.status} · {job.percent}% · {job.message}
        </p>
      )}
      {job.error ? (
        job.error_code && authFailureNeedsToken(job.error_code) ? (
          <AuthAccess message={job.error} action={hfSectionAuth(auth).action} />
        ) : (
          <p className="error">{job.error}</p>
        )
      ) : null}
      {job.status === "running" ? (
        <button type="button" className="quiet" onClick={onCancel}>
          Cancel
        </button>
      ) : null}
    </div>
  );
}

function LibraryNav({ section }: { section: LibrarySection }) {
  const huggingface = section === "discover" || section === "downloaded" || section === "import" || section === "token";
  const cloud = section === "providers" || section === "models" || section === "capabilities";
  return (
    <div className="subnav">
      <button type="button" className={huggingface ? "on" : ""} onClick={() => go("#/library/huggingface")}>
        Hugging Face
      </button>
      <button type="button" className={cloud ? "on" : ""} onClick={() => go("#/library/cloud/models")}>
        Cloud Models
      </button>
    </div>
  );
}

function HuggingFaceNav({ section }: { section: LibrarySection | null }) {
  const items: { id: LibrarySection; label: string; hash: string }[] = [
    { id: "discover", label: "Discover", hash: "#/library/huggingface" },
    { id: "downloaded", label: "Downloaded", hash: "#/library/huggingface/downloaded" },
    { id: "import", label: "Import Local", hash: "#/library/huggingface/import" },
  ];
  return (
    <div className="hf-bar">
      <div className="subnav minor">
        {items.map((item) => (
          <button type="button" key={item.id} className={section === item.id ? "on" : ""} onClick={() => go(item.hash)}>
            {item.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function CloudNav({ section }: { section: LibrarySection }) {
  const items: { id: LibrarySection; label: string; hash: string }[] = [
    { id: "providers", label: "Providers", hash: "#/library/cloud/providers" },
    { id: "models", label: "Models", hash: "#/library/cloud/models" },
  ];
  return (
    <div className="subnav minor">
      {items.map((item) => (
        <button type="button" key={item.id} className={section === item.id ? "on" : ""} onClick={() => go(item.hash)}>
          {item.label}
        </button>
      ))}
    </div>
  );
}

function ImportLocal({ models }: { models: LocalModel[] }) {
  const [path, setPath] = useState("");
  return (
    <section className="card">
      <h2>Import Local</h2>
      <p className="note">Open a checkpoint that is already on this computer. The file is not copied, and it is read only when it sits inside a configured directory. Local inspection does not require an HF Token.</p>
      <label htmlFor="local-path">Checkpoint path</label>
      <div className="row">
        <input id="local-path" type="text" value={path} onChange={(event) => setPath(event.target.value)} placeholder="/absolute/path/model.safetensors" />
        <button type="button" disabled={!path.trim()} onClick={() => go(`#/file?path=${encodeURIComponent(path.trim())}`)}>
          Inspect
        </button>
      </div>
      <h3>Already on this computer</h3>
      <ModelList models={models} empty="No packaged checkpoint is available outside the Books collection." />
    </section>
  );
}

function CloudPath({
  provider,
  modelId,
  keyPage,
}: {
  provider?: { id: string; name: string } | null;
  modelId?: string;
  keyPage?: boolean;
}) {
  const providerCurrent = Boolean(provider) && !modelId && !keyPage;
  return (
    <p className="crumb">
      <button type="button" onClick={() => go("#/library/cloud/models")}>
        Cloud Models
      </button>
      {provider ? (
        <>
          {" / "}
          {providerCurrent ? (
            provider.name
          ) : (
            <button type="button" onClick={() => go(`#/library/cloud/models/${provider.id}`)}>
              {provider.name}
            </button>
          )}
        </>
      ) : null}
      {keyPage ? " / Key" : null}
      {modelId ? ` / ${modelId}` : null}
    </p>
  );
}

function CloudSection({
  apiBase,
  section,
  cloudProvider,
  cloudModel,
}: {
  apiBase: string;
  section: LibrarySection;
  cloudProvider?: string;
  cloudModel?: string;
}) {
  if (section === "capabilities") return <CloudHome apiBase={apiBase} helpOpen />;
  if (section === "providers" && cloudProvider) return <CloudKeyPage apiBase={apiBase} providerId={cloudProvider} />;
  if (section === "models" && cloudProvider && cloudModel) {
    return <CloudModelPage apiBase={apiBase} providerId={cloudProvider} modelId={cloudModel} />;
  }
  if (section === "models" && cloudProvider) return <CloudCatalogPage apiBase={apiBase} providerId={cloudProvider} />;
  return <CloudHome apiBase={apiBase} />;
}

const CLOUD_PROVIDERS: {
  id: string;
  name: string;
  credential: string;
  href?: string;
  link?: string;
  configurable: boolean;
}[] = [
  {
    id: "openai",
    name: "OpenAI",
    credential: "API Key",
    href: "https://platform.openai.com/api-keys",
    link: "Get Key",
    configurable: true,
  },
  {
    id: "anthropic",
    name: "Anthropic",
    credential: "API Key",
    href: "https://console.anthropic.com/",
    link: "Get Key",
    configurable: true,
  },
  {
    id: "google",
    name: "Google Gemini",
    credential: "API Key",
    href: "https://aistudio.google.com/apikey",
    link: "Get Key",
    configurable: true,
  },
  { id: "hosted", name: "Hosted inference", credential: "Provider-specific", configurable: false },
];

function CloudHome({ apiBase, helpOpen = false }: { apiBase: string; helpOpen?: boolean }) {
  const [providers, setProviders] = useState<CredentialProvider[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const token = await loadControllerCredential();
      if (!token || cancelled) {
        if (!cancelled) setError("Cloud Models needs the local controller credential on this computer. Nothing was saved.");
        return;
      }
      const result = await listCredentials(apiBase, token);
      if (cancelled) return;
      if (result.ok !== true) {
        setError(result.message);
        return;
      }
      setError("");
      setProviders(result.providers);
    })();
    return () => {
      cancelled = true;
    };
  }, [apiBase]);

  return (
    <section className="card">
      <h2>Cloud Models</h2>
      <p className="note">
        Each provider has its own key and connection status. A failure with one provider does not change the others. Saved keys remain in the macOS Keychain and are not shown.
      </p>
      <p className="note">
        View Models lists what that account's API exposes. A listed model does not mean inference, chat, or training is available. This page does not send an inference request, and an API key does not download or inspect proprietary model weights.
      </p>
      {error ? <p className="error">{error}</p> : null}
      <table className="providers cloud">
        <thead>
          <tr>
            <th>Provider</th>
            <th>Authentication</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          {CLOUD_PROVIDERS.filter((item) => item.configurable).map((item) => {
            const stored = providers.find((row) => row.id === item.id);
            const shown = cloudSectionAuth(stored?.status);
            return (
              <tr key={item.id}>
                <td>{item.name}</td>
                <td>
                  {shown.access}
                  {shown.access === "Connected" ? <span className="connected-mark"> ✓</span> : null}
                </td>
                <td>
                  <div className="cloud-actions">
                    <button type="button" onClick={() => go(`#/library/cloud/providers/${item.id}`)}>
                      {shown.action}
                    </button>
                    {shown.models ? (
                      <button type="button" onClick={() => go(`#/library/cloud/models/${item.id}`)}>
                        View Models
                      </button>
                    ) : null}
                    {item.href ? (
                      <a href={item.href} target="_blank" rel="noreferrer">
                        Get Key<span className="visually-hidden"> for {item.name}</span>
                      </a>
                    ) : null}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="note">Get API Key opens the provider's own site. Hosted inference: Not supported yet.</p>
      <AboutCloudInspection key={helpOpen ? "open" : "closed"} open={helpOpen} />
    </section>
  );
}

function CloudKeyPage({ apiBase, providerId }: { apiBase: string; providerId: string }) {
  const provider = CLOUD_PROVIDERS.find((item) => item.id === providerId && item.configurable) || null;
  const [providers, setProviders] = useState<CredentialProvider[]>([]);
  const [draft, setDraft] = useState("");
  const [reveal, setReveal] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState("");
  const controllerToken = useRef<string | null>(null);
  const selected = providers.find((item) => item.id === providerId);
  const shown = cloudSectionAuth(selected?.status);

  async function load() {
    const token = await loadControllerCredential();
    controllerToken.current = token;
    if (!token) {
      setError("Cloud Models needs the local controller credential on this computer. Nothing was saved.");
      return;
    }
    const result = await listCredentials(apiBase, token);
    if (result.ok !== true) {
      setError(result.message);
      return;
    }
    setError("");
    setProviders(result.providers);
  }

  useEffect(() => {
    if (provider) void load();
  }, [apiBase, providerId]);

  async function withToken(
    label: string,
    action: (token: string) => Promise<CredentialProvider | { ok: false; message: string }>,
  ) {
    const token = controllerToken.current || (await loadControllerCredential());
    if (!token) {
      setError("Cloud Models needs the local controller credential on this computer. Nothing was saved.");
      return;
    }
    controllerToken.current = token;
    setBusy(label);
    setError("");
    setNotice("");
    try {
      const result = await action(token);
      if (result.ok !== true) {
        setError(result.message);
        return;
      }
      setDraft("");
      setReveal(false);
      setNotice(result.message || "");
      await load();
    } finally {
      setBusy("");
    }
  }

  if (!provider) {
    return (
      <section className="card">
        <h2>Cloud Models</h2>
        <p className="error">That provider is not part of Cloud Models.</p>
        <CloudPath />
      </section>
    );
  }

  return (
    <section className="card">
      <CloudPath provider={provider} keyPage />
      <h2>{shown.action}</h2>
      <div className="credential-facts">
        <div>
          <span>Provider</span>
          <strong>{provider.name}</strong>
        </div>
        <div>
          <span>Credential</span>
          <strong>API Key</strong>
        </div>
        <div>
          <span>Get API Key</span>
          {provider.href ? (
            <a href={provider.href} target="_blank" rel="noreferrer">
              {provider.link || "Get Key"}
            </a>
          ) : (
            <strong>—</strong>
          )}
        </div>
        <div>
          <span>Status</span>
          <strong>
            {shown.access}
            {shown.access === "Connected" ? <span className="connected-mark"> ✓</span> : null}
          </strong>
        </div>
      </div>
      <p className="note">Saved keys remain in the macOS Keychain. The saved key is not shown. Show key reveals only the text in this box. This key is separate from the HF Token and from the other cloud providers.</p>
      <p className="note">Test Connection asks {provider.name} to accept the key. A failed test does not remove it. View Models appears after the provider accepts the key.</p>
      <p className="note">A model list does not run inference, chat, or training, and it does not reveal internal weights.</p>
      {error ? <p className="error">{error}</p> : null}
      {notice ? <p className="note">{notice}</p> : null}
      <label htmlFor={`cloud-key-${provider.id}`}>API Key</label>
      <div className="secret-field">
        <input
          id={`cloud-key-${provider.id}`}
          type={reveal ? "text" : "password"}
          autoComplete="off"
          spellCheck={false}
          value={draft}
          onChange={(event) => {
            const next = event.target.value;
            setDraft(next);
            if (!next) setReveal(false);
          }}
          placeholder="Paste a key"
        />
        <button
          type="button"
          className="quiet"
          aria-pressed={reveal}
          aria-controls={`cloud-key-${provider.id}`}
          disabled={!draft}
          onClick={() => setReveal((current) => !current)}
        >
          {reveal ? "Hide key" : "Show key"}
        </button>
      </div>
      {selected?.saved ? <p className="note">Saved in the macOS Keychain on this Mac. The key is not shown.</p> : null}
      <div className="row">
        <button
          type="button"
          disabled={busy !== "" || !draft.trim()}
          onClick={() =>
            void withToken(`Saving the ${provider.name} key on this computer.`, (token) =>
              saveCloudCredential(apiBase, token, provider.id, draft.trim()),
            )
          }
        >
          {selected?.saved ? "Replace" : "Save"}
        </button>
        <button
          type="button"
          className="quiet"
          disabled={busy !== "" || !selected || selected.status === "not_configured"}
          onClick={() =>
            void withToken(`Testing the saved ${provider.name} key.`, (token) => testCloudCredential(apiBase, token, provider.id))
          }
        >
          Test Connection
        </button>
        <button
          type="button"
          className="quiet"
          disabled={busy !== "" || !selected?.saved}
          onClick={() =>
            void withToken(`Removing the saved ${provider.name} key from this Mac.`, (token) =>
              removeCloudCredential(apiBase, token, provider.id),
            )
          }
        >
          Remove
        </button>
        {shown.models ? (
          <button type="button" className="quiet" onClick={() => go(`#/library/cloud/models/${provider.id}`)}>
            View Models
          </button>
        ) : null}
      </div>
      {busy ? <WorkProgress label={busy} /> : null}
    </section>
  );
}

function CloudCatalogPage({ apiBase, providerId }: { apiBase: string; providerId: string }) {
  const provider = CLOUD_PROVIDERS.find((item) => item.id === providerId && item.configurable) || null;
  const [catalog, setCatalog] = useState<CloudCatalog | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!provider) return;
    let cancelled = false;
    setCatalog(null);
    setError("");
    setBusy(true);
    void (async () => {
      const token = await loadControllerCredential();
      if (!token || cancelled) {
        if (!cancelled) {
          setError("Cloud Models needs the local controller credential on this computer. Nothing was requested.");
          setBusy(false);
        }
        return;
      }
      const result = await listCloudModels(apiBase, token, provider.id);
      if (cancelled) return;
      if (result.ok !== true) setError(result.message);
      else setCatalog(result);
      setBusy(false);
    })();
    return () => {
      cancelled = true;
    };
  }, [apiBase, providerId]);

  if (!provider) {
    return (
      <section className="card">
        <h2>Cloud Models</h2>
        <p className="error">That provider is not part of Cloud Models.</p>
        <CloudPath />
      </section>
    );
  }

  return (
    <section className="card">
      <CloudPath provider={provider} />
      <div className="row">
        <button type="button" className="quiet" onClick={() => go(`#/library/cloud/providers/${provider.id}`)}>
          Manage Key
        </button>
      </div>
      <h2>{provider.name} models</h2>
      {error ? <p className="error">{error}</p> : null}
      {busy ? <WorkProgress label={`Retrieving the ${provider.name} model list.`} /> : null}
      {catalog ? (
        <>
          <p className="note">{catalog.note}</p>
          {catalog.truncated ? <p className="note">The list was cut off after the first pages returned by the provider.</p> : null}
          {catalog.models.length === 0 ? (
            <p className="meta">This account did not expose any models.</p>
          ) : (
            <>
              <p className="meta">
                {catalog.models.length} {catalog.models.length === 1 ? "model" : "models"}
              </p>
              <div className="results-scroll">
                <table className="catalog">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Reported by the provider</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...catalog.models]
                      .sort((a, b) => a.id.localeCompare(b.id, "en"))
                      .map((model) => (
                        <tr key={model.id}>
                          <td>{model.display_name ? `${model.display_name} (${model.id})` : model.id}</td>
                          <td>{model.detail || "—"}</td>
                          <td>
                            <button type="button" className="quiet" onClick={() => go(cloudModelHash(provider.id, model.id))}>
                              View Details
                            </button>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </>
      ) : null}
    </section>
  );
}

function RecordTable({ rows }: { rows: SourcedValue[] }) {
  return (
    <div className="results-scroll">
      <table className="record">
        <thead>
          <tr>
            <th>Field</th>
            <th>Value</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.field}>
              <td>{row.field}</td>
              <td>{row.value}</td>
              <td>{row.source}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CloudModelPage({ apiBase, providerId, modelId }: { apiBase: string; providerId: string; modelId: string }) {
  const provider = CLOUD_PROVIDERS.find((item) => item.id === providerId && item.configurable) || null;
  const [catalog, setCatalog] = useState<CloudCatalog | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!provider) return;
    let cancelled = false;
    setCatalog(null);
    setError("");
    setBusy(true);
    void (async () => {
      const token = await loadControllerCredential();
      if (!token || cancelled) {
        if (!cancelled) {
          setError("Cloud Models needs the local controller credential on this computer. Nothing was requested.");
          setBusy(false);
        }
        return;
      }
      const result = await listCloudModels(apiBase, token, provider.id);
      if (cancelled) return;
      if (result.ok !== true) setError(result.message);
      else setCatalog(result);
      setBusy(false);
    })();
    return () => {
      cancelled = true;
    };
  }, [apiBase, providerId]);

  if (!provider) {
    return (
      <section className="card">
        <h2>Cloud Models</h2>
        <p className="error">That provider is not part of Cloud Models.</p>
        <CloudPath />
      </section>
    );
  }

  const model = catalog?.models.find((item) => item.id === modelId) || null;
  const details = model ? cloudModelDetails(provider.name, model) : null;

  return (
    <section className="card">
      <CloudPath provider={provider} modelId={modelId} />
      <div className="row">
        <button type="button" className="quiet" onClick={() => go(`#/library/cloud/providers/${provider.id}`)}>
          Manage Key
        </button>
      </div>
      <h2>{model?.display_name || modelId}</h2>
      {model?.display_name ? <p className="meta">{modelId}</p> : null}
      <p className="note">
        This page records what the {provider.name} model list reports for this account. It does not run inference, and it does not inspect internal weights.
      </p>
      {error ? <p className="error">{error}</p> : null}
      {busy ? <WorkProgress label={`Retrieving the ${provider.name} model record.`} /> : null}
      {catalog && !model ? <p className="meta">This model was not in the current account list.</p> : null}
      {details ? (
        <>
          <h3>Model identity</h3>
          <RecordTable rows={details.identity} />
          <h3>Published capabilities</h3>
          <p className="note">A capability appears here only when this provider's model list reports it.</p>
          <RecordTable rows={details.capabilities} />
          <h3>Internal model inspection</h3>
          <p className="note">Weights and tensors: Not accessible through the {provider.name} API.</p>
          <RecordTable rows={details.internal} />
        </>
      ) : null}
    </section>
  );
}

function AuthAccess({ message, action }: { message: string; action: "Configure Token" | "Manage Token" }) {
  return (
    <p className="error">
      {message}{" "}
      <button type="button" className="linkish" onClick={() => go("#/library/huggingface/token")}>
        {action}
      </button>
    </p>
  );
}

function AboutCloudInspection({ open = false }: { open?: boolean }) {
  const [expanded, setExpanded] = useState(open);
  const rows = [
    ["Model information", "Yes", "When exposed"],
    ["Architecture details", "When available", "Usually limited"],
    ["Tensor inspection", "With accessible weights", "Generally unavailable"],
    ["Weight magnitude", "With compatible checkpoints", "Generally unavailable"],
    ["Weight direction", "With compatible checkpoints", "Generally unavailable"],
    ["Behavioral evaluation", "A separate instrument, not weight inspection", "A separate instrument, not weight inspection"],
  ];
  return (
    <details
      className="technical about-cloud"
      open={expanded}
      onToggle={(event) => setExpanded(event.currentTarget.open)}
    >
      <summary>About Cloud Inspection</summary>
      <p className="note">Hugging Face can supply weights. An API-only cloud model cannot. An API key does not make those weights available. Behavioral comparison is not a measurement of internal weights.</p>
      <table>
        <thead>
          <tr>
            <th>Capability</th>
            <th>Hugging Face open weights</th>
            <th>API-only cloud model</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row[0]}>
              <td>{row[0]}</td>
              <td>{row[1]}</td>
              <td>{row[2]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}

function CapabilityList({ capabilities }: { capabilities?: LocalModel["capabilities"] }) {
  if (!capabilities) return null;
  return (
    <p className="meta">
      Model information: {capabilities.model_information ? "yes" : "not exposed"}. Architecture:{" "}
      {capabilities.architecture_details ? "when the card or file provides it" : "not from this access method"}. Tensor inspection:{" "}
      {capabilities.tensor_inspection ? "yes" : "after a compatible file is on this computer"}.
    </p>
  );
}

function DownloadedTable({
  models,
  apiBase,
  activeRepo,
  onChanged,
}: {
  models: LocalModel[];
  apiBase: string;
  activeRepo: string;
  onChanged: () => void;
}) {
  const [pending, setPending] = useState<null | { scope: "file" | "model"; model: LocalModel; fileName?: string; bytes: number }>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  if (models.length === 0) return <p>No Hugging Face files have been downloaded yet.</p>;

  function ask(scope: "file" | "model", model: LocalModel, fileName?: string) {
    const bytes =
      scope === "model"
        ? model.storage_bytes ?? model.files.reduce((sum, file) => sum + (file.size_bytes ?? 0), 0)
        : model.files.find((file) => file.name === fileName)?.size_bytes ?? 0;
    setError("");
    setPending({ scope, model, fileName, bytes });
  }

  async function confirmDelete() {
    if (!pending) return;
    setBusy(true);
    setError("");
    const result = await deleteDownloaded(apiBase, pending.model.repo, pending.scope, pending.fileName);
    setBusy(false);
    if (!result.ok) {
      setError(result.message);
      return;
    }
    setNotice(
      result.freed_bytes === pending.bytes
        ? ""
        : `Freed ${result.freed_bytes.toLocaleString()} bytes. The estimate was ${pending.bytes.toLocaleString()} bytes.`,
    );
    setPending(null);
    onChanged();
  }

  return (
    <>
      {notice ? <p className="note">{notice}</p> : null}
      <div className="results-scroll">
        <table className="downloaded">
          <thead>
            <tr>
              <th>Model</th>
              <th>Downloaded files</th>
              <th>Size</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => {
              const weights = model.files.filter((file) => file.inspectable);
              const weightBytes = weights.reduce((sum, file) => sum + (file.size_bytes ?? 0), 0);
              const downloading = activeRepo === model.repo;
              const title = model.repo.split("/")[1] || model.repo;
              return (
                <tr key={`${model.repo}-${model.revision}`}>
                  <td>
                    <strong>{title}</strong>
                    <p className="meta">{model.repo}</p>
                    {model.checkpoint_label ? <p className="meta">{model.checkpoint_label}</p> : null}
                    {model.integrity_label ? <p className="meta">{model.integrity_label}</p> : null}
                    {model.inspection_label ? <p className="meta">{model.inspection_label}</p> : null}
                  </td>
                  <td>{downloadedFilesLabel(model)}</td>
                  <td>{weights.length ? formatDownloadSize(weightBytes) : model.storage_label || "size unknown"}</td>
                  <td>
                    {downloading ? <p className="note">A download for this model is still running. Cancel it before deleting.</p> : null}
                    {weights.map((file) => (
                      <div className="row" key={file.path || file.name}>
                        <button type="button" onClick={() => go(`#/file?path=${encodeURIComponent(file.path)}`)}>
                          Inspect
                        </button>
                        <button type="button" className="delete" disabled={downloading} onClick={() => ask("file", model, file.name)}>
                          Delete
                        </button>
                      </div>
                    ))}
                    <button type="button" className="delete" disabled={downloading} onClick={() => ask("model", model)}>
                      Delete Entire Model
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {pending ? (
        <div className="dialog-backdrop">
          <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="delete-title">
            <h2 id="delete-title">{pending.scope === "model" ? "Delete entire model" : "Delete file"}</h2>
            <p className="note">This removes a local copy from the Inspector library. The Hugging Face repository is not changed.</p>
            <p className="meta">Repository {pending.model.repo}</p>
            <p className="meta">{pending.scope === "model" ? "Every local file for this model" : `File ${pending.fileName}`}</p>
            <p className="meta">Disk space freed {formatDownloadSize(pending.bytes)} ({pending.bytes.toLocaleString()} bytes)</p>
            {error ? <p className="error">{error}</p> : null}
            <div className="row">
              <button type="button" disabled={busy} onClick={() => setPending(null)}>
                Cancel
              </button>
              <button type="button" className="delete" disabled={busy} onClick={() => void confirmDelete()}>
                {pending.scope === "model" ? "Delete Entire Model" : "Delete file"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}

function ModelList({ models, empty }: { models: LocalModel[]; empty?: string }) {
  if (models.length === 0) return <p>{empty || "No models are listed yet."}</p>;
  return (
    <div className="cards">
      {models.map((model) => (
        <article key={`${model.repo}-${model.revision}`}>
          <h3>{model.title || model.repo}</h3>
          <p className="meta">{model.repo}</p>
          <p className="meta">Revision {model.revision || "unknown"}</p>
          {model.checkpoint_label ? <p className="meta">{model.checkpoint_label}</p> : null}
          {model.files.filter((file) => file.inspectable).map((file) => (
            <div className="row" key={file.path}>
              <span>
                {file.name} · {file.size_human || "size unknown"}
              </span>
              <button type="button" onClick={() => go(`#/file?path=${encodeURIComponent(file.path)}`)}>
                Inspect
              </button>
            </div>
          ))}
        </article>
      ))}
    </div>
  );
}

function HuggingFaceToken({ apiBase, onChanged }: { apiBase: string; onChanged: () => void }) {
  const [providers, setProviders] = useState<CredentialProvider[]>([]);
  const [precedence, setPrecedence] = useState("");
  const [draft, setDraft] = useState("");
  const [reveal, setReveal] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState("");
  const controllerToken = useRef<string | null>(null);

  async function load() {
    const token = await loadControllerCredential();
    controllerToken.current = token;
    if (!token) {
      setError("HF Token needs the local controller credential on this computer. Nothing was saved.");
      return;
    }
    const result = await listCredentials(apiBase, token);
    if (result.ok !== true) {
      setError(result.message);
      return;
    }
    setError("");
    setProviders(result.providers);
    setPrecedence(result.precedence);
  }

  useEffect(() => {
    void load();
  }, [apiBase]);

  const selected = providers.find((provider) => provider.id === "huggingface");

  async function withToken(
    label: string,
    action: (token: string) => Promise<CredentialProvider | { ok: false; message: string }>,
  ) {
    const token = controllerToken.current || (await loadControllerCredential());
    if (!token) {
      setError("HF Token needs the local controller credential on this computer. Nothing was saved.");
      return;
    }
    controllerToken.current = token;
    setBusy(label);
    setError("");
    setNotice("");
    try {
      const result = await action(token);
      if (result.ok !== true) {
        setError(result.message);
        return;
      }
      setDraft("");
      setReveal(false);
      setNotice(result.message || "");
      onChanged();
      await load();
    } finally {
      setBusy("");
    }
  }

  return (
    <section className="card">
      <h2>{selected && selected.status !== "not_configured" ? "Manage Token" : "Configure Token"}</h2>
      <div className="credential-facts">
        <div>
          <span>Provider</span>
          <strong>Hugging Face</strong>
        </div>
        <div>
          <span>Credential</span>
          <strong>HF Token</strong>
        </div>
        <div>
          <span>Get Token</span>
          <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
            Get HF Token
          </a>
        </div>
        <div>
          <span>Status</span>
          <strong>{hfSectionAuth(selected?.status).access}</strong>
        </div>
      </div>
      <p className="note">An HF Token is optional for public models. Private or gated repositories may require authentication and access approval.</p>
      <p className="note">Test Connection asks Hugging Face to accept the token. After it succeeds, search includes private repositories this token can access.</p>
      <p className="note">Saved tokens remain in your macOS Keychain.</p>
      <p className="limits">{precedence || "INSPECTOR_HF_TOKEN, when set for the API process, is used before a token saved in the macOS Keychain. A failed connection test does not remove a saved token."}</p>
      {error ? <p className="error">{error}</p> : null}
      {notice ? <p className="note">{notice}</p> : null}
      <div className="label-row">
        <label htmlFor="hf-token">HF Token</label>
        <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
          Get HF Token
        </a>
      </div>
      <div className="secret-field">
        <input
          id="hf-token"
          type={reveal ? "text" : "password"}
          autoComplete="off"
          spellCheck={false}
          value={draft}
          onChange={(event) => {
            const next = event.target.value;
            setDraft(next);
            if (!next) setReveal(false);
          }}
          placeholder="Paste a token"
        />
        <button
          type="button"
          className="quiet"
          aria-pressed={reveal}
          aria-controls="hf-token"
          disabled={!draft}
          onClick={() => setReveal((current) => !current)}
        >
          {reveal ? "Hide token" : "Show token"}
        </button>
      </div>
      {selected?.source === "environment" ? (
        <p className="note">
          INSPECTOR_HF_TOKEN is set for this API process and is used before the Keychain. Removing a Keychain token does not remove that environment value. The value is not shown.
        </p>
      ) : null}
      {selected?.source === "keychain" ? (
        <p className="note">Saved in the macOS Keychain on this Mac. The token is not shown.</p>
      ) : null}
      <div className="row">
        <button
          type="button"
          disabled={busy !== "" || !draft.trim()}
          onClick={() => void withToken("Saving the token on this computer.", (token) => saveCredential(apiBase, token, draft.trim()))}
        >
          {selected?.saved ? "Replace" : "Save"}
        </button>
        <button
          type="button"
          className="quiet"
          disabled={busy !== "" || !selected || selected.status === "not_configured"}
          onClick={() => void withToken("Testing the saved Hugging Face token.", (token) => testCredential(apiBase, token))}
        >
          Test Connection
        </button>
        <button
          type="button"
          className="quiet"
          disabled={busy !== "" || !selected?.saved}
          onClick={() => void withToken("Removing the saved token from this Mac.", (token) => removeCredential(apiBase, token))}
        >
          Remove
        </button>
      </div>
      {busy ? <WorkProgress label={busy} /> : null}
    </section>
  );
}

function FileInspection({
  apiBase,
  path,
  models,
  onInspected,
}: {
  apiBase: string;
  path: string;
  models: LocalModel[];
  onInspected: () => void;
}) {
  const model = models.find((item) => item.files.some((file) => file.path === path));
  const file = model?.files.find((item) => item.path === path);
  const fileName = file?.name || path.split("/").pop() || path;
  const partial = model?.checkpoint_status === "partial";
  const [parsed, setParsed] = useState(false);
  useEffect(() => {
    setParsed(false);
  }, [path]);

  return (
    <section>
      <div className="row">
        <button type="button" className="quiet" onClick={() => go(path.includes("/library/") ? "#/library/huggingface/downloaded" : "#/library/huggingface/import")}>
          Back
        </button>
        <h2>Checkpoint</h2>
      </div>
      {partial ? (
        <p className="note">Complete model analysis is unavailable until all required shards are present. Stored weights describe this file only.</p>
      ) : null}
      <CheckpointInspector
        apiBase={apiBase}
        path={path}
        title={fileName}
        onReport={() => {
          setParsed(true);
          onInspected();
        }}
        provenance={
          <>
            <p className="meta">Repository {model?.repo || "This file is not in the downloaded library."}</p>
            <p className="meta">File {fileName}</p>
            <p className="meta">Size {file?.size_human || "reading"}</p>
            <p className="meta">{model?.checkpoint_label || "Checkpoint completeness is recorded for downloaded library files."}</p>
            <p className="meta">{model?.integrity_label || "Integrity not verified — no published checksum was available."}</p>
            <p className="meta">
              {parsed
                ? "Inspectable — the checkpoint was parsed and its tensors can be examined."
                : model?.inspection_label || "Not yet inspected."}
            </p>
          </>
        }
      />
    </section>
  );
}
