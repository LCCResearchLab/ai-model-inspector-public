/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * Books screens. Every number on these screens comes from the Inspector API.
 */
import { useEffect, useState, type ReactNode } from "react";
import {
  fetchEffectiveUpdate,
  fetchWeightStatistics,
  MATCHING_BASE_PROJECTIONS_HELP,
  MATCHING_BASE_PROJECTIONS_LABEL,
  SHARED_LAYER_SCALE_TITLE,
  UPDATE_DIRECTION_HELP,
  UPDATE_DIRECTION_TITLE,
  barPercent,
  fetchUpdateAlignment,
  formatMeasure,
  sharedLayerMaximum,
  listFreshmanResearch,
  listLocalModels,
  type AlignmentComparison,
  type UpdateAlignmentReport,
  type EffectiveUpdateReport,
  type ResearchSubject,
  type WeightStatisticsReport,
} from "./inspectorClient";
import { WorkProgress } from "./progress";
import { weightStepLabel } from "./workProgress";
import { RESEARCH_COLLECTIONS, collectionById, experimentTitle, studyLine, type ResearchCollection } from "./provenance";
import {
  BASE_RECORDED_NOTE,
  BOOK_PAGES,
  CHECKPOINT_MISSING,
  QWEN_RECORDED_NOTE,
  RECORDED_RESULTS_LABEL,
  recordedDirectionView,
  recordedMeasurementLines,
  recordedPairLabel,
  formatRecorded,
  type PublishedCosine,
  type PublishedResearch,
} from "./publishedResults";
import { go, type Route, type WorkspaceTab } from "./route";
import { CheckpointInspector } from "./checkpointInspector";
import { cachedInspection, rememberInspection } from "./inspectionCache";
import { ProjectPath } from "./report";

const COMPARE = ["F5-1", "F5-2", "F5-3", "F6"];

function roleName(role: string): string {
  if (role === "lora_A") return "LoRA A";
  if (role === "lora_B") return "LoRA B";
  return role;
}

function projectionName(label: string | number): string {
  const names: Record<string, string> = { q_proj: "Q", k_proj: "K", v_proj: "V", o_proj: "O" };
  return names[String(label)] || String(label);
}

export function BookConsole({ apiBase, route }: { apiBase: string; route: Route }) {
  const [limits, setLimits] = useState("");
  const [subjects, setSubjects] = useState<ResearchSubject[]>([]);
  const [published, setPublished] = useState<PublishedResearch | null>(null);
  const [error, setError] = useState("");
  const [registryLoading, setRegistryLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setRegistryLoading(true);
    listFreshmanResearch(apiBase)
      .then((result) => {
        if (cancelled) return;
        setRegistryLoading(false);
        if (result.ok) {
          setLimits(result.limits);
          setSubjects(result.subjects);
          setPublished(result.published);
          setError("");
        } else {
          setError(result.message);
        }
      })
      .catch((caught) => {
        if (cancelled) return;
        setRegistryLoading(false);
        setError(caught instanceof Error ? caught.message : "The research registry did not respond.");
      });
    return () => {
      cancelled = true;
    };
  }, [apiBase]);

  const workspaceSubject = route.view === "workspace" && route.collection === "freshman" ? subjects.find((item) => item.id === route.id) : undefined;
  const focus = workspaceSubject?.artifact_type === "lora_adapter" ? workspaceSubject : subjects.find((item) => item.id === "F5-1");
  const onIndex = route.view === "book" && route.section === "subjects" && (route.collection === "index" || route.collection === "freshman");

  return (
    <>
      {onIndex ? (
        <>
          <h1>Books</h1>
          <p className="lede">Books &amp; Research. Collections follow the experiments. The Model Library follows where a file came from.</p>
          <p className="note">Published research results stay available when the checkpoint files are not on this computer. Inspect Local Checkpoint calculates a new measurement only when those files are present.</p>
          <p className="note">A later book is another research collection. It does not require a new section of the instrument.</p>
        </>
      ) : null}
      {error ? <p className="error">{error}</p> : null}
      {registryLoading && subjects.length === 0 && !error ? (
        <WorkProgress label="Checking the research files on this computer." />
      ) : null}
      {onIndex ? <CollectionBrowser subjects={subjects} published={published} limits={limits} registryLoading={registryLoading} /> : null}
      {route.view === "workspace" && route.collection === "freshman" && workspaceSubject ? (
        <FreshmanWorkspace apiBase={apiBase} subject={workspaceSubject} subjects={subjects} published={published} tab={route.tab} />
      ) : null}
      {route.view === "workspace" && route.collection === "freshman" && !workspaceSubject && !registryLoading ? (
        <p className="error">That research subject is not in Freshman Research.</p>
      ) : null}
      {route.view === "workspace" && route.collection === "qwen" ? <QwenResearch apiBase={apiBase} tab={route.tab} /> : null}
      {route.view === "book" && route.section === "findings" ? (
        <>
          <p className="crumb">
            <button type="button" onClick={() => go("#/book")}>Books</button> / Research findings
          </p>
          <PublishedBookThree subjects={subjects} published={published} />
          {focus ? <WeightsPanel apiBase={apiBase} subject={focus} subjects={subjects} mode="findings" embedded /> : null}
        </>
      ) : null}
    </>
  );
}

function freshmanHash(id: string, tab: WorkspaceTab): string {
  const subject = encodeURIComponent(id);
  if (tab === "checkpoint") return `#/book/${subject}/checkpoint`;
  if (tab === "magnitudes") return `#/book/${subject}/weights`;
  if (tab === "directions") return `#/book/${subject}/directions`;
  return `#/book/${subject}`;
}

function qwenHash(tab: WorkspaceTab): string {
  if (tab === "checkpoint") return "#/book/qwen/checkpoint";
  if (tab === "magnitudes") return "#/book/qwen/magnitudes";
  if (tab === "directions") return "#/book/qwen/directions";
  return "#/book/qwen";
}

function verificationLine(subject: ResearchSubject): string {
  const verified = subject.hash_status === "verified_match" ? "SHA-256 verified" : subject.status;
  return `${subject.artifact_label} · ${verified}`;
}

function CollectionBrowser({
  subjects,
  published,
  limits,
  registryLoading,
}: {
  subjects: ResearchSubject[];
  published: PublishedResearch | null;
  limits: string;
  registryLoading: boolean;
}) {
  const qwen = collectionById("qwen");
  const bookOne = BOOK_PAGES[0];
  const bookTwo = BOOK_PAGES[1];
  const bookThree = BOOK_PAGES[2];
  return (
    <>
      <section className="card">
        <h2>{bookOne.title}</h2>
        <p className="note">{bookOne.note}</p>
      </section>
      <section className="card">
        <h2>{bookTwo.title}</h2>
        <p className="note">{bookTwo.note}</p>
      </section>
      <details className="card collection" open>
        <summary>{bookThree.title}</summary>
        <p className="meta">Freshman Research. {bookThree.note}</p>
        <p className="meta">BASE is the starting model. F5-1, F5-2, F5-3, and F6 are trained adapters from Book Two.</p>
        <p className="meta">{registryLoading ? "Checking the research files." : `${subjects.length} subjects.`}</p>
        <button type="button" className="quiet" onClick={() => go("#/book/findings")}>
          Research findings
        </button>
        <PublishedBookThree subjects={subjects} published={published} />
        {limits ? (
          <details className="technical">
            <summary>Research limits</summary>
            <p className="limits">{limits}</p>
          </details>
        ) : null}
        {subjects.map((subject) => (
          <div className="subject-row" key={subject.id}>
            <div>
              <h3>{subject.id} — Freshman</h3>
              <p className="meta">{studyLine(subject.experiment)}</p>
              {subject.seed != null ? <p className="meta">Seed: {subject.seed} · Epoch: {subject.epoch}</p> : <p className="meta">Before the Chapter 5 and Chapter 6 lessons</p>}
              <p className="meta">{verificationLine(subject)}</p>
            </div>
            <button type="button" onClick={() => go(freshmanHash(subject.id, "overview"))}>
              Open →
            </button>
          </div>
        ))}
      </details>
      <details className="card collection" open>
        <summary>Qwen Research</summary>
        <p className="meta">{qwen?.researchContext}</p>
        <div className="subject-row">
          <div>
            <h3>Qwen — Qwen Research</h3>
            <p className="meta">No book chapter is recorded</p>
            <p className="meta">LoRA adapter · packaged revision 23001001-epoch-3</p>
          </div>
          <button type="button" onClick={() => go(qwenHash("overview"))}>
            Inspect →
          </button>
        </div>
      </details>
    </>
  );
}

function WorkspaceFrame({
  subjectId,
  title,
  lines,
  tab,
  hashes,
  children,
}: {
  subjectId: string;
  title: string;
  lines: string[];
  tab: WorkspaceTab;
  hashes: (next: WorkspaceTab) => string;
  children: ReactNode;
}) {
  const tabs: { id: WorkspaceTab; label: string }[] = [
    { id: "overview", label: "Overview" },
    { id: "checkpoint", label: "Checkpoint" },
    { id: "magnitudes", label: "Magnitudes" },
    { id: "directions", label: "Directions" },
  ];
  const place = tab === "overview" ? "" : tab;
  return (
    <>
      <p className="crumb">
        <button type="button" onClick={() => go("#/book")}>Books</button>
        {" / "}
        {place ? (
          <button type="button" onClick={() => go(hashes("overview"))}>{subjectId}</button>
        ) : (
          subjectId
        )}
        {place ? ` / ${place}` : ""}
      </p>
      <h2>{title}</h2>
      {lines.map((line) => (
        <p className="meta" key={line}>{line}</p>
      ))}
      <div className="subnav">
        {tabs.map((item) => (
          <button type="button" key={item.id} className={tab === item.id ? "on" : ""} onClick={() => go(hashes(item.id))}>
            {item.label}
          </button>
        ))}
      </div>
      {children}
    </>
  );
}

function FreshmanWorkspace({
  apiBase,
  subject,
  subjects,
  published,
  tab,
}: {
  apiBase: string;
  subject: ResearchSubject;
  subjects: ResearchSubject[];
  published: PublishedResearch | null;
  tab: WorkspaceTab;
}) {
  const lines = [
    studyLine(subject.experiment),
    subject.seed != null ? `Seed: ${subject.seed} · Epoch: ${subject.epoch}` : "Before the Chapter 5 and Chapter 6 lessons",
    verificationLine(subject),
  ];
  const measureSubject = tab === "directions" && subject.artifact_type !== "lora_adapter" ? subjects.find((item) => item.id === "F5-1") ?? subject : subject;
  return (
    <WorkspaceFrame
      subjectId={subject.id}
      title={`${subject.id} — Freshman`}
      lines={lines}
      tab={tab}
      hashes={(next) => freshmanHash(subject.id, next)}
    >
      <RecordedResearchResults subject={subject} published={published} />
      {tab === "overview" ? <SubjectDetail subject={subject} /> : null}
      {tab === "overview" ? <LocalCheckpointNotice subject={subject} /> : null}
      {tab === "checkpoint" ? (
        <section className="card">
          <h3>Inspect Local Checkpoint</h3>
          <p className="note">A measurement on this tab is newly calculated from a checkpoint on this computer. It is separate from the recorded research results.</p>
          {subject.location ? <CheckpointPanel apiBase={apiBase} subject={subject} embedded /> : <p className="note">{CHECKPOINT_MISSING}</p>}
        </section>
      ) : null}
      {tab === "magnitudes" ? <WeightsPanel apiBase={apiBase} subject={subject} subjects={subjects} mode="magnitudes" embedded /> : null}
      {tab === "directions" ? (
        <>
          {subject.artifact_type !== "lora_adapter" ? <p className="note">BASE is the starting model. The direction comparison is among the trained adapters.</p> : null}
          <WeightsPanel apiBase={apiBase} subject={measureSubject} subjects={subjects} mode="directions" embedded />
        </>
      ) : null}
    </WorkspaceFrame>
  );
}

function LocalCheckpointNotice({ subject }: { subject: ResearchSubject }) {
  return (
    <section className="card">
      <h3>Inspect Local Checkpoint</h3>
      <p className="note">A measurement started here is newly calculated from a checkpoint on this computer. It is separate from the recorded research results.</p>
      {subject.location ? (
        <button type="button" onClick={() => go(freshmanHash(subject.id, "checkpoint"))}>Inspect →</button>
      ) : (
        <p className="note">{CHECKPOINT_MISSING}</p>
      )}
    </section>
  );
}

function RecordedResearchResults({ subject, published }: { subject: ResearchSubject; published: PublishedResearch | null }) {
  return (
    <section className="card" data-testid="recorded-results">
      <h3>{RECORDED_RESULTS_LABEL}</h3>
      {!published ? <p className="note">The published measurement record was not returned.</p> : <RecordedSubjectBody subject={subject} published={published} />}
    </section>
  );
}

function RecordedSubjectBody({ subject, published }: { subject: ResearchSubject; published: PublishedResearch }) {
  const row = published.subjects.find((item) => item.id === subject.id);
  const directions = recordedDirectionView(subject.id, published.chapter_five_cosines);
  return (
    <>
      <p className="note">These values are the published record ({published.record}, {published.measured_on}). They were not calculated from a file on this computer.</p>
      {row ? (
        <>
          <div className="facts">
            {recordedMeasurementLines(row).map((line) => (
              <div key={line.testId} className={line.testId === "sha256" ? "wide" : undefined}>
                <span>{line.label}</span>
                <strong className={line.testId === "sha256" ? "hash" : undefined} data-testid={`recorded-${subject.id}-${line.testId}`}>{line.value}</strong>
              </div>
            ))}
          </div>
          <details className="technical">
            <summary>What is the denominator?</summary>
            <p className="limits">{MATCHING_BASE_PROJECTIONS_HELP}</p>
          </details>
        </>
      ) : (
        <>
          <p className="note">{BASE_RECORDED_NOTE}</p>
          <Field label="Weight SHA-256" value={subject.expected_sha256 || "not recorded"} />
        </>
      )}
      <RecordedDirectionBlock directions={directions} />
    </>
  );
}

function RecordedDirectionBlock({ directions }: { directions: { rows: PublishedCosine[]; note: string | null } }) {
  if (directions.rows.length === 0 && !directions.note) return null;
  return (
    <>
      <h3>Recorded Chapter 5 direction comparisons</h3>
      {directions.note ? <p className="note">{directions.note}</p> : null}
      {directions.rows.length > 0 ? (
        <>
          <p className="note">Each row compares the recorded effective-update direction of the two named Freshman adapters.</p>
          <RecordedDirectionTable rows={directions.rows} />
          <details className="technical">
            <summary>What do these cosines compare?</summary>
            <p className="limits">{UPDATE_DIRECTION_HELP}</p>
          </details>
        </>
      ) : null}
    </>
  );
}

function RecordedDirectionTable({ rows }: { rows: PublishedCosine[] }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Freshman pair</th>
          <th>Overall cosine</th>
          <th>Mean pair cosine</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={`${row.left}-${row.right}`}>
            <td>{recordedPairLabel(row)}</td>
            <td data-testid={`recorded-cosine-${row.left}-${row.right}`}>{formatRecorded(row.cosine)}</td>
            <td>{formatRecorded(row.mean_pair_cosine)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function PublishedBookThree({ subjects, published }: { subjects: ResearchSubject[]; published: PublishedResearch | null }) {
  if (!published) return <p className="note">The published measurement record was not returned.</p>;
  const base = subjects.find((item) => item.id === "BASE");
  return (
    <section className="card">
      <h3>{RECORDED_RESULTS_LABEL}</h3>
      <p className="note">These values are the published record ({published.record}, {published.measured_on}). They were not calculated from a file on this computer.</p>
      <table>
        <thead>
          <tr>
            <th>Subject</th>
            <th>LoRA rank</th>
            <th>Alpha</th>
            <th>Scaling</th>
            <th>Effective-update magnitude (L2)</th>
            <th>Relative to matching BASE projections</th>
            <th>Weight SHA-256</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>BASE</td>
            <td colSpan={4}>{BASE_RECORDED_NOTE}</td>
            <td />
            <td className="hash">{base?.expected_sha256 || "not recorded"}</td>
          </tr>
          {published.subjects.map((row) => (
            <tr key={row.id}>
              <td>{row.id}</td>
              <td>{formatRecorded(row.rank)}</td>
              <td>{formatRecorded(row.lora_alpha)}</td>
              <td>{formatRecorded(row.scale)}</td>
              <td data-testid={`recorded-table-${row.id}-magnitude`}>{formatRecorded(row.effective_update_norm)}</td>
              <td>{formatRecorded(row.relative_to_base)}</td>
              <td className="hash">{row.weight_sha256}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <details className="technical">
        <summary>What is the denominator?</summary>
        <p className="limits">{MATCHING_BASE_PROJECTIONS_HELP}</p>
      </details>
      <RecordedDirectionBlock directions={{ rows: published.chapter_five_cosines, note: "No direction comparison is recorded for F6 or for BASE." }} />
    </section>
  );
}

function Field({ label, value, wide }: { label: string; value: string; wide?: boolean }) {
  return (
    <p className={wide ? "meta span" : "meta"}>
      <strong>{label}: </strong>
      {value}
    </p>
  );
}

const FRESHMAN = RESEARCH_COLLECTIONS[0];
const QWEN = collectionById("qwen") ?? RESEARCH_COLLECTIONS[1];

function QwenResearch({ apiBase, tab }: { apiBase: string; tab: WorkspaceTab }) {
  const [repo, setRepo] = useState("");
  const [revision, setRevision] = useState("");
  const [adapterPath, setAdapterPath] = useState("");
  const [error, setError] = useState("");
  const [work, setWork] = useState("Locating the packaged Qwen adapter.");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const listed = await listLocalModels(apiBase);
        if (cancelled) return;
        if (!listed.ok) {
          setError(listed.message);
          return;
        }
        const model = listed.bundled.find((item) => item.repo === "book-three/qwen");
        const file = model?.files.find((item) => item.name === "adapter_model.safetensors" && item.size_bytes);
        if (!model || !file?.path) {
          setError("The Qwen adapter is not included in this package. Place the local research copy in inspector-engine/bundled/qwen/ to inspect it.");
          return;
        }
        setRepo(model.repo);
        setRevision(model.revision || "");
        setAdapterPath(file.path);
      } catch (caught) {
        if (!cancelled) setError(caught instanceof Error ? caught.message : "The Qwen adapter could not be located.");
      } finally {
        if (!cancelled) setWork("");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [apiBase]);

  const saved = adapterPath ? cachedInspection(adapterPath) : null;
  return (
    <WorkspaceFrame
      subjectId="Qwen"
      title="Qwen — Qwen Research"
      lines={["No book chapter is recorded", `LoRA adapter · packaged revision ${revision || "23001001-epoch-3"}`]}
      tab={tab}
      hashes={qwenHash}
    >
      <section className="card">
        <h3>{RECORDED_RESULTS_LABEL}</h3>
        <p className="note">{QWEN_RECORDED_NOTE}</p>
      </section>
      {work ? <WorkProgress label={work} /> : null}
      {error ? <p className="error">{error}</p> : null}
      {tab === "overview" ? (
        <section className="card">
          <h3>Research provenance</h3>
          <Field label="Artifact" value="Qwen adapter" wide />
          <Field label="Package record" value={repo || "book-three/qwen"} wide />
          <Field label="Recorded revision" value={revision || "not recorded"} wide />
          <Field label="Experiment" value="No book or chapter is recorded for this adapter." wide />
          <Field label="Research context" value={QWEN.researchContext} wide />
          <Field label="Measurement" value={QWEN.measurement} wide />
          <p className="note">These values are read from the packaged adapter. They are not copied from Freshman Research. This adapter is not part of the public package until its redistribution rights are confirmed.</p>
          <p className="note">Open Checkpoint to read the file, then choose Stored weights or Effective update. A saved result stays available for this session.</p>
          <QwenSources update={saved?.update ?? null} />
        </section>
      ) : null}
      {tab === "checkpoint" && !adapterPath && !work ? (
        <section className="card">
          <h3>Inspect Local Checkpoint</h3>
          <p className="note">{CHECKPOINT_MISSING}</p>
        </section>
      ) : null}
      {tab === "checkpoint" && adapterPath ? (
        <CheckpointInspector
          apiBase={apiBase}
          path={adapterPath}
          title="Qwen adapter"
          provenance={<p className="meta">Package record {repo || "book-three/qwen"}</p>}
        />
      ) : null}
      {tab === "magnitudes" && !adapterPath && !work ? <p className="note">{CHECKPOINT_MISSING}</p> : null}
      {tab === "magnitudes" && adapterPath ? (
        <CheckpointInspector
          apiBase={apiBase}
          path={adapterPath}
          title="Qwen adapter"
          provenance={<QwenSources update={saved?.update ?? null} />}
        />
      ) : null}
      {tab === "directions" ? <p className="note">This collection has one adapter. Direction comparison needs a second compatible adapter, so no direction was calculated.</p> : null}
    </WorkspaceFrame>
  );
}

function QwenSources({ update }: { update: EffectiveUpdateReport | null }) {
  if (!update?.sources) return null;
  return (
    <div className="sources">
      <p className="meta"><strong>Alpha. </strong>{update.sources.lora_alpha}</p>
      <p className="meta"><strong>Rank. </strong>{update.sources.r}</p>
      <p className="meta"><strong>Scale. </strong>{update.sources.scale}</p>
      {update.relative_note ? <p className="meta"><strong>Relative magnitude. </strong>{update.relative_note}</p> : null}
    </div>
  );
}

function ExperimentText({ experiment }: { experiment: string }) {
  const title = experimentTitle(experiment);
  if (!title) return <>{experiment}</>;
  return (
    <>
      {experiment.slice(0, experiment.length - title.length)}
      <em>{title}</em>
    </>
  );
}

function Provenance({ experiment, collection }: { experiment: string; collection: ResearchCollection }) {
  return (
    <>
      <p className="meta span">
        <strong>Experiment: </strong>
        <ExperimentText experiment={experiment} />
      </p>
      <p className="meta span">
        <strong>Measurement: </strong>
        {collection.measurement}
      </p>
      <p className="meta span">
        <strong>Research context: </strong>
        {collection.researchContext}
      </p>
    </>
  );
}

function SubjectDetail({ subject }: { subject: ResearchSubject }) {
  const provenance = subject.provenance;
  return (
    <section className="card">
      <div className="identity">
        <Provenance experiment={subject.experiment} collection={FRESHMAN} />
        <Field label="Role" value={subject.experiment_role} />
        <Field label="Unit" value={subject.unit_id || "none recorded"} />
        <Field label="Base model" value={subject.base_model} />
        <Field label="Seed" value={subject.seed == null ? "not a trained run" : String(subject.seed)} />
        <Field label="Epoch" value={subject.epoch == null ? "not a trained run" : String(subject.epoch)} />
        <Field label="Artifact" value={subject.artifact_label} />
        <Field label="Full student checkpoint" value={subject.full_student_label} />
        <Field label="File" value={subject.file_name || "none"} />
        <Field label="Size" value={subject.size_human || "unknown"} />
        <Field label="Status" value={subject.status} />
      </div>
      {provenance?.packaged_configuration?.matches_original === false ? (
        <p className="note">This copy is not the original research file.</p>
      ) : null}
      {provenance?.base_weights ? <p className="note">{provenance.base_weights.note}</p> : null}
      <details className="technical">
        <summary>Technical details</summary>
        <ProjectPath label={subject.location_label} display={subject.location_display} />
        <Field label="Weight SHA-256" value={subject.observed_sha256 || subject.expected_sha256 || "not recorded"} />
        {provenance?.original_configuration ? (
          <Field label="Original configuration SHA-256" value={provenance.original_configuration.sha256 || "not recorded"} />
        ) : null}
        {provenance?.packaged_configuration ? (
          <Field label="Packaged configuration SHA-256" value={provenance.packaged_configuration.sha256 || "not recorded"} />
        ) : null}
      </details>
    </section>
  );
}

function CheckpointPanel({ apiBase, subject }: { apiBase: string; subject: ResearchSubject; embedded?: boolean }) {
  const location = subject.location || "";
  return (
    <CheckpointInspector
      apiBase={apiBase}
      path={location}
      title={`${subject.id} — ${subject.artifact_label}`}
      measurements={subject.artifact_type === "lora_adapter"}
      provenance={
        <>
          <p className="meta">{studyLine(subject.experiment)}</p>
          <p className="meta">{verificationLine(subject)}</p>
          <p className="note">SHA-256 is the fingerprint returned with this inspection.</p>
        </>
      }
    />
  );
}

function WeightsPanel({
  apiBase,
  subject,
  subjects,
  mode,
  embedded = false,
}: {
  apiBase: string;
  subject: ResearchSubject;
  subjects: ResearchSubject[];
  mode: "magnitudes" | "directions" | "findings";
  embedded?: boolean;
}) {
  const [stats, setStats] = useState<WeightStatisticsReport | null>(null);
  const [updates, setUpdates] = useState<Record<string, EffectiveUpdateReport>>({});
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(0);
  const [alignment, setAlignment] = useState<UpdateAlignmentReport | null>(null);
  const [directionError, setDirectionError] = useState("");
  const measureOrder = [subject.id, ...COMPARE.filter((item) => item !== subject.id)];
  const magnitudeTotal = 1 + measureOrder.length;
  const comparingDirection = step >= magnitudeTotal;

  useEffect(() => {
    const next: Record<string, EffectiveUpdateReport> = {};
    for (const id of COMPARE) {
      const other = subjects.find((item) => item.id === id);
      const saved = other?.location ? cachedInspection(other.location) : null;
      if (saved?.update) next[id] = saved.update;
      if (id === subject.id && saved?.stats) setStats(saved.stats);
    }
    if (Object.keys(next).length > 0) setUpdates(next);
  }, [subject.id, subjects]);

  async function compareDirections() {
    const order = [subject.id, ...COMPARE.filter((item) => item !== subject.id)];
    setLoading(true);
    setStep(0);
    setDirectionError("");
    setError("");
    try {
      if (!subject.location) {
        setError(subject.status);
        return;
      }
      const measured = await fetchWeightStatistics(apiBase, subject.location);
      if (!measured.ok) {
        setError(measured.message);
        return;
      }
      setStats(measured);
      rememberInspection(subject.location, { stats: measured });
      setStep(1);
      const next: Record<string, EffectiveUpdateReport> = {};
      for (let index = 0; index < order.length; index += 1) {
        const id = order[index];
        const other = subjects.find((item) => item.id === id);
        if (!other?.location) {
          setError(other?.status || `${id} has no verified checkpoint.`);
          return;
        }
        const update = await fetchEffectiveUpdate(apiBase, other.location);
        if (!update.ok) {
          setError(update.message);
          return;
        }
        next[id] = update;
        rememberInspection(other.location, { update });
        setUpdates({ ...next });
        setStep(index + 2);
      }
      const paths = order.map((id) => subjects.find((item) => item.id === id)?.location || "");
      const aligned = await fetchUpdateAlignment(apiBase, paths);
      if (!aligned.ok) setDirectionError(aligned.message);
      else setAlignment(aligned);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The measurement did not finish.");
    } finally {
      setLoading(false);
    }
  }

  if (subject.artifact_type !== "lora_adapter") {
    return (
      <section className="card">
        {embedded ? null : <h2>Weight Magnitudes · {subject.id}</h2>}
        <p>
          {subject.id} is a full-model checkpoint: the starting weights. This screen does not calculate stored-weight
          statistics or an effective LoRA update for it. Those measurements apply to LoRA adapters, which are compared
          with this file.
        </p>
      </section>
    );
  }

  const own = updates[subject.id];
  const maxLayer = sharedLayerMaximum([own?.by_layer.map((row) => row.frobenius_norm) ?? []]);
  const sharedReady = COMPARE.every((id) => (updates[id]?.by_layer.length ?? 0) > 0);
  const sharedMax = sharedReady
    ? sharedLayerMaximum(COMPARE.map((id) => updates[id].by_layer.map((row) => row.frobenius_norm)))
    : 0;

  return (
    <section>
      {!embedded && mode === "magnitudes" ? (
        <div className="row">
          <button type="button" className="quiet" onClick={() => go(`#/book/${encodeURIComponent(subject.id)}`)}>
            Subject
          </button>
          <h2>Weight Magnitudes · {subject.id}</h2>
          {COMPARE.map((id) => (
            <button type="button" key={id} className={id === subject.id ? "" : "quiet"} onClick={() => go(`#/book/${encodeURIComponent(id)}/weights`)}>
              {id}
            </button>
          ))}
        </div>
      ) : null}
      {mode === "findings" ? <h2>Research Findings</h2> : null}
      {!embedded && mode === "directions" ? <h2>Weight Directions</h2> : null}
      {loading ? (
        comparingDirection ? (
          <WorkProgress label="Comparing the direction of the effective updates." />
        ) : (
          <WorkProgress label={weightStepLabel(step, subject.id, measureOrder)} completed={step} total={magnitudeTotal} />
        )
      ) : null}
      {error ? <p className="error">{error}</p> : null}
      {directionError ? <p className="error">{directionError}</p> : null}
      {subject.location && mode === "magnitudes" ? (
        <>
          <h3>Newly calculated measurement</h3>
          <p className="note">This measurement is calculated from the checkpoint on this computer. It is separate from the recorded research results.</p>
          <CheckpointInspector
            apiBase={apiBase}
            path={subject.location}
            title={`${subject.id} — ${subject.artifact_label}`}
            onMeasured={(result) => {
              if (result.stats) setStats(result.stats);
              if (result.update) setUpdates((current) => ({ ...current, [subject.id]: result.update as EffectiveUpdateReport }));
            }}
          />
        </>
      ) : null}
      {mode === "magnitudes" && !subject.location ? <p className="note">{CHECKPOINT_MISSING}</p> : null}
      <div className="row">
        <button type="button" disabled={loading || !subject.location} onClick={() => void compareDirections()}>
          {alignment ? "Compare directions again" : "Compare directions"}
        </button>
      </div>
      <p className="note">Compare directions calculates a new comparison from local checkpoints. It does not replace the recorded research results.</p>
      {mode === "directions" ? <DirectionComparison alignment={alignment} /> : null}
      {mode === "findings" ? <FindingsView subjects={subjects} updates={updates} alignment={alignment} /> : null}
      {mode === "magnitudes" && subject.location ? <MeasuredCheckpointCard subject={subject} stats={stats} update={own} updates={updates} /> : null}
      {mode === "magnitudes" && own ? (
        <div className="facts">
          <div>
            <span>Stored tensors</span>
            <strong data-testid="stored-tensor-count">{stats ? stats.groups.reduce((sum, group) => sum + group.tensor_count, 0) : "…"}</strong>
          </div>
          <div>
            <span>LoRA A/B pairs</span>
            <strong data-testid="pair-count">{own.pair_count}</strong>
          </div>
          <div>
            <span>Rank</span>
            <strong data-testid="rank">{own.configuration_agreement.r}</strong>
          </div>
          <div>
            <span>Scale</span>
            <strong data-testid="scale">{formatMeasure(own.configuration_agreement.scale)}</strong>
          </div>
          <div>
            <span>Effective update norm</span>
            <strong data-testid="effective-norm">{formatMeasure(own.total_frobenius_norm)}</strong>
          </div>
          <div className="wide">
            <span>{MATCHING_BASE_PROJECTIONS_LABEL}</span>
            <strong data-testid="relative-magnitude">{formatMeasure(own.total_relative_frobenius)}</strong>
            <span className="meta" data-testid="base-denominator">
              Denominator {formatMeasure(own.total_base_frobenius_norm)}
            </span>
          </div>
        </div>
      ) : null}
      {mode === "magnitudes" && own ? (
        <p className="note">
          Displayed values keep four significant figures. Values below 0.001 are shown in scientific notation. The measurement response is unchanged.
        </p>
      ) : null}
      {mode === "magnitudes" && stats ? (
        <section className="card">
          <h2>Stored weights</h2>
          <p>
            These are the A and B matrices saved in {stats.checkpoint?.name || "this adapter"}. A and B are separate
            populations. This table is not the scaled product BA.
          </p>
          <p className="meta">{fingerprintLine(stats.checkpoint?.sha256, stats.research?.hash_status)}</p>
          <details className="technical">
            <summary>Instrument limits</summary>
            <p className="limits">{stats.limits}</p>
          </details>
          <table>
            <thead>
              <tr>
                <th>Group</th>
                <th>Tensors</th>
                <th>Parameters</th>
                <th>Min</th>
                <th>Max</th>
                <th>Mean</th>
                <th>Std</th>
                <th>L1</th>
                <th>L2</th>
                <th>Zeros</th>
                <th>NaN</th>
                <th>Inf</th>
              </tr>
            </thead>
            <tbody>
              {stats.groups.map((group) => (
                <tr key={group.role}>
                  <td>{roleName(group.role)}</td>
                  <td>{group.tensor_count}</td>
                  <td>{group.parameter_count.toLocaleString()}</td>
                  <td>{formatMeasure(group.minimum)}</td>
                  <td>{formatMeasure(group.maximum)}</td>
                  <td>{formatMeasure(group.mean)}</td>
                  <td>{formatMeasure(group.standard_deviation)}</td>
                  <td>{formatMeasure(group.l1_norm)}</td>
                  <td data-testid={`l2-${group.role}`}>{formatMeasure(group.l2_norm)}</td>
                  <td>{group.zero_count}</td>
                  <td>{group.nan_count}</td>
                  <td>{group.inf_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : null}
      {mode === "magnitudes" && own ? (
        <section className="card">
          <h2>Effective update</h2>
          <p>
            Each value is the effective update, scale times B times A. This file uses scale{" "}
            {formatMeasure(own.configuration_agreement.scale)} ({own.configuration_agreement.scale_rule}).
          </p>
          <p className="meta">{fingerprintLine(subject.observed_sha256, own.research?.hash_status || subject.hash_status)}</p>
          <p className="note">
            These magnitudes describe the saved adapter relative to the verified BASE weights. They are not a measurement
            of acquired knowledge. BASE together with this adapter is not an independently saved full-student checkpoint.
            Full student checkpoint: {subject.full_student_label}.
          </p>
          <details className="technical">
            <summary>Instrument limits</summary>
            <p className="limits">{own.limits}</p>
          </details>
          <p className="meta">Alpha {shownNumber(own.configuration_agreement.lora_alpha)}. {own.sources?.lora_alpha}</p>
          <p className="meta">Rank {own.configuration_agreement.r}. {own.sources?.r}</p>
          {own.base_checkpoint?.path_display ? (
            <p className="meta">Base checkpoint {own.base_checkpoint.path_display}</p>
          ) : null}
          <p className="meta">Configuration agreement {own.configuration_agreement.agreed ? "yes" : "no"}</p>
          <p className="meta">Total Frobenius {formatMeasure(own.total_frobenius_norm)}</p>
          <p className="meta">
            {MATCHING_BASE_PROJECTIONS_LABEL} {formatMeasure(own.total_relative_frobenius)}. Denominator{" "}
            {formatMeasure(own.total_base_frobenius_norm)}.
          </p>
          <details className="technical">
            <summary>What is the denominator?</summary>
            <p className="limits">{MATCHING_BASE_PROJECTIONS_HELP}</p>
          </details>
          <h3>By projection</h3>
          <table>
            <thead>
              <tr>
                <th>Projection</th>
                {COMPARE.map((id) => (
                  <th key={id}>{id}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {["q_proj", "k_proj", "v_proj", "o_proj"].map((projection) => (
                <tr key={projection}>
                  <td>{projectionName(projection)}</td>
                  {COMPARE.map((id) => {
                    const row = updates[id]?.by_projection.find((item) => String(item.label) === projection);
                    return <td key={id}>{row ? formatMeasure(row.frobenius_norm) : "…"}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <h3>By layer</h3>
          <p className="note">
            Bars in this table use {subject.id}’s own largest layer, {formatMeasure(maxLayer)}. They are not on the shared scale.
          </p>
          <table>
            <thead>
              <tr>
                <th>Layer</th>
                <th>{subject.id}</th>
                {COMPARE.filter((id) => id !== subject.id).map((id) => (
                  <th key={id}>{id}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(own.by_layer || []).map((row) => (
                <tr key={String(row.label)}>
                  <td>{String(row.label)}</td>
                  <td>
                    <div className="bar" title={formatMeasure(row.frobenius_norm)}>
                      <span style={{ width: `${barPercent(row.frobenius_norm, maxLayer)}%` }} />
                    </div>
                    {formatMeasure(row.frobenius_norm)}
                  </td>
                  {COMPARE.filter((id) => id !== subject.id).map((id) => {
                    const match = updates[id]?.by_layer.find((item) => item.label === row.label);
                    return <td key={id}>{match ? formatMeasure(match.frobenius_norm) : "…"}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          {sharedReady ? (
            <>
              <h3>{SHARED_LAYER_SCALE_TITLE}</h3>
              <p className="note">
                Every bar uses one maximum, the largest per-layer effective-update norm among F5-1, F5-2, F5-3, and F6:{" "}
                {formatMeasure(sharedMax)}. A subject is not scaled on its own.
              </p>
              <table>
                <thead>
                  <tr>
                    <th>Layer</th>
                    {COMPARE.map((id) => (
                      <th key={id}>{id}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {own.by_layer.map((row) => (
                    <tr key={`shared-${String(row.label)}`}>
                      <td>{String(row.label)}</td>
                      {COMPARE.map((id) => {
                        const match = updates[id].by_layer.find((item) => item.label === row.label);
                        const value = match?.frobenius_norm ?? 0;
                        return (
                          <td key={id}>
                            <div className="bar" data-testid={`shared-bar-${id}-${String(row.label)}`}>
                              <span style={{ width: `${barPercent(value, sharedMax)}%` }} />
                            </div>
                            {match ? formatMeasure(value) : "…"}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          ) : null}
          <h3>Research subjects</h3>
          <p className="note">Each column is the same effective-update measurement on another saved adapter. Fingerprints for those files are in Measured files.</p>
          <table>
            <thead>
              <tr>
                <th>Subject</th>
                <th>Pairs</th>
                <th>Effective update norm</th>
                <th>{MATCHING_BASE_PROJECTIONS_LABEL}</th>
              </tr>
            </thead>
            <tbody>
              {COMPARE.map((id) => (
                <tr key={id}>
                  <td>{id}</td>
                  <td>{updates[id]?.pair_count ?? "…"}</td>
                  <td>{updates[id] ? formatMeasure(updates[id].total_frobenius_norm) : "…"}</td>
                  <td>{updates[id] ? formatMeasure(updates[id].total_relative_frobenius) : "…"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : null}
    </section>
  );
}

const SAME_LESSON: [string, string][] = [
  ["F5-1", "F5-2"],
  ["F5-1", "F5-3"],
  ["F5-2", "F5-3"],
];

const OTHER_LESSON: [string, string][] = [
  ["F5-1", "F6"],
  ["F5-2", "F6"],
  ["F5-3", "F6"],
];

function BaselineNote({ baseline }: { baseline: UpdateAlignmentReport["baseline"] }) {
  return (
    <section className="card">
      <h3>Chance baseline</h3>
      <p className="note">{baseline.limits}</p>
      <div className="facts">
        <div>
          <span>Update entries</span>
          <strong>{baseline.entry_count.toLocaleString()}</strong>
        </div>
        <div>
          <span>Dense random scale</span>
          <strong>{formatMeasure(baseline.dense_isotropic_scale)}</strong>
        </div>
        <div>
          <span>Random-factor median</span>
          <strong data-testid="baseline-median">{formatMeasure(baseline.random_factor_median_absolute)}</strong>
        </div>
        <div>
          <span>Random-factor maximum</span>
          <strong>{formatMeasure(baseline.random_factor_maximum_absolute)}</strong>
        </div>
      </div>
      <p className="meta">
        Seed {baseline.seed}. {baseline.draws} draws. The shapes come from the measured updates. This baseline is exploratory.
      </p>
    </section>
  );
}

function FindingsView({
  subjects,
  updates,
  alignment,
}: {
  subjects: ResearchSubject[];
  updates: Record<string, EffectiveUpdateReport>;
  alignment: UpdateAlignmentReport | null;
}) {
  const ready = COMPARE.every((id) => updates[id]?.ok) && alignment;
  if (!ready || !alignment) return null;
  const chapter = SAME_LESSON.map(([left, right]) => findComparison(alignment.comparisons, left, right)?.cosine).filter((value): value is number => value != null);
  const other = OTHER_LESSON.map(([left, right]) => findComparison(alignment.comparisons, left, right)?.cosine).filter((value): value is number => value != null);
  const largest = chapter.length ? Math.max(...chapter) : null;
  const aboveChance = largest != null && alignment.baseline.random_factor_maximum_absolute != null && largest > alignment.baseline.random_factor_maximum_absolute;
  return (
    <>
      <h3>Measurements</h3>
      <p className="note">These numbers are Research Inspector measurements for the {FRESHMAN.label} collection. They are not copied from the book.</p>
      <table>
        <thead>
          <tr>
            <th>Subject</th>
            <th>Experiment</th>
            <th>Effective update norm</th>
            <th>{MATCHING_BASE_PROJECTIONS_LABEL}</th>
          </tr>
        </thead>
        <tbody>
          {COMPARE.map((id) => {
            const subject = subjects.find((item) => item.id === id);
            return (
              <tr key={id}>
                <td>{id}</td>
                <td>{subject?.experiment || ""}</td>
                <td>{formatMeasure(updates[id].total_frobenius_norm)}</td>
                <td>{formatMeasure(updates[id].total_relative_frobenius)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <CosineSummary pairs={SAME_LESSON} found={SAME_LESSON.map(([left, right]) => findComparison(alignment.comparisons, left, right))} />
      <h3>Different lesson</h3>
      <p className="note">Each value compares F6 with one Chapter 5 run.</p>
      <CosineSummary pairs={OTHER_LESSON} found={OTHER_LESSON.map(([left, right]) => findComparison(alignment.comparisons, left, right))} />
      <BaselineNote baseline={alignment.baseline} />
      <h3>Observations</h3>
      <p>
        F5-1, F5-2, and F5-3 have effective-update norms {COMPARE.slice(0, 3).map((id) => formatMeasure(updates[id].total_frobenius_norm)).join(", ")}. F6 is{" "}
        {formatMeasure(updates.F6.total_frobenius_norm)}. The Chapter 5 pairwise cosines are {chapter.map((value) => formatMeasure(value)).join(", ")}. The F6 comparisons with F5-1, F5-2, and F5-3 have overall cosines{" "}
        {other.map((value) => formatMeasure(value)).join(", ")}.
        {aboveChance
          ? " The largest Chapter 5 cosine is larger than every draw in this exploratory baseline, and it is far below 1."
          : " Those cosines are not above the random-factor maximum in this exploratory baseline."}
      </p>
      <h3>Interpretations</h3>
      <p>
        A similar update size does not mean a similar update direction. The Chapter 5 effective-update vectors have very small positive cosine similarities, indicating that their directions are nearly orthogonal in the measured parameter space. A small positive alignment remains. It is larger than the random-factor draws in this exploratory baseline. That comparison does not establish a statistically significant shared learning direction.
      </p>
      <p>The magnitude of a weight change and the direction of that change are two different properties. Neither alone tells us what knowledge was acquired.</p>
      <p>This does not identify why the directions differ. It does not measure acquired knowledge, and it does not say whether the runs produce similar answers.</p>
      <h3>Limitations</h3>
      <p className="limits">
        These statements describe BASE, F5-1, F5-2, F5-3, and F6 in the {FRESHMAN.label} collection. Layer count, projection count, and rank were read from these files. The baseline uses {alignment.baseline.draws} random-factor draws. It is exploratory. It does not establish statistical significance, and it is not the distribution of independently trained adapters. It is not a model of training. {FRESHMAN.researchContext} is the research context, and that book stays a separate document.
      </p>
    </>
  );
}

function findComparison(comparisons: AlignmentComparison[], left: string, right: string): AlignmentComparison | undefined {
  return comparisons.find((item) => (item.left === left && item.right === right) || (item.left === right && item.right === left));
}

function DirectionComparison({ alignment }: { alignment: UpdateAlignmentReport | null }) {
  if (!alignment) return null;
  const same = SAME_LESSON.map(([left, right]) => findComparison(alignment.comparisons, left, right));
  const other = OTHER_LESSON.map(([left, right]) => findComparison(alignment.comparisons, left, right));
  return (
    <>
      <h3>{UPDATE_DIRECTION_TITLE}</h3>
      <p className="note">{UPDATE_DIRECTION_HELP}</p>
      <p className="note">
        F5-1, F5-2, and F5-3 are three runs of the same lesson with different seeds. F6 is a different lesson. This comparison does not say whether the runs produce similar answers.
      </p>
      <BaselineNote baseline={alignment.baseline} />
      <h3>Same lesson, different seeds</h3>
      <CosineSummary pairs={SAME_LESSON} found={same} />
      <h3>By projection</h3>
      <CosineGrid pairs={SAME_LESSON} found={same} kind="by_projection" />
      <h3>By layer</h3>
      <CosineGrid pairs={SAME_LESSON} found={same} kind="by_layer" />
      <h3>Different lesson</h3>
      <p className="note">Each value compares F6 with one Chapter 5 run.</p>
      <CosineSummary pairs={OTHER_LESSON} found={other} />
      <CosineGrid pairs={OTHER_LESSON} found={other} kind="by_layer" />
      <details className="technical">
        <summary>Instrument limits</summary>
        <p className="limits">{alignment.limits}</p>
      </details>
    </>
  );
}

function CosineSummary({ pairs, found }: { pairs: [string, string][]; found: (AlignmentComparison | undefined)[] }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Pair</th>
          <th>Overall cosine</th>
          <th>Mean pair cosine</th>
        </tr>
      </thead>
      <tbody>
        {pairs.map(([left, right], index) => (
          <tr key={`${left}-${right}`}>
            <td>
              {left} · {right}
            </td>
            <td data-testid={`cosine-${left}-${right}`}>{formatMeasure(found[index]?.cosine)}</td>
            <td>{formatMeasure(found[index]?.mean_pair_cosine)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function CosineGrid({
  pairs,
  found,
  kind,
}: {
  pairs: [string, string][];
  found: (AlignmentComparison | undefined)[];
  kind: "by_layer" | "by_projection";
}) {
  const labels = found.find((item) => item)?.[kind].map((row) => row.label) ?? [];
  if (labels.length === 0) return null;
  return (
    <table>
      <thead>
        <tr>
          <th>{kind === "by_layer" ? "Layer" : "Projection"}</th>
          {pairs.map(([left, right]) => (
            <th key={`${left}-${right}`}>
              {left} · {right}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {labels.map((label) => (
          <tr key={String(label)}>
            <td>{kind === "by_projection" ? projectionName(label) : String(label)}</td>
            {found.map((comparison, index) => {
              const row = comparison?.[kind].find((item) => item.label === label);
              return <td key={`${pairs[index][0]}-${pairs[index][1]}`}>{formatMeasure(row?.cosine)}</td>;
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function shownNumber(value: number | null): string {
  return value == null ? "unknown" : String(value);
}

function recordLabel(status: string | undefined): string {
  if (status === "verified_match") return "SHA-256 matches the research record";
  if (status === "mismatch") return "SHA-256 does not match the research record";
  return status || "Fingerprint status was not returned with this measurement";
}

function fingerprintLine(sha: string | null | undefined, status: string | undefined): string {
  return sha ? `${recordLabel(status)}. ${sha}` : recordLabel(status);
}

function MeasuredCheckpointCard({
  subject,
  stats,
  update,
  updates,
}: {
  subject: ResearchSubject;
  stats: WeightStatisticsReport | null;
  update: EffectiveUpdateReport | undefined;
  updates: Record<string, EffectiveUpdateReport>;
}) {
  const measured = stats?.checkpoint;
  const updateSha = update?.research?.provenance?.weight?.observed_sha256;
  const base = update?.research?.provenance?.base_weights;
  const sameFile = !measured?.sha256 || !updateSha || measured.sha256 === updateSha;
  return (
    <section className="card">
      <h2>Measured checkpoint</h2>
      <div className="identity">
        <Field label="Subject" value={subject.id} />
        <Provenance experiment={subject.experiment} collection={FRESHMAN} />
        <Field label="Seed" value={subject.seed == null ? "not a trained run" : String(subject.seed)} />
        <Field label="Epoch" value={subject.epoch == null ? "not a trained run" : String(subject.epoch)} />
        <Field label="File" value={measured?.name || subject.file_name || "unknown"} />
        <Field label="Record" value={subject.status} />
        <Field label="Full student checkpoint" value={subject.full_student_label} />
      </div>
      {stats || update ? (
        <p className="note">
          The stored-weight statistics and the effective update below were measured on this adapter. Relative magnitude also uses the verified BASE weights. This is a newly calculated measurement. It is separate from the recorded research results.
        </p>
      ) : (
        <p className="note">No newly calculated measurement is shown yet.</p>
      )}
      {sameFile ? null : <p className="error">The stored-weight measurement and the effective-update measurement named different fingerprints.</p>}
      <details className="technical">
        <summary>File path</summary>
        <ProjectPath
          label={measured?.path_label || subject.location_label}
          display={measured?.path_display || subject.location_display}
        />
      </details>
      <h3>Measured files</h3>
      <table>
        <thead>
          <tr>
            <th>Subject</th>
            <th>Role on this screen</th>
            <th>SHA-256</th>
            <th>Record</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>{subject.id}</td>
            <td>Stored weights and effective update</td>
            <td className="hash">{measured?.sha256 || updateSha || "not returned"}</td>
            <td>{recordLabel(stats?.research?.hash_status || update?.research?.hash_status)}</td>
          </tr>
          <tr>
            <td>BASE</td>
            <td>Reference for relative magnitude</td>
            <td className="hash">{base?.observed_sha256 || "not returned"}</td>
            <td>{recordLabel(base?.status)}</td>
          </tr>
          {COMPARE.filter((id) => id !== subject.id).map((id) => {
            const other = updates[id];
            const weight = other?.research?.provenance?.weight;
            return (
              <tr key={id}>
                <td>{id}</td>
                <td>Comparison column</td>
                <td className="hash">{weight?.observed_sha256 || (other ? "not returned" : "…")}</td>
                <td>{other ? recordLabel(weight?.status || other.research?.hash_status) : "…"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}
