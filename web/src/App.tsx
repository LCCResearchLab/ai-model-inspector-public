/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * Local web console for AI Model Inspector.
 * Start and Stop call the local controller on this computer.
 */
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import logo from "./assets/lcc-research-logo.png";
import {
  controllerAction,
  packagedApiBase,
  isLocalApiHost,
  parseApiAddress,
  probeHealth,
  readControllerProcess,
  type ControllerProcessState,
  type InspectorHealth,
} from "./inspectorClient";
import { RESERVED_PORTS, controlPillLabel, interpretControl, type ControlPhase, type ControlView, type Pill } from "./serverControl";
import { BookConsole } from "./book";
import { WorkProgress } from "./progress";
import { loadControllerCredential } from "./credential";
import { LibraryConsole } from "./library";
import { go, parseRoute, type Route } from "./route";
import { readTextZoom, stepTextZoom, TEXT_ZOOM_DEFAULT, TEXT_ZOOM_KEY, TEXT_ZOOM_MAX, TEXT_ZOOM_MIN } from "./textZoom";

const STORAGE_KEY = "aimi-web-api-base";

export function App() {
  const [route, setRoute] = useState<Route>(parseRoute);
  const [apiBase, setApiBase] = useState(() => localStorage.getItem(STORAGE_KEY) || packagedApiBase());
  const [draft, setDraft] = useState(apiBase);
  const [health, setHealth] = useState<InspectorHealth | null>(null);
  const [view, setView] = useState<ControlView>({
    startEnabled: false,
    stopEnabled: false,
    featuresEnabled: false,
    pill: "checking",
    message: null,
  });
  const phaseRef = useRef<ControlPhase>("idle");
  const tokenRef = useRef<string | null>(null);
  const mainRef = useRef<HTMLElement>(null);
  const [textZoom, setTextZoom] = useState(() => readTextZoom(typeof localStorage === "undefined" ? null : localStorage));

  useLayoutEffect(() => {
    document.documentElement.style.setProperty("--text-zoom", String(textZoom));
    localStorage.setItem(TEXT_ZOOM_KEY, String(textZoom));
  }, [textZoom]);

  useLayoutEffect(() => {
    mainRef.current?.scrollTo(0, 0);
  }, [route]);

  useEffect(() => {
    const onHash = () => setRoute(parseRoute());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const probe = useCallback(async () => {
    if (phaseRef.current !== "idle") return;
    const parsed = parseApiAddress(apiBase);
    const localHost = parsed != null && isLocalApiHost(parsed.host);
    const reserved = parsed != null && RESERVED_PORTS.has(parsed.port);
    const found = parsed && !reserved ? await probeHealth(apiBase) : null;
    const apiReachable = Boolean(
      found?.ok === true && found.service === "ai-model-inspector" && parsed && found.port === parsed.port,
    );
    let controllerReachable = false;
    let credentialPresent = false;
    let authorized = false;
    let processState: ControllerProcessState | null = null;
    if (parsed && localHost && !reserved) {
      const token = await loadControllerCredential();
      tokenRef.current = token;
      const controller = await readControllerProcess(parsed.port, token);
      controllerReachable = controller.controllerReachable;
      credentialPresent = controller.credentialPresent;
      authorized = controller.authorized;
      processState = controller.processState;
    } else {
      tokenRef.current = null;
    }
    if (phaseRef.current !== "idle") return;
    setHealth(apiReachable ? found : null);
    setView(
      interpretControl({
        phase: "idle",
        addressValid: parsed != null,
        localHost,
        port: parsed?.port ?? null,
        controllerReachable,
        credentialPresent,
        authorized,
        processState,
        apiReachable,
      }),
    );
  }, [apiBase]);

  useEffect(() => {
    void probe();
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") void probe();
    }, 2000);
    return () => window.clearInterval(timer);
  }, [probe]);

  async function changeServer(action: "start" | "stop") {
    const parsed = parseApiAddress(apiBase);
    const token = tokenRef.current;
    if (!parsed || !token || phaseRef.current !== "idle") return;
    const nextPhase: ControlPhase = action === "start" ? "starting" : "stopping";
    phaseRef.current = nextPhase;
    setView({
      startEnabled: false,
      stopEnabled: false,
      featuresEnabled: false,
      pill: nextPhase,
      message: nextPhase === "starting" ? "Starting Inspector API…" : "Stopping Inspector API…",
    });
    try {
      await controllerAction(action, parsed.port, token, action === "start" ? 180000 : 30000);
    } finally {
      phaseRef.current = "idle";
      await probe();
    }
  }

  return (
    <div className="app">
      <nav>
        <img src={logo} alt="LCC Research" />
        <p>AI Model Inspector</p>
        <p className="mark">Local web console</p>
        <button type="button" className={route.view === "server" ? "on" : ""} onClick={() => go("#/home")}>
          Home
        </button>
        <button type="button" className={route.view === "book" || route.view === "workspace" ? "on" : ""} onClick={() => go("#/book")}>
          Books
        </button>
        <button type="button" className={route.view === "library" || route.view === "file" ? "on" : ""} onClick={() => go("#/library")}>
          Model Library
        </button>
        <button type="button" className={route.view === "about" ? "on" : ""} onClick={() => go("#/about")}>
          About
        </button>
        <TextZoomControls zoom={textZoom} onChange={setTextZoom} />
      </nav>
      <main ref={mainRef}>
        {route.view === "about" ? <AboutView health={health} /> : null}
        {route.view === "server" ? (
          <ServerView
            draft={draft}
            setDraft={setDraft}
            view={view}
            health={health}
            onSave={() => {
              const next = draft.trim().replace(/\/+$/, "");
              localStorage.setItem(STORAGE_KEY, next);
              setApiBase(next);
            }}
            onStart={() => void changeServer("start")}
            onStop={() => void changeServer("stop")}
          />
        ) : null}
        {route.view === "book" || route.view === "workspace" ? (
          view.pill === "checking" ? <p>Checking the measurement API.</p> : view.featuresEnabled ? <BookConsole apiBase={apiBase} route={route} /> : <Unavailable message={view.message} />
        ) : null}
        {route.view === "library" || route.view === "file" ? (
          view.pill === "checking" ? (
            <p>Checking the measurement API.</p>
          ) : view.featuresEnabled ? (
            <LibraryConsole
              apiBase={apiBase}
              section={route.view === "library" ? route.section : "import"}
              path={route.view === "file" ? route.path : undefined}
              repo={route.view === "library" ? route.repo : undefined}
              cloudProvider={route.view === "library" ? route.cloudProvider : undefined}
              cloudModel={route.view === "library" ? route.cloudModel : undefined}
            />
          ) : (
            <Unavailable message={view.message} />
          )
        ) : null}
      </main>
    </div>
  );
}

function TextZoomControls({ zoom, onChange }: { zoom: number; onChange: (next: number) => void }) {
  const percent = Math.round(zoom * 100);
  return (
    <div className="text-zoom">
      <p id="text-zoom-label">Text size</p>
      <div className="text-zoom-row" role="group" aria-labelledby="text-zoom-label">
        <button type="button" aria-label="Decrease text size" disabled={zoom <= TEXT_ZOOM_MIN} onClick={() => onChange(stepTextZoom(zoom, -1))}>
          A−
        </button>
        <span aria-live="polite">{percent}%</span>
        <button type="button" aria-label="Increase text size" disabled={zoom >= TEXT_ZOOM_MAX} onClick={() => onChange(stepTextZoom(zoom, 1))}>
          A+
        </button>
      </div>
      <button type="button" className="text-zoom-reset" aria-label="Reset text size" onClick={() => onChange(TEXT_ZOOM_DEFAULT)}>
        Reset
      </button>
    </div>
  );
}

function AboutView({ health }: { health: InspectorHealth | null }) {
  const version = health?.version?.trim();
  return (
    <section className="card about-page">
      <h1>AI Model Inspector</h1>
      <p className="lede">
        <em>An LCC Research Lab research instrument</em>
      </p>
      <h2>About LCC Research Lab</h2>
      <p>
        LCC Research Lab explores how artificial intelligence models learn, how their internal structures change, and what experimental evidence can tell us about those changes.
      </p>
      <p>
        Our research is documented in the <em>Adventures in AI</em> book series.
      </p>
      <h2>About AI Model Inspector</h2>
      <p>AI Model Inspector is a free research tool developed by LCC Research Lab.</p>
      <p>
        It provides tools for examining accessible model weights, comparing model updates, and exploring information exposed by Hugging Face and cloud AI providers.
      </p>
      <p>
        The Inspector distinguishes measured evidence from interpretation. Access to a cloud model's API does not imply access to its internal weights.
      </p>
      <h2>Contact LCC Research Lab</h2>
      <ul>
        <li>
          Website: <a href="https://lccresearchlab.com">https://lccresearchlab.com</a>
        </li>
        <li>
          Email: <a href="mailto:lccresearchlab@gmail.com">lccresearchlab@gmail.com</a>
        </li>
        <li>
          GitHub: <a href="https://github.com/LCCResearchLab">https://github.com/LCCResearchLab</a>
        </li>
      </ul>
      <p className="meta">Version: {version || "The Inspector API has not reported a version."}</p>
    </section>
  );
}

function Unavailable({ message }: { message: string | null }) {
  return (
    <section>
      <h1>Inspector API</h1>
      <p>{message || "The measurement API is not available."}</p>
      <button type="button" onClick={() => go("#/home")}>
        Home
      </button>
    </section>
  );
}

function pillClass(pill: Pill): string {
  if (pill === "running") return "pill ok";
  if (pill === "checking" || pill === "starting" || pill === "stopping") return "pill wait";
  return "pill bad";
}

function ServerView({
  draft,
  setDraft,
  view,
  health,
  onSave,
  onStart,
  onStop,
}: {
  draft: string;
  setDraft: (value: string) => void;
  view: ControlView;
  health: InspectorHealth | null;
  onSave: () => void;
  onStart: () => void;
  onStop: () => void;
}) {
  return (
    <section>
      <h1>Home</h1>
      <p className="lede">AI Model Inspector measures model weights. The books supply its first research datasets.</p>
      <p className="note">Start and Stop use the local server controller on this computer.</p>
      <p>
        <span className={pillClass(view.pill)}>{controlPillLabel(view.pill)}</span>
      </p>
      {health ? (
        <p className="meta">
          {health.service} {health.version} · port {health.port}
        </p>
      ) : null}
      {view.pill === "starting" || view.pill === "stopping" ? (
        <WorkProgress label={view.message || (view.pill === "starting" ? "Starting Inspector API." : "Stopping Inspector API.")} />
      ) : view.message ? (
        <p className="note">{view.message}</p>
      ) : null}
      <div className="row">
        <button type="button" disabled={!view.startEnabled} onClick={onStart}>
          Start Inspector API
        </button>
        <button type="button" disabled={!view.stopEnabled} onClick={onStop}>
          Stop Inspector API
        </button>
      </div>
      <p className="limits">They manage only AI Model Inspector, and only after the process state is verified. The controller credential is not shown.</p>
      <label htmlFor="api">API address</label>
      <div className="row">
        <input id="api" type="url" value={draft} onChange={(event) => setDraft(event.target.value)} />
        <button type="button" onClick={onSave}>
          Use this address
        </button>
      </div>
      <p className="limits">The address is stored in this browser only. It does not change the server, and it is not published.</p>
    </section>
  );
}
