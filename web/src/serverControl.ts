/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * Decide which home-screen controls are usable from verified process state.
 * An API address that does not respond is not the same as a stopped Inspector.
 */

export const RESERVED_PORTS = new Set([8787, 8799, 8811]);

export type ControlPhase = "idle" | "starting" | "stopping";
export type ProcessState = "running" | "stopped" | "foreign" | "unresponsive" | "error" | "unknown";
export type Pill = "checking" | "running" | "stopped" | "unreachable" | "starting" | "stopping";

export type ControlInput = {
  phase: ControlPhase;
  addressValid: boolean;
  localHost: boolean;
  port: number | null;
  controllerReachable: boolean;
  credentialPresent: boolean;
  authorized: boolean;
  processState: ProcessState | null;
  apiReachable: boolean;
};

export type ControlView = {
  startEnabled: boolean;
  stopEnabled: boolean;
  featuresEnabled: boolean;
  pill: Pill;
  message: string | null;
};

export function interpretControl(input: ControlInput): ControlView {
  if (input.phase === "starting" || input.phase === "stopping") {
    return {
      startEnabled: false,
      stopEnabled: false,
      featuresEnabled: false,
      pill: input.phase,
      message: input.phase === "starting" ? "Starting Inspector API…" : "Stopping Inspector API…",
    };
  }
  if (!input.addressValid || input.port == null) {
    return blocked("Enter an API address that includes a host and a port.");
  }
  if (RESERVED_PORTS.has(input.port)) {
    return blocked(`Port ${input.port} is reserved for another local service. Nothing was changed.`);
  }
  if (!input.localHost) {
    return {
      startEnabled: false,
      stopEnabled: false,
      featuresEnabled: input.apiReachable,
      pill: input.apiReachable ? "running" : "unreachable",
      message: input.apiReachable
        ? "Start and Stop manage the Inspector on this computer. This address points to another host."
        : "The API address did not respond. That does not mean the Inspector on this computer is stopped.",
    };
  }
  const verified =
    input.controllerReachable && input.credentialPresent && input.authorized && input.processState != null;
  if (!input.controllerReachable) {
    return {
      startEnabled: false,
      stopEnabled: false,
      featuresEnabled: input.apiReachable,
      pill: input.apiReachable ? "running" : "unreachable",
      message: input.apiReachable
        ? "The API is responding. Start and Stop stay disabled until the server controller can confirm the process."
        : "Server controller is not available, and the API did not respond. The process state was not verified. This does not mean the API is stopped.",
    };
  }
  if (!input.credentialPresent || !input.authorized) {
    return {
      startEnabled: false,
      stopEnabled: false,
      featuresEnabled: input.apiReachable,
      pill: input.apiReachable ? "running" : "unreachable",
      message: input.credentialPresent
        ? "The server controller rejected the credential. The process state was not verified. This does not mean the API is stopped."
        : "The console does not have the local controller credential, so the process state was not verified. This does not mean the API is stopped.",
    };
  }
  if (!verified || input.processState == null || input.processState === "unknown" || input.processState === "error") {
    return {
      startEnabled: false,
      stopEnabled: false,
      featuresEnabled: input.apiReachable,
      pill: input.apiReachable ? "running" : "unreachable",
      message: "The process state could not be verified. This does not mean the API is stopped.",
    };
  }
  if (input.processState === "foreign") {
    return {
      startEnabled: false,
      stopEnabled: false,
      featuresEnabled: false,
      pill: "unreachable",
      message: `Port ${input.port} is in use by another program. The Inspector was not started or stopped.`,
    };
  }
  if (input.processState === "unresponsive") {
    return {
      startEnabled: false,
      stopEnabled: true,
      featuresEnabled: false,
      pill: "unreachable",
      message: "The Inspector process is running, but the API is not responding.",
    };
  }
  if (input.processState === "stopped") {
    return {
      startEnabled: !input.apiReachable,
      stopEnabled: false,
      featuresEnabled: input.apiReachable,
      pill: input.apiReachable ? "running" : "stopped",
      message: input.apiReachable
        ? "The API responded, but the controller did not find an Inspector process. Start and Stop stay disabled."
        : "Inspector API is stopped.",
    };
  }
  return {
    startEnabled: false,
    stopEnabled: input.processState === "running",
    featuresEnabled: input.apiReachable,
    pill: input.apiReachable ? "running" : "unreachable",
    message: input.apiReachable ? null : "The controller found the Inspector process, but the console could not reach the API.",
  };
}

export function controlPillLabel(pill: Pill): string {
  if (pill === "checking") return "…";
  if (pill === "running") return "API running";
  if (pill === "stopped") return "API stopped";
  if (pill === "starting") return "Starting";
  if (pill === "stopping") return "Stopping";
  return "API unreachable";
}

function blocked(message: string): ControlView {
  return {
    startEnabled: false,
    stopEnabled: false,
    featuresEnabled: false,
    pill: "unreachable",
    message,
  };
}
