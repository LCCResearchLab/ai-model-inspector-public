/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 */
export type BookSection = "subjects" | "checkpoints" | "magnitudes" | "directions" | "findings";

export type BookCollection = "index" | "freshman" | "qwen";

export type LibrarySection = "discover" | "downloaded" | "import" | "token" | "providers" | "models" | "capabilities";

export type WorkspaceTab = "overview" | "checkpoint" | "magnitudes" | "directions";

export type Route =
  | { view: "server" }
  | { view: "about" }
  | { view: "book"; section: BookSection; collection: BookCollection }
  | { view: "workspace"; collection: "freshman" | "qwen"; id: string; tab: WorkspaceTab }
  | { view: "library"; section: LibrarySection; repo?: string; cloudProvider?: string; cloudModel?: string }
  | { view: "file"; path: string };

function workspaceTab(part: string | undefined): WorkspaceTab {
  if (part === "checkpoint") return "checkpoint";
  if (part === "weights" || part === "magnitudes") return "magnitudes";
  if (part === "directions") return "directions";
  return "overview";
}

export function go(hash: string) {
  window.location.hash = hash.startsWith("#") ? hash : `#${hash}`;
}

export function cloudModelHash(provider: string, modelId: string): string {
  const id = modelId.split("/").filter(Boolean).map((part) => encodeURIComponent(part)).join("/");
  return `#/library/cloud/models/${encodeURIComponent(provider)}/${id}`;
}

export function modelHash(repo: string): string {
  const parts = repo.split("/").filter(Boolean).map((part) => encodeURIComponent(part));
  return `#/library/huggingface/model/${parts.join("/")}`;
}

export function parseRoute(): Route {
  const hash = window.location.hash.replace(/^#/, "") || "/home";
  const [pathPart, queryPart] = hash.split("?");
  const parts = pathPart.split("/").filter(Boolean);
  const params = new URLSearchParams(queryPart || "");
  if (parts[0] === "book" && parts[1] === "freshman" && !parts[2]) return { view: "book", section: "subjects", collection: "freshman" };
  if (parts[0] === "book" && parts[1] === "qwen") return { view: "workspace", collection: "qwen", id: "Qwen", tab: workspaceTab(parts[2]) };
  if (parts[0] === "book" && parts[1] === "checkpoints" && !parts[2]) return { view: "workspace", collection: "freshman", id: "F5-1", tab: "checkpoint" };
  if (parts[0] === "book" && parts[1] === "magnitudes" && !parts[2]) return { view: "workspace", collection: "freshman", id: "F5-1", tab: "magnitudes" };
  if (parts[0] === "book" && parts[1] === "directions" && !parts[2]) return { view: "workspace", collection: "freshman", id: "F5-1", tab: "directions" };
  if (parts[0] === "book" && parts[1] === "findings" && !parts[2]) return { view: "book", section: "findings", collection: "freshman" };
  if (parts[0] === "book" && parts[1] && parts[2]) return { view: "workspace", collection: "freshman", id: decodeURIComponent(parts[1]), tab: workspaceTab(parts[2]) };
  if (parts[0] === "book" && parts[1]) return { view: "workspace", collection: "freshman", id: decodeURIComponent(parts[1]), tab: "overview" };
  if (parts[0] === "book") return { view: "book", section: "subjects", collection: "index" };
  if (parts[0] === "library" && (parts[1] === "keys" || (parts[1] === "huggingface" && parts[2] === "token"))) {
    return { view: "library", section: "token" };
  }
  if (parts[0] === "library" && parts[1] === "huggingface" && parts[2] === "downloaded") return { view: "library", section: "downloaded" };
  if (parts[0] === "library" && parts[1] === "huggingface" && parts[2] === "import") return { view: "library", section: "import" };
  if (parts[0] === "library" && parts[1] === "huggingface" && parts[2] === "model" && parts.length >= 5) {
    return { view: "library", section: "discover", repo: parts.slice(3).map((part) => decodeURIComponent(part)).join("/") };
  }
  if (parts[0] === "library" && parts[1] === "huggingface") return { view: "library", section: "discover" };
  if (parts[0] === "library" && parts[1] === "cloud" && parts[2] === "providers" && parts[3]) {
    return { view: "library", section: "providers", cloudProvider: decodeURIComponent(parts[3]) };
  }
  if (parts[0] === "library" && parts[1] === "cloud" && parts[2] === "models" && parts[3] && parts.length > 4) {
    return {
      view: "library",
      section: "models",
      cloudProvider: decodeURIComponent(parts[3]),
      cloudModel: parts.slice(4).map((part) => decodeURIComponent(part)).join("/"),
    };
  }
  if (parts[0] === "library" && parts[1] === "cloud" && parts[2] === "models" && parts[3]) {
    return { view: "library", section: "models", cloudProvider: decodeURIComponent(parts[3]) };
  }
  if (parts[0] === "library" && parts[1] === "cloud" && parts[2] === "models") return { view: "library", section: "models" };
  if (parts[0] === "library" && parts[1] === "cloud" && parts[2] === "capabilities") return { view: "library", section: "capabilities" };
  if (parts[0] === "library" && (parts[1] === "cloud" || parts[1] === "providers")) return { view: "library", section: "providers" };
  if (parts[0] === "library" && (parts[1] === "inspect" || parts[1] === "local")) return { view: "library", section: "import" };
  if (parts[0] === "library") return { view: "library", section: "discover" };
  if (parts[0] === "file") return { view: "file", path: params.get("path") || "" };
  if (parts[0] === "about") return { view: "about" };
  return { view: "server" };
}
