/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * Read the local controller credential for this browser session.
 * It is held in memory and passed to the local controller API.
 */

export async function loadControllerCredential(): Promise<string | null> {
  try {
    const res = await fetch("/controller-credential", { cache: "no-store" });
    if (!res.ok) return null;
    const token = (await res.text()).trim();
    if (!token || token.length > 200 || token.includes("\n")) return null;
    return token;
  } catch {
    return null;
  }
}
