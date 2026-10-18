/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * Start the local web console. If it is already running, ask before restarting it.
 */
import { execFileSync, spawn } from "node:child_process";
import net from "node:net";
import path from "node:path";
import readline from "node:readline";
import { fileURLToPath } from "node:url";

const PORT = 5173;
const HOST = "127.0.0.1";
const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function listening() {
  return new Promise((resolve) => {
    const socket = net.connect({ port: PORT, host: HOST }, () => {
      socket.end();
      resolve(true);
    });
    socket.on("error", () => resolve(false));
  });
}

function listenerPids() {
  try {
    const output = execFileSync("lsof", ["-nP", "-t", `-iTCP:${PORT}`, "-sTCP:LISTEN"], { encoding: "utf8" });
    return output.split(/\s+/).map((item) => item.trim()).filter(Boolean);
  } catch {
    return [];
  }
}

function commandOf(pid) {
  try {
    return execFileSync("ps", ["-p", pid, "-o", "command="], { encoding: "utf8" }).trim();
  } catch {
    return "";
  }
}

function isWebConsole(pid) {
  const command = commandOf(pid);
  return command.includes("vite") && command.includes(webRoot);
}

function ask() {
  return new Promise((resolve) => {
    const prompts = readline.createInterface({ input: process.stdin, output: process.stdout });
    prompts.question("Would you like to restart? y/n ", (answer) => {
      prompts.close();
      resolve(answer.trim().toLowerCase());
    });
  });
}

function startVite() {
  const vite = path.join(webRoot, "node_modules", ".bin", "vite");
  const child = spawn(vite, [], { cwd: webRoot, stdio: "inherit" });
  child.on("exit", (code, signal) => {
    if (signal) process.kill(process.pid, signal);
    else process.exit(code ?? 0);
  });
}

async function waitUntilFree() {
  for (let attempt = 0; attempt < 25; attempt += 1) {
    if (!(await listening())) return true;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  return false;
}

if (!(await listening())) {
  startVite();
} else {
  const pids = listenerPids().filter(isWebConsole);
  if (pids.length === 0) {
    console.error(`Port ${PORT} is already in use by another program. The web console was not started.`);
    process.exit(1);
  }
  console.log(`The server is running on port ${PORT}.`);
  const answer = await ask();
  if (answer !== "y" && answer !== "yes") {
    console.log(`Left the server running at http://${HOST}:${PORT}`);
    process.exit(0);
  }
  for (const pid of pids) {
    try {
      process.kill(Number(pid), "SIGTERM");
    } catch {
      // The process already exited.
    }
  }
  if (!(await waitUntilFree())) {
    console.error(`Port ${PORT} did not become free. The web console was not restarted.`);
    process.exit(1);
  }
  startVite();
}
