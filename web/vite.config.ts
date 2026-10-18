/**
 * Authors: LCC Research Lab
 * Project: From Document to Evidence
 *
 * Local browser console. The credential route is served only to this computer.
 * The page then calls the local controller API.
 */
import fs from "node:fs";
import path from "node:path";
import { defineConfig, type Connect, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

function controllerCredential(): Plugin {
  const tokenPath = path.resolve(__dirname, "../.inspector-runtime/controller-token");
  const attach: Connect.NextHandleFunction = (req, res, next) => {
    const url = (req.url || "").split("?")[0];
    if (url !== "/controller-credential") {
      next();
      return;
    }
    const host = (req.headers.host || "").split(":")[0];
    if (req.method !== "GET" || (host !== "127.0.0.1" && host !== "localhost")) {
      res.statusCode = 403;
      res.end();
      return;
    }
    try {
      const token = fs.readFileSync(tokenPath, "utf8").trim();
      if (!token || token.length > 200) {
        res.statusCode = 404;
        res.end();
        return;
      }
      res.statusCode = 200;
      res.setHeader("Cache-Control", "no-store");
      res.setHeader("Content-Type", "text/plain; charset=utf-8");
      res.end(token);
    } catch {
      res.statusCode = 404;
      res.end();
    }
  };
  return {
    name: "inspector-controller-credential",
    configureServer(server) {
      server.middlewares.use(attach);
    },
    configurePreviewServer(server) {
      server.middlewares.use(attach);
    },
  };
}

export default defineConfig({
  plugins: [react(), controllerCredential()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
  },
  preview: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
  },
});
