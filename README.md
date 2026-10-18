<!--
Authors: LCC Research Lab
Project: From Document to Evidence
-->

# AI Model Inspector

**Version:** 0.1.0

This package is the public source distribution. It does not include model weights. The Book Three Freshman validation record lists the published measurement values and checkpoint hashes. The original research configurations stay in the private archive. A Safetensors file you already have can be inspected when it sits in a configured directory. The Qwen adapter and the Freshman adapter weights are not included; each screen says so when the file is absent.

**Python:** 3.12 or 3.13. Python 3.14 can run the current suite and reports deprecation warnings from Starlette and FastAPI. Those warnings are not measurement failures.

AI Model Inspector is an independent measurement instrument. The books supply its first research datasets, and a later book can be added without renaming the instrument. Version 1.0 is a desktop-first web console backed by a local Python measurement engine. Model files and measurements stay on the researcher’s computer. The API listens on `127.0.0.1` only. `INSPECTOR_HOST` cannot move it onto the local network.

Commands below are relative to the project directory.

## Layout

| Path | Role |
|------|------|
| `web/` | Web console for the browser on this computer. It calls the Inspector API. |
| `inspector-engine/` | Python API. Default port 8810. |
| `inspector-server.sh` | Start, stop, restart, and check the API. |
| `inspector-controller.sh` | Keep the local server controller available on port 8811. |
| `start-inspector-api.sh` | Install the API environment and start the server in the foreground. |

## Run

```bash
./start-inspector-api.sh
```

The API listens at `http://127.0.0.1:8810` unless another port is selected. The host is always `127.0.0.1`. Health is `GET /health`, and the response includes the listening port. Checkpoint inspection is `POST /inspect/checkpoint` with a JSON body `{ "path": "/absolute/path/file.safetensors" }`. The file must be a `.safetensors` file inside a directory listed in `inspector-engine/checkpoint_roots.txt`, or inside `inspector-engine/library/` or `inspector-engine/bundled/`. A path cannot leave those directories through `..` or a symlink. The filesystem root, the home directory, and other broad directories are not accepted as inspection roots.

The report keeps three kinds of evidence apart. Declared architecture is read from a local `config.json`. An adapter declaration is read from `adapter_config.json` when that file is present. Observed tensors come from the safetensors header. Differences are warnings. The inspection does not modify the checkpoint or the configuration files.

The Models screen can look up a Hugging Face repository and list its files without downloading them. A download starts only after the researcher selects files. Weights go to `inspector-engine/library/`, which the checkpoint inspector is allowed to read. A gated or private repository can use a Hugging Face token. The API process uses `INSPECTOR_HF_TOKEN` when that environment variable is set. Otherwise it uses a token saved from Model Library → Hugging Face → HF Token. Cloud provider API keys are separate from the Hugging Face token. Each saved credential uses its own Keychain account. On this Mac the saved token is in the login keychain. Windows and Linux do not have a credential store in the Inspector yet, so those computers use `INSPECTOR_HF_TOKEN` until a store is added. Removing a Keychain token does not unset `INSPECTOR_HF_TOKEN`. The API does not return either value, and it does not read a plaintext token file.

```bash
./inspector-server.sh start
./inspector-server.sh start --port 9000
INSPECTOR_PORT=9000 ./inspector-server.sh status
```

`--port` overrides `INSPECTOR_PORT`. When neither is set, the port is 8810. Each port has its own PID file and log in `.inspector-runtime/`.

The built web console is served by that same API process. Build it once, then open the API address. The port stays configurable; 8810 is the default.

```bash
cd web
npm install
npm run build
cd ..
./inspector-server.sh start
```

Open `http://127.0.0.1:8810`. If the API was started on another port, open that port instead. The page calls the API on the address it was opened from, unless this browser has a saved API address.

`npm run dev` remains the development server at `http://127.0.0.1:5173`. It talks to the API at `http://127.0.0.1:8810`.

The web console address is editable in the browser on this computer. It must match the local server. Changing the address in the console does not change the server’s port. The console uses `127.0.0.1`.

Start and Stop in the console talk to a separate server controller at `http://127.0.0.1:8811`. The controller stays running when the measurement API is stopped. It accepts only local requests, requires a credential stored on this Mac, and can only ask `inspector-server.sh` to check, start, or stop an Inspector process. An address that does not respond is not treated as a stopped API unless the controller has verified that the Inspector process is stopped.

```bash
./inspector-controller.sh start
```

The web console uses that controller for Start and Stop.

## Web console

From the project directory, install the browser dependencies once, then start the local page:

```bash
cd web
npm install
npm run dev
```

Open `http://127.0.0.1:5173` in a browser on this Mac. The page is not published. It reads the Inspector API at `http://127.0.0.1:8810` and, for Start and Stop, the controller at `http://127.0.0.1:8811`.

If the web console is already running, `npm run dev` reports that and asks whether to restart it. Answer `n` to leave the current server running.

## Ports

| Service | Port |
|---------|------|
| AI Model Inspector API | 8810 by default; selectable |
| AI Model Inspector controller | 8811 |
| AI Model Inspector web console | 5173 |
