<!--
Authors: LCC Research Lab
Project: From Document to Evidence
-->

<p align="center">
  <a href="https://lccresearchlab.com">
    <img src="docs/lcc-research-logo.png" alt="LCC Research Lab" width="128">
  </a>
</p>

# AI Model Inspector

Free, open-source scientific instrument from [LCC Research Lab](https://lccresearchlab.com) for examining local model weights and LoRA updates.

The software measures. The [*Adventures in AI*](https://lccresearchlab.com/books.html) books explain the experiments, methodology, and findings. The Inspector remains free: it does not require a book purchase or an account.

The instrument and the laboratory's publications: [lccresearchlab.com/ai-model-inspector](https://lccresearchlab.com/ai-model-inspector.html)

**Version:** 1.0.1

This package is the public source distribution. It does not include model weights. The Book Three Freshman validation record lists the published measurement values and checkpoint hashes. The original research configurations stay in the private archive. A Safetensors file you already have can be inspected when it sits in a configured directory. The Qwen adapter and the Freshman adapter weights are not included; each screen says so when the file is absent.

**Python:** 3.12 or 3.13. Python 3.14 can run the current suite and reports deprecation warnings from Starlette and FastAPI. Those warnings are not measurement failures.

**Node.js:** A current Node.js LTS release with `npm` (tested with Node.js 22). The web console installs from the committed `web/package-lock.json`.

AI Model Inspector is an independent measurement instrument. The books supply its first research datasets, and a later book can be added without renaming the instrument. Version 1.0 is a desktop web console backed by a local Python measurement engine. Model files and measurements stay on your computer. The API listens on `127.0.0.1` only. `INSPECTOR_HOST` cannot move it onto the local network.

## Research

AI Model Inspector accompanies *Adventures in AI*, the book series from LCC Research Lab. The instrument examines model weights and LoRA updates on your computer. The books document the reproducible experiments behind those measurements: how the evidence was gathered, how it was interpreted, and what the internal measurements showed.

This package does not include model weights. Recorded research results that ship with the instrument can be read without the original checkpoints. The publications are available at [lccresearchlab.com](https://lccresearchlab.com).

## Platforms

| Platform | Status |
|----------|--------|
| macOS | Tested. This is the platform used for development and for the v1.0 release checks. |
| Windows | Planned. Not validated, and not a supported platform for this release. |
| Linux | Planned. Not validated, and not a supported platform for this release. |

The React console and the Python measurement engine are written to run on more than one operating system. The launch scripts, the macOS Keychain credential store, and the local controller have been exercised on macOS only. Do not treat Windows or Linux as supported until those platforms have been tested.

Commands below are relative to the project directory after a fresh copy of this package.

## Running AI Model Inspector locally

Use one terminal. Commands are relative to the project directory after you unzip the named release asset
`ai-model-inspector-1.0.1-macos-source.zip` (or after you clone this repository).

1. Install Python 3.12 or 3.13, and Node.js with `npm`.
2. Build the web console from the lockfile:

```bash
cd web
npm ci
npm run build
cd ..
```

3. Start the Inspector API (creates `inspector-engine/.venv` on first run, installs `inspector-engine/requirements.txt`, and serves the built console):

```bash
./inspector-server.sh start
```

4. Optional — enable Start and Stop in the console (listens on `127.0.0.1:8811` only):

```bash
./inspector-controller.sh start
```

5. Open `http://127.0.0.1:8810` in your browser. If you started the API on another port, open that port instead.

For console development, `npm run dev` (from `web/`) serves the page at `http://127.0.0.1:5173`. Start the API first, then open that address.

The application communicates with two local services:

- Inspector API: `http://127.0.0.1:8810`
- Local Controller: `http://127.0.0.1:8811` (application start/stop management)

All of these services run on your computer and are accessible through the loopback interface. `127.0.0.1` is the standard loopback address. It is not a public network address. AI Model Inspector does not require a publicly accessible server.

Hugging Face discovery and downloads, and cloud-provider model listings, do contact external services when you use those screens. Checkpoint inspection, stored-weight measurement, and effective-update measurement read files on your computer and do not send those files out.

## Layout

| Path | Role |
|------|------|
| `web/` | Web console. It calls the Inspector API. |
| `inspector-engine/` | Python API. Default port 8810. |
| `inspector-server.sh` | Start, stop, restart, and check the API. |
| `inspector-controller.sh` | Keep the local server controller available on port 8811. |
| `start-inspector-api.sh` | Foreground / debug only: create the API environment and start uvicorn in this terminal. Prefer `inspector-server.sh start` for normal use. |

## Fresh installation

`./inspector-server.sh start` creates `inspector-engine/.venv` when that environment is absent, installs the packages in `inspector-engine/requirements.txt`, and starts the API in the background. An existing environment is reused. Use `./start-inspector-api.sh` only when you want the API attached to the current terminal.

The API listens at `http://127.0.0.1:8810` unless another port is selected. The host is always `127.0.0.1`. Health is `GET /health`, and the response includes the listening port. Checkpoint inspection is `POST /inspect/checkpoint` with a JSON body `{ "path": "/absolute/path/file.safetensors" }`. The file must be a `.safetensors` file inside a directory listed in `inspector-engine/checkpoint_roots.txt`, or inside `inspector-engine/library/` or `inspector-engine/bundled/`. A path cannot leave those directories through `..` or a symlink. The filesystem root, the home directory, and other broad directories are not accepted as inspection roots.

The report keeps three kinds of evidence apart. Declared architecture is read from a local `config.json`. An adapter declaration is read from `adapter_config.json` when that file is present. Observed tensors come from the safetensors header. Differences are warnings. The inspection does not modify the checkpoint or the configuration files.

```bash
./inspector-server.sh start
./inspector-server.sh start --port 9000
INSPECTOR_PORT=9000 ./inspector-server.sh status
```

`--port` overrides `INSPECTOR_PORT`. When neither is set, the port is 8810. Each port has its own PID file and log in `.inspector-runtime/`.

The built web console is served by that same API process. The page calls the API on the address it was opened from, unless this browser has a saved API address. The web console address must match the local server. Changing the address in the console does not change the server’s port. The console uses `127.0.0.1`.

Start and Stop in the console talk to the local controller at `http://127.0.0.1:8811`. The controller stays running when the measurement API is stopped. It accepts only local requests, requires the local controller credential created in `.inspector-runtime/`, and can only ask `inspector-server.sh` to check, start, or stop an Inspector process. An address that does not respond is not treated as a stopped API unless the controller has verified that the Inspector process is stopped.

```bash
./inspector-controller.sh start
```

If the development console is already running, `npm run dev` reports that and asks whether to restart it. Answer `n` to leave the current server running.

## Models and credentials

The Models screen can look up a Hugging Face repository and list its files without downloading them. That lookup contacts Hugging Face. A download starts only after you select files, and the files are requested from Hugging Face. Weights go to `inspector-engine/library/`, which the checkpoint inspector is allowed to read.

A gated or private repository can use a Hugging Face token. The API process uses `INSPECTOR_HF_TOKEN` when that environment variable is set. Otherwise, on the tested macOS build, it uses a token saved from Model Library → Hugging Face → HF Token. Saved Hugging Face tokens and cloud provider keys are stored in the macOS Keychain, one account per provider. The API does not return those values, and it does not read a plaintext token file. Removing a Keychain token does not unset `INSPECTOR_HF_TOKEN`.

Cloud Models reads public model metadata from the provider for the account whose key is saved. Those requests leave your computer. They do not run inference, chat, or training, and they do not read model weights. Windows and Linux do not have a credential store in the Inspector yet. Those platforms are not supported for this release.

### API Key Security

AI Model Inspector can store credentials for OpenAI, Anthropic, Google Gemini, and Hugging Face. On this macOS release, a credential saved from the console goes into the macOS Keychain. The operating system holds that entry. The Inspector does not write the saved key into a project file, and it does not send keys to LCC Research Lab.

| Credential | Purpose | Storage |
|---|---|---|
| OpenAI API key | OpenAI model discovery | macOS Keychain |
| Anthropic API key | Anthropic model discovery | macOS Keychain |
| Google Gemini API key | Gemini model discovery | macOS Keychain |
| Hugging Face token | Access to gated or private repositories, and the Hugging Face connection test | macOS Keychain, unless `INSPECTOR_HF_TOKEN` is set for the API process |
| Controller credential | Start and stop the local API | A file created in `.inspector-runtime/`, readable only by the account running the Inspector |

`INSPECTOR_HF_TOKEN` applies only to Hugging Face. When it is set, Hugging Face requests use that environment value and do not read the Keychain token. OpenAI, Anthropic, and Gemini keys are never taken from that variable. Each provider has its own Keychain entry. Removing one does not remove the others. Removing the Hugging Face Keychain entry does not unset `INSPECTOR_HF_TOKEN`.

The local API retrieves a saved key when you test it or ask for that provider’s model list. The key is sent only to that provider, over HTTPS, and it is not placed in the URL. A saved key is not shown in the web console and is not included in the API’s replies. You can remove a saved credential from the Hugging Face or Cloud Models screen. Saving, testing, removing, and listing credentials requires the local controller credential and a browser on this computer.

The controller credential is not an API key. It is created on this computer for Start and Stop. It is not sent to Hugging Face or to a cloud provider.

The Inspector API and the controller listen on `127.0.0.1` only.

## Books

Books shows the published research results without requiring the model files.

Book One and Book Two do not include a measurement record. Book Three includes the Freshman record: LoRA rank, alpha, scaling, effective-update magnitude, relative magnitude, weight SHA-256, and the Chapter 5 direction comparisons. Those values are labeled Recorded Research Results. They are read from `inspector-engine/research/freshman_validation.json` and the Freshman registry. They are not calculated from a checkpoint.

Relative magnitude is the effective-update magnitude divided by the combined Frobenius norm of the 96 BASE attention projection matrices that correspond to the LoRA updates: Q, K, V, and O across 24 transformer layers. That denominator is not the norm of the entire BASE model. Each Chapter 5 comparison names the two Freshman adapters, such as F5-1 compared with F5-2.

Inspect Local Checkpoint calculates a new measurement when the original checkpoint files are on this computer. If a file is missing, the page says so and the recorded results stay visible. The Qwen adapter has no recorded measurement in this distribution. The base model named by the Freshman record is HuggingFaceTB/SmolLM2-1.7B-Instruct. This repository does not include that model's weights.

## Ports

| Service | Address |
|---------|---------|
| Inspector API | `http://127.0.0.1:8810` by default; the port is selectable |
| Local controller | `http://127.0.0.1:8811` |
| Web console during development | `http://127.0.0.1:5173` |
