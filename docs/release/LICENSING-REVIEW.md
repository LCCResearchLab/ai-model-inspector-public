<!--
Authors: LCC Research Lab
Project: From Document to Evidence
-->

# Licensing review for the public distribution

Prepared 18 October 2026 for the public snapshot at `42ec28e`. This file is an inventory. The license grant is the root `LICENSE`.

## Software and model licenses are separate

Apache-2.0 is the license for the original source code in this distribution: the measurement engine, the web console, the tests, the launch scripts, and the documentation written for this project. The copyright holder is LCC Research Lab.

That license does not grant redistribution rights for third-party model weights or adapters. A user who wants to inspect a checkpoint obtains that file under its own license and points the Inspector at it. Shipping this source code does not ship those weights.

Apache-2.0 is appropriate for the original source. The Python and JavaScript dependencies used to build and run it are permissive: MIT, BSD, ISC, and Apache-2.0, plus the CC-BY-4.0 data file named below. None of them is GPL, AGPL, or SSPL. `NOTICE` carries the attributions those licenses require.

## What this distribution includes

| Included file | What it is | License treatment |
|---|---|---|
| Python engine, web console, tests, and launch scripts | Original Inspector source | Apache-2.0, root `LICENSE` |
| `NOTICE` | Third-party attributions | Kept with the source, as Apache-2.0 requires |
| `inspector-engine/research/freshman_validation.json` | Recorded hashes and measurement values | Lab record. Not a weight file. |
| `inspector-engine/research/book_three_freshman.json` | Registry of expected hashes and relative paths | Lab record. Not a weight file. |
| `distribution/book-three/freshman/F5-1` through `F6` `adapter_config.json` | Packaged adapter settings. The base path is the relative `../BASE`. | Lab configuration. Not adapter weights. |
| `distribution/book-three/freshman/BASE/config.json` | Architecture declaration for the starting model | Configuration only. The weight file is not included. |
| `bundled/freshman/adapter_config.json` | Training settings with local paths replaced by `withheld-local-training-path` | Lab configuration. The adapter weights are not included. |
| `bundled/freshman/manifest.json` and `bundled/qwen/manifest.json` | Notes that those weights are not in this package | Lab documentation |
| `web/src/assets/lcc-research-logo.png` | LCC Research Lab mark | No third-party notice was found in the file |

## What the user obtains separately

| Not in this Git tree | Why |
|---|---|
| Book Three `BASE/model.safetensors` and the F5-1, F5-2, F5-3, and F6 adapter weights | Model weights. Their hashes are recorded. The files are not redistributed here. |
| `bundled/freshman/adapters.safetensors` | Freshman adapter weights. Not included. |
| Qwen `adapter_model.safetensors` and `adapter_config.json` | Redistribution is not cleared. The manifest says where a local copy can be placed. |
| `adapter_config.research.json` | Original research configurations. They are not in this distribution. |

No `.safetensors`, `.gguf`, `.pt`, `.bin`, or `.onnx` file is tracked.

SmolLM2-1.7B-Instruct, when a user obtains it from its publisher, is published under Apache-2.0. That model license is not granted by this repository. The Qwen adapter remains withheld.

## Third-party software

Direct Python packages in `inspector-engine/requirements.txt`: FastAPI 0.115.6 (MIT), Uvicorn 0.34.0 (BSD-3-Clause), NumPy 2.5.4 (BSD-3-Clause, with portions under 0BSD, MIT, Zlib, and CC0-1.0).

Development packages in `requirements-dev.txt`: pytest 8.3.5 (MIT), HTTPX 0.28.1 (BSD-3-Clause).

Direct web packages: React and React DOM 18.3.1 (MIT), Vite (MIT), `@vitejs/plugin-react` (MIT), TypeScript (Apache-2.0).

`web/package-lock.json` records 121 JavaScript packages: 112 MIT, 5 ISC, 2 Apache-2.0, 1 BSD-3-Clause (`source-map-js`), and 1 CC-BY-4.0 (`caniuse-lite`). No GPL, AGPL, or SSPL package is listed. `caniuse-lite` is browser-compatibility data used while building the console. `NOTICE` attributes it.

## Reproducibility and this license

The validation record lists the published checkpoint hashes and measurement values. Five tests recompute those values from the weight files. They skip when the files are absent. Their skip does not change the license and does not mean the recorded numbers were omitted. Repeating the measurement requires the user to supply the weight files under the model licenses above.
