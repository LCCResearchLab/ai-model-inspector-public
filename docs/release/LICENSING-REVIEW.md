<!--
Authors: LCC Research Lab
Project: From Document to Evidence

Draft for review. This file does not apply a license. Do not treat the repository as Apache-2.0 until this draft is approved and a root LICENSE file is added.
-->

# Licensing review for AI Model Inspector

Prepared 18 October 2026. Not approved. Not a public release.

## Proposed license

Apache License 2.0 for the original console and measurement source. The draft text is `LICENSE-Apache-2.0.draft` in this directory. The copyright holder in that draft is LCC Research Lab.

Apache-2.0 fits the direct dependencies: FastAPI and pytest are MIT, uvicorn and httpx are BSD-3-Clause, NumPy is BSD-3-Clause with 0BSD, MIT, Zlib, and CC0 portions, React and Vite are MIT, and TypeScript is Apache-2.0. None of the 71 installed JavaScript packages is GPL, AGPL, or SSPL. `caniuse-lite` is CC-BY-4.0 and is a development data file. Keep its attribution if a public package redistributes it.

## Bundled assets

| Asset | In Git today | Redistribution |
|---|---|---|
| `inspector-engine/bundled/freshman/adapters.safetensors` | Yes | Lab LoRA. Candidate to ship with the source. |
| `inspector-engine/bundled/freshman/adapter_config.json` | Yes | Packaged training settings. Absolute machine paths were replaced with `withheld-local-training-path`. |
| `inspector-engine/bundled/qwen/adapter_model.safetensors` | Local copy only after this change | Withheld. Qwen redistribution rights are not confirmed. |
| `inspector-engine/bundled/qwen/adapter_config.json` | Local copy only after this change | Withheld with the weights. It names a local training directory. |
| `distribution/book-three/freshman/*/adapter_config.research.json` | Yes | Scientific originals. Their SHA-256 values are the validation fingerprints, so those bytes stay unchanged. They still contain the original training paths. |
| `distribution/book-three/freshman/*/adapter_config.json` | Yes | Packaged copies. The base path is the relative `../BASE`. |
| `distribution/book-three/freshman/BASE/model.safetensors` and the adapter weight files | No. Gitignored. | SmolLM2-1.7B-Instruct weights are not in the repository. |
| `web/src/assets/lcc-research-logo.png` | Yes | Lab mark. No third-party notice was found. |

A clone without the Qwen files can still run. The Qwen page says the adapter is not part of the public package and where to place a local copy.

## Before approval

Confirm that the Freshman LoRA may be published with the source, and that the research configuration files may be published with their original training paths. Those paths are part of the recorded hashes. Replacing them would change the scientific record.
