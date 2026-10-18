<!--
Authors: LCC Research Lab
Project: From Document to Evidence
-->

# AI Model Inspector release candidate

Release candidate: 1.0.0
Status: not tagged and not published
Pre-release checkpoint: `ccad4c7247c9f6a489f56193a68120913bddbaa1`

This file describes the public source distribution. The operative license is the root `LICENSE`: Apache License, Version 2.0, copyright 2026 LCC Research Lab.

Scientific validation is recorded in `inspector-engine/research/freshman_validation.json`:

- Freshman F5-1
- Freshman F5-2
- Freshman F5-3
- Freshman F6

The measurement formulas in `weight_statistics.py` and `effective_lora.py` are unchanged. The validation record lists the published checkpoint hashes and measurement values. Packaged adapter settings use the relative base path `../BASE`. This distribution does not include model weights. A Safetensors file supplied by the user can be inspected when it is in a configured directory.

`v1.0.0` has not been tagged. This repository has not been published.
