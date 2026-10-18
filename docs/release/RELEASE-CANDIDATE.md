# AI Model Inspector release candidate

Release candidate: 1.0.0
Private source commit: f8a596a364a676f8a8933bb6e4d8ddf1520b59e7
Public distribution commit: the initial commit of the public snapshot
Distribution branch: distribution/v1.0

Scientific validation, recorded in inspector-engine/research/freshman_validation.json:

- Freshman F5-1
- Freshman F5-2
- Freshman F5-3
- Freshman F6

Measurement engine: frozen algorithms unchanged. weight_statistics.py and effective_lora.py are the same bytes as the private source commit.

The original research configuration files and checkpoint weights remain in the private archive. This candidate ships the validation numbers, the relative packaged adapter configurations, and the source required to inspect a Safetensors file the user supplies.
