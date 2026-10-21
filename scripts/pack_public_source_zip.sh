#!/usr/bin/env bash
# Authors: LCC Research Lab
# Project: From Document to Evidence
#
# Build the named macOS source ZIP for a public release asset.
# Usage:
#   ./scripts/pack_public_source_zip.sh 1.0.1
#   ./scripts/pack_public_source_zip.sh 1.0.1 --ref HEAD
#
# Output (gitignored):
#   .release-build/ai-model-inspector-<version>-macos-source.zip
#   .release-build/ai-model-inspector-<version>-macos-source.zip.sha256
#
# The ZIP is built from a git tree (default HEAD). It does not include
# dependencies, runtime state, credentials, or model weights.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

VERSION="${1:-}"
REF="HEAD"
if [[ $# -ge 3 && "$2" == "--ref" ]]; then
  REF="$3"
elif [[ $# -ge 2 && "$2" == --ref=* ]]; then
  REF="${2#--ref=}"
elif [[ $# -ge 2 && "$2" != --ref ]]; then
  echo "Usage: $0 <version> [--ref <git-ref>]" >&2
  exit 1
fi

if [[ -z "${VERSION}" ]] || ! printf '%s' "${VERSION}" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
  echo "Version must look like 1.2.3, without a leading v." >&2
  exit 1
fi

if [[ "${VERSION}" == "1.0.0" ]]; then
  echo "v1.0.0 is frozen. Refusing to pack a replacement asset for that release." >&2
  exit 1
fi

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need git
need zip
need shasum
need python3

PREFIX="ai-model-inspector-${VERSION}"
ASSET_BASENAME="ai-model-inspector-${VERSION}-macos-source"
OUT_DIR="${ROOT}/.release-build"
STAGE="${OUT_DIR}/stage"
TREE="${STAGE}/${PREFIX}"
ZIP_PATH="${OUT_DIR}/${ASSET_BASENAME}.zip"
SHA_PATH="${OUT_DIR}/${ASSET_BASENAME}.zip.sha256"

rm -rf "${OUT_DIR}"
mkdir -p "${STAGE}"

echo "Exporting git tree ${REF} → ${PREFIX}/"
git archive --format=tar --prefix="${PREFIX}/" "${REF}" | tar -x -C "${STAGE}"

# Not required for the documented macOS source install path.
rm -rf "${TREE}/.github"

# Stale pre-publish checkpoint note; release notes live in docs/release/vX.Y.Z.md.
rm -f "${TREE}/docs/release/RELEASE-CANDIDATE.md"

echo "Running package safety checks"
python3 - <<'PY' "${TREE}" "${VERSION}"
import re
import sys
from pathlib import Path

tree = Path(sys.argv[1])
version = sys.argv[2]
errors: list[str] = []

forbidden_suffixes = {
    ".safetensors",
    ".gguf",
    ".bin",
    ".pt",
    ".pth",
    ".onnx",
    ".pem",
    ".env",
}
forbidden_names = {
    ".env",
    "credentials.json",
    "secrets.json",
    ".ds_store",
}
forbidden_dir_names = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".inspector-runtime",
    ".mypy_cache",
    ".ruff_cache",
}

secret_re = re.compile(
    rb"(sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY|AKIA[0-9A-Z]{16})"
)

required = [
    "README.md",
    "LICENSE",
    "NOTICE",
    "start-inspector-api.sh",
    "inspector-server.sh",
    "inspector-controller.sh",
    "inspector-engine/requirements.txt",
    "inspector-engine/server.py",
    "inspector-engine/inspector_engine/__init__.py",
    "web/package.json",
    "web/package-lock.json",
    f"docs/release/v{version}.md",
]

for rel in required:
    if not (tree / rel).is_file():
        errors.append(f"missing required file: {rel}")

readme = (tree / "README.md").read_text(encoding="utf-8", errors="replace") if (tree / "README.md").is_file() else ""
if f"**Version:** {version}" not in readme:
    errors.append(f"README.md does not report Version {version}")
quick = ""
if "## Running AI Model Inspector locally" in readme and "## Layout" in readme:
    quick = readme.split("## Running AI Model Inspector locally", 1)[1].split("## Layout", 1)[0]
if "npm ci" not in quick or "./inspector-server.sh start" not in quick:
    errors.append("README quick-start must use npm ci and inspector-server.sh start")
if "./start-inspector-api.sh" in quick:
    errors.append("README quick-start must not call start-inspector-api.sh (foreground/debug only)")

init_py = tree / "inspector-engine/inspector_engine/__init__.py"
if init_py.is_file():
    text = init_py.read_text(encoding="utf-8", errors="replace")
    if f'ENGINE_VERSION = "{version}"' not in text:
        errors.append(f"ENGINE_VERSION is not {version}")

server_py = tree / "inspector-engine/server.py"
if server_py.is_file():
    text = server_py.read_text(encoding="utf-8", errors="replace")
    if f'API_VERSION = "{version}"' not in text:
        errors.append(f"API_VERSION is not {version}")

pkg = tree / "web/package.json"
if pkg.is_file():
    if f'"version": "{version}"' not in pkg.read_text(encoding="utf-8", errors="replace"):
        errors.append(f"web/package.json version is not {version}")

for path in tree.rglob("*"):
    rel = path.relative_to(tree).as_posix()
    name = path.name.lower()
    if path.is_dir():
        if name in forbidden_dir_names:
            errors.append(f"forbidden directory in package: {rel}")
        continue
    if name in forbidden_names or name.startswith(".env."):
        errors.append(f"forbidden file in package: {rel}")
        continue
    lower = name
    for suf in forbidden_suffixes:
        if lower.endswith(suf):
            errors.append(f"forbidden weight/credential-like file: {rel}")
            break
    if "adapter_model" in lower or lower.endswith(".safetensors"):
        errors.append(f"forbidden model weight file: {rel}")
    # Skip binary-ish scans for known images.
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico"}:
        continue
    try:
        data = path.read_bytes()
    except OSError as exc:
        errors.append(f"unreadable file {rel}: {exc}")
        continue
    if secret_re.search(data):
        errors.append(f"secret-like pattern in {rel}")

if errors:
    print("PACKAGE_CHECK_FAILED", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    sys.exit(1)
print("Package safety checks passed")
PY

echo "Creating ${ZIP_PATH}"
rm -f "${ZIP_PATH}" "${SHA_PATH}"
(
  cd "${STAGE}"
  zip -X -r "${ZIP_PATH}" "${PREFIX}" >/dev/null
)

(
  cd "${OUT_DIR}"
  shasum -a 256 "${ASSET_BASENAME}.zip" | tee "${ASSET_BASENAME}.zip.sha256"
)

echo "Packed ${ZIP_PATH}"
echo "Checksum ${SHA_PATH}"
