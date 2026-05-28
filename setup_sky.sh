#!/usr/bin/env bash
# Bootstrap a Sky/Ubuntu training box for the neurosity training harness.
#
# Run AFTER cloning the repo (with submodules) and `cd`-ing into it:
#
#     git clone --recurse-submodules https://github.com/colbhben/neurosity.git
#     cd neurosity
#     ./setup_sky.sh
#
# The script:
#   1. installs Python 3.12 + build tooling (via apt, sudo)
#   2. creates a project-local venv at ./.venv
#   3. installs torch with CUDA 12.4 wheels
#   4. installs the rest of the training deps + the dance submodule
#
# Sudo is invoked only for the apt step. The venv + pip work runs as the
# invoking user; if the script itself is run via sudo, the venv would end
# up root-owned, so we explicitly avoid that.

set -euo pipefail

# ----- preflight ------------------------------------------------------------

if [[ "${EUID}" -eq 0 ]]; then
    echo "error: do NOT run setup_sky.sh under sudo / as root." >&2
    echo "       run it as your normal user; the script will sudo when it needs to." >&2
    exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
    echo "error: sudo not found; this script needs sudo for apt installs." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

if [[ ! -d .git ]]; then
    echo "error: ${REPO_ROOT} is not a git checkout. Clone the repo (with --recurse-submodules) first, then run this script from its root." >&2
    exit 1
fi

if [[ ! -f third_party/dance/pyproject.toml ]]; then
    echo "error: submodule third_party/dance is empty. Re-clone with --recurse-submodules, or run:" >&2
    echo "       git submodule update --init --recursive" >&2
    exit 1
fi

PYTHON_BIN="python3.12"
CUDA_INDEX="https://download.pytorch.org/whl/cu124"

# ----- 1. system python -----------------------------------------------------

echo "[1/4] Installing Python 3.12 + build tooling via apt..."
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    "${PYTHON_BIN}" \
    "${PYTHON_BIN}-venv" \
    "${PYTHON_BIN}-dev" \
    git \
    build-essential \
    ca-certificates

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "error: ${PYTHON_BIN} not on PATH after install. On older Ubuntu (<24.04) you may need the deadsnakes PPA:" >&2
    echo "       sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt-get update" >&2
    exit 1
fi

# ----- 2. venv --------------------------------------------------------------

echo "[2/4] Creating venv at ${REPO_ROOT}/.venv..."
if [[ ! -d .venv ]]; then
    "${PYTHON_BIN}" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# ----- 3. torch (CUDA 12.4) -------------------------------------------------

echo "[3/4] Installing torch (CUDA 12.4 wheels)..."
pip install --index-url "${CUDA_INDEX}" torch

# ----- 4. project + dance ---------------------------------------------------

echo "[4/4] Installing training requirements + neurosity SDK + dance submodule..."
pip install -r requirements.txt -r dev-requirements.txt
pip install -e .
pip install -e third_party/dance

echo
echo "Setup complete. Activate with:"
echo "    source ${REPO_ROOT}/.venv/bin/activate"
