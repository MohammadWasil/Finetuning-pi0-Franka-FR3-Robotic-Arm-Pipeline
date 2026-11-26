#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root dynamically regardless of where the script is called from
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"
OPENPI_DIR="${REPO_DIR}/third_party/openpi"

echo "📂 Repo directory: ${REPO_DIR}"

# 1. Install uv & create a local Python 3.11 virtual environment
python3 -m pip install -q --upgrade pip uv
uv venv "${VENV_DIR}" --python 3.11 --seed --clear

export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH}"

# 2. Install Poetry and project dependencies directly into .venv
pip install -q poetry
poetry config virtualenvs.create true

cd "${REPO_DIR}"
poetry lock
poetry install

# 3. Clone forked openpi into third_party/openpi and install into the same .venv
rm -rf "${OPENPI_DIR}"
mkdir -p "$(dirname "${OPENPI_DIR}")"
git clone --recurse-submodules https://github.com/MohammadWasil/openpi.git "${OPENPI_DIR}"

cd "${OPENPI_DIR}"
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "${VENV_DIR}/bin/python" \
  -e ./packages/openpi-client \
  -e . \
  "ml-dtypes==0.4.1" \
  "tensorstore==0.1.74" \
  "pyarrow<18.0.0" \
  "datasets>=2.19.0,<4.0.0"

# 4. Allow datasets 3.x to load Parquet metadata created with datasets 4.x ('List' -> 'Sequence')
SITE_PACKAGES="$("${VENV_DIR}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
cat << 'EOF' > "${SITE_PACKAGES}/sitecustomize.py"
try:
    import datasets.features.features as _ff
    if "List" not in _ff._FEATURE_TYPES and "Sequence" in _ff._FEATURE_TYPES:
        _ff._FEATURE_TYPES["List"] = _ff._FEATURE_TYPES["Sequence"]
except Exception:
    pass
EOF

echo "✅ Setup complete! Python version: $("${VENV_DIR}/bin/python" --version)"
