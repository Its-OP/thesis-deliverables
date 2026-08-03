#!/bin/bash
# =============================================================================
# Server setup script for the low-pT tau thesis-deliverables project.
#
# Designed to be uploaded directly to the workspace directory of an external
# GPU server. It will:
#   1. Clone the thesis-deliverables repository
#   2. Download and extract the extended dataset (train/ + eval/ + test/
#      parquet shards: ~16.5 GB zipped, ~18.8 GB extracted, so ~36 GB of
#      free disk is needed while extracting and ~19 GB once the zip is
#      deleted)
#   3. Set up conda and install all dependencies (editable pip package)
#
# Data is downloaded BEFORE conda setup because the dataset zip requires
# temporary disk space that may not be available once the conda
# environment (~8 GB) is installed.
#
# Usage:
#   chmod +x setup_server.sh
#   ./setup_server.sh
# =============================================================================
set -euo pipefail

# ---- Configuration ----
ORIGIN_REPO="https://github.com/Its-OP/thesis-deliverables.git"
REPO_DIR="thesis-deliverables"
CONDA_ENV_NAME="part"
PYTHON_VERSION="3.13"
# Google Drive file ID for the EXTENDED (76-column) dataset archive: it
# holds train/ (10 shards), eval/ (9 shards) and test/ (5 shards, still the
# legacy 17-column schema). Clearing this makes the script print
# manual-transfer instructions instead of downloading. (The old ID
# 1xupCmSJtjKtWB0jbemXfW4a9pW7Pl7Mj is the LEGACY 17-column dataset — do
# not reuse it.)
GDRIVE_DATA_ZIP_ID="1Yq7ITL7cSV1YD6kEHyyJjV5N177-e9mG"

EXPECTED_TRAIN_SHARDS=10
EXPECTED_EVAL_SHARDS=9

echo "============================================"
echo "  Low-pT Tau Deliverables — Server Setup"
echo "============================================"

# ---- Step 1: Fetch repository ----
echo ""
echo "[1/4] Fetching repository..."

if [ -d ".git" ] && git remote get-url origin 2>/dev/null | grep -q "thesis-deliverables"; then
    echo "  Already inside the repository, pulling latest changes..."
    git fetch origin
    git checkout main
    git pull origin main
else
    if [ -d "${REPO_DIR}" ]; then
        echo "  '${REPO_DIR}/' already exists, pulling latest changes..."
        cd "${REPO_DIR}" && git pull && cd ..
    else
        git clone "$ORIGIN_REPO" "${REPO_DIR}"
    fi
    cd "${REPO_DIR}"
fi

echo "  Repository ready."

# ---- Step 2: Download and extract dataset ----
echo ""
echo "[2/4] Checking dataset (extended 76-column shards)..."

DATASET_DIR="data/low-pt"
TRAIN_DIR="${DATASET_DIR}/train"
EVAL_DIR="${DATASET_DIR}/eval"
TEST_DIR="${DATASET_DIR}/test"
mkdir -p "$TRAIN_DIR" "$EVAL_DIR" "$TEST_DIR"

TRAIN_COUNT=$(find "${TRAIN_DIR}" -maxdepth 1 -name "train_*.parquet" 2>/dev/null | wc -l | tr -d ' ')
EVAL_COUNT=$(find "${EVAL_DIR}" -maxdepth 1 -name "eval_*.parquet" 2>/dev/null | wc -l | tr -d ' ')

if [ "$TRAIN_COUNT" -lt "$EXPECTED_TRAIN_SHARDS" ] || [ "$EVAL_COUNT" -lt "$EXPECTED_EVAL_SHARDS" ]; then
    if [ -z "$GDRIVE_DATA_ZIP_ID" ]; then
        echo "  Data missing (${TRAIN_COUNT}/${EXPECTED_TRAIN_SHARDS} train,"
        echo "  ${EVAL_COUNT}/${EXPECTED_EVAL_SHARDS} eval) and no Google Drive"
        echo "  ID is configured. Transfer the shards manually, e.g. from the"
        echo "  local machine:"
        echo "    zip then scp a single archive into ${REPO_DIR}/${DATASET_DIR}/"
        echo "    and unzip so that train_*.parquet land in ${TRAIN_DIR}/ and"
        echo "    eval_*.parquet in ${EVAL_DIR}/"
        echo "  Then re-run this script."
        exit 1
    fi

    ZIP_PATH="${DATASET_DIR}/dataset.zip"

    # Install gdown into system python (not conda — env doesn't exist yet)
    pip install -q gdown 2>/dev/null || pip3 install -q gdown

    # -c resumes a partial download: the archive is ~16.5 GB, so a dropped
    # connection must not restart from zero
    echo "  Downloading extended dataset (~16.5 GB: train + eval + test)..."
    gdown -c "https://drive.google.com/uc?id=${GDRIVE_DATA_ZIP_ID}" -O "${ZIP_PATH}"

    # Validate download is not an HTML error page
    FILE_SIZE_BYTES=$(wc -c < "${ZIP_PATH}" | tr -d ' ')
    if [ "$FILE_SIZE_BYTES" -lt 100000 ]; then
        echo "  ERROR: Downloaded file is only ${FILE_SIZE_BYTES} bytes — likely an HTML error page"
        echo "  (a public file this large can hit the Google Drive download quota)."
        rm -f "${ZIP_PATH}"
        echo "  Please scp the dataset archive manually into ${DATASET_DIR}/ and extract it."
        exit 1
    fi

    echo "  Extracting..."
    unzip -o "${ZIP_PATH}" -d "${DATASET_DIR}/"
    rm "${ZIP_PATH}"

    TRAIN_COUNT=$(find "${TRAIN_DIR}" -maxdepth 1 -name "train_*.parquet" | wc -l | tr -d ' ')
    EVAL_COUNT=$(find "${EVAL_DIR}" -maxdepth 1 -name "eval_*.parquet" | wc -l | tr -d ' ')
    TEST_COUNT=$(find "${TEST_DIR}" -maxdepth 1 -name "test_*.parquet" | wc -l | tr -d ' ')
    echo "  Extracted: ${TRAIN_COUNT} train, ${EVAL_COUNT} eval, ${TEST_COUNT} test shards"
else
    TEST_COUNT=$(find "${TEST_DIR}" -maxdepth 1 -name "test_*.parquet" 2>/dev/null | wc -l | tr -d ' ')
    echo "  Data already present: ${TRAIN_COUNT} train, ${EVAL_COUNT} eval, ${TEST_COUNT} test shards"
fi

# ---- Step 3: Set up conda and install dependencies ----
echo ""
echo "[3/4] Setting up conda..."

# Find conda installation
if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base)
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_BASE="$HOME/anaconda3"
elif [ -d "/opt/miniconda3" ]; then
    CONDA_BASE="/opt/miniconda3"
else
    echo "  Conda not found. Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    CONDA_BASE="$HOME/miniconda3"
fi

# Initialize conda for this shell session
source "${CONDA_BASE}/etc/profile.d/conda.sh"
echo "  Using conda at: $CONDA_BASE"

# Accept Anaconda ToS non-interactively (required since conda 25.x for main/r channels)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "  Environment '${CONDA_ENV_NAME}' already exists. Activating..."
else
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
    echo "  Environment created."
fi

conda activate "$CONDA_ENV_NAME"
echo "  Active Python: $(python --version) at $(which python)"

# Ensure pip exists in the env (conda create sometimes omits it on vast.ai;
# always use `python -m pip` — bare `pip` may resolve to the system python)
python -m pip --version >/dev/null 2>&1 || python -m ensurepip --upgrade

# ---- Step 3.5: Install system packages ----
echo ""
echo "  Installing system packages..."

if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y -qq screen
    echo "  screen installed."
elif command -v yum &>/dev/null; then
    sudo yum install -y -q screen
    echo "  screen installed."
else
    echo "  WARNING: Could not detect package manager. Please install screen manually."
fi

# ---- Step 4: Install dependencies ----
echo ""
echo "[4/4] Installing dependencies..."

# Install PyTorch with CUDA support (detect CUDA version)
echo "  Installing PyTorch..."
CUDA_VERSION_STRING=""
if command -v nvidia-smi &>/dev/null; then
    # Extract CUDA version from nvidia-smi (e.g., "12.4" -> "cu124")
    CUDA_FULL=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)
    if [ -n "$CUDA_FULL" ]; then
        CUDA_MAJOR=$(echo "$CUDA_FULL" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_FULL" | cut -d. -f2)
        CUDA_VERSION_STRING="cu${CUDA_MAJOR}${CUDA_MINOR}"
        echo "  Detected CUDA $CUDA_FULL (${CUDA_VERSION_STRING})"
        # PyTorch publishes wheels only for selected CUDA versions — a 13.1
        # driver has no cu131 index — so fall back to the default PyPI
        # wheel, which already bundles a recent CUDA runtime
        if ! wget -q --spider "https://download.pytorch.org/whl/${CUDA_VERSION_STRING}/torch/"; then
            echo "  No wheel index for ${CUDA_VERSION_STRING}; using the default PyPI build."
            CUDA_VERSION_STRING=""
        fi
    fi
else
    echo "  WARNING: no nvidia-smi found — this box may have no GPU."
fi

if [ -n "$CUDA_VERSION_STRING" ]; then
    python -m pip install torch --extra-index-url "https://download.pytorch.org/whl/${CUDA_VERSION_STRING}"
else
    python -m pip install torch
fi

# Install conda-forge packages
echo "  Installing pyarrow via conda-forge..."
conda install -y -c conda-forge pyarrow

# Install the deliverables package (editable) with test + condor extras —
# this provides utils/, networks/, scripts/ and the vendored weaver package
echo "  Installing thesis-deliverables (editable)..."
python -m pip install -e '.[test,condor]'

# Remaining pinned/runtime dependencies (sklearn pinned to 1.8.0: the
# third-pion GBDT joblib artifacts were pickled with that version)
echo "  Installing runtime dependencies..."
python -m pip install -r requirements.txt
python -m pip install 'scikit-learn==1.8.0' tensorboard pandas h5py vector requests

echo "  All dependencies installed."

# ---- Done ----
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "To activate the environment:"
echo "  conda activate ${CONDA_ENV_NAME}"
echo ""
echo "To start training:"
echo "  bash train_prefilter.sh"
echo ""
echo "Train data: ${TRAIN_DIR}/ (${TRAIN_COUNT} shards)"
echo "Eval data:  ${EVAL_DIR}/ (${EVAL_COUNT} shards)"
echo "Test data:  ${TEST_DIR}/ (${TEST_COUNT} shards)"
