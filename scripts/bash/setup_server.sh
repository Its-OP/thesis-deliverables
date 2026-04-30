#!/bin/bash
# =============================================================================
# Server setup for the low-pT tau cascade (thesis-deliverables).
#
# Uploaded directly to the GPU server's workspace, run once to:
#   1. Clone the deliverables repo
#   2. Download and extract the parquet dataset (~20 GB temp disk)
#   3. Create conda env `part` (python 3.13) with torch (CUDA), pyarrow, deps
#   4. pip install -e . so the repo's packages are importable
#
# Usage:
#   chmod +x setup_server.sh
#   ./setup_server.sh
# =============================================================================
set -euo pipefail

# ---- Configuration ----
ORIGIN_REPO="https://github.com/Its-OP/thesis-deliverables.git"
REPO_DIR="deliverables"
CONDA_ENV_NAME="part"
PYTHON_VERSION="3.13"
# Google Drive file ID for the split parquet data archive (train/ + val/)
GDRIVE_DATA_ZIP_ID="1xupCmSJtjKtWB0jbemXfW4a9pW7Pl7Mj"

echo "============================================"
echo "  thesis-deliverables — Server Setup"
echo "============================================"

# ---- Step 1: Fetch deliverables repo ----
echo ""
echo "[1/4] Fetching deliverables repo..."

if [ -d ".git" ] && git remote get-url origin 2>/dev/null | grep -q "thesis-deliverables"; then
    echo "  Already inside the deliverables repo, pulling latest changes..."
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

echo "  Repo ready."

# ---- Step 2: Download and extract dataset ----
echo ""
echo "[2/4] Downloading dataset from Google Drive..."

DATASET_DIR="data/low-pt"
TRAIN_DIR="${DATASET_DIR}/train"
VAL_DIR="${DATASET_DIR}/val"
mkdir -p "$TRAIN_DIR" "$VAL_DIR"

TRAIN_COUNT=$(find "${TRAIN_DIR}" -maxdepth 1 -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
VAL_COUNT=$(find "${VAL_DIR}" -maxdepth 1 -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')

if [ "$TRAIN_COUNT" -lt 10 ] || [ "$VAL_COUNT" -lt 7 ]; then
    ZIP_PATH="${DATASET_DIR}/dataset.zip"

    # Install gdown into system python (not conda — env doesn't exist yet)
    pip install -q gdown 2>/dev/null || pip3 install -q gdown

    echo "  Downloading split data (train + val)..."
    gdown "https://drive.google.com/uc?id=${GDRIVE_DATA_ZIP_ID}" -O "${ZIP_PATH}"

    # Validate download is not an HTML error page
    FILE_SIZE_BYTES=$(wc -c < "${ZIP_PATH}" | tr -d ' ')
    if [ "$FILE_SIZE_BYTES" -lt 100000 ]; then
        echo "  ERROR: Downloaded file is only ${FILE_SIZE_BYTES} bytes — likely an HTML error page."
        rm -f "${ZIP_PATH}"
        echo "  Please scp the dataset.zip manually into ${DATASET_DIR}/ and re-run."
        exit 1
    fi

    echo "  Extracting..."
    unzip -o "${ZIP_PATH}" -d "${DATASET_DIR}/"
    rm "${ZIP_PATH}"

    TRAIN_COUNT=$(find "${TRAIN_DIR}" -maxdepth 1 -name "*.parquet" | wc -l | tr -d ' ')
    VAL_COUNT=$(find "${VAL_DIR}" -maxdepth 1 -name "*.parquet" | wc -l | tr -d ' ')
    echo "  Extracted: ${TRAIN_COUNT} train files, ${VAL_COUNT} val files"
else
    echo "  Data already present: ${TRAIN_COUNT} train files, ${VAL_COUNT} val files"
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
elif [ -d "/root/miniconda3" ]; then
    CONDA_BASE="/root/miniconda3"
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

# ---- Step 3.5: Install system packages ----
echo ""
echo "  Installing system packages..."

if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y -qq screen unzip
    echo "  screen + unzip installed."
elif command -v yum &>/dev/null; then
    sudo yum install -y -q screen unzip
    echo "  screen + unzip installed."
else
    echo "  WARNING: Could not detect package manager. Please install screen + unzip manually."
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
    fi
fi

if [ -n "$CUDA_VERSION_STRING" ]; then
    pip install torch --extra-index-url "https://download.pytorch.org/whl/${CUDA_VERSION_STRING}"
else
    echo "  WARNING: No CUDA detected, installing CPU-only PyTorch."
    pip install torch
fi

# Install conda-forge packages
echo "  Installing pyarrow via conda-forge..."
conda install -y -c conda-forge pyarrow

# Install remaining runtime + training deps
echo "  Installing runtime + training deps..."
pip install requests tqdm h5py pandas numpy awkward uproot vector pyyaml
pip install tensorboard matplotlib scikit-learn pytest pytest-cov

# Install the deliverables package itself (editable)
echo "  Installing deliverables (editable)..."
pip install -e .

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
echo "To start training Stage 1 prefilter:"
echo "  python train_prefilter.py \\"
echo "      --data-config data/low-pt/lowpt_tau_trackfinder.yaml \\"
echo "      --data-dir ${TRAIN_DIR}/ --val-data-dir ${VAL_DIR}/ \\"
echo "      --network networks/lowpt_tau_TrackPreFilter.py \\"
echo "      --experiments-dir experiments --epochs 40 --batch-size 256 \\"
echo "      --device cuda:0 --amp --checkpoint-criterion perfect_at_256"
echo ""
echo "Train data: ${TRAIN_DIR}/ (${TRAIN_COUNT} files)"
echo "Val data:   ${VAL_DIR}/ (${VAL_COUNT} files)"
