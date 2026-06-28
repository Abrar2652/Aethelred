#!/usr/bin/env bash
# =============================================================================
#  setup.sh — One-time environment setup for Aethelred
#
#  Usage:
#    bash setup.sh            # CUDA 12.1 (default)
#    bash setup.sh cu118      # CUDA 11.8
#    bash setup.sh cpu        # CPU-only (no GPU)
#
#  What it does:
#    1. Verifies Python 3.9+
#    2. Creates ./venv
#    3. Installs PyTorch 2.2.0 (matching CUDA)
#    4. Installs torch-geometric + sparse extensions (exact build tags)
#    5. Installs all remaining Python dependencies
#    6. Downloads all datasets
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SCRIPT_DIR}/venv"

# ── Resolve CUDA tag from first argument ────────────────────────────────────
CUDA_TAG="${1:-cu121}"
case "${CUDA_TAG}" in
    cu121|cu118|cpu) ;;
    *)
        echo "ERROR: Unsupported argument '${CUDA_TAG}'."
        echo "       Use: cu121 | cu118 | cpu"
        exit 1
        ;;
esac

case "${CUDA_TAG}" in
    cu121) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    cu118) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
    cpu)   TORCH_INDEX="https://download.pytorch.org/whl/cpu"   ;;
esac

PYG_INDEX="https://data.pyg.org/whl/torch-2.2.0+${CUDA_TAG}.html"

echo ""
echo "========================================================================"
echo "  Aethelred — Environment Setup"
echo "  CUDA tag    : ${CUDA_TAG}"
echo "  PyTorch idx : ${TORCH_INDEX}"
echo "  PyG idx     : ${PYG_INDEX}"
echo "========================================================================"
echo ""

# ── Step 1: Python 3.9+ check ───────────────────────────────────────────────
echo "[1/6] Checking Python version..."
PY=$(command -v python3 2>/dev/null || command -v python)
PY_VER=$("${PY}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "      Found: ${PY}  (${PY_VER})"
"${PY}" -c "
import sys
if sys.version_info < (3, 9):
    print(f'ERROR: Python 3.9+ required. Found {sys.version_info.major}.{sys.version_info.minor}.')
    sys.exit(1)
print('      Version OK.')
"

# ── Step 2: Create virtual environment ──────────────────────────────────────
echo ""
echo "[2/6] Setting up virtual environment at ${VENV}..."
if [[ -d "${VENV}" ]]; then
    echo "      venv already exists — skipping creation."
else
    "${PY}" -m venv "${VENV}"
    echo "      Created."
fi

VPYTHON="${VENV}/bin/python"
VPIP="${VENV}/bin/pip"

# ── Step 3: Upgrade pip / build tools ───────────────────────────────────────
echo ""
echo "[3/6] Upgrading pip, setuptools, wheel..."
"${VPIP}" install --quiet --upgrade pip setuptools wheel
echo "      Done."

# ── Step 4: PyTorch 2.2.0 ───────────────────────────────────────────────────
echo ""
echo "[4/6] Installing PyTorch 2.2.0 + torchvision (${CUDA_TAG})..."
"${VPIP}" install --quiet \
    "torch==2.2.0" \
    "torchvision==0.17.0" \
    --extra-index-url "${TORCH_INDEX}"
echo "      Done."

# ── Step 5: torch-geometric + sparse extensions ─────────────────────────────
echo ""
echo "[5/6] Installing torch-geometric 2.5.3 + sparse extensions..."
"${VPIP}" install --quiet "torch_geometric==2.5.3"
"${VPIP}" install --quiet \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f "${PYG_INDEX}"
echo "      Done."

# ── Step 6: Remaining dependencies ──────────────────────────────────────────
echo ""
echo "[6/6] Installing remaining dependencies..."
"${VPIP}" install --quiet \
    "numpy>=1.24.0" \
    "scipy>=1.10.0" \
    "scikit-learn>=1.2.0" \
    "matplotlib>=3.7.0" \
    "networkx>=3.0" \
    "tqdm>=4.64.0" \
    "deeprobust>=0.2.9" \
    "requests>=2.28.0" \
    "pandas>=1.5.0"
echo "      Done."

# ── Download datasets ────────────────────────────────────────────────────────
echo ""
echo "[+] Downloading datasets..."
"${VPYTHON}" "${SCRIPT_DIR}/download_dataset.py"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "  Setup complete."
echo ""
echo "  Activate the environment:"
echo "    source ${VENV}/bin/activate"
echo ""
echo "  Run all experiments:"
echo "    bash run_all.sh"
echo ""
echo "  Run with forced retraining:"
echo "    bash run_all.sh --retrain"
echo "========================================================================"
echo ""
