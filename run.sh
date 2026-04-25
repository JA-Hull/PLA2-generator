#!/usr/bin/env bash
set -euo pipefail

# PLA2 Domain Generator — Full Pipeline
# Usage: chmod +x run.sh && ./run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Environment setup ---

VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "=== Creating virtual environment ==="
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "=== Installing dependencies ==="
pip install --quiet --upgrade pip
pip install --quiet \
    numpy \
    torch --index-url https://download.pytorch.org/whl/cpu \
    fair-esm \
    openpyxl \
    matplotlib

# --- Create output directories ---

mkdir -p data output/figures

# --- Step 1: Import sequences ---

echo ""
echo "=== Step 1: Import PLA2 domains from xlsx ==="
if [ -f "data/natural_pla2_domains.fasta" ]; then
    echo "  data/natural_pla2_domains.fasta already exists, skipping."
    echo "  Delete it to re-import from xlsx."
else
    python3 src/import_xlsx.py
fi

# --- Step 2: Generate sequences ---

echo ""
echo "=== Step 2: Generate novel PLA2 domains ==="
python3 src/run_pipeline.py

# --- Step 3: Generate figures ---

echo ""
echo "=== Step 3: Generate figures ==="
python3 src/make_figures.py

echo ""
echo "=== Done ==="
echo "Outputs:"
echo "  output/generated.fasta       — Generated PLA2 domain sequences"
echo "  output/figures/              — All figures (contact maps, heatmaps)"
echo "  output/contact_*.npy/npz    — Contact model metadata"
