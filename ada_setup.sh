#!/bin/bash
# =============================================================================
# ADA SETUP SCRIPT — Run this ONCE after cloning the repo on ADA
# Usage: bash ada_setup.sh
# =============================================================================

set -e  # exit on any error

echo "============================================================"
echo "  ANLP Project — ADA Setup (Worker 4: Abhinav)"
echo "============================================================"

# 1. Load conda (ADA uses module system)
echo "[1/5] Loading conda module..."
module load conda 2>/dev/null || module load anaconda3 2>/dev/null || {
    echo "  Trying to source conda directly..."
    source ~/.bashrc
}

# 2. Create conda environment
echo "[2/5] Creating conda environment 'anlp_proj'..."
conda create -n anlp_proj python=3.10 -y 2>/dev/null || echo "  (env already exists, skipping)"
conda activate anlp_proj

# 3. Install dependencies
echo "[3/5] Installing Python dependencies..."
pip install -r requirements.txt --quiet

# 4. Download spaCy model
echo "[4/5] Downloading spaCy English model..."
python -m spacy download en_core_web_sm --quiet

# 5. Run Steps 1 & 2 to generate data files
echo "[5/5] Generating data files (Steps 1 & 2)..."
echo "  Running 01_load_data.py (downloads from HuggingFace)..."
python src/dataset_creation/01_load_data.py
echo "  Running 02_extract_spans.py (extracts candidate spans)..."
python src/dataset_creation/02_extract_spans.py

echo ""
echo "============================================================"
echo "  Setup complete! Data files generated in ./data/"
echo "  Now submit the labeling jobs with:"
echo "    sbatch jobs/run_qwen.sh"
echo "    sbatch jobs/run_glm.sh"
echo "    sbatch jobs/run_gptoss.sh"
echo "============================================================"
