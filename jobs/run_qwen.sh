#!/bin/bash
#SBATCH --job-name=anlp_qwen_w4
#SBATCH --output=logs/slurm_qwen_%j.log
#SBATCH --error=logs/slurm_qwen_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=cpu

# No GPU needed — this job only makes HTTP API calls

echo "========================================"
echo "  Job: Qwen 3.7-max labeling (Worker 4)"
echo "  Node: $(hostname)"
echo "  Time: $(date)"
echo "========================================"

# Load conda and activate environment
module load conda 2>/dev/null || module load anaconda3 2>/dev/null || source ~/.bashrc
conda activate anlp_proj

# Set working directory
cd $SLURM_SUBMIT_DIR

# Load API keys from .env
set -a; source .env; set +a

echo "Worker ID: $WORKER_ID"
echo "Starting Qwen labeling..."

python src/dataset_creation/03_counterfactual_labeling.py --model qwen3.7-max

echo "Qwen labeling finished at $(date)"
