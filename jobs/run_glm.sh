#!/bin/bash
#SBATCH --job-name=anlp_glm_w4
#SBATCH --output=logs/slurm_glm_%j.log
#SBATCH --error=logs/slurm_glm_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=cpu

echo "========================================"
echo "  Job: GLM 5.3-flash labeling (Worker 4)"
echo "  Node: $(hostname)"
echo "  Time: $(date)"
echo "========================================"

module load conda 2>/dev/null || module load anaconda3 2>/dev/null || source ~/.bashrc
conda activate anlp_proj

cd $SLURM_SUBMIT_DIR
set -a; source .env; set +a

echo "Worker ID: $WORKER_ID"
echo "Starting GLM labeling..."

python src/dataset_creation/03_counterfactual_labeling.py --model glm-5.3-flash

echo "GLM labeling finished at $(date)"
