#!/bin/bash
#SBATCH --job-name=anlp_gptoss_w4
#SBATCH --output=logs/slurm_gptoss_%j.log
#SBATCH --error=logs/slurm_gptoss_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=cpu

echo "========================================"
echo "  Job: GPT-OSS 120B labeling (Worker 4)"
echo "  Node: $(hostname)"
echo "  Time: $(date)"
echo "========================================"

module load conda 2>/dev/null || module load anaconda3 2>/dev/null || source ~/.bashrc
conda activate anlp_proj

cd $SLURM_SUBMIT_DIR
set -a; source .env; set +a

echo "Worker ID: $WORKER_ID"
echo "Starting GPT-OSS labeling..."

python src/dataset_creation/03_counterfactual_labeling.py --model openai/gpt-oss-120b

echo "GPT-OSS labeling finished at $(date)"
