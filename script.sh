#!/bin/bash

# SLURM OPTIONS
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu-h100 #Partition is a queue for jobs
#SBATCH --time=24:00:00         # Time limit for the job
#SBATCH --job-name=llm   # Name of your job
#SBATCH --error=log/job-%A_%a.err
#SBATCH --output=log/job-%A_%a.out
#SBATCH --nodes=1               # Number of nodes you want to run your process on
#SBATCH --ntasks-per-node=1     # Number of CPU cores
#SBATCH --mem=40GB
#SBATCH --gres=gpu:2      # Number of GPUs
#SBATCH --exclusive
#SBATCH --exclude=node-1

module load nvidia/cuda/12.8
source ~/scratch/venv-llm/bin/activate

TSV_FILE="batchs_semeval.tsv"

echo "Running task index $SLURM_ARRAY_TASK_ID from $TSV_FILE"
python task_run.py "$TSV_FILE" "$SLURM_ARRAY_TASK_ID"

#python task_run.py "$TSV_FILE" 30
