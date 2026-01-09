NUM_TASKS=$(($(wc -l < batchs.tsv) - 1))  # minus header
sbatch --array=0-$(($NUM_TASKS-1)) script.sh

