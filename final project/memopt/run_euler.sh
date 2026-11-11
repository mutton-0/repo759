#!/usr/bin/env bash
# Submit from the memopt directory:  sbatch run_euler.sh
# Runs every version at the main problem size (N = 4096, d = 64).
#SBATCH -p instruction
#SBATCH -J memopt
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gpus-per-task=1
#SBATCH --mem=8G
#SBATCH -t 0-00:20:00
#SBATCH -o memopt-%j.out -e memopt-%j.err

module load nvidia/cuda
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
make

for v in $(./memopt list); do
  ./memopt "$v" 4096 64
  echo
done
