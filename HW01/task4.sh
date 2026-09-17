#!/usr/bin/env bash
#SBATCH --job-name=FirstSlurm
#SBATCH --output=FirstSlurm.out
#SBATCH --error=FirstSlurm.err
#SBATCH --cpus-per-task=2
#SBATCH --time=00:01:00
#SBATCH --partition=instruction

hostname
