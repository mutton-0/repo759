#!/usr/bin/env bash
# Submit from the baseline directory:  sbatch run_euler.sh
# CPU part of the benchmark: serial and OpenMP, prefill sweep, thread scaling,
# and the decoding case (1 query against a long KV cache), d = 64.
#SBATCH -p instruction
#SBATCH -J baseline
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=4G
#SBATCH -t 0-00:30:00
#SBATCH -o baseline-%j.out -e baseline-%j.err

cd "$SLURM_SUBMIT_DIR"
make
D=64

echo "## prefill, 8 threads"
for N in 256 512 1024 2048 4096 8192; do
  ./bench cpu_serial $N $D
  OMP_NUM_THREADS=8 ./bench cpu_omp $N $D
done

echo "## thread scaling, N = 4096"
for T in 1 2 4 8; do
  OMP_NUM_THREADS=$T ./bench cpu_omp 4096 $D
done

echo "## decoding, Nq = 1"
for NK in 1024 4096 16384 65536; do
  ./bench cpu_serial $NK $D 1
  OMP_NUM_THREADS=8 ./bench cpu_omp $NK $D 1
done

# inputs and outputs for the PyTorch check (torch_ref.py)
mkdir -p dump
./bench cpu_serial 1024 $D --dump dump
