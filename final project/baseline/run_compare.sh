#!/usr/bin/env bash
# Submit from the baseline directory:  sbatch run_compare.sh
# Same problem sizes on the CPU versions, PyTorch (CPU) and the CUDA versions
# of ../memopt (first and latest), so they can go in one plot (plot.py).
# PyTorch is expected in ~/venv-torch (CPU build).
#SBATCH -p instruction
#SBATCH -J compare
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus-per-task=1
#SBATCH --mem=8G
#SBATCH -t 0-00:40:00
#SBATCH -o compare-%j.out -e compare-%j.err

module load nvidia/cuda
cd "$SLURM_SUBMIT_DIR"
make
(cd ../memopt && make)
PY=$HOME/venv-torch/bin/python
D=64
GPU="v0 v3"

run_point() {  # Nk Nq
  NK=$1; NQ=$2
  mkdir -p dump
  OMP_NUM_THREADS=8 ./bench cpu_omp $NK $D $NQ --dump dump
  ./bench cpu_serial $NK $D $NQ
  $PY torch_ref.py dump $NK $D $NQ --threads 8
  for v in $GPU; do ../memopt/memopt $v $NK $D $NQ; done
}

echo "## prefill"
for N in 256 512 1024 2048 4096 8192; do run_point $N $N; done
echo "## decoding"
for NK in 1024 4096 16384 65536; do run_point $NK 1; done
