#!/usr/bin/env bash
# Full comparison on one machine with a GPU (no Slurm): bash run_5090.sh > out.txt
# Needs nvcc and PyTorch (CUDA) on PATH, e.g. export PATH=$HOME/micromamba/envs/vla/bin:$PATH
# GPU architecture: ARCH=sm_120 (RTX 5090); CPU threads: T (default: all cores).
cd "$(dirname "$0")"
ARCH=${ARCH:-sm_120}
T=${T:-$(nproc)}
D=64
make
(cd ../memopt && make ARCH=$ARCH)
GPU=$(../memopt/memopt list)
mkdir -p dump

run_point() {  # Nk Nq
  NK=$1; NQ=$2
  OMP_NUM_THREADS=$T ./bench cpu_omp $NK $D $NQ --dump dump
  ./bench cpu_serial $NK $D $NQ
  python torch_ref.py dump $NK $D $NQ --threads $T
  python torch_ref.py dump $NK $D $NQ --device cuda
  for v in $GPU; do ../memopt/memopt $v $NK $D $NQ; done
}

echo "## machine"; nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader; echo "cpu threads $T"
echo "## prefill"
for N in 256 512 1024 2048 4096 8192; do run_point $N $N; done
echo "## decoding"
for NK in 1024 4096 16384 65536; do run_point $NK 1; done
echo "## thread scaling, N = 4096"
for t in 1 2 4 8 16 $T; do OMP_NUM_THREADS=$t ./bench cpu_omp 4096 $D; done
