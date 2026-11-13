#!/usr/bin/env bash
# Submit from the memopt directory:  sbatch ncu_euler.sh v0 v1
# Nsight Compute counters for the first call of each given version
# (N = 2048, d = 64): global sectors per request, bytes used per sector,
# DRAM traffic and shared-memory bank conflicts, one block per kernel.
#SBATCH -p instruction
#SBATCH -J memopt-ncu
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gpus-per-task=1
#SBATCH --mem=8G
#SBATCH -t 0-00:20:00
#SBATCH -o memopt-ncu-%j.out -e memopt-ncu-%j.err

module load nvidia/cuda
cd "$SLURM_SUBMIT_DIR"
make

metrics=gpu__time_duration.sum
metrics+=,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
metrics+=,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
metrics+=,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
metrics+=,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum
metrics+=,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
metrics+=,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct
metrics+=,dram__bytes_read.sum,dram__bytes_write.sum
metrics+=,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum

for v in "$@"; do
  echo "===== $v"
  ncu --metrics "$metrics" --launch-count 4 ./memopt "$v" 2048 64
done
