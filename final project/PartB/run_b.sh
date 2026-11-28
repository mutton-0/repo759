#!/bin/bash
#SBATCH --partition=instruction
#SBATCH --job-name=attention_B
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=b_%j.out
#SBATCH --error=b_%j.err

set -euo pipefail  
export LC_ALL=C  
cd "${SLURM_SUBMIT_DIR:?请在 cuda_basic 目录使用 sbatch run_b.sh 提交}" 
module purge 
module load nvidia/cuda  

repeats=${REPEATS:-20}  
if ! [[ "$repeats" =~ ^[0-9]+$ ]] || (( repeats < 1 )); then  
    echo "REPEATS 必须是正整数。" >&2  
    exit 1  
fi
job=${SLURM_JOB_ID:?没有 Slurm 作业编号，请使用 sbatch 提交}  
result_dir="results/job_${job}" 
mkdir -p "$result_dir" 
architecture=${CUDA_ARCH:-sm_89}  
torch_python=${TORCH_PYTHON:-python3}  
summary_python=${SUMMARY_PYTHON:-python3} 
command -v "$summary_python" >/dev/null 

{  # 将本次真实运行环境保存到文件，同时显示到 Slurm 日志。
    echo "job_id=$job"  # 记录作业编号。
    echo "host=$(hostname)"  # 记录实际计算节点。
    echo "date_utc=$(date -u +%FT%TZ)"  # 记录 UTC 时间。
    echo "architecture=$architecture repeats=$repeats"  # 记录编译架构和重复次数。
    echo "timing=CUDA events; three kernels; allocation and copies excluded"  # 明确计时范围。
    nvcc --version  # 记录真实 CUDA 编译器版本。
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv  # 记录实际分配到的 GPU。
} | tee "$result_dir/environment.txt"  # 保存环境信息。

nvcc -O3 -std=c++17 -lineinfo -arch="$architecture" -Xcompiler=-Wall bench_b.cu naive.cu tiled.cu adapter.cu -o bench_b  # 编译全部源文件，不启用 fast-math。
./bench_b --demo | tee "$result_dir/demo.txt"  # 打印S/P/O
./bench_b --self-test | tee "$result_dir/self_test.txt"  # 检查边缘尺寸、稳定 softmax 和 decoding。

if [[ "${SANITIZE:-0}" == 1 ]]; then  # 检查 GPU 越界
    for tool in memcheck racecheck synccheck; do  # 依次运行三种检查工具。
        compute-sanitizer --tool "$tool" --error-exitcode 1 ./bench_b --self-test > "$result_dir/${tool}.txt" 2>&1 
fi

use_torch=0  
if "$torch_python" -c 'import torch' >/dev/null 2>&1; then  
    use_torch=1  
    "$torch_python" -c 'import torch; print("PyTorch version:", torch.__version__)' | tee "$result_dir/reference_mode.txt" 
else  
    echo "当前 Python 没有 PyTorch；本次结果仅与 CPU double 对照，还未完成 PyTorch 验收。" | tee "$result_dir/reference_mode.txt"  
    if [[ "${REQUIRE_TORCH:-0}" == 1 ]]; then  
        echo "请通过 TORCH_PYTHON 指定已有 PyTorch 环境中的 Python。" >&2  
        exit 1  
fi

shapes=("128 128 64" "256 256 64" "512 512 64" "1024 1024 64" "1 1024 64" "1 4096 64" "3 33 17")  # 覆盖序列增长、解码和不规则尺寸。
for shape in "${shapes[@]}"; do  # 依次运行每种矩阵形状。
    read -r nq nkv d <<< "$shape"  # 将一行尺寸拆成三个整数。
    benchmark_command=(./bench_b all "$nq" "$nkv" "$d" "$repeats" "$result_dir/timings.csv")  # 先准备始终需要的命令和参数。
    if (( use_torch == 1 )); then  # 可用 PyTorch 时准备同一份输入和答案。
        reference_dir="$result_dir/ref_${nq}_${nkv}_${d}"  # 每个形状使用独立的参考目录。
        "$torch_python" generate_reference.py --nq "$nq" --nkv "$nkv" --d "$d" --out "$reference_dir"  # 在 CPU 上生成参考数据。
        benchmark_command+=("$reference_dir")  # 有 PyTorch 时再追加参考目录，避免空数组在旧版 Bash 中报错。
    fi
    "${benchmark_command[@]}" | tee -a "$result_dir/benchmark.txt"  # 执行准备好的命令，三个版本使用相同输入和计时规则。
done

if (( use_torch == 1 )); then  # 再用 PyTorch 检查两种特殊数值情况。
    for mode in uniform large; do  # 一种产生均匀权重，一种产生很大的正分数。
        reference_dir="$result_dir/ref_${mode}"  # 特殊用例使用独立目录。
        "$torch_python" generate_reference.py --nq 3 --nkv 5 --d 4 --mode "$mode" --out "$reference_dir"  # 生成特殊用例的 PyTorch 正确答案。
        ./bench_b all 3 5 4 3 "$result_dir/special_checks.csv" "$reference_dir" | tee -a "$result_dir/benchmark.txt"  # 单独保存，不混入普通输入的加速比。
    done
fi

"$summary_python" summarize.py "$result_dir/timings.csv" "$result_dir/speedups.csv" | tee "$result_dir/summary.txt"  # 根据真实耗时计算逐项加速比。
echo "作业完成，结果目录：$result_dir"  
echo "把 b_${job}.out、b_${job}.err 和 $result_dir 整个目录拉下来。"  
