// attention_tiled：两个矩阵乘法改成共享内存tiled，softmax 仍然使用naive。
// attention_tiled_warp：沿用相同的tiled矩阵乘法，再换成 warp 归约 softmax。
// 一个线程仍负责一个矩阵结果元素；tiled改变的是输入数据的加载和复用方式。
#include "attention_b.cuh"  // 读取公共接口、错误检查和朴素 softmax 的声明。
#include <cmath>  // 提供 expf、sqrtf 和 fmaxf。
#include <cstddef>  // 提供 std::size_t，防止数组位置使用过小的整数类型。

constexpr int TILE = 16;  // 每个矩阵小块有 16 行、16 列。
constexpr int WARP_SIZE = 32;  // NVIDIA GPU 的一个 warp 包含 32 个线程。
constexpr int WARPS_PER_BLOCK = 4;  // 一个 softmax 线程块处理 4 行，共 128 个线程。

// 第一步的tiled版本：计算 S = QK^T / sqrt(d)。
__global__ void scores_tiled_kernel(const float* Q, const float* K, float* S,
                                    int Nq, int Nkv, int d) {
    __shared__ float q_tile[TILE][TILE];  // 当前线程块共同使用的 Q 小块。
    __shared__ float k_tile[TILE][TILE];  // 当前线程块共同使用的 K 小块。
    int tx = threadIdx.x;  // 线程在线程块中的列编号，范围是 0 到 15。
    int ty = threadIdx.y;  // 线程在线程块中的行编号，范围是 0 到 15。
    int row = blockIdx.y * TILE + ty;  // 当前线程负责S矩阵的哪一行。
    int col = blockIdx.x * TILE + tx;  // 当前线程负责S矩阵的哪一列。
    int key_row = blockIdx.x * TILE + ty;  // 当前线程协作加载 K 的哪一行；加载分工与计算分工不同。
    float sum = 0.0f;  // 当前输出元素的部分和，一直保留在线程自己的寄存器里。
    for (int start = 0; start < d; start += TILE) {  // 沿向量长度方向，每轮只处理 16 个分量。
        int feature = start + tx;  // 当前线程这一轮负责加载第几个分量。
        q_tile[ty][tx] = (row < Nq && feature < d) ? Q[std::size_t(row) * d + feature] : 0.0f;  // 有效位置读 Q，边缘补零。
        k_tile[ty][tx] = (key_row < Nkv && feature < d) ? K[std::size_t(key_row) * d + feature] : 0.0f;  // 同样加载 K。
        __syncthreads();  // 等组内所有线程搬完数据，再开始共同使用这些数据。
        for (int k = 0; k < TILE; ++k) {  // 使用共享内存中的当前小块。
            sum += q_tile[ty][k] * k_tile[tx][k];  // Q 的一行与 K 的一行做点积，相当于乘 K 的转置。
        }
        __syncthreads();  // 等所有线程用完当前小块，才能在下一轮覆盖它。
    }
    if (row < Nq && col < Nkv) {  // 只允许有效线程写结果；前面的同步所有线程都必须参与。
        S[std::size_t(row) * Nkv + col] = sum / sqrtf(float(d));  // 合并完所有小块后，再做缩放。
    }
}

// 第三步的tiled版本：计算 O = PV。
__global__ void output_tiled_kernel(const float* P, const float* V, float* O,
                                    int Nq, int Nkv, int d) {
    __shared__ float p_tile[TILE][TILE];  // 当前线程块共同使用的权重小块。
    __shared__ float v_tile[TILE][TILE];  // 当前线程块共同使用的 V 小块。
    int tx = threadIdx.x;  // 线程在线程块内的列编号。
    int ty = threadIdx.y;  // 线程在线程块内的行编号。
    int row = blockIdx.y * TILE + ty;  // 当前线程负责 O 的哪一行。
    int col = blockIdx.x * TILE + tx;  // 当前线程负责 O 的哪一列。
    float sum = 0.0f;  // 保存当前输出元素的部分和。
    for (int start = 0; start < Nkv; start += TILE) {  // 每轮处理 16 个 key/value。
        int p_col = start + tx;  // 当前线程负责加载 P 的哪一列。
        int v_row = start + ty;  // 当前线程负责加载 V 的哪一行。
        p_tile[ty][tx] = (row < Nq && p_col < Nkv) ? P[std::size_t(row) * Nkv + p_col] : 0.0f;  // 加载权重，边缘补零。
        v_tile[ty][tx] = (v_row < Nkv && col < d) ? V[std::size_t(v_row) * d + col] : 0.0f;  // 加载 V，边缘补零。
        __syncthreads();  // 等待组内所有输入加载完成。
        for (int k = 0; k < TILE; ++k) {  // 对当前两个小块做矩阵乘法。
            sum += p_tile[ty][k] * v_tile[k][tx];  // 一行乘一列，累加到当前线程的结果中。
        }
        __syncthreads();  // 等大家用完当前小块，再加载下一块。
    }
    if (row < Nq && col < d) {  // 只写实际存在的输出元素。
        O[std::size_t(row) * d + col] = sum;  // 写出累加后的最终结果。
    }
}

// 每个 warp 合作算一行 softmax，行长可以小于、等于或大于 32。
__global__ void softmax_warp_kernel(const float* S, float* P, int Nq, int Nkv) {
    int lane = threadIdx.x % WARP_SIZE;  // 当前线程在 warp 中的位置，范围是 0 到 31。
    int row = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / WARP_SIZE;  // 当前 warp 负责哪一行。
    if (row >= Nq) return;  // 同一个 warp 的线程一起退出，不会留下缺少成员的归约。
    std::size_t base = std::size_t(row) * Nkv;  // 当前行的起始数组位置。
    float maximum = -CUDART_INF_F;  // 没有分到元素的线程用负无穷参与求最大值。
    for (int col = lane; col < Nkv; col += WARP_SIZE) {  // 线程依次负责 lane、lane+32、lane+64 等位置。
        maximum = fmaxf(maximum, S[base + col]);  // 先求自己负责的那些元素的局部最大值。
    }
    for (int offset = 16; offset > 0; offset /= 2) {  // 依次与距离 16、8、4、2、1 的线程合并。
        float other = __shfl_down_sync(0xffffffffu, maximum, offset);  // 读取同一 warp 中另一个线程的局部结果。
        maximum = fmaxf(maximum, other);  // shuffle 只负责取数；这一行才是真正比较最大值。
    }
    maximum = __shfl_sync(0xffffffffu, maximum, 0);  // 整行最大值在 lane 0，把它广播给所有线程。
    float total = 0.0f;  // 保存当前线程负责的指数结果之和。
    for (int col = lane; col < Nkv; col += WARP_SIZE) {  // 各线程同时处理各自负责的元素。
        float value = expf(S[base + col] - maximum);  // 使用整行最大值，稳定地计算指数。
        P[base + col] = value;  // 保存自己的指数结果，后面直接除以总和。
        total += value;  // 先得到自己的局部和。
    }
    for (int offset = 16; offset > 0; offset /= 2) {  // 再做一次相同结构的归约，这次是求和。
        total += __shfl_down_sync(0xffffffffu, total, offset);  // 取另一个线程的部分和，并加到自己的部分和上。
    }
    total = __shfl_sync(0xffffffffu, total, 0);  // lane 0 得到整行总和，再广播给所有线程。
    for (int col = lane; col < Nkv; col += WARP_SIZE) {  // 各线程再次处理自己刚才写入的位置。
        P[base + col] /= total;  // 用自己的指数除以整行总和，得到最终 softmax。
    }
}

// 单独启动 warp softmax；每块 4 个 warp，因此每块处理 4 行。
void launch_softmax_warp(const float* S, float* P, int Nq, int Nkv) {
    softmax_warp_kernel<<<(Nq + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, WARP_SIZE * WARPS_PER_BLOCK>>>(S, P, Nq, Nkv);  // 覆盖全部行。
    CUDA_CHECK(cudaGetLastError());  // 检查内核启动错误。
}

// 用一个内部函数复用两套相同的tiled矩阵乘法；布尔开关只选择 softmax 版本。
static void run_tiled(const float* Q, const float* K, const float* V,
                      float* O, float* S, float* P, int Nq, int Nkv, int d, bool use_warp) {
    dim3 block(TILE, TILE);  // 一个线程块负责 16×16 个输出元素。
    dim3 score_grid((Nkv + TILE - 1) / TILE, (Nq + TILE - 1) / TILE);  // 覆盖整个分数矩阵。
    scores_tiled_kernel<<<score_grid, block>>>(Q, K, S, Nq, Nkv, d);  // 第一步：tiled计算分数。
    CUDA_CHECK(cudaGetLastError());  // 检查tiled分数内核的启动错误。
    if (use_warp) {  // 根据测试版本决定是否使用 warp softmax。
        launch_softmax_warp(S, P, Nq, Nkv);  // 优化版：一个 warp 合作处理一行。
    } else {  // 不使用 warp 时，保留和基线完全相同的 softmax。
        launch_softmax_naive(S, P, Nq, Nkv);  // 中间版：只测量矩阵tiled带来的变化。
    }
    dim3 output_grid((d + TILE - 1) / TILE, (Nq + TILE - 1) / TILE);  // 覆盖整个输出矩阵。
    output_tiled_kernel<<<output_grid, block>>>(P, V, O, Nq, Nkv, d);  // 第三步：tiled计算权重乘 V。
    CUDA_CHECK(cudaGetLastError());  // 检查tiled输出内核的启动错误。
}

void attention_tiled(const float* Q, const float* K, const float* V,
                     float* O, float* S, float* P, int Nq, int Nkv, int d) {
    run_tiled(Q, K, V, O, S, P, Nq, Nkv, d, false);  // 只开启tiled，保留naive softmax。
}

void attention_tiled_warp(const float* Q, const float* K, const float* V,
                          float* O, float* S, float* P, int Nq, int Nkv, int d) {
    run_tiled(Q, K, V, O, S, P, Nq, Nkv, d, true);  // 同时开启矩阵tiled和 warp softmax。
}
