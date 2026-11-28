// Naive CUDA 写法计算 O = softmax(QK^T / sqrt(d)) V。
// 一共启动三个 kernel，中间的 S/P 保存在显存里。
// 第一个和第三个 kernel：一个线程算一个输出元素。
// 第二个 kernel：一个线程依次算完一整行 softmax。
#include "attention_b.cuh"  // 读取公共函数声明和 CUDA 错误检查工具。
#include <cmath>  // 提供平方根、指数和最大值函数。
#include <cstddef>  // 提供适合表示数组位置的 std::size_t 类型。

// 第一步：计算分数矩阵 S = QK^T / sqrt(d)。
__global__ void scores_naive_kernel(const float* Q, const float* K, float* S,
                                    int Nq, int Nkv, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 当前线程负责 Q 的第几行。
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 当前线程负责 K 的第几行，也就是 S 的第几列。
    if (row >= Nq || col >= Nkv) return;  // 多出来的线程不访问数组，避免越界。
    float sum = 0.0f;  // 保存当前这个分数的累加结果。
    for (int k = 0; k < d; ++k) {  // 依次处理向量中的每个分量。
        sum += Q[std::size_t(row) * d + k] * K[std::size_t(col) * d + k];  // 对应分量相乘，再加起来。
    }
    S[std::size_t(row) * Nkv + col] = sum / sqrtf(float(d));  // 除以 sqrt(d)，写入分数矩阵。
}

// 第二步：对 S 的每一行做 softmax；这里每个线程独立处理一整行。
__global__ void softmax_naive_kernel(const float* S, float* P, int Nq, int Nkv) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // 当前线程负责哪一行。
    if (row >= Nq) return;  // 超过实际行数的线程直接结束。
    std::size_t base = std::size_t(row) * Nkv;  // 当前行第一个元素在一维数组中的位置。
    float maximum = -CUDART_INF_F;  // 从负无穷开始找最大值，负数输入也能正确处理。
    for (int col = 0; col < Nkv; ++col) {  // 依次查看这一行的所有分数。
        maximum = fmaxf(maximum, S[base + col]);  // 保留到目前为止最大的分数。
    }
    float total = 0.0f;  // 保存指数结果的总和，也就是 softmax 的分母。
    for (int col = 0; col < Nkv; ++col) {  // 再依次处理这一行的每个分数。
        float value = expf(S[base + col] - maximum);  // 先减最大值，再取指数，避免数值太大。
        P[base + col] = value;  // 先把未归一化的指数结果保存在 P 中。
        total += value;  // 把这个指数结果加入总和。
    }
    for (int col = 0; col < Nkv; ++col) {  // 最后再次遍历这一行。
        P[base + col] /= total;  // 每个指数除以总和，得到最终权重。
    }
}

// 第三步：计算输出矩阵 O = PV。
__global__ void output_naive_kernel(const float* P, const float* V, float* O,
                                    int Nq, int Nkv, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 当前线程负责 O 的第几行。
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 当前线程负责 O 的第几列。
    if (row >= Nq || col >= d) return;  // 处理不能整除线程块大小的矩阵边缘。
    float sum = 0.0f;  // 保存一个输出元素的累加结果。
    for (int k = 0; k < Nkv; ++k) {  // 遍历所有 key/value。
        sum += P[std::size_t(row) * Nkv + k] * V[std::size_t(k) * d + col];  // 用权重乘 V，再累加。
    }
    O[std::size_t(row) * d + col] = sum;  // 写出当前线程负责的结果。
}

// 这是 CPU 调用的启动函数；真正的 softmax 计算在 GPU 内核中进行。
void launch_softmax_naive(const float* S, float* P, int Nq, int Nkv) {
    softmax_naive_kernel<<<(Nq + 127) / 128, 128>>>(S, P, Nq, Nkv);  // 每块 128 个线程，向上取整覆盖所有行。
    CUDA_CHECK(cudaGetLastError());  // 检查刚才的内核是否成功启动。
}

// 按顺序启动三个内核；默认 CUDA stream 会保证前一步完成后再执行下一步。
void attention_naive(const float* Q, const float* K, const float* V,
                     float* O, float* S, float* P, int Nq, int Nkv, int d) {
    dim3 block(16, 16);  // 每个线程块是 16×16，共 256 个线程。
    dim3 score_grid((Nkv + 15) / 16, (Nq + 15) / 16);  // 分数矩阵需要多少个线程块。
    scores_naive_kernel<<<score_grid, block>>>(Q, K, S, Nq, Nkv, d);  // 第一个内核：算分数。
    CUDA_CHECK(cudaGetLastError());  // 检查第一个内核的启动错误。
    launch_softmax_naive(S, P, Nq, Nkv);  // 第二个内核：算每一行的 softmax。
    dim3 output_grid((d + 15) / 16, (Nq + 15) / 16);  // 输出矩阵需要多少个线程块。
    output_naive_kernel<<<output_grid, block>>>(P, V, O, Nq, Nkv, d);  // 第三个内核：权重乘 V。
    CUDA_CHECK(cudaGetLastError());  // 检查第三个内核的启动错误。
}
