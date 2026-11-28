// 文件用途：兼容分工表中 attention_xxx(Q,K,V,O,N,d) 的六参数函数签名。
// 此接口内部申请/释放临时显存；bench_b 使用九参数接口，避免把这些开销计入内核时间。
#include "attention_b.cuh"  // 读取六参数和九参数接口的声明。
#include <cstddef>  // 提供 std::size_t，用于计算显存字节数。

using AttentionFunction = void (*)(const float*, const float*, const float*, float*, float*, float*, int, int, int);  // 表示九参数函数的地址类型。

// 用统一逻辑管理方形 attention 所需的两个临时矩阵。
static void run_square(AttentionFunction function, const float* Q, const float* K,
                       const float* V, float* O, int N, int d) {
    if (N <= 0 || N > 1048560 || d <= 0) {  // 输入尺寸必须为正，且不能超过二维 grid 的行数范围。
        std::fprintf(stderr, "attention 的 N/d 尺寸无效\n");  // 告诉调用者参数存在问题。
        std::exit(EXIT_FAILURE);  // 停止，避免启动无效的内核。
    }
    std::size_t bytes = std::size_t(N) * N * sizeof(float);  // S 和 P 都是 N×N 个 float。
    float* S = nullptr;  // 准备保存分数矩阵的 GPU 地址。
    float* P = nullptr;  // 准备保存权重矩阵的 GPU 地址。
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&S), bytes));  // 申请分数矩阵。
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&P), bytes));  // 申请权重矩阵。
    function(Q, K, V, O, S, P, N, N, d);  // 调用对应的九参数版本；方形情况 Nq=Nkv=N。
    CUDA_CHECK(cudaDeviceSynchronize());  // 等待计算结束，检查运行时错误后再释放临时数组。
    CUDA_CHECK(cudaFree(P));  // 释放权重矩阵，最终输出 O 仍由调用者保留。
    CUDA_CHECK(cudaFree(S));  // 释放分数矩阵。
}

void attention_naive(const float* Q, const float* K, const float* V, float* O, int N, int d) {
    run_square(attention_naive, Q, K, V, O, N, d);  // C++ 根据函数地址类型选择九参数的同名函数。
}

void attention_tiled(const float* Q, const float* K, const float* V, float* O, int N, int d) {
    run_square(attention_tiled, Q, K, V, O, N, d);  // 使用分块矩阵乘法的六参数适配器。
}

void attention_tiled_warp(const float* Q, const float* K, const float* V, float* O, int N, int d) {
    run_square(attention_tiled_warp, Q, K, V, O, N, d);  // 使用分块加 warp softmax 的六参数适配器。
}
