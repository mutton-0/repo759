// 文件用途：声明 B 部分的三个 attention 版本，方便测试程序和队友调用。
// 所有矩阵都按行连续存放，数据类型是 float（32 位浮点数）。
// Q 的形状是 Nq×d，K/V 的形状是 Nkv×d，O 的形状是 Nq×d。
// S/P 是调用者提前申请的 Nq×Nkv 显存，分别保存分数和 softmax 权重。
// 所有指针都指向 GPU 显存；函数只启动计算，不申请显存，也不复制数据。
// Nq=Nkv=N 就是普通 attention；Nq=1 就可以测试 decoding。
#pragma once  // 同一个源文件重复包含此头文件时，只读取一次。

#include <cuda_runtime.h>  // 提供 CUDA 的类型、显存操作和内核启动支持。
#include <cstdio>  // 提供打印错误信息的函数。
#include <cstdlib>  // 提供退出程序的函数。

// 检查一次 CUDA 调用是否成功；失败时打印出错位置并停止程序。
inline void check_cuda(cudaError_t status, const char* file, int line) {
    if (status != cudaSuccess) {  // 状态不是成功，说明这次 CUDA 调用出错。
        std::fprintf(stderr, "CUDA 错误：%s，位置：%s:%d\n", cudaGetErrorString(status), file, line);  // 打印原因。
        std::exit(EXIT_FAILURE);  // 立即停止，避免继续输出错误的结果。
    }
}

#define CUDA_CHECK(call) check_cuda((call), __FILE__, __LINE__)  // 自动附上文件名和行号。

// 三个函数的参数顺序完全相同；S 和 P 都由调用者管理。
void attention_naive(const float* Q, const float* K, const float* V,
                     float* O, float* S, float* P, int Nq, int Nkv, int d);  // 朴素矩阵乘法 + 朴素 softmax。
void attention_tiled(const float* Q, const float* K, const float* V,
                     float* O, float* S, float* P, int Nq, int Nkv, int d);  // 分块矩阵乘法 + 朴素 softmax。
void attention_tiled_warp(const float* Q, const float* K, const float* V,
                          float* O, float* S, float* P, int Nq, int Nkv, int d);  // 分块矩阵乘法 + warp softmax。

// 单独声明 softmax，方便 tiled.cu 复用 naive.cu 中的朴素实现。
void launch_softmax_naive(const float* S, float* P, int Nq, int Nkv);  // 一线程负责一行。
void launch_softmax_warp(const float* S, float* P, int Nq, int Nkv);  // 一个 warp 负责一行。

// 兼容分工表的六参数接口：Q/K/V/O 仍然是 GPU 指针，N 表示相同的 query/key 行数。
// 这三个重载函数会申请和释放 S/P，适合先接入队友代码；正式计时请用上面的预分配接口。
void attention_naive(const float* Q, const float* K, const float* V, float* O, int N, int d);  // 六参数朴素版。
void attention_tiled(const float* Q, const float* K, const float* V, float* O, int N, int d);  // 六参数分块版。
void attention_tiled_warp(const float* Q, const float* K, const float* V, float* O, int N, int d);  // 六参数分块加 warp 版。
