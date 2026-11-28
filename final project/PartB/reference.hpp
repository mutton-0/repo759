// 文件用途：在 CPU 上准备输入、计算独立的 double 精度参考答案、读取 PyTorch 数据。
// 这个文件只供测试使用，不属于要优化的 GPU 算法，也不参加 GPU 计时。
#pragma once  // 防止重复读取这个头文件。
#include <algorithm>  // 提供最大值、填充等操作。
#include <cmath>  // 提供平方根、指数和有限数检查。
#include <fstream>  // 提供二进制文件读写。
#include <random>  // 提供固定种子的随机数。
#include <stdexcept>  // 提供错误信息和异常。
#include <string>  // 提供文件路径字符串。
#include <vector>  // 提供自动管理内存的数组。

// 保存一个测试用例的尺寸、输入和参考输出；这些数组都在 CPU 内存中。
struct CaseData {
    int nq, nkv, d;  // query 行数、key/value 行数、每行的特征数。
    std::vector<float> q, k, v, expected;  // Q、K、V 和正确输出 O。
    std::string reference = "cpu_double";  // 记录正确答案来自 CPU double 计算还是 PyTorch。
};

// 生成固定的随机输入；同一用例的三个 GPU 版本共用完全相同的输入。
inline CaseData make_case(int nq, int nkv, int d) {
    CaseData data;  // 新建一个保存测试数据的对象。
    data.nq = nq; data.nkv = nkv; data.d = d;  // 保存三个矩阵尺寸。
    data.q.resize(std::size_t(nq) * d);  // Q 一共有 nq×d 个数。
    data.k.resize(std::size_t(nkv) * d);  // K 一共有 nkv×d 个数。
    data.v.resize(std::size_t(nkv) * d);  // V 的形状与 K 相同。
    std::mt19937 generator(759);  // 固定随机种子，让测试可以重复。
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);  // 输入数值在 -1 到 1 之间。
    for (float& x : data.q) x = distribution(generator);  // 逐个填充 Q。
    for (float& x : data.k) x = distribution(generator);  // 逐个填充 K。
    for (float& x : data.v) x = distribution(generator);  // 逐个填充 V。
    return data;  // 把准备好的数据交给调用者。
}

// 在 CPU 上用 double 中间结果计算 attention，作为独立的正确性对照。
inline std::vector<float> cpu_reference(const CaseData& data) {
    std::vector<float> output(std::size_t(data.nq) * data.d);  // 最终结果仍然保存为 float，方便与 GPU 对照。
    std::vector<double> weights(data.nkv);  // 每次只保存一行分数和权重，节省 CPU 内存。
    double scale = 1.0 / std::sqrt(double(data.d));  // 预先算出 1/sqrt(d)。
    for (int row = 0; row < data.nq; ++row) {  // 依次处理每个 query。
        double maximum = -INFINITY;  // 初始化这一行的最大分数。
        for (int key = 0; key < data.nkv; ++key) {  // 依次计算这个 query 与每个 key 的分数。
            double sum = 0.0;  // 当前点积的累加结果。
            for (int feature = 0; feature < data.d; ++feature) {  // 遍历向量的全部分量。
                sum += double(data.q[std::size_t(row) * data.d + feature]) * double(data.k[std::size_t(key) * data.d + feature]);  // double 精度乘加。
            }
            weights[key] = sum * scale;  // 保存缩放后的分数。
            maximum = std::max(maximum, weights[key]);  // 同时更新这一行的最大分数。
        }
        double total = 0.0;  // 初始化指数结果的总和。
        for (double& x : weights) {  // 逐个处理当前行的分数。
            x = std::exp(x - maximum);  // 减最大值后取指数。
            total += x;  // 累加 softmax 的分母。
        }
        for (double& x : weights) x /= total;  // 把指数转换成权重。
        for (int feature = 0; feature < data.d; ++feature) {  // 计算当前输出行的每个分量。
            double sum = 0.0;  // 当前输出分量的累加结果。
            for (int key = 0; key < data.nkv; ++key) {  // 用所有权重组合 V。
                sum += weights[key] * double(data.v[std::size_t(key) * data.d + feature]);  // 权重乘对应的 V 分量。
            }
            output[std::size_t(row) * data.d + feature] = float(sum);  // 写出一个参考结果。
        }
    }
    return output;  // 返回完整的参考输出。
}

// 从文件读取指定数量的 float；文件太短、太长或不存在都报错。
inline std::vector<float> read_floats(const std::string& path, std::size_t count) {
    std::ifstream file(path, std::ios::binary);  // 以二进制方式打开输入文件。
    if (!file) throw std::runtime_error("无法读取文件：" + path);  // 防止把缺失的参考数据当成正确答案。
    std::vector<float> values(count);  // 申请对应数量的 CPU 数组空间。
    file.read(reinterpret_cast<char*>(values.data()), std::streamsize(count * sizeof(float)));  // 直接读取 float 的原始字节。
    if (!file || file.peek() != std::char_traits<char>::eof()) throw std::runtime_error("文件大小不符合矩阵尺寸：" + path);  // 检查长度。
    for (float x : values) if (!std::isfinite(x)) throw std::runtime_error("参考数据包含 NaN/Inf：" + path);  // 拒绝无效参考数据。
    return values;  // 返回文件里的数组。
}

// 读取 generate_reference.py 生成的同一套输入和 PyTorch fp32 参考输出。
inline CaseData load_case(const std::string& directory, int nq, int nkv, int d) {
    std::ifstream shape(directory + "/shape.txt");  // 读取参考数据对应的矩阵尺寸。
    int saved_nq = 0, saved_nkv = 0, saved_d = 0;  // 先给待读取的尺寸设置初值。
    if (!(shape >> saved_nq >> saved_nkv >> saved_d) || saved_nq != nq || saved_nkv != nkv || saved_d != d) {  // 确认使用的是正确用例。
        throw std::runtime_error("参考数据的 shape.txt 与命令行尺寸不一致");  // 避免误用其他规模的答案。
    }
    CaseData data;  // 新建一个测试用例。
    data.nq = nq; data.nkv = nkv; data.d = d;  // 保存当前用例的尺寸。
    data.q = read_floats(directory + "/q.bin", std::size_t(nq) * d);  // 读取 Q。
    data.k = read_floats(directory + "/k.bin", std::size_t(nkv) * d);  // 读取 K。
    data.v = read_floats(directory + "/v.bin", std::size_t(nkv) * d);  // 读取 V。
    data.expected = read_floats(directory + "/o.bin", std::size_t(nq) * d);  // 读取 PyTorch 的正确输出。
    data.reference = "pytorch_fp32";  // 在结果中明确标记参考答案来源。
    return data;  // 返回完整测试数据。
}
