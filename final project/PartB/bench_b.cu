// 文件用途：检查三个版本是否正确，并在 Euler GPU 上测量三个 kernel 合起来的时间。
// --demo 打印课堂小例子的 S、P、O；--self-test 检查边缘尺寸、长行和 decoding。
// 普通用法：./bench_b all Nq Nkv d 重复次数 结果.csv [PyTorch参考目录]
// 申请显存、CPU/GPU 数据复制和参考答案计算都放在计时外面。
#include "attention_b.cuh"  // 三个 CUDA 实现和错误检查函数。
#include "reference.hpp"  // CPU 参考计算和二进制参考数据读取。
#include <iomanip>  // 控制打印的小数位数。
#include <iostream>  // 提供标准输出和错误输出。
#include <limits>  // 提供 int 的最大范围。

constexpr int GUARD = 16;  // 在每个显存数组末尾多放 16 个检查值，用于发现部分越界写入。
constexpr float GUARD_VALUE = 1234567.0f;  // 正常计算不应该改变这个检查值。
constexpr double TOLERANCE = 1e-4;  // 分工表要求最大绝对误差严格小于 1e-4。

// 管理一个 GPU 数组；对象结束时自动释放显存，减少忘记释放的风险。
struct DeviceBuffer {
    float* pointer = nullptr;  // 保存 GPU 数组地址，nullptr 表示还没有申请。
    std::size_t count;  // 保存实际需要的元素数量，不包含尾部检查值。
    explicit DeviceBuffer(std::size_t n) : count(n) {  // 创建对象时申请 n+16 个 float。
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), (count + GUARD) * sizeof(float)));  // 真正申请 GPU 显存。
        reset();  // 把数据区设为 NaN，把尾部设为固定检查值。
    }
    DeviceBuffer(const DeviceBuffer&) = delete;  // 不允许复制显存所有权，避免重复释放。
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;  // 同样禁止通过赋值复制。
    ~DeviceBuffer() { cudaFree(pointer); }  // 对象离开作用域时释放显存；析构函数不抛出异常。
    void reset() {  // 每个版本开始前，清除上一个版本留下的数据。
        CUDA_CHECK(cudaMemset(pointer, 0xff, count * sizeof(float)));  // 全 1 字节组成 NaN，漏写结果会被正确性检查发现。
        std::vector<float> tail(GUARD, GUARD_VALUE);  // 在 CPU 上准备尾部检查值。
        CUDA_CHECK(cudaMemcpy(pointer + count, tail.data(), GUARD * sizeof(float), cudaMemcpyHostToDevice));  // 把检查值放到数组尾部。
    }
    void upload(const std::vector<float>& values) {  // 把 CPU 输入复制到当前 GPU 数组。
        if (values.size() != count) throw std::runtime_error("输入数组长度不一致");  // 先确认复制长度正确。
        CUDA_CHECK(cudaMemcpy(pointer, values.data(), count * sizeof(float), cudaMemcpyHostToDevice));  // 执行 CPU 到 GPU 的复制。
    }
    std::vector<float> download() const {  // 把 GPU 结果复制回 CPU。
        std::vector<float> values(count);  // 申请 CPU 接收空间。
        CUDA_CHECK(cudaMemcpy(values.data(), pointer, count * sizeof(float), cudaMemcpyDeviceToHost));  // 等待计算完成并取回数据。
        return values;  // 返回取回的数组。
    }
    void check_guard() const {  // 检查数组后面的 16 个值有没有被错误覆盖。
        std::vector<float> tail(GUARD);  // 在 CPU 上准备接收尾部检查值。
        CUDA_CHECK(cudaMemcpy(tail.data(), pointer + count, GUARD * sizeof(float), cudaMemcpyDeviceToHost));  // 取回检查值。
        for (float x : tail) if (x != GUARD_VALUE) throw std::runtime_error("显存数组尾部被修改，可能存在越界写入");  // 发现异常立即停止。
    }
};

// 一个 attention 测试需要 Q、K、V、S、P、O 六个显存数组。
struct DeviceCase {
    DeviceBuffer q, k, v, s, p, o;  // 分别保存六个数组的地址和长度。
    explicit DeviceCase(const CaseData& data)
        : q(data.q.size()), k(data.k.size()), v(data.v.size()),  // 为三个输入申请显存。
          s(std::size_t(data.nq) * data.nkv), p(std::size_t(data.nq) * data.nkv),  // 为分数和权重申请显存。
          o(std::size_t(data.nq) * data.d) {  // 为最终输出申请显存。
        q.upload(data.q); k.upload(data.k); v.upload(data.v);  // 输入只复制一次，三个版本共同使用。
    }
};

// 按名字选择版本；这里只决定调用谁，不执行算法细节。
void launch(const std::string& variant, DeviceCase& gpu, const CaseData& data) {
    if (variant == "naive") {  // 最直接的基线版本。
        attention_naive(gpu.q.pointer, gpu.k.pointer, gpu.v.pointer, gpu.o.pointer, gpu.s.pointer, gpu.p.pointer, data.nq, data.nkv, data.d);  // 启动三个朴素内核。
    } else if (variant == "tiled") {  // 只优化两个矩阵乘法。
        attention_tiled(gpu.q.pointer, gpu.k.pointer, gpu.v.pointer, gpu.o.pointer, gpu.s.pointer, gpu.p.pointer, data.nq, data.nkv, data.d);  // 启动分块版。
    } else if (variant == "tiled_warp") {  // 分块加上 warp softmax。
        attention_tiled_warp(gpu.q.pointer, gpu.k.pointer, gpu.v.pointer, gpu.o.pointer, gpu.s.pointer, gpu.p.pointer, data.nq, data.nkv, data.d);  // 启动两项优化版。
    } else {  // 输入了程序不认识的版本名字。
        throw std::runtime_error("版本只能是 all、naive、tiled、tiled_warp");  // 报错，避免误测其他内容。
    }
}

// 比较 GPU 输出与独立参考结果，返回最大的绝对误差。
double check_output(const std::vector<float>& actual, const std::vector<float>& expected) {
    double error = 0.0;  // 初始化最大误差。
    for (std::size_t i = 0; i < actual.size(); ++i) {  // 检查每一个输出元素，而不只检查首尾。
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) throw std::runtime_error("输出或参考结果出现 NaN/Inf");  // 拒绝无效数值。
        error = std::max(error, std::abs(double(actual[i]) - double(expected[i])));  // 更新最大绝对误差。
    }
    if (error >= TOLERANCE) throw std::runtime_error("最大绝对误差没有达到 <1e-4：" + std::to_string(error));  // 正确性不合格时立即停止。
    return error;  // 把真实误差写入日志和 CSV。
}

// 顺便检查 softmax 的基本性质：每个值非负，每行总和接近 1。
void check_probabilities(const std::vector<float>& probabilities, int nq, int nkv) {
    for (int row = 0; row < nq; ++row) {  // 依次检查每一行。
        double total = 0.0;  // 用 double 累加，减少检查程序自己的误差。
        for (int col = 0; col < nkv; ++col) {  // 查看当前行的每一个权重。
            float x = probabilities[std::size_t(row) * nkv + col];  // 读取一个权重。
            if (!std::isfinite(x) || x < 0.0f || x > 1.00001f) throw std::runtime_error("softmax 权重无效");  // 检查范围和 NaN。
            total += x;  // 累加本行的权重。
        }
        if (std::abs(total - 1.0) >= TOLERANCE) throw std::runtime_error("softmax 一行的权重之和不接近 1");  // 检查归一化。
    }
}

// 打印 demo 中的一个数组，方便对照手算过程。
void print_values(const char* name, const std::vector<float>& values) {
    std::cout << name << " = [";  // 打印数组名字和左括号。
    for (std::size_t i = 0; i < values.size(); ++i) {  // 按顺序输出所有数字。
        if (i != 0) std::cout << ", ";  // 数字之间加逗号。
        std::cout << std::fixed << std::setprecision(6) << values[i];  // 每个数字显示 6 位小数。
    }
    std::cout << "]\n";  // 打印右括号并换行。
}

// 在同一份输入上检查并测量指定版本；repeats=0 表示只检查，不计时。
void run_case(CaseData data, const std::string& selected, int repeats,
              const std::string& csv_path, bool demo = false) {
    if (data.expected.empty()) data.expected = cpu_reference(data);  // 没有 PyTorch 文件时，使用独立 CPU double 参考。
    DeviceCase gpu(data);  // 提前申请全部显存并复制输入，计时不包括这一步。
    for (const std::string variant : {"naive", "tiled", "tiled_warp"}) {  // 按固定顺序比较三个版本。
        if (selected != "all" && selected != variant) continue;  // 只运行用户指定的版本。
        gpu.s.reset(); gpu.p.reset(); gpu.o.reset();  // 清除中间结果，防止后一个版本借用前一个版本的答案。
        launch(variant, gpu, data);  // 先运行一次，用于正确性检查。
        double error = check_output(gpu.o.download(), data.expected);  // 取回所有输出并与参考比较。
        check_probabilities(gpu.p.download(), data.nq, data.nkv);  // 检查每一行的 softmax。
        if (demo) {  // 小例子额外打印中间结果，方便学习。
            std::cout << "\n版本：" << variant << "\n";  // 标明当前使用的算法版本。
            print_values("S（缩放后的分数）", gpu.s.download());  // 应得到 [1,2,3,4]。
            print_values("P（softmax 权重）", gpu.p.download());  // 应得到约 [0.0321,0.0871,0.2369,0.6439]。
            print_values("O（最终输出）", gpu.o.download());  // demo 的 V 是单位矩阵，所以 O=P。
        }
        double minimum = 0.0, median = 0.0;  // 只检查的用例不记录耗时。
        if (repeats > 0) {  // 正式性能测试才执行下面的计时。
            for (int warmup = 0; warmup < 3; ++warmup) launch(variant, gpu, data);  // 预热 3 次，减少首次运行影响。
            CUDA_CHECK(cudaDeviceSynchronize());  // 确认预热全部完成。
            cudaEvent_t start, stop;  // 声明 GPU 计时的起点和终点事件。
            CUDA_CHECK(cudaEventCreate(&start));  // 创建起点事件。
            CUDA_CHECK(cudaEventCreate(&stop));  // 创建终点事件。
            std::vector<float> times;  // 保存每次重复测量的毫秒数。
            for (int repeat = 0; repeat < repeats; ++repeat) {  // 多次测量同一个版本。
                CUDA_CHECK(cudaEventRecord(start));  // 在默认 stream 中记录计时起点。
                launch(variant, gpu, data);  // 计时区间内只启动这套 attention 的三个内核。
                CUDA_CHECK(cudaEventRecord(stop));  // 记录计时终点。
                CUDA_CHECK(cudaEventSynchronize(stop));  // 等待这一轮 GPU 计算结束。
                float milliseconds = 0.0f;  // 准备接收本次耗时。
                CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));  // 读取两个 GPU 事件之间的毫秒数。
                times.push_back(milliseconds);  // 保存本次样本。
            }
            CUDA_CHECK(cudaEventDestroy(start));  // 释放起点事件。
            CUDA_CHECK(cudaEventDestroy(stop));  // 释放终点事件。
            std::sort(times.begin(), times.end());  // 从小到大排列重复测量的耗时。
            minimum = times.front();  // 分工表采用多次测量的最小值。
            median = (double(times[(times.size() - 1) / 2]) + double(times[times.size() / 2])) / 2.0;  // 同时保留中位数。
            error = check_output(gpu.o.download(), data.expected);  // 计时结束后再确认最终输出仍然正确。
        }
        gpu.q.check_guard(); gpu.k.check_guard(); gpu.v.check_guard();  // 检查三个输入数组的尾部。
        gpu.s.check_guard(); gpu.p.check_guard(); gpu.o.check_guard();  // 检查三个输出/临时数组的尾部。
        std::cout << "PASS " << variant << " Nq=" << data.nq << " Nkv=" << data.nkv << " d=" << data.d;  // 打印版本和尺寸。
        std::cout << " max_abs_error=" << std::scientific << error << " reference=" << data.reference;  // 打印实际误差和参考来源。
        if (repeats > 0) std::cout << " min_ms=" << std::fixed << std::setprecision(6) << minimum << " median_ms=" << median;  // 打印实测时间。
        std::cout << "\n";  // 结束当前版本的日志。
        if (!csv_path.empty()) {  // 用户提供了 CSV 路径时才保存数据。
            std::ifstream existing(csv_path);  // 检查结果文件是否已经存在且有内容。
            bool needs_header = !existing || existing.peek() == std::char_traits<char>::eof();  // 新文件需要表头。
            std::ofstream csv(csv_path, std::ios::app);  // 追加结果，不覆盖之前的测试用例。
            if (!csv) throw std::runtime_error("无法写入 CSV：" + csv_path);  // 文件写入失败时停止。
            if (needs_header) csv << "nq,nkv,d,variant,repeats,min_ms,median_ms,max_abs_error,reference\n";  // 写出列名。
            csv << data.nq << ',' << data.nkv << ',' << data.d << ',' << variant << ',' << repeats << ',';  // 保存尺寸和版本。
            csv << std::setprecision(10) << minimum << ',' << median << ',' << error << ',' << data.reference << '\n';  // 保存测量值。
            if (!csv) throw std::runtime_error("CSV 写入失败：" + csv_path);  // 防止文件系统写入失败却报告成功。
        }
    }
}

// 课堂小例子：Q/K 产生 [1,2,3,4]；V 取单位矩阵，因此 O 就等于 softmax 权重。
void run_demo() {
    CaseData data = make_case(1, 4, 4);  // 一个 query、四个 key，每个向量有四个分量。
    data.q = {1, 1, 1, 1};  // 与讲解中的 Q 完全相同。
    data.k = {1, 0, 1, 0, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2};  // 按行依次放入四个 key。
    data.v = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};  // 4×4 单位矩阵，相当于不改变权重。
    run_case(data, "all", 0, "", true);  // 三个版本都运行，并打印 S/P/O。
}

// 一组小测试覆盖不能整除 tile/warp 的尺寸和 query/key 行数不同的情况。
void run_self_test() {
    const int shapes[][3] = {{1,1,1}, {3,5,7}, {2,31,17}, {3,32,33}, {5,33,15}, {17,65,31}, {1,257,64}, {33,17,65}, {1,4097,7}};  // 包括 31/32/33 和长行。
    for (const auto& shape : shapes) run_case(make_case(shape[0], shape[1], shape[2]), "all", 0, "");  // 每个形状都检查全部三个版本。
    CaseData uniform = make_case(5, 33, 17);  // 准备分数全零的特殊用例。
    std::fill(uniform.q.begin(), uniform.q.end(), 0.0f);  // Q 全零，所有分数都为零，softmax 应为均匀分布。
    run_case(uniform, "all", 0, "");  // 验证均匀权重也能正确计算。
    CaseData large = make_case(3, 5, 4);  // 准备较大正分数的特殊用例。
    std::fill(large.q.begin(), large.q.end(), 1.0f);  // 所有 query 分量都设为 1。
    for (int row = 0; row < large.nkv; ++row) {  // 依次填写每个 key。
        for (int feature = 0; feature < large.d; ++feature) large.k[std::size_t(row) * large.d + feature] = 250.0f + row;  // 分数约 500，直接取指数会溢出。
    }
    run_case(large, "all", 0, "");  // 验证减去最大值的稳定 softmax。
    std::cout << "全部 11 个 attention 用例通过，每个用例检查 3 个版本。\n";  // 总计 33 次 attention 正确性检查。
}

// 把命令行文字转换成正整数；拒绝负数、零、超范围和带多余字符的输入。
int positive_integer(const char* text) {
    std::size_t consumed = 0;  // 保存实际读了多少个字符。
    long long value = std::stoll(text, &consumed);  // 尝试读取一个整数，格式错误会抛出异常。
    if (consumed != std::string(text).size() || value <= 0 || value > std::numeric_limits<int>::max()) throw std::runtime_error("参数必须是正整数");  // 检查完整性和范围。
    return int(value);  // 返回可以安全存进 int 的值。
}

int main(int argc, char** argv) {
    try {  // 集中处理参数、文件和正确性检查的错误。
        if (argc == 2 && std::string(argv[1]) == "--demo") { run_demo(); return 0; }  // 运行手算小例子。
        if (argc == 2 && std::string(argv[1]) == "--self-test") { run_self_test(); return 0; }  // 运行边缘用例检查。
        if (argc != 7 && argc != 8) {  // 普通测试需要六个参数，参考目录可以省略。
            std::cerr << "用法：./bench_b all|naive|tiled|tiled_warp Nq Nkv d repeats result.csv [reference_dir]\n";  // 显示参数顺序。
            std::cerr << "或者：./bench_b --demo / ./bench_b --self-test\n";  // 显示两个学习和检查入口。
            return 1;  // 参数数量不对，返回失败状态。
        }
        std::string selected = argv[1];  // 保存用户选择的版本。
        if (selected != "all" && selected != "naive" && selected != "tiled" && selected != "tiled_warp") throw std::runtime_error("版本名字无效");  // 防止一个版本都没有实际运行。
        int nq = positive_integer(argv[2]);  // 读取 query 行数。
        int nkv = positive_integer(argv[3]);  // 读取 key/value 行数。
        int d = positive_integer(argv[4]);  // 读取每个向量的维度。
        int repeats = positive_integer(argv[5]);  // 读取计时重复次数。
        if (nq > 1048560) throw std::runtime_error("Nq 超过此版本二维 grid 的支持范围");  // 保护 CUDA grid 的 y 维上限。
        CaseData data = (argc == 8) ? load_case(argv[7], nq, nkv, d) : make_case(nq, nkv, d);  // 优先使用指定的 PyTorch 数据。
        run_case(data, selected, repeats, argv[6]);  // 运行检查和正式计时，保存 CSV。
        return 0;  // 所有指定版本都通过检查，正常结束。
    } catch (const std::exception& error) {  // 接住参数、文件和数值检查错误。
        std::cerr << "测试失败：" << error.what() << '\n';  // 打印具体失败原因。
        return 1;  // 用非零退出码通知 Slurm 作业失败。
    }
}
