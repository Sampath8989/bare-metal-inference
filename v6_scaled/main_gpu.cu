// GPU & Complete Benchmark Runner for V6 Scaled (Colab execution of all V1-V5 & V4 CUDA variants)
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>

bool file_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

std::string exec_and_capture(const std::string& cmd) {
    std::string result = "";
    char buffer[256];
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return "Error opening pipe for command: " + cmd + "\n";
    }
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    return result;
}

void run_scale_gpu(int scale, std::ofstream& out_file) {
    std::string banner = "======================================================================\n"
                         "             V6 GPU & FULL SCALED BENCHMARK SUITE (" + std::to_string(scale) + "x" + std::to_string(scale) + ")\n"
                         "======================================================================\n\n";
    std::cout << banner;
    out_file << banner;

    std::string scale_str = std::to_string(scale);
    std::vector<std::pair<std::string, std::string>> targets = {
        {"V1 Float32 Baseline", "v1_float32_exe"},
        {"V2 Int8 Quantized", "v2_int8_exe"},
        {"V3 Int8 AVX2 SIMD", "v3_simd_exe"},
        {"V4 CUDA INT8 Inference", "v4_cuda_exe"},
        {"V4 CUDA Shared Tiled Matmul", "v4_cuda_shared_exe"},
        {"V4 CUDA Batched Sweep", "v4_cuda_batched_exe"},
        {"V4 CUDA Pipelined Overlap", "v4_cuda_pipeline_exe"},
        {"V5 Transformer Block + KV Cache", "v5_transformer_exe"}
    };

    for (const auto& target : targets) {
        std::string exe_name = target.second;
        std::string path;
        if (file_exists("./" + exe_name)) {
            path = "./" + exe_name;
        } else if (file_exists("./v6_scaled/" + exe_name)) {
            path = "./v6_scaled/" + exe_name;
        } else {
            std::string msg = "Executable " + exe_name + " not found. Skipping.\n\n";
            std::cout << msg;
            out_file << msg;
            continue;
        }

        std::string cmd = path + " " + scale_str;
        std::string header = ">>> Running " + target.first + " (" + cmd + ")...\n";
        std::cout << header;
        out_file << header;

        std::string output = exec_and_capture(cmd);
        std::cout << output << "\n";
        out_file << output << "\n";
    }
}

int main(int argc, char* argv[]) {
    std::vector<int> scales;
    if (argc >= 2) {
        int s = std::atoi(argv[1]);
        if (s > 0) scales.push_back(s);
    } else {
        scales = {512, 1024};
    }

    std::ofstream out_file("gpu.txt");
    if (!out_file.is_open()) {
        std::cerr << "Error: Could not open gpu.txt for writing.\n";
        return 1;
    }

    for (int scale : scales) {
        run_scale_gpu(scale, out_file);
    }

    std::string footer = "======================================================================\n"
                         "             V6 GPU & FULL BENCHMARKS COMPLETE (Saved to gpu.txt)\n"
                         "======================================================================\n";
    std::cout << footer;
    out_file << footer;
    out_file.close();

    int unused __attribute__((unused)) = std::system("cp gpu.txt ../gpu.txt 2>/dev/null");

    return 0;
}
