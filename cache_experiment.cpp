#include <iostream>
#include <chrono>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Us    = std::chrono::microseconds;

int main() {
    constexpr int N = 1 << 20;
    std::vector<float> arr(N, 1.0f);
    volatile float sink = 0.0f;

    // sequential — walks through cache lines in order
    auto t0 = Clock::now();
    for (int i = 0; i < N; i += 16) {
        float s = 0;
        for (int j = 0; j < 16; j++) s += arr[i+j];
        sink += s;
    }
    auto t1 = Clock::now();
    long long seq = std::chrono::duration_cast<Us>(t1-t0).count();

    // strided — jumps 16 floats each time, cache miss on every access
    auto t2 = Clock::now();
    for (int rep = 0; rep < 16; rep++)
        for (int i = rep; i < N; i += 16)
            sink += arr[i];
    auto t3 = Clock::now();
    long long str = std::chrono::duration_cast<Us>(t3-t2).count();

    std::cout << "sequential : " << seq << " us\n";
    std::cout << "strided    : " << str << " us\n";
    std::cout << "ratio      : " << (float)str/seq << "x\n";

    (void)sink;
    return 0;
}
