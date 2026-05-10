#include <iostream>
#include <cstdint>

int main() {
    float unaligned[8]        = {1,2,3,4,5,6,7,8};
    alignas(32) float aligned[8] = {1,2,3,4,5,6,7,8};

    uintptr_t u = reinterpret_cast<uintptr_t>(unaligned);
    uintptr_t a = reinterpret_cast<uintptr_t>(aligned);

    std::cout << "unaligned % 32 = " << u % 32 << "\n";
    std::cout << "aligned   % 32 = " << a % 32 << " (AVX2 ready)\n";
    return 0;
}
