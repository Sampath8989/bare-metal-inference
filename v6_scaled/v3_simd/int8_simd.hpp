#pragma once
// V3 SIMD inference header — uses shared types from common.hpp.

#include "common.hpp"

inline int argmax(const float* data, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (data[i] > data[best]) best = i;
    return best;
}
