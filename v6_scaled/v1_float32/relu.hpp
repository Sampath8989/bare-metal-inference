#pragma once
#include <algorithm>

inline float relu(float x) {
    return std::max(0.0f, x);
}
