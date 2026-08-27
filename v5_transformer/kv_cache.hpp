#pragma once

// KVCache + cached attention forward pass.
#include <vector>
#include <cstring>
#include <algorithm>
#include "attention.hpp"

// Pre-allocated [max_seq_len, d_head] K/V; append() writes fixed slots — no heap alloc.
struct KVCache {
    std::vector<float> key_cache;   // [max_seq_len, d_head]
    std::vector<float> val_cache;   // [max_seq_len, d_head]
    int seq_len;
    int d_head;
    int max_seq_len;

    KVCache(int d_head, int max_seq_len) : seq_len(0), d_head(d_head), max_seq_len(max_seq_len) {
        key_cache.resize(max_seq_len * d_head, 0.0f);
        val_cache.resize(max_seq_len * d_head, 0.0f);
    }

    KVCache(int d_head) : seq_len(0), d_head(d_head), max_seq_len(0) {}

    void reset() {
        seq_len = 0;   // only entries up to seq_len are ever read
    }

    void preallocate(int max_len) {
        max_seq_len = max_len;
        key_cache.resize(max_seq_len * d_head, 0.0f);
        val_cache.resize(max_seq_len * d_head, 0.0f);
    }

    void append(const float* k, const float* v) {
        float* k_dst = key_cache.data() + seq_len * d_head;
        float* v_dst = val_cache.data() + seq_len * d_head;
        std::memcpy(k_dst, k, d_head * sizeof(float));
        std::memcpy(v_dst, v, d_head * sizeof(float));
        seq_len++;
    }

    size_t memory_bytes() const {
        return max_seq_len * d_head * 2 * sizeof(float);
    }
};

Tensor attention_forward_cached(const Tensor& input, const AttentionHead& head, KVCache* cache = nullptr) {
    int seq_len = input.rows;
    int d_model = head.d_model;
    int d_head = head.d_head;

    Tensor Q = matmul(input, head.Wq);
    Tensor K_current = matmul(input, head.Wk);
    Tensor V_current = matmul(input, head.Wv);

    int full_seq_len;
    const float* K_ptr;
    const float* V_ptr;

    if (cache != nullptr) {

        for (int i = 0; i < seq_len; i++) {
            cache->append(&K_current.data[i * d_head], &V_current.data[i * d_head]);
        }

        K_ptr = cache->key_cache.data();
        V_ptr = cache->val_cache.data();
        full_seq_len = cache->seq_len;
    } else {

        K_ptr = K_current.data;
        V_ptr = V_current.data;
        full_seq_len = seq_len;
    }

    Tensor scores(seq_len, full_seq_len);
    scores.zero();
    float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
    for (int i = 0; i < seq_len; i++) {
        // q_id hoisted out of the p-loop for ILP.
        #pragma unroll
        for (int d = 0; d < d_head; d++) {
            float q_id = Q.data[i * d_head + d];
            for (int p = 0; p < full_seq_len; p++) {
                scores.data[i * full_seq_len + p] += q_id * K_ptr[p * d_head + d];
            }
        }
    }

    for (int i = 0; i < seq_len; i++) {
        for (int p = 0; p < full_seq_len; p++) {
            scores.data[i * full_seq_len + p] *= scale;
        }
    }

    for (int i = 0; i < seq_len; i++) {
        softmax_stable(&scores.data[i * full_seq_len], full_seq_len);
    }

    Tensor output(seq_len, d_head);
    output.zero();
    // d-innermost: V_ptr access is sequential (p-outer would stride d_head → cache misses).
    for (int i = 0; i < seq_len; i++) {
        for (int p = 0; p < full_seq_len; p++) {
            float si_p = scores.data[i * full_seq_len + p];
            #pragma unroll
            for (int d = 0; d < d_head; d++) {
                output.data[i * d_head + d] += si_p * V_ptr[p * d_head + d];
            }
        }
    }

    Tensor final_output = matmul(output, head.Wo);

    return final_output;
}
