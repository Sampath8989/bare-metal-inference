// Continuous batching simulator (vLLM-style scheduler, CPU-only).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <random>
#include <thread>
#include <vector>

using Clock  = std::chrono::steady_clock;
using Micros = std::chrono::microseconds;

static int      g_numRequests   = 1000;
static int      g_maxBatchSize  = 32;
static int      g_minTokens     = 8;
static int      g_maxTokens     = 64;
static double   g_stepComputeUs = 100.0;
static double   g_meanArrivalUs = 100.0;
static uint32_t g_seed          = 20260810;

static double now_us() {
    return std::chrono::duration<double, std::micro>(Clock::now().time_since_epoch()).count();
}

struct Request {
    int    id               = 0;
    double arrival_time     = 0.0;
    int    max_tokens       = 0;
    int    generated_tokens = 0;
    bool   finished         = false;
    double completion_us    = 0.0;
};

class ContinuousBatcher {
public:
    ContinuousBatcher(int max_batch, bool continuous)
        : max_batch_(max_batch), continuous_(continuous) {}

    void begin_mode() { base_us_ = now_us(); }

    double sim_now() const { return now_us() - base_us_; }

    void load_arrival_schedule(const std::vector<std::pair<double, int>>& schedule) {
        arrivals_ = schedule;
    }

    void materialize_arrivals() {
        while (next_idx_ < (int)arrivals_.size() &&
               arrivals_[next_idx_].first <= sim_now()) {
            auto req             = std::make_unique<Request>();
            req->id              = next_idx_;
            req->arrival_time    = arrivals_[next_idx_].first;
            req->max_tokens      = arrivals_[next_idx_].second;
            waiting_queue_.push_back(std::move(req));
            ++next_idx_;
        }
    }

    bool has_future_arrivals() const { return next_idx_ < (int)arrivals_.size(); }

    double next_arrival_time() const {
        return has_future_arrivals() ? arrivals_[next_idx_].first : 0.0;
    }

    bool all_done() const {
        return active_batch_.empty() && waiting_queue_.empty() &&
               next_idx_ >= (int)arrivals_.size();
    }

    bool can_step() const {
        if (!active_batch_.empty()) return true;
        if (waiting_queue_.empty()) return false;
        if (continuous_) return true;

        // Static: start a wave once full, or when arrivals are exhausted.
        return (int)waiting_queue_.size() >= max_batch_ || !has_future_arrivals();
    }

    void step() {
        const auto cpu0 = Clock::now();

        active_batch_.erase(
            std::remove_if(active_batch_.begin(), active_batch_.end(),
                           [](const std::unique_ptr<Request>& r) { return r->finished; }),
            active_batch_.end());

        const bool batch_started_empty = active_batch_.empty();
        while ((int)active_batch_.size() < max_batch_ &&
               !waiting_queue_.empty() && can_admit_(batch_started_empty)) {
            active_batch_.push_back(std::move(waiting_queue_.front()));
            waiting_queue_.pop_front();
        }

        if (active_batch_.empty()) return;

        for (auto& r : active_batch_) ++r->generated_tokens;

        scheduler_overhead_us_ +=
            std::chrono::duration<double, std::micro>(Clock::now() - cpu0).count();

        // Simulated GPU time — constant per step regardless of batch size.
        std::this_thread::sleep_for(Micros((int64_t)std::llround(g_stepComputeUs)));
        busy_us_ += (double)active_batch_.size() * g_stepComputeUs;
        ++steps_;

        for (auto& r : active_batch_) {
            if (!r->finished && r->generated_tokens >= r->max_tokens) {
                r->finished       = true;
                r->completion_us  = sim_now();
                latencies_.push_back(r->completion_us - r->arrival_time);
            }
        }
    }

    const std::vector<double>& latencies() const { return latencies_; }
    double busy_us()     const { return busy_us_; }
    double overhead_us() const { return scheduler_overhead_us_; }
    int64_t steps()      const { return steps_; }
    int max_batch()      const { return max_batch_; }

private:

    bool can_admit_(bool batch_started_empty) const {
        if (continuous_) return true;
        return batch_started_empty &&
               ((int)waiting_queue_.size() >= max_batch_ || !has_future_arrivals());
    }

    int max_batch_  = 0;
    bool continuous_ = false;
    double base_us_  = 0.0;
    int next_idx_    = 0;

    std::vector<std::pair<double, int>>     arrivals_;
    std::deque<std::unique_ptr<Request>>    waiting_queue_;
    std::vector<std::unique_ptr<Request>>   active_batch_;
    std::vector<double>                     latencies_;

    double busy_us_             = 0.0;
    double scheduler_overhead_us_ = 0.0;
    int64_t steps_              = 0;
};

struct ModeStats {
    double total_us        = 0.0;
    double busy_us         = 0.0;
    double overhead_us     = 0.0;
    int64_t steps          = 0;
    std::vector<double> latencies;
};

static double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / (double)v.size();
}

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = std::min(v.size() - 1, (size_t)std::llround(p * (double)(v.size() - 1)));
    return v[idx];
}

static ModeStats run_mode(bool continuous,
                          const std::vector<std::pair<double, int>>& schedule) {
    ContinuousBatcher batcher(g_maxBatchSize, continuous);
    batcher.begin_mode();
    batcher.load_arrival_schedule(schedule);

    while (!batcher.all_done()) {
        batcher.materialize_arrivals();
        if (batcher.can_step()) {
            batcher.step();
        } else if (batcher.has_future_arrivals()) {
            // GPU idle: sleep until the next arrival.
            const double until = batcher.next_arrival_time() - batcher.sim_now();
            if (until > 0.0) {
                std::this_thread::sleep_for(Micros((int64_t)std::llround(until)));
            }
        }

    }

    ModeStats s;
    s.total_us    = batcher.sim_now();
    s.busy_us     = batcher.busy_us();
    s.overhead_us = batcher.overhead_us();
    s.steps       = batcher.steps();
    s.latencies   = batcher.latencies();
    return s;
}

static void table_sep() {
    printf("+----------------------------+-------------+-------------+--------------+\n");
}
static void table_row(const char* label, const char* a, const char* b, const char* imp) {
    printf("| %-26s | %11s | %11s | %12s |\n", label, a, b, imp);
}

int main(int argc, char** argv) {
    if (argc > 1) g_numRequests   = std::max(1, std::atoi(argv[1]));
    if (argc > 2) g_maxBatchSize  = std::max(1, std::atoi(argv[2]));
    if (argc > 3) g_meanArrivalUs = std::max(1.0, std::atof(argv[3]));
    if (argc > 4) g_seed          = (uint32_t)std::atoi(argv[4]);

    printf("====================================================================\n");
    printf("  Continuous Batching Simulator  (vLLM-style scheduler, CPU-only)\n");
    printf("====================================================================\n");
    printf("  Requests            : %d\n", g_numRequests);
    printf("  Max batch size      : %d decode slots\n", g_maxBatchSize);
    printf("  Tokens / request    : uniform [%d, %d]\n", g_minTokens, g_maxTokens);
    printf("  Compute / step      : %.1f us (constant, batched GPU model)\n", g_stepComputeUs);
    printf("  Mean inter-arrival  : %.1f us (Poisson), seed = %u\n", g_meanArrivalUs, g_seed);
    printf("====================================================================\n\n");

    std::mt19937 rng(g_seed);
    std::uniform_int_distribution<int> tokens(g_minTokens, g_maxTokens);
    std::exponential_distribution<double> gap(1.0 / g_meanArrivalUs);

    std::vector<std::pair<double, int>> schedule;
    schedule.reserve((size_t)g_numRequests);
    double t = 0.0;
    for (int i = 0; i < g_numRequests; ++i) {
        schedule.emplace_back(t, tokens(rng));
        t += gap(rng);
    }

    const ModeStats static_s  = run_mode(false, schedule);
    const ModeStats cont_s    = run_mode(true,  schedule);

    const double s_total_ms   = static_s.total_us / 1000.0;
    const double c_total_ms   = cont_s.total_us   / 1000.0;
    const double s_avg_ms     = mean(static_s.latencies) / 1000.0;
    const double c_avg_ms     = mean(cont_s.latencies)   / 1000.0;
    const double s_p50_ms     = percentile(static_s.latencies, 0.50) / 1000.0;
    const double c_p50_ms     = percentile(cont_s.latencies,   0.50) / 1000.0;
    const double s_p95_ms     = percentile(static_s.latencies, 0.95) / 1000.0;
    const double c_p95_ms     = percentile(cont_s.latencies,   0.95) / 1000.0;

    const double s_util       = static_s.busy_us / (static_s.total_us * g_maxBatchSize);
    const double c_util       = cont_s.busy_us   / (cont_s.total_us   * g_maxBatchSize);
    const double s_avg_active = static_s.busy_us / std::max<int64_t>(1, static_s.steps) / g_stepComputeUs;
    const double c_avg_active = cont_s.busy_us   / std::max<int64_t>(1, cont_s.steps)   / g_stepComputeUs;
    const double s_overhead   = static_s.overhead_us / std::max<int64_t>(1, static_s.steps);
    const double c_overhead   = cont_s.overhead_us   / std::max<int64_t>(1, cont_s.steps);

    char val[2][64];
    char imp[8][64];

    snprintf(imp[0], sizeof(imp[0]), "%.2fx", s_total_ms / c_total_ms);
    snprintf(imp[1], sizeof(imp[1]), "%.2fx", s_avg_ms / c_avg_ms);
    snprintf(imp[2], sizeof(imp[2]), "%.2fx", s_p50_ms / c_p50_ms);
    snprintf(imp[3], sizeof(imp[3]), "%.2fx", s_p95_ms / c_p95_ms);
    snprintf(imp[4], sizeof(imp[4]), "%.2fx", c_util / std::max(s_util, 1e-9));
    snprintf(imp[5], sizeof(imp[5]), "%.2fx", (double)static_s.steps / (double)cont_s.steps);
    snprintf(imp[6], sizeof(imp[6]), "%.2fx", c_avg_active / std::max(s_avg_active, 1e-9));
    snprintf(imp[7], sizeof(imp[7]), "%.2fx", s_overhead / std::max(c_overhead, 1e-9));

    table_sep();
    table_row("Metric", "Static", "Continuous", "Improvement");
    table_sep();
    snprintf(val[0], sizeof(val[0]), "%.2f ms", s_total_ms);
    snprintf(val[1], sizeof(val[1]), "%.2f ms", c_total_ms);
    table_row("Total time (1000 requests)", val[0], val[1], imp[0]);
    snprintf(val[0], sizeof(val[0]), "%.2f ms", s_avg_ms);
    snprintf(val[1], sizeof(val[1]), "%.2f ms", c_avg_ms);
    table_row("Avg latency / request", val[0], val[1], imp[1]);
    snprintf(val[0], sizeof(val[0]), "%.2f ms", s_p50_ms);
    snprintf(val[1], sizeof(val[1]), "%.2f ms", c_p50_ms);
    table_row("p50 latency", val[0], val[1], imp[2]);
    snprintf(val[0], sizeof(val[0]), "%.2f ms", s_p95_ms);
    snprintf(val[1], sizeof(val[1]), "%.2f ms", c_p95_ms);
    table_row("p95 latency", val[0], val[1], imp[3]);
    snprintf(val[0], sizeof(val[0]), "%.1f %%", s_util * 100.0);
    snprintf(val[1], sizeof(val[1]), "%.1f %%", c_util * 100.0);
    table_row("Avg GPU utilization", val[0], val[1], imp[4]);
    snprintf(val[0], sizeof(val[0]), "%lld", (long long)static_s.steps);
    snprintf(val[1], sizeof(val[1]), "%lld", (long long)cont_s.steps);
    table_row("Decode steps", val[0], val[1], imp[5]);
    snprintf(val[0], sizeof(val[0]), "%.1f", s_avg_active);
    snprintf(val[1], sizeof(val[1]), "%.1f", c_avg_active);
    table_row("Avg active batch (slots)", val[0], val[1], imp[6]);
    snprintf(val[0], sizeof(val[0]), "%.3f us", s_overhead);
    snprintf(val[1], sizeof(val[1]), "%.3f us", c_overhead);
    table_row("Scheduler overhead / step", val[0], val[1], imp[7]);
    snprintf(val[0], sizeof(val[0]), "%zu", static_s.latencies.size());
    snprintf(val[1], sizeof(val[1]), "%zu", cont_s.latencies.size());
    table_row("Requests completed", val[0], val[1], "—");
    table_sep();
    printf("\n");

    printf("Interpretation\n");
    printf("-------------\n");
    printf("  * Continuous batching finishes %d requests %.2fx faster\n"
           "    (%6.1f ms vs %6.1f ms) by back-filling decode slots instead of\n"
           "    letting the GPU drain between waves.\n",
           g_numRequests, s_total_ms / c_total_ms, c_total_ms, s_total_ms);
    printf("  * Average per-request latency falls %.2fx\n"
           "    (%6.2f ms -> %6.2f ms); p95 falls %.2fx\n"
           "    (%6.2f ms -> %6.2f ms) — requests no longer queue behind an\n"
           "    entire wave.\n",
           s_avg_ms / c_avg_ms, s_avg_ms, c_avg_ms, s_p95_ms / c_p95_ms, s_p95_ms, c_p95_ms);
    printf("  * GPU utilization rises from %.1f%% to %.1f%%: the average active\n"
           "    batch grows from %.1f to %.1f of %d slots.\n",
           s_util * 100.0, c_util * 100.0, s_avg_active, c_avg_active, g_maxBatchSize);
    printf("  * Scheduler overhead is negligible in both modes (%.3f vs %.3f us/step):\n"
           "    the win comes from the admission policy, not from cheaper steps.\n",
           s_overhead, c_overhead);
    printf("  * Model: 1 step == 1 token/request; each step costs a fixed %.0f us\n"
           "    regardless of batch size. Utilization is the time-weighted mean of\n"
           "    (active_batch_size / max_batch_size) over the full run, idle included.\n",
           g_stepComputeUs);

    return 0;
}
