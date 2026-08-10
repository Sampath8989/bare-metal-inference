// Paged KV Cache memory manager simulator (vLLM-style PagedAttention, CPU-only).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

// Configuration — edit here, or override on the command line
static constexpr int NUM_PAGES         = 1024;
static constexpr int TOKENS_PER_PAGE   = 16;
static constexpr int KV_DIM            = 64;
static constexpr int MAX_SEQ_LEN       = 256;
static constexpr int PAGES_PER_REQUEST = MAX_SEQ_LEN / TOKENS_PER_PAGE;

static int      g_numRequests     = 2000;
static int      g_minTokens       = 20;
static int      g_maxTokens       = 180;
static double   g_arrivalsPerStep = 1.5;
static uint32_t g_seed            = 20260810;

// One LLM inference job; physical pages are granted by the pool, not owned here.
struct Request {
    int id        = 0;
    int length    = 0;
    int generated = 0;
    int stalls    = 0;
};

// Flat physical pool + free-page FIFO + per-request page tables
class KVCachePool {
public:
    explicit KVCachePool(int num_pages)
        : num_pages_(num_pages),
          data_(new float[(size_t)num_pages * TOKENS_PER_PAGE * 2 * KV_DIM]) {
        std::fill(data_.get(),
                  data_.get() + (size_t)num_pages * TOKENS_PER_PAGE * 2 * KV_DIM,
                  0.0f);
        for (int p = 0; p < num_pages_; ++p) free_pages_.push_back(p);
    }

    // Flat-array offset of K[t]; V[t] sits +KV_DIM.
    size_t k_offset(int page, int token_in_page) const {
        return ((size_t)page * TOKENS_PER_PAGE + (size_t)token_in_page) * 2 * KV_DIM;
    }
    size_t v_offset(int page, int token_in_page) const {
        return k_offset(page, token_in_page) + KV_DIM;
    }

    int allocate_page(int request_id) {
        if (free_pages_.empty()) return -1;   // pool exhausted
        const int page = free_pages_.front();
        free_pages_.pop_front();
        page_tables_[request_id].push_back(page);
        ++allocated_pages_;
        return page;
    }

    // Return every page a request holds to the free pool (call once, on completion).
    void free_request(int request_id) {
        auto it = page_tables_.find(request_id);
        if (it == page_tables_.end()) return;
        for (int page : it->second) free_pages_.push_back(page);
        allocated_pages_ -= (int)it->second.size();
        page_tables_.erase(it);
    }

    const std::vector<int>& page_table(int request_id) const {
        return page_tables_.at(request_id);
    }
    bool has_page_table(int request_id) const {
        return page_tables_.count(request_id) != 0;
    }

    int   free_pages()      const { return (int)free_pages_.size(); }
    int   allocated_pages() const { return allocated_pages_; }
    float* data()           const { return data_.get(); }

    size_t pool_bytes() const {
        return (size_t)num_pages_ * TOKENS_PER_PAGE * 2 * KV_DIM * sizeof(float);
    }

private:
    int                          num_pages_ = 0;
    std::unique_ptr<float[]>     data_;
    std::deque<int>              free_pages_;
    std::unordered_map<int, std::vector<int>> page_tables_;
    int                          allocated_pages_ = 0;
};

// Deterministic value so page-table reads can be verified exactly.
static float token_value(int request_id, int token_idx) {
    uint32_t h = (uint32_t)request_id * 2654435761u;
    h ^= (uint32_t)(token_idx + 1) * 40503u;
    h ^= h >> 16;
    return (float)(h % 1000u) * 0.001f;
}

struct PreallocResult {
    int        admitted       = 0;
    int        rejected       = 0;
    long long  reserved_slots = 0;
    long long  used_slots     = 0;
    long long  wasted_slots   = 0;
    double     utilization    = 0.0;
};

static PreallocResult run_preallocated(const std::vector<int>& lengths) {
    KVCachePool pool(NUM_PAGES);
    PreallocResult r;
    long long used = 0;
    int admitted = 0;

    for (int L : lengths) {

        std::vector<int> reserved;
        reserved.reserve(PAGES_PER_REQUEST);
        for (int k = 0; k < PAGES_PER_REQUEST; ++k) {
            const int page = pool.allocate_page(admitted);
            if (page < 0) break;
            reserved.push_back(page);
        }
        if ((int)reserved.size() < PAGES_PER_REQUEST) {

            pool.free_request(admitted);
            break;
        }
        used += L;
        ++admitted;
    }

    r.admitted       = admitted;
    r.rejected       = (int)lengths.size() - admitted;
    r.reserved_slots = (long long)admitted * MAX_SEQ_LEN;
    r.used_slots     = used;
    r.wasted_slots   = r.reserved_slots - r.used_slots;
    r.utilization    = r.reserved_slots ? (double)r.used_slots / (double)r.reserved_slots : 0.0;
    return r;
}

struct PagedResult {
    int        served        = 0;
    int        peak_active   = 0;
    double     avg_active    = 0.0;
    int        peak_pages    = 0;
    long long  peak_used     = 0;
    long long  steps         = 0;
    long long  tokens        = 0;
    long long  stalls        = 0;
    long long  frag_slots    = 0;
    long long  wasted_slots  = 0;
    double     utilization   = 0.0;
    bool       verify_ok     = true;
};

static PagedResult run_paged(const std::vector<int>& lengths,
                             double arrivals_per_step, uint32_t seed) {
    KVCachePool pool(NUM_PAGES);
    std::mt19937 rng(seed);
    std::poisson_distribution<int> arrivals(arrivals_per_step);

    std::vector<Request> active;
    active.reserve(512);
    int  next    = 0;
    int  pending = 0;
    long long steps = 0, tokens = 0, stalls = 0, frag = 0, active_acc = 0;
    long long completed = 0;
    int  peak_active = 0, peak_pages = 0;
    long long peak_used = 0;
    long long idle_steps = 0;
    bool all_ok = true;

    while (next < (int)lengths.size() || !active.empty()) {
        const long long tokens_before = tokens;
        int admitted_this_step = 0, completed_this_step = 0;

        // Arrivals grab their first page now; defer to `pending` if no pages free.
        pending += arrivals(rng);
        while (pending > 0 && next < (int)lengths.size() && pool.free_pages() > 0) {
            Request r;
            r.id     = next;
            r.length = lengths[next];
            pool.allocate_page(r.id);
            active.push_back(r);
            ++next;
            --pending;
            ++admitted_this_step;
        }

        // Crossing a page boundary needs a fresh page; stall until one is free.
        for (auto& r : active) {
            const int pages_held  = (int)pool.page_table(r.id).size();
            const int pages_need  = (r.generated / TOKENS_PER_PAGE) + 1;
            if (pages_need > pages_held) {
                if (pool.allocate_page(r.id) < 0) { ++r.stalls; ++stalls; continue; }
            }
            const int t    = r.generated;
            const int page = pool.page_table(r.id)[t / TOKENS_PER_PAGE];
            const int tin  = t % TOKENS_PER_PAGE;
            const float v  = token_value(r.id, t);
            pool.data()[pool.k_offset(page, tin)] =  v;
            pool.data()[pool.v_offset(page, tin)] = -v;
            ++r.generated;
            ++tokens;
        }

        // Verify K/V read back through the page table, then free the pages.
        for (auto it = active.begin(); it != active.end();) {
            if (it->generated >= it->length) {
                const std::vector<int>& pt = pool.page_table(it->id);
                frag += (long long)pt.size() * TOKENS_PER_PAGE - it->length;
                for (int t = 0; t < it->length; ++t) {
                    const float expect = token_value(it->id, t);
                    if (pool.data()[pool.k_offset(pt[t / TOKENS_PER_PAGE], t % TOKENS_PER_PAGE)] != expect ||
                        pool.data()[pool.v_offset(pt[t / TOKENS_PER_PAGE], t % TOKENS_PER_PAGE)] != -expect) {
                        all_ok = false;
                        break;
                    }
                }
                pool.free_request(it->id);
                it = active.erase(it);
                ++completed_this_step;
                ++completed;
            } else {
                ++it;
            }
        }

        ++steps;
        const int now_active = (int)active.size();
        active_acc += now_active;
        peak_active = std::max(peak_active, now_active);
        peak_pages  = std::max(peak_pages, pool.allocated_pages());
        long long used_now = 0;
        for (const auto& r : active) used_now += r.generated;
        peak_used = std::max(peak_used, used_now);

        // Defensive: stop if nothing progresses for several steps.
        const bool progressed = (tokens > tokens_before) || completed_this_step > 0 ||
                                admitted_this_step > 0;
        idle_steps = progressed ? 0 : idle_steps + 1;
        if (idle_steps > 16) break;
    }

    PagedResult r;
    r.served       = (int)completed;

    r.peak_active  = std::max(peak_active, (int)active.size());
    r.avg_active   = steps ? (double)active_acc / (double)steps : 0.0;
    r.peak_pages   = peak_pages;
    r.peak_used    = peak_used;
    r.steps        = steps;
    r.tokens       = tokens;
    r.stalls       = stalls;
    r.frag_slots   = frag;
    r.wasted_slots = (long long)peak_pages * TOKENS_PER_PAGE - peak_used;
    r.utilization  = peak_pages ? (double)peak_used / ((double)peak_pages * TOKENS_PER_PAGE) : 0.0;
    r.verify_ok    = all_ok;
    return r;
}

static void table_sep() {
    printf("+--------------------------------+----------------+-----------------+-----------------+\n");
}
static void table_row(const char* label, const char* a, const char* b, const char* imp) {
    printf("| %-30s | %-14s | %-15s | %-15s |\n", label, a, b, imp);
}

int main(int argc, char** argv) {
    if (argc > 1) g_numRequests     = std::max(1, std::atoi(argv[1]));
    if (argc > 2) g_arrivalsPerStep = std::max(0.01, std::atof(argv[2]));
    if (argc > 3) g_seed            = (uint32_t)std::atoi(argv[3]);

    const size_t pool_bytes = (size_t)NUM_PAGES * TOKENS_PER_PAGE * 2 * KV_DIM * sizeof(float);

    printf("================================================================================\n");
    printf("  Paged KV Cache Simulator  (vLLM-style PagedAttention memory manager, CPU-only)\n");
    printf("================================================================================\n");
    printf("  Physical pool        : %d pages x %d tokens x 2 (K,V) x %d dims = %.2f MB\n",
           NUM_PAGES, TOKENS_PER_PAGE, KV_DIM, (double)pool_bytes / (1024.0 * 1024.0));
    printf("  Max seq len (reserve): %d tokens = %d pages / request\n", MAX_SEQ_LEN, PAGES_PER_REQUEST);
    printf("  Requests             : %d  (output length uniform [%d, %d])\n",
           g_numRequests, g_minTokens, g_maxTokens);
    printf("  Arrivals / step      : %.2f (Poisson), seed = %u\n",
           g_arrivalsPerStep, (unsigned)g_seed);
    printf("                        (lengths RNG uses seed, arrivals RNG uses seed+7)\n");
    printf("================================================================================\n\n");

    std::mt19937 rng(g_seed);
    std::uniform_int_distribution<int> dist(g_minTokens, g_maxTokens);
    std::vector<int> lengths;
    lengths.reserve((size_t)g_numRequests);
    for (int i = 0; i < g_numRequests; ++i) lengths.push_back(dist(rng));

    // [1/2] Pre-allocated: reserve max_seq_len pages up front
    const PreallocResult pre = run_preallocated(lengths);
    printf("[1/2] PRE-ALLOCATED  — every request reserves %d pages (%d tokens) up front\n",
           PAGES_PER_REQUEST, MAX_SEQ_LEN);
    printf("  • Pool exhausted after request #%d: the pool holds %d pages in total, so\n"
           "    only %d concurrent requests fit and the remaining %d were REJECTED.\n",
           pre.admitted, NUM_PAGES, pre.admitted, pre.rejected);
    printf("  • Reserved capacity : %lld token slots (all %d pages handed out)\n",
           pre.reserved_slots, NUM_PAGES);
    printf("  • Actually written  : %lld token slots (the requests' real lengths)\n", pre.used_slots);
    printf("  • Wasted            : %lld slots — %.1f%% of reserved memory sits idle\n\n",
           pre.wasted_slots, (1.0 - pre.utilization) * 100.0);

    // [2/2] Paged: demand-paging, token-by-token
    const PagedResult paged = run_paged(lengths, g_arrivalsPerStep, g_seed + 7u);
    printf("[2/2] PAGED  — pages granted token-by-token, freed the instant a request finishes\n");
    printf("  • Served %d requests in %lld steps at %.1f avg / %d peak concurrent\n",
           paged.served, paged.steps, paged.avg_active, paged.peak_active);
    printf("  • Peak physical pages in use : %d of %d\n", paged.peak_pages, NUM_PAGES);
    printf("  • Peak capacity at that moment : %lld slots, %lld stored\n",
           (long long)paged.peak_pages * TOKENS_PER_PAGE, paged.peak_used);
    printf("  • Last-page fragmentation (internal, per request) : %.1f slots avg over %d requests\n",
           paged.served ? (double)paged.frag_slots / (double)paged.served : 0.0, paged.served);
    printf("  • Memory-pressure stalls (request waited for a free page) : %lld\n", paged.stalls);
    printf("  %s\n\n",
           paged.verify_ok
               ? "  ✓ Page-table verification: every request's K/V read back EXACTLY through "
                 "its page table (physical pages reused safely)"
               : "  ✗ Page-table verification FAILED — memory corruption in the pool!");

    const double conc_improve = (double)paged.peak_active / (double)std::max(1, pre.admitted);
    const double waste_improve = (double)pre.wasted_slots / (double)std::max<long long>(1, paged.wasted_slots);
    const double util_improve  = paged.utilization / std::max(pre.utilization, 1e-9);

    char val[64];
    char imp[3][64];
    char a[64], b[64];

    printf("Comparison — the SAME %.2f MB physical pool, the SAME %d-request workload\n",
           (double)pool_bytes / (1024.0 * 1024.0), g_numRequests);
    table_sep();
    table_row("Metric", "Pre-allocated", "Paged", "Improvement");
    table_sep();

    snprintf(a, sizeof(a), "%d", pre.admitted);
    snprintf(b, sizeof(b), "%d", paged.peak_active);
    snprintf(imp[0], sizeof(imp[0]), "%.1fx more", conc_improve);
    table_row("Max concurrent requests", a, b, imp[0]);

    snprintf(a, sizeof(a), "%d of %d", pre.admitted, g_numRequests);
    snprintf(b, sizeof(b), "%d of %d", paged.served, g_numRequests);
    snprintf(imp[1], sizeof(imp[1]), "%.1fx more", (double)paged.served / (double)std::max(1, pre.admitted));
    table_row("Requests served", a, b, imp[1]);

    snprintf(a, sizeof(a), "%lld slots  (%.1f%%)", pre.wasted_slots, (1.0 - pre.utilization) * 100.0);
    snprintf(b, sizeof(b), "%lld slots  (%.1f%%)", paged.wasted_slots, (1.0 - paged.utilization) * 100.0);
    snprintf(imp[2], sizeof(imp[2]), "%.1fx less", waste_improve);
    table_row("Memory wasted (alloc. unused)", a, b, imp[2]);

    snprintf(a, sizeof(a), "%.1f %%", pre.utilization * 100.0);
    snprintf(b, sizeof(b), "%.1f %%", paged.utilization * 100.0);
    snprintf(val, sizeof(val), "%.1fx", util_improve);
    table_row("Memory utilization", a, b, val);

    snprintf(a, sizeof(a), "%lld", pre.reserved_slots);
    snprintf(b, sizeof(b), "%lld", (long long)paged.peak_pages * TOKENS_PER_PAGE);
    table_row("Capacity allocated (peak)", a, b, "—");

    snprintf(a, sizeof(a), "%.2f MB", (double)pool_bytes / (1024.0 * 1024.0));
    snprintf(b, sizeof(b), "%.2f MB", (double)pool_bytes / (1024.0 * 1024.0));
    table_row("Physical pool size", a, b, "—");
    table_sep();
    printf("\n");

    printf("Interpretation\n");
    printf("-------------\n");
    printf("  * Pre-allocation reserves %d tokens/request up front, so the pool admits\n"
           "    only %d concurrent requests before memory is exhausted — the %dth request\n"
           "    and every one after it (%d of %d) is REJECTED. %.1f%% of the reserved\n"
           "    capacity (%lld of %lld slots) never holds a single token.\n",
           MAX_SEQ_LEN, pre.admitted, pre.admitted + 1, pre.rejected, g_numRequests,
           (1.0 - pre.utilization) * 100.0, pre.wasted_slots, pre.reserved_slots);
    printf("  * Paged allocation grows each request's page_table token-by-token and\n"
           "    returns pages the moment a request finishes, so the same pool sustains\n"
           "    %d concurrent requests (peak, %.1f avg) and serves all %d requests.\n",
           paged.peak_active, paged.avg_active, paged.served);
    printf("  * Memory utilization rises from %.1f%% to %.1f%%: the only waste left is the\n"
           "    partially-filled last page of each request (%.1f slots avg — internal\n"
           "    fragmentation). External fragmentation is impossible because pages are\n"
           "    the only allocation unit and free pages are reusable by any request.\n",
           pre.utilization * 100.0, paged.utilization * 100.0,
           paged.served ? (double)paged.frag_slots / (double)paged.served : 0.0);
    printf("  * %s\n",
           paged.verify_ok
               ? "The page-table indirection was verified end-to-end: every token written\n"
                 "    through a logical page table was read back bit-exact, even after pages\n"
                 "    were freed, recycled, and overwritten by other requests."
               : "Page-table verification FAILED — inspect the pool implementation.");

    return 0;
}
