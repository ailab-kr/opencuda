// mmvq_pool.cpp -- a small persistent OS-thread pool for the loop forms of
// the region-loop benchmark (mmvq_rl.h). Like the CPU backend's pool, the
// caller works too and workers spin for a while before sleeping, so the
// launch cost is comparable.
#include "mmvq_rl.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace {

struct pool {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::thread> workers;
    std::atomic<uint64_t> gen{0};
    std::atomic<int> pending{0};
    int active = 0; // workers that take part in the current launch
    void (*fn)(void *) = nullptr;
    void *arg = nullptr;

    void worker(int id)
    {
        uint64_t seen = 0;
        for (;;) {
            uint64_t g;
            auto t0 = std::chrono::steady_clock::now();
            int spins = 0;
            while ((g = gen.load(std::memory_order_acquire)) == seen) {
                if (++spins % 256 == 0 &&
                    std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(200)) {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return gen.load(std::memory_order_acquire) != seen; });
                }
            }
            seen = g;
            if (id < active) {
                fn(arg);
                pending.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }

    void run(int n, void (*f)(void *), void *a)
    {
        int want = n - 1;
        while (int(workers.size()) < want) {
            int id = int(workers.size());
            workers.emplace_back([this, id] { worker(id); });
            workers.back().detach();
        }
        fn = f;
        arg = a;
        active = want;
        pending.store(want, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mu);
            gen.fetch_add(1, std::memory_order_acq_rel);
        }
        cv.notify_all();
        f(a);
        while (pending.load(std::memory_order_acquire) != 0) {
        }
    }
};

pool &the_pool()
{
    static pool *p = new pool; // never destroyed: workers are detached
    return *p;
}

void nothing(void *) {}

} // namespace

extern "C" void mmvq_pool_run(int nthreads, void (*fn)(void *arg), void *arg)
{
    the_pool().run(nthreads < 1 ? 1 : nthreads, fn, arg);
}

extern "C" void mmvq_pool_empty(int nthreads) { mmvq_pool_run(nthreads, nothing, nullptr); }
