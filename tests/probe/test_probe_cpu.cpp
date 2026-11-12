// End-to-end probe: unmodified .cu files, rewritten by
// compiler/probe/cpu_rewrite.py and compiled as plain C++, run on the CPU
// backend through the backend interface. PROBE ONLY (PLAN P0-W1-11).
#include "probe_cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// From probe_kernel.cu
void run(int n, float a, const float *x, float *y);
__global__ void block_sum(const float *in, float *out);
// From probe_features.cu
void run_square(const float *in, float *out, int w, int h);
void run_square_int(const int *in, int *out, int w, int h);
void run_reverse(int *data, int blocks, int threads);
void run_histogram(const int *values, int n, unsigned *bins);
void run_affine(float *v, int n);
// From probe_review.cu
void run_rotate(float *data, int blocks);
void run_fill(int *a, double *b, int n);
void run_maybe_scale(float *v, const float *factor, int n);

namespace ocu_cpu {
static ocu_device *g_dev;
const ocu_backend *backend() { return ocu_backend_cpu(); }
ocu_device *device() { return g_dev; }
} // namespace ocu_cpu

static int g_fail;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

template <typename T>
static T *dev_alloc(std::size_t n)
{
    void *p = nullptr;
    if (ocu_cpu::backend()->mem_alloc(ocu_cpu::device(), n * sizeof(T), &p) != OCU_OK) {
        std::abort();
    }
    return static_cast<T *>(p);
}

static void dev_free(void *p) { ocu_cpu::backend()->mem_free(ocu_cpu::device(), p); }

int main()
{
    if (ocu_cpu::backend()->dev_open(0, &ocu_cpu::g_dev) != OCU_OK) {
        return 1;
    }

    { // probe_kernel.cu: saxpy through the host function, unchanged source
        const int n = 1000;
        float *x = dev_alloc<float>(n), *y = dev_alloc<float>(n);
        for (int i = 0; i < n; ++i) { x[i] = float(i); y[i] = 0.5f; }
        run(n, 3.0f, x, y);
        CHECK(ocu_cpu::last_status() == OCU_OK);
        int bad = 0;
        for (int i = 0; i < n; ++i) bad += y[i] != 3.0f * float(i) + 0.5f;
        CHECK(bad == 0);
        dev_free(x); dev_free(y);
    }
    { // probe_kernel.cu: block_sum uses a static __shared__ array + barrier
        const int blocks = 9, threads = 256;
        float *in = dev_alloc<float>(blocks * threads), *out = dev_alloc<float>(blocks);
        for (int i = 0; i < blocks * threads; ++i) in[i] = float(i % 5);
        ocu_cpu::launch(ocu_cpu::cfg(blocks, threads),
                        [](auto &&...a) { block_sum(a...); }, in, out);
        CHECK(ocu_cpu::last_status() == OCU_OK);
        int bad = 0;
        for (int b = 0; b < blocks; ++b) {
            float want = 0;
            for (int t = 0; t < threads; ++t) want += float((b * threads + t) % 5);
            bad += out[b] != want;
        }
        CHECK(bad == 0);
        dev_free(in); dev_free(out);
    }
    { // template kernel, 2-D grid, float and int instantiations
        const int w = 37, h = 21;
        float *in = dev_alloc<float>(w * h), *out = dev_alloc<float>(w * h);
        int *ii = dev_alloc<int>(w * h), *io = dev_alloc<int>(w * h);
        for (int i = 0; i < w * h; ++i) { in[i] = float(i) * 0.5f; ii[i] = i - 300; }
        run_square(in, out, w, h);
        CHECK(ocu_cpu::last_status() == OCU_OK);
        run_square_int(ii, io, w, h);
        CHECK(ocu_cpu::last_status() == OCU_OK);
        int bad = 0;
        for (int i = 0; i < w * h; ++i) {
            bad += out[i] != in[i] * in[i];
            bad += io[i] != ii[i] * ii[i];
        }
        CHECK(bad == 0);
        dev_free(in); dev_free(out); dev_free(ii); dev_free(io);
    }
    { // extern __shared__ (dynamic shared memory) + barrier
        const int blocks = 5, threads = 64;
        int *d = dev_alloc<int>(blocks * threads);
        for (int i = 0; i < blocks * threads; ++i) d[i] = i;
        run_reverse(d, blocks, threads);
        CHECK(ocu_cpu::last_status() == OCU_OK);
        int bad = 0;
        for (int b = 0; b < blocks; ++b)
            for (int t = 0; t < threads; ++t)
                bad += d[b * threads + t] != b * threads + (threads - 1 - t);
        CHECK(bad == 0);
        dev_free(d);
    }
    { // atomicAdd across blocks running on several OS threads
        const int n = 100000;
        int *v = dev_alloc<int>(n);
        unsigned *bins = dev_alloc<unsigned>(16);
        std::memset(bins, 0, 16 * sizeof(unsigned));
        for (int i = 0; i < n; ++i) v[i] = i * 7;
        run_histogram(v, n, bins);
        CHECK(ocu_cpu::last_status() == OCU_OK);
        unsigned want[16] = {};
        for (int i = 0; i < n; ++i) want[(i * 7) & 15]++;
        int bad = 0;
        for (int b = 0; b < 16; ++b) bad += bins[b] != want[b];
        CHECK(bad == 0);
        dev_free(v); dev_free(bins);
    }
    { // struct passed by value, globally qualified kernel name
        const int n = 300;
        float *v = dev_alloc<float>(n);
        for (int i = 0; i < n; ++i) v[i] = float(i);
        run_affine(v, n);
        CHECK(ocu_cpu::last_status() == OCU_OK);
        int bad = 0;
        for (int i = 0; i < n; ++i) bad += v[i] != float(i) * 2.0f - 1.0f;
        CHECK(bad == 0);
        dev_free(v);
    }

    { // file-scope extern __shared__ shared by blocks on several OS threads
        const int blocks = 40, threads = 32;
        float *d = dev_alloc<float>(blocks * threads);
        for (int i = 0; i < blocks * threads; ++i) d[i] = float(i);
        run_rotate(d, blocks);
        CHECK(ocu_cpu::last_status() == OCU_OK);
        int bad = 0;
        for (int b = 0; b < blocks; ++b)
            for (int t = 0; t < threads; ++t)
                bad += d[b * threads + t] != float(b * threads + (t + 1) % threads);
        CHECK(bad == 0);
        dev_free(d);
    }
    { // template argument deduction, launch inside a multi-line macro
        const int n = 100;
        int *a = dev_alloc<int>(n);
        double *b = dev_alloc<double>(n);
        run_fill(a, b, n);
        CHECK(ocu_cpu::last_status() == OCU_OK);
        int bad = 0;
        for (int i = 0; i < n; ++i) bad += a[i] != 7 || b[i] != 2.5;
        CHECK(bad == 0);
        dev_free(a); dev_free(b);
    }
    { // NULL kernel argument
        const int n = 50;
        float *v = dev_alloc<float>(n), *f = dev_alloc<float>(1);
        for (int i = 0; i < n; ++i) v[i] = float(i);
        f[0] = 3.0f;
        run_maybe_scale(v, f, n);
        run_maybe_scale(v, nullptr, n);
        CHECK(ocu_cpu::last_status() == OCU_OK);
        int bad = 0;
        for (int i = 0; i < n; ++i) bad += v[i] != -3.0f * float(i);
        CHECK(bad == 0);
        dev_free(v); dev_free(f);
    }

    ocu_cpu::backend()->dev_close(ocu_cpu::device());
    if (g_fail == 0) {
        std::printf("test_probe_cpu: ok\n");
        return 0;
    }
    return 1;
}
