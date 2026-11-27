// OCU_MALLOC_FILL=B fills every new device allocation with byte B (a
// checker for reads of memory nothing wrote; 0xff makes f32/f16/bf16 NaN).
// Unset, or not a byte value, allocations are not touched.
//   ocu-nvcc tests/runtime/rt_malloc_fill.cu -o rt_malloc_fill
//   ./rt_malloc_fill                       (checks nothing is filled: no fill expected)
//   OCU_MALLOC_FILL=0xff ./rt_malloc_fill   (expects 0xff)
//   OCU_MALLOC_FILL=7 ./rt_malloc_fill      (expects 7)
// The value is read once per process, so each setting is its own run.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

int main()
{
    const char *s = std::getenv("OCU_MALLOC_FILL");
    char *end = nullptr;
    long b = s ? std::strtol(s, &end, 0) : -1;
    const int want = (s && end != s && *end == '\0' && b >= 0 && b <= 255) ? (int)b : -1;

    for (size_t n : {size_t(1), size_t(4096), size_t(3 << 20)}) {
        unsigned char *d = nullptr;
        CHECK(cudaMalloc(&d, n) == cudaSuccess);
        if (want < 0) {
            // not filled: write a pattern and read it back (allocation usable as before)
            CHECK(cudaMemset(d, 0x5a, n) == cudaSuccess);
        }
        unsigned char *h = static_cast<unsigned char *>(std::malloc(n));
        CHECK(cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost) == cudaSuccess);
        const int expect = want < 0 ? 0x5a : want;
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) bad += h[i] != expect;
        CHECK(bad == 0);
        std::free(h);
        CHECK(cudaFree(d) == cudaSuccess);
    }
    std::printf("%s: %d checks, %d failed (fill %d)\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail, want);
    return g_fail ? 1 : 0;
}
