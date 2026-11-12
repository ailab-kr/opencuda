// Program for the ocu_cc.py end-to-end test: calls run() from the
// unmodified probe_kernel.cu, built by `ocu_cc.py --target cpu`.
#include "probe_cpu.h"

#include <cstdio>

void run(int n, float a, const float *x, float *y);

int main()
{
    const int n = 4096;
    void *px = nullptr, *py = nullptr;
    ocu_cpu::backend()->mem_alloc(ocu_cpu::device(), n * sizeof(float), &px);
    ocu_cpu::backend()->mem_alloc(ocu_cpu::device(), n * sizeof(float), &py);
    float *x = static_cast<float *>(px), *y = static_cast<float *>(py);
    for (int i = 0; i < n; ++i) { x[i] = float(i); y[i] = 1.0f; }
    run(n, 0.5f, x, y);
    int bad = ocu_cpu::last_status() != OCU_OK;
    for (int i = 0; i < n; ++i) bad += y[i] != 0.5f * float(i) + 1.0f;
    std::printf("main_saxpy: %s (%d bad)\n", bad ? "FAIL" : "ok", bad);
    return bad ? 1 : 0;
}
