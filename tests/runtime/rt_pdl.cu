// Programmatic dependent launch declarations and cudaLaunchKernelEx
// (card libs-cub-subset, test list items 1-6). Expected behaviour comes from
// the public CUDA Runtime API reference (cudaLaunchKernelEx, cudaLaunchConfig_t,
// cudaLaunchAttributeID) and the CUDA C++ Programming Guide (programmatic
// dependent launch). Runs on the CPU path and on the NVIDIA path.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) do { cudaError_t e_ = (call); ++g_checks; if (e_ != cudaSuccess) { std::printf("FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__, #call, (int)e_); ++g_fail; } } while (0)
#define ERR(call, want) do { cudaError_t e_ = (call); ++g_checks; if (e_ != (want)) { std::printf("FAIL %s:%d: %s -> %d, want %d\n", __FILE__, __LINE__, #call, (int)e_, (int)(want)); ++g_fail; } (void)cudaGetLastError(); } while (0)

// ---- kernels -----------------------------------------------------------

// Records the launch geometry it sees and proves that dynamic shared memory
// of the requested size is usable.
__global__ void geometry(int *out, int dyn_ints)
{
    extern __shared__ int s[];
    int t = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    int nt = blockDim.x * blockDim.y * blockDim.z;
    for (int i = t; i < dyn_ints; i += nt) s[i] = i + 1;
    __syncthreads();
    if (t == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        out[0] = gridDim.x; out[1] = gridDim.y; out[2] = gridDim.z;
        out[3] = blockDim.x; out[4] = blockDim.y; out[5] = blockDim.z;
        int sum = 0;
        for (int i = 0; i < dyn_ints; ++i) sum += s[i];
        out[6] = sum;
    }
}

// Mixed parameter types, to see the arguments converted to the parameter types.
__global__ void mixed(float *fo, double *dout, long long *lo, unsigned char *co,
                      const float *in, float a, double d, long long ll, unsigned char c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fo[i] = in[i] * a;
    if (i == 0) { dout[0] = d; lo[0] = ll; co[0] = c; }
}

__device__ int g_hits;
__global__ void no_args() { atomicAdd(&g_hits, 1); }

__global__ void set_flag(int *p, int v) { if (threadIdx.x == 0 && blockIdx.x == 0) *p = v; }

// Producer / consumer pair of the programmatic dependent launch pattern.
__global__ void producer(int *t, const int *x, int n, int round)
{
    cudaTriggerProgrammaticLaunchCompletion();   // early trigger, then the real work
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) t[i] = x[i] + round;
}
__global__ void consumer(int *x, const int *t, int n)
{
    cudaGridDependencySynchronize();             // t is only valid after this
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = (t[i] * 3) & 0xffff;
}

// A kernel that writes the result late; used with the event attributes.
__global__ void slow_write(int *p, int n, int v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int a = 0;
        for (int k = 0; k < 2000; ++k) a += (i ^ k) & 1;
        p[i] = v + (a & 0);
    }
}

__global__ void sync_calls(int *p)
{
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
    if (threadIdx.x == 0) atomicAdd(p, 1);
}

static std::vector<int> down(const int *d, int n)
{
    std::vector<int> h(n);
    if (n) cudaMemcpy(h.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost);
    return h;
}

static cudaLaunchConfig_t make_cfg(dim3 g, dim3 b, size_t dyn, cudaStream_t s)
{
    cudaLaunchConfig_t c = {};
    c.gridDim = g; c.blockDim = b; c.dynamicSmemBytes = dyn; c.stream = s;
    return c;
}

int main()
{
    // ---- item 1: declarations ------------------------------------------
    {
        cudaLaunchConfig_t cfg = {};
        CHECK(cfg.numAttrs == 0 && cfg.attrs == nullptr && cfg.stream == nullptr);
        CHECK(cfg.dynamicSmemBytes == 0 && cfg.gridDim.x == 1);
        cudaLaunchAttribute a;
        std::memset(&a, 0, sizeof a);
        a.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        a.val.programmaticStreamSerializationAllowed = 1;
        CHECK(a.id == 6 && a.val.programmaticStreamSerializationAllowed == 1);
        a.val.clusterDim.x = 2; a.val.clusterDim.y = 3; a.val.clusterDim.z = 4;
        CHECK(a.val.clusterDim.x == 2 && a.val.clusterDim.y == 3 && a.val.clusterDim.z == 4);
        a.val.priority = -1;
        CHECK(a.val.priority == -1);
        CHECK(sizeof(cudaLaunchAttributeValue) >= sizeof(int) * 3);
        CHECK((int)cudaLaunchAttributeIgnore == 0);
        CHECK((int)cudaLaunchAttributeAccessPolicyWindow == 1);
        CHECK((int)cudaLaunchAttributeCooperative == 2);
        CHECK((int)cudaLaunchAttributeSynchronizationPolicy == 3);
        CHECK((int)cudaLaunchAttributeClusterDimension == 4);
        CHECK((int)cudaLaunchAttributeClusterSchedulingPolicyPreference == 5);
        CHECK((int)cudaLaunchAttributeProgrammaticStreamSerialization == 6);
        CHECK((int)cudaLaunchAttributeProgrammaticEvent == 7);
        CHECK((int)cudaLaunchAttributePriority == 8);
        CHECK((int)cudaLaunchAttributeMemSyncDomainMap == 9);
        CHECK((int)cudaLaunchAttributeMemSyncDomain == 10);
        CHECK((int)cudaLaunchAttributePreferredClusterDimension == 11);
        CHECK((int)cudaLaunchAttributeLaunchCompletionEvent == 12);
        CHECK((int)cudaLaunchAttributeDeviceUpdatableKernelNode == 13);
        CHECK((int)cudaLaunchAttributePreferredSharedMemoryCarveout == 14);
    }

    int *d_out = nullptr;
    OK(cudaMalloc(&d_out, 64 * sizeof(int)));
    cudaStream_t st;
    OK(cudaStreamCreate(&st));

    // ---- item 2: geometry, dynamic shared memory, stream ----------------
    {
        struct G { dim3 g, b; int dyn; } cases[] = {
            {dim3(1), dim3(1), 1},        {dim3(3), dim3(64), 100},
            {dim3(2, 3), dim3(8, 4), 32}, {dim3(2, 2, 2), dim3(4, 2, 2), 16},
            {dim3(5), dim3(33), 1000},    {dim3(1), dim3(128), 0},
        };
        for (const G &c : cases) {
            std::vector<int> init(64, -1);
            OK(cudaMemcpy(d_out, init.data(), 64 * sizeof(int), cudaMemcpyHostToDevice));
            cudaLaunchConfig_t cfg = make_cfg(c.g, c.b, c.dyn * sizeof(int), st);
            OK(cudaLaunchKernelEx(&cfg, geometry, d_out, c.dyn));
            OK(cudaStreamSynchronize(st));
            std::vector<int> r = down(d_out, 7);
            CHECK(r[0] == (int)c.g.x && r[1] == (int)c.g.y && r[2] == (int)c.g.z);
            CHECK(r[3] == (int)c.b.x && r[4] == (int)c.b.y && r[5] == (int)c.b.z);
            CHECK(r[6] == c.dyn * (c.dyn + 1) / 2);
        }
        // Same launch with 0 attributes and a non-null attrs pointer.
        cudaLaunchAttribute dummy;
        std::memset(&dummy, 0, sizeof dummy);
        cudaLaunchConfig_t cfg = make_cfg(dim3(2), dim3(32), 8 * sizeof(int), st);
        cfg.attrs = &dummy; cfg.numAttrs = 0;
        OK(cudaLaunchKernelEx(&cfg, geometry, d_out, 8));
        OK(cudaStreamSynchronize(st));
        CHECK(down(d_out, 7)[6] == 36);
    }

    // Stream order: a launch on a stream follows earlier work on it and
    // precedes later work (device-side ordering seen by the host).
    {
        OK(cudaMemset(d_out, 0, 4 * sizeof(int)));
        cudaLaunchConfig_t cfg = make_cfg(dim3(1), dim3(1), 0, st);
        OK(cudaLaunchKernelEx(&cfg, set_flag, d_out, 11));
        OK(cudaLaunchKernelEx(&cfg, set_flag, d_out + 1, 22));
        void *args[] = {&d_out, nullptr};
        int v = 33;
        args[1] = &v;
        OK(cudaLaunchKernel(set_flag, dim3(1), dim3(1), args, 0, st));
        OK(cudaStreamSynchronize(st));
        std::vector<int> r = down(d_out, 2);
        CHECK(r[0] == 33 && r[1] == 22);
    }

    // Argument conversion: int literals to float / double / long long /
    // unsigned char parameters; float* to const float*.
    {
        const int n = 200;
        std::vector<float> h(n);
        for (int i = 0; i < n; ++i) h[i] = float(i) * 0.5f;
        float *in, *fo;
        double *dd;
        long long *ll;
        unsigned char *cc;
        OK(cudaMalloc(&in, n * sizeof(float)));
        OK(cudaMalloc(&fo, n * sizeof(float)));
        OK(cudaMalloc(&dd, sizeof(double)));
        OK(cudaMalloc(&ll, sizeof(long long)));
        OK(cudaMalloc(&cc, 1));
        OK(cudaMemcpy(in, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        cudaLaunchConfig_t cfg = make_cfg(dim3((n + 63) / 64), dim3(64), 0, st);
        // 3 (int) -> float, 7.25f (float) -> double, 5 (int) -> long long, 300 (int) -> unsigned char
        OK(cudaLaunchKernelEx(&cfg, mixed, fo, dd, ll, cc, in, 3, 7.25f, 5, 300, n));
        OK(cudaStreamSynchronize(st));
        std::vector<float> r(n);
        OK(cudaMemcpy(r.data(), fo, n * sizeof(float), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < n; ++i) bad += r[i] != h[i] * 3.0f;
        CHECK(bad == 0);
        double dv = 0; long long lv = 0; unsigned char cv = 0;
        OK(cudaMemcpy(&dv, dd, sizeof dv, cudaMemcpyDeviceToHost));
        OK(cudaMemcpy(&lv, ll, sizeof lv, cudaMemcpyDeviceToHost));
        OK(cudaMemcpy(&cv, cc, 1, cudaMemcpyDeviceToHost));
        CHECK(dv == 7.25);
        CHECK(lv == 5);
        CHECK(cv == (unsigned char)300);
        // The arguments are copied at the call: later changes have no effect.
        float a = 2.0f; int m = n;
        OK(cudaLaunchKernelEx(&cfg, mixed, fo, dd, ll, cc, in, a, 1.0, 1LL, 1, m));
        a = 100.0f; m = 0;
        OK(cudaStreamSynchronize(st));
        OK(cudaMemcpy(r.data(), fo, n * sizeof(float), cudaMemcpyDeviceToHost));
        bad = 0;
        for (int i = 0; i < n; ++i) bad += r[i] != h[i] * 2.0f;
        CHECK(bad == 0);
        OK(cudaFree(in)); OK(cudaFree(fo)); OK(cudaFree(dd)); OK(cudaFree(ll)); OK(cudaFree(cc));
    }

    // Kernel without parameters.
    {
        int zero = 0;
        OK(cudaMemcpyToSymbol(g_hits, &zero, sizeof zero));
        cudaLaunchConfig_t cfg = make_cfg(dim3(4), dim3(8), 0, st);
        OK(cudaLaunchKernelEx(&cfg, no_args));
        OK(cudaStreamSynchronize(st));
        int hits = -1;
        OK(cudaMemcpyFromSymbol(&hits, g_hits, sizeof hits));
        CHECK(hits == 32);
    }

    // ---- item 3: producer / consumer chains with the PDL attribute -------
    {
        const int n = 1000, rounds = 60;
        int *x, *t;
        OK(cudaMalloc(&x, n * sizeof(int)));
        OK(cudaMalloc(&t, n * sizeof(int)));
        std::vector<int> hx(n), ref(n);
        for (int i = 0; i < n; ++i) hx[i] = ref[i] = i;
        OK(cudaMemcpy(x, hx.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        for (int use_stream = 0; use_stream < 2; ++use_stream) {
            cudaStream_t s = use_stream ? st : nullptr;
            OK(cudaMemcpy(x, hx.data(), n * sizeof(int), cudaMemcpyHostToDevice));
            for (int i = 0; i < n; ++i) ref[i] = i;
            cudaLaunchAttribute attr;
            std::memset(&attr, 0, sizeof attr);
            attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr.val.programmaticStreamSerializationAllowed = 1;
            cudaLaunchConfig_t cfg = make_cfg(dim3((n + 127) / 128), dim3(128), 0, s);
            cfg.attrs = &attr; cfg.numAttrs = 1;
            for (int r = 0; r < rounds; ++r) {
                OK(cudaLaunchKernelEx(&cfg, producer, t, (const int *)x, n, r));
                OK(cudaLaunchKernelEx(&cfg, consumer, x, (const int *)t, n));
                for (int i = 0; i < n; ++i) ref[i] = ((ref[i] + r) * 3) & 0xffff;
            }
            if (s) OK(cudaStreamSynchronize(s)); else OK(cudaDeviceSynchronize());
            CHECK(down(x, n) == ref);
        }
        OK(cudaFree(x)); OK(cudaFree(t));
    }

    // ---- item 4: errors and accepted attributes --------------------------
    {
        OK(cudaMemset(d_out, 0, 4 * sizeof(int)));
        cudaLaunchConfig_t cfg = make_cfg(dim3(1), dim3(1), 0, st);
        ERR(cudaLaunchKernelEx((const cudaLaunchConfig_t *)nullptr, set_flag, d_out, 1), cudaErrorInvalidValue);
        cfg.numAttrs = 1; cfg.attrs = nullptr;
        ERR(cudaLaunchKernelEx(&cfg, set_flag, d_out, 1), cudaErrorInvalidValue);
        cfg.numAttrs = 0;

        cudaLaunchAttribute a;
        std::memset(&a, 0, sizeof a);
        cfg.attrs = &a; cfg.numAttrs = 1;

        a.id = (cudaLaunchAttributeID)999;
        ERR(cudaLaunchKernelEx(&cfg, set_flag, d_out, 1), cudaErrorInvalidValue);
        a.id = cudaLaunchAttributeCooperative; a.val.cooperative = 1;
        ERR(cudaLaunchKernelEx(&cfg, set_flag, d_out, 1), cudaErrorNotSupported);
        a.id = cudaLaunchAttributeClusterDimension;
        a.val.clusterDim.x = 2; a.val.clusterDim.y = 1; a.val.clusterDim.z = 1;
        ERR(cudaLaunchKernelEx(&cfg, set_flag, d_out, 1), cudaErrorNotSupported);
        // None of the failed launches ran the kernel.
        OK(cudaStreamSynchronize(st));
        CHECK(down(d_out, 1)[0] == 0);

        // Accepted attributes (hints) run the kernel.
        int expect = 0;
        a.id = cudaLaunchAttributeIgnore;
        OK(cudaLaunchKernelEx(&cfg, set_flag, d_out, ++expect));
        OK(cudaStreamSynchronize(st));
        CHECK(down(d_out, 1)[0] == expect);
        std::memset(&a, 0, sizeof a);
        a.id = cudaLaunchAttributePriority; a.val.priority = 0;
        OK(cudaLaunchKernelEx(&cfg, set_flag, d_out, ++expect));
        OK(cudaStreamSynchronize(st));
        CHECK(down(d_out, 1)[0] == expect);
        std::memset(&a, 0, sizeof a);
        a.id = cudaLaunchAttributeAccessPolicyWindow;
        a.val.accessPolicyWindow.base_ptr = d_out;
        a.val.accessPolicyWindow.num_bytes = 64;
        a.val.accessPolicyWindow.hitRatio = 0.5f;
        OK(cudaLaunchKernelEx(&cfg, set_flag, d_out, ++expect));
        OK(cudaStreamSynchronize(st));
        CHECK(down(d_out, 1)[0] == expect);
        // Cooperative = 0 and a cluster of one block are not requests.
        std::memset(&a, 0, sizeof a);
        a.id = cudaLaunchAttributeCooperative; a.val.cooperative = 0;
        OK(cudaLaunchKernelEx(&cfg, set_flag, d_out, ++expect));
        std::memset(&a, 0, sizeof a);
        a.id = cudaLaunchAttributeClusterDimension;
        a.val.clusterDim.x = 1; a.val.clusterDim.y = 1; a.val.clusterDim.z = 1;
        OK(cudaLaunchKernelEx(&cfg, set_flag, d_out, ++expect));
        OK(cudaStreamSynchronize(st));
        CHECK(down(d_out, 1)[0] == expect);

        // Several attributes in one launch; a bad one among good ones fails.
        cudaLaunchAttribute many[3];
        std::memset(many, 0, sizeof many);
        many[0].id = cudaLaunchAttributePriority;
        many[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        many[1].val.programmaticStreamSerializationAllowed = 1;
        many[2].id = cudaLaunchAttributeIgnore;
        cfg.attrs = many; cfg.numAttrs = 3;
        OK(cudaLaunchKernelEx(&cfg, set_flag, d_out, ++expect));
        OK(cudaStreamSynchronize(st));
        CHECK(down(d_out, 1)[0] == expect);
        many[2].id = (cudaLaunchAttributeID)999;
        ERR(cudaLaunchKernelEx(&cfg, set_flag, d_out, 1234), cudaErrorInvalidValue);
        OK(cudaStreamSynchronize(st));
        CHECK(down(d_out, 1)[0] == expect);
    }

#if !defined(OCU_CPU_PATH)
    // cudaLaunchKernelExC (void** form). Kernel handles are launch stubs.
    {
        OK(cudaMemset(d_out, 0, 4 * sizeof(int)));
        cudaLaunchConfig_t cfg = make_cfg(dim3(1), dim3(1), 0, st);
        int v = 77;
        void *args[] = {&d_out, &v};
        OK(cudaLaunchKernelExC(&cfg, (const void *)set_flag, args));
        OK(cudaStreamSynchronize(st));
        CHECK(down(d_out, 1)[0] == 77);
        ERR(cudaLaunchKernelExC(nullptr, (const void *)set_flag, args), cudaErrorInvalidValue);
        cfg.numAttrs = 1; cfg.attrs = nullptr;
        ERR(cudaLaunchKernelExC(&cfg, (const void *)set_flag, args), cudaErrorInvalidValue);
    }
#endif

    // ---- item 5: event attributes ----------------------------------------
    for (int which = 0; which < 2; ++which) {
        const int n = 4096;
        int *p;
        OK(cudaMalloc(&p, n * sizeof(int)));
        OK(cudaMemset(p, 0, n * sizeof(int)));
        cudaEvent_t ev;
        OK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        cudaLaunchAttribute a;
        std::memset(&a, 0, sizeof a);
        if (which == 0) {
            a.id = cudaLaunchAttributeProgrammaticEvent;
            a.val.programmaticEvent.event = ev;
            a.val.programmaticEvent.flags = 0;
            a.val.programmaticEvent.triggerAtBlockStart = 0;
        } else {
            a.id = cudaLaunchAttributeLaunchCompletionEvent;
            a.val.launchCompletionEvent.event = ev;
            a.val.launchCompletionEvent.flags = 0;
        }
        cudaLaunchConfig_t cfg = make_cfg(dim3(n / 128), dim3(128), 0, st);
        cfg.attrs = &a; cfg.numAttrs = 1;
        OK(cudaLaunchKernelEx(&cfg, slow_write, p, n, 5 + which));
        OK(cudaEventSynchronize(ev));
        // Launch completion event: the kernel has finished. Programmatic
        // event: triggered by the kernel; with no explicit trigger in the
        // kernel it fires at the latest at completion.
        std::vector<int> r = down(p, n);
        int bad = 0;
        for (int i = 0; i < n; ++i) bad += r[i] != 5 + which;
        CHECK(bad == 0);
        // A NULL event is an invalid handle.
        if (which == 0) a.val.programmaticEvent.event = nullptr;
        else a.val.launchCompletionEvent.event = nullptr;
        ERR(cudaLaunchKernelEx(&cfg, slow_write, p, n, 9), cudaErrorInvalidResourceHandle);
        OK(cudaEventDestroy(ev));
        OK(cudaFree(p));
    }

    // ---- item 6: device functions return at once --------------------------
    {
        OK(cudaMemset(d_out, 0, 4 * sizeof(int)));
        sync_calls<<<3, 16, 0, st>>>(d_out);
        OK(cudaStreamSynchronize(st));
        CHECK(down(d_out, 1)[0] == 3);
        cudaLaunchConfig_t cfg = make_cfg(dim3(2), dim3(16), 0, st);
        OK(cudaLaunchKernelEx(&cfg, sync_calls, d_out));
        OK(cudaStreamSynchronize(st));
        CHECK(down(d_out, 1)[0] == 5);
    }

    OK(cudaStreamDestroy(st));
    OK(cudaFree(d_out));
    std::printf("rt_pdl: %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "ok", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
