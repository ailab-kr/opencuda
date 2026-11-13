// __align__(n) written after a struct's closing brace, as real programs do
// ("struct S { ... } __align__(8);"), next to the prefix forms. Documented
// meaning: the type's alignment is at least n bytes (CUDA C++ Programming
// Guide, "__align__"). Checks alignof, sizeof, placement in arrays, in
// kernel parameters, in local, shared and global memory, and through
// copies to and from the device.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) do { ++g_checks; cudaError_t e_ = (call); if (e_ != cudaSuccess) { std::printf("FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__, #call, cudaGetErrorName(e_)); ++g_fail; } } while (0)

struct S8 { int a; int b; } __align__(8);
struct S16 { float x; char c; } __align__(16);
struct S4 { char c; } __align__(4);
struct __align__(8) P8 { int a; int b; };                 // prefix form, control
struct S8b {
    int a;
    int b;
}
__align__(8);                                              // attribute on its own line
struct N8 { struct { int q; } inner; short s; } __align__(8);   // nested braces before the attribute
template <typename T> struct Box { T v; } __align__(16);
typedef struct { double d; } V8d;
struct S32 { char c[3]; } __align__(32);

static_assert(alignof(S8) == 8 && sizeof(S8) == 8, "S8");
static_assert(alignof(S16) == 16 && sizeof(S16) == 16, "S16");
static_assert(alignof(S4) == 4 && sizeof(S4) == 4, "S4");
static_assert(alignof(P8) == 8, "P8");
static_assert(alignof(S8b) == 8 && sizeof(S8b) == 8, "S8b");
static_assert(alignof(N8) == 8 && sizeof(N8) == 8, "N8");
static_assert(alignof(Box<char>) == 16 && sizeof(Box<char>) == 16, "Box<char>");
static_assert(alignof(S32) == 32 && sizeof(S32) == 32, "S32");

struct Holder { char pad; S8 s; char pad2; S16 t; };       // members get their types' alignment
static_assert(offsetof(Holder, s) == 8, "member alignment of S8");
static_assert(offsetof(Holder, t) == 32, "member alignment of S16");
static_assert(sizeof(Holder) == 48, "Holder size");

__device__ S8 g_s8;
__device__ S16 g_s16[3];

__global__ void kern(const S8 *in, S8 *out, S16 byval, S8 byval8, int *info, int n)
{
    __shared__ S16 sh;
    S8 local;
    S16 loc16;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i].a = in[i].a + byval8.a;
        out[i].b = in[i].b * 2;
    }
    if (i == 0) {
        sh.x = byval.x;
        local.a = 1; loc16.x = 2;
        info[0] = (int)(((uintptr_t)&sh) % 16);
        info[1] = (int)(((uintptr_t)&local) % 8);
        info[2] = (int)(((uintptr_t)&loc16) % 16);
        info[3] = (int)(((uintptr_t)&g_s8) % 8);
        info[4] = (int)(((uintptr_t)&g_s16[1]) % 16);
        info[5] = (int)(byval.x);
        info[6] = (int)sizeof(S16);
        g_s8.a = 41;
        g_s16[2].x = 3.5f;
    }
}

int main()
{
    const int n = 100;
    std::vector<S8> in(n), out(n);
    for (int i = 0; i < n; ++i) { in[i].a = i; in[i].b = 3 * i; }
    S8 *din, *dout; int *dinfo;
    OK(cudaMalloc(&din, n * sizeof(S8)));
    OK(cudaMalloc(&dout, n * sizeof(S8)));
    OK(cudaMalloc(&dinfo, 8 * sizeof(int)));
    // device allocations are aligned for any type, so S8 arrays meet their alignment
    CHECK(((uintptr_t)din) % alignof(S8) == 0);
    OK(cudaMemcpy(din, in.data(), n * sizeof(S8), cudaMemcpyHostToDevice));
    S16 b16; b16.x = 9.f; b16.c = 1;
    S8 b8; b8.a = 100; b8.b = 0;
    kern<<<2, 64>>>(din, dout, b16, b8, dinfo, n);
    OK(cudaDeviceSynchronize());
    OK(cudaMemcpy(out.data(), dout, n * sizeof(S8), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < n; ++i) ok = ok && out[i].a == i + 100 && out[i].b == 6 * i;
    CHECK(ok);
    int info[8] = {};
    OK(cudaMemcpy(info, dinfo, sizeof info, cudaMemcpyDeviceToHost));
    CHECK(info[0] == 0);
    CHECK(info[1] == 0);
    CHECK(info[2] == 0);
    CHECK(info[3] == 0);
    CHECK(info[4] == 0);
    CHECK(info[5] == 9);
    CHECK(info[6] == 16);
    // host alignment of objects
    S8 hs; S16 h16; S32 h32; Box<char> hb;
    CHECK(((uintptr_t)&hs) % 8 == 0);
    CHECK(((uintptr_t)&h16) % 16 == 0);
    CHECK(((uintptr_t)&h32) % 32 == 0);
    CHECK(((uintptr_t)&hb) % 16 == 0);
    S8 *arr = new S8[5];
    CHECK(((uintptr_t)&arr[1]) - ((uintptr_t)&arr[0]) == 8);
    delete[] arr;
    // device globals through the symbol interface
    S8 gs = {0, 0};
    OK(cudaMemcpyFromSymbol(&gs, g_s8, sizeof gs));
    CHECK(gs.a == 41);
    S16 g16[3];
    std::memset(g16, 0, sizeof g16);
    OK(cudaMemcpyFromSymbol(g16, g_s16, sizeof g16));
    CHECK(g16[2].x == 3.5f);
    cudaFree(din); cudaFree(dout); cudaFree(dinfo);
#ifdef OCU_MUTANT
    CHECK(sizeof(S8) == 16);
#endif
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
