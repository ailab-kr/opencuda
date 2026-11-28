// Kernel arguments larger than OpenCL drivers take (CL_DEVICE_MAX_PARAMETER_SIZE,
// 2048 bytes on an Arc Pro B70; CUDA takes 4 KB, 32 KB since 12.1), in the
// shape of PyTorch's / Apex's multi_tensor_apply: a struct of tensor
// addresses, sizes and block maps passed by value (~3 KB). Checked: every
// field reaches the kernel, through a device function too; 20 launches in
// a row on one stream, each with different contents, the host struct
// rewritten right after each launch (a launch copies its arguments); two
// large structs in one kernel next to small arguments; cudaLaunchKernel
// with an argument array; a 12 KB struct.
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_big_args.cu -o rt_big_args
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

static int g_fail, g_checks;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { if (g_fail++ < 20) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } } while (0)

constexpr int NT = 48, NB = 320;
struct TensorList {
    float *addresses[3][NT];
    long long sizes[NT];
    unsigned char block_to_tensor[NB];
    int block_to_chunk[NB];
    int start;
};
static_assert(sizeof(TensorList) > 3000, "larger than the driver's limit");
struct Huge { int v[3072]; }; // 12 KB

__device__ float fetch(const TensorList &tl, int b) { return tl.addresses[0][tl.block_to_tensor[b]][tl.block_to_chunk[b]]; }

// out[b] = x[tensor][chunk] + sizes[tensor] + start, per block b
__global__ void apply(int nblocks, TensorList tl, float *out)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < nblocks) out[b] = fetch(tl, b) + (float)tl.sizes[tl.block_to_tensor[b]] + (float)tl.start;
}
__global__ void two(float scale, TensorList a, int *flag, TensorList b, float *out)
{
    int i = threadIdx.x;
    out[i] = scale * (float)(a.block_to_chunk[i] - b.block_to_chunk[i]) + (float)(a.start + b.start);
    if (i == 0) *flag = 1;
}
__global__ void huge(Huge h, long long *out)
{
    long long s = 0;
    for (int i = threadIdx.x; i < 3072; i += blockDim.x) s += h.v[i];
    atomicAdd((unsigned long long *)out, (unsigned long long)s);
}

int main()
{
    const int nchunk = 64;
    float *data;
    cudaMalloc(&data, NT * nchunk * sizeof(float));
    static float h[NT * nchunk];
    for (int i = 0; i < NT * nchunk; ++i) h[i] = (float)(i % 1000) * 0.5f;
    cudaMemcpy(data, h, sizeof h, cudaMemcpyHostToDevice);
    float *out;
    cudaMalloc(&out, 20 * NB * sizeof(float));
    cudaStream_t s;
    cudaStreamCreate(&s);

    // 20 launches back to back, the host struct changed after each
    static TensorList tl;
    static TensorList sent[20];
    for (int r = 0; r < 20; ++r) {
        std::memset(&tl, 0, sizeof tl);
        for (int t = 0; t < NT; ++t) {
            tl.addresses[0][t] = data + t * nchunk;
            tl.sizes[t] = 1000 * (t + r);
        }
        for (int b = 0; b < NB; ++b) {
            tl.block_to_tensor[b] = (unsigned char)((b * 7 + r) % NT);
            tl.block_to_chunk[b] = (b * 13 + r) % nchunk;
        }
        tl.start = r * 3;
        sent[r] = tl;
        apply<<<(NB + 63) / 64, 64, 0, s>>>(NB, tl, out + r * NB);
        std::memset(&tl, 0xff, sizeof tl); // the launch already copied it
    }
    CHECK(cudaStreamSynchronize(s) == cudaSuccess, "launches");
    static float got[20 * NB];
    cudaMemcpy(got, out, sizeof got, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int r = 0; r < 20; ++r)
        for (int b = 0; b < NB; ++b) {
            const TensorList &t = sent[r];
            int tt = t.block_to_tensor[b];
            float want = h[tt * nchunk + t.block_to_chunk[b]] + (float)t.sizes[tt] + (float)t.start;
            if (got[r * NB + b] != want && bad++ < 3) std::printf("  round %d block %d: %g want %g\n", r, b, got[r * NB + b], want);
        }
    CHECK(bad == 0, "%d wrong values in 20 launches", bad);

    // two large structs and small arguments around them
    TensorList a = sent[3], b = sent[11];
    int *flag;
    cudaMalloc(&flag, sizeof(int));
    cudaMemset(flag, 0, sizeof(int));
    two<<<1, 256>>>(2.0f, a, flag, b, out);
    CHECK(cudaDeviceSynchronize() == cudaSuccess, "two");
    float g2[256];
    int f = 0;
    cudaMemcpy(g2, out, sizeof g2, cudaMemcpyDeviceToHost);
    cudaMemcpy(&f, flag, sizeof f, cudaMemcpyDeviceToHost);
    bad = 0;
    for (int i = 0; i < 256; ++i) bad += g2[i] != 2.0f * (float)(a.block_to_chunk[i] - b.block_to_chunk[i]) + (float)(a.start + b.start);
    CHECK(bad == 0 && f == 1, "two structs: %d wrong, flag %d", bad, f);

    // cudaLaunchKernel with an argument array (the typed form: the CPU path
    // cannot launch a kernel known only as a const void *)
    int nb = NB;
    float *o2 = out + NB;
    void *args[] = {&nb, &a, &o2};
    CHECK(cudaLaunchKernel(apply, dim3((NB + 63) / 64), dim3(64), args, 0, s) == cudaSuccess, "cudaLaunchKernel");
    cudaStreamSynchronize(s);
    cudaMemcpy(got, o2, NB * sizeof(float), cudaMemcpyDeviceToHost);
    bad = 0;
    for (int bb = 0; bb < NB; ++bb) {
        int tt = a.block_to_tensor[bb];
        bad += got[bb] != h[tt * nchunk + a.block_to_chunk[bb]] + (float)a.sizes[tt] + (float)a.start;
    }
    CHECK(bad == 0, "cudaLaunchKernel: %d wrong", bad);

    // 12 KB
    static Huge hg;
    long long want = 0;
    for (int i = 0; i < 3072; ++i) { hg.v[i] = i * 3 - 1000; want += hg.v[i]; }
    long long *dsum, sum = 0;
    cudaMalloc(&dsum, sizeof sum);
    cudaMemset(dsum, 0, sizeof sum);
    huge<<<1, 128>>>(hg, dsum);
    CHECK(cudaDeviceSynchronize() == cudaSuccess, "huge");
    cudaMemcpy(&sum, dsum, sizeof sum, cudaMemcpyDeviceToHost);
    CHECK(sum == want, "12 KB struct: sum %lld want %lld", sum, want);

    cudaFree(data); cudaFree(out); cudaFree(flag); cudaFree(dsum);
    cudaStreamDestroy(s);
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
