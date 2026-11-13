// 300 kernels of one type, launched through pointers from this translation
// unit (the pass lets a pointer launch loop for at most 256 kernels of a
// type; the rest must still run, on fibers, with the right results).
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

typedef void (*kfn)(int *);

template <int N> __global__ void k_ptr_many(int *o) { o[blockIdx.x * blockDim.x + threadIdx.x] = N * 1000 + (int)threadIdx.x; }

template <size_t... I> static void fill_many(std::vector<kfn> &v, std::index_sequence<I...>) { v = {k_ptr_many<(int)I>...}; }

unsigned long long run_many(bool *okp)
{
    const int B = 2, N = 32;
    int *o = nullptr;
    cudaMalloc((void **)&o, B * N * 4);
    std::vector<kfn> many;
    fill_many(many, std::make_index_sequence<300>());
    bool ok = true;
    unsigned long long h = 1469598103934665603ull;
    for (size_t i = 0; i < many.size(); ++i) {
        cudaMemset(o, 0, B * N * 4);
        kfn km = many[i];
        km<<<B, N>>>(o);
        cudaDeviceSynchronize();
        std::vector<int> v(B * N);
        cudaMemcpy(v.data(), o, B * N * 4, cudaMemcpyDeviceToHost);
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < N; ++t) ok &= v[b * N + t] == (int)i * 1000 + t;
        const unsigned char *c = (const unsigned char *)v.data();
        for (size_t k = 0; k < v.size() * 4; ++k) { h ^= c[k]; h *= 1099511628211ull; }
    }
    cudaFree(o);
    *okp = ok;
    return h;
}
