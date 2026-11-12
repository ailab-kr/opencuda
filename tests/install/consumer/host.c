/* C host code using the runtime API from the installed headers. */
#include <cuda_runtime_api.h>
#include <stdio.h>

int main(void)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) return 1;
    void *p = NULL;
    if (cudaMalloc(&p, 1024) != cudaSuccess || cudaMemset(p, 0, 1024) != cudaSuccess) return 1;
    if (cudaFree(p) != cudaSuccess) return 1;
    printf("consumer_host: ok (%d device)\n", n);
    return 0;
}
