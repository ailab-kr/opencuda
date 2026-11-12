// BLAS conformance driver for the BLAS library API (cublas_v2.h): reads the
// reference cases on stdin, runs each through the library as a user
// program would (device memory, a handle, host scalars), writes outputs on
// stdout. Same protocol as blas_driver.c (the internal BLAS layer's
// driver); run it with
//   python3 -m ocu_conf.blas_run <this> --only gemm,gemv:f32,axpy:f32,dot:f32,scal:f32,nrm2:f32,trsm:f32
// since the library offers only those functions. OCU_DRIVER_STREAM=1 runs
// every call on a non-default stream.
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int MAX_FIELDS = 32;

struct field {
    char kind;
    char name[32];
    long i;
    char c;
    double r;
    std::vector<double> v;
};

struct kase {
    char id[256], fn[32], kind[8];
    field f[MAX_FIELDS];
    int nf;
};

field &get(kase &k, const char *name)
{
    for (int i = 0; i < k.nf; ++i)
        if (std::strcmp(k.f[i].name, name) == 0) return k.f[i];
    std::fprintf(stderr, "driver: case %s has no field %s\n", k.id, name);
    std::exit(2);
}

long I(kase &k, const char *n) { return get(k, n).i; }
double R(kase &k, const char *n) { return get(k, n).r; }
char C(kase &k, const char *n) { return get(k, n).c; }

bool read_case(kase &k)
{
    char tok[256];
    if (std::scanf("%255s", tok) != 1) return false;
    if (std::strcmp(tok, "case") != 0) std::exit(2);
    if (std::scanf("%255s %31s %7s", k.id, k.fn, k.kind) != 3) std::exit(2);
    k.nf = 0;
    for (;;) {
        if (std::scanf("%255s", tok) != 1) std::exit(2);
        if (std::strcmp(tok, "end") == 0) return true;
        if (k.nf == MAX_FIELDS) std::exit(2);
        field &f = k.f[k.nf++];
        f.kind = tok[0];
        f.v.clear();
        if (std::scanf("%31s", f.name) != 1) std::exit(2);
        switch (f.kind) {
        case 'i': if (std::scanf("%ld", &f.i) != 1) std::exit(2); break;
        case 'c': if (std::scanf(" %c", &f.c) != 1) std::exit(2); break;
        case 'r': if (std::scanf("%255s", tok) != 1) std::exit(2); f.r = std::strtod(tok, nullptr); break;
        case 'v': {
            long count;
            if (std::scanf("%ld", &count) != 1 || count < 0) std::exit(2);
            f.v.resize(count);
            for (long j = 0; j < count; ++j) {
                if (std::scanf("%255s", tok) != 1) std::exit(2);
                f.v[j] = std::strtod(tok, nullptr);
            }
            break;
        }
        default: std::exit(2);
        }
    }
}

cublasHandle_t H;
cudaStream_t S;
bool f32;

void *to_dev(kase &k, const char *name)
{
    field &f = get(k, name);
    const size_t es = f32 ? 4 : 8, n = f.v.empty() ? 1 : f.v.size();
    std::vector<unsigned char> host(n * es, 0);
    for (size_t j = 0; j < f.v.size(); ++j) {
        if (f32) { float x = float(f.v[j]); std::memcpy(&host[j * 4], &x, 4); }
        else std::memcpy(&host[j * 8], &f.v[j], 8);
    }
    void *d = nullptr;
    if (cudaMalloc(&d, host.size()) != cudaSuccess ||
        cudaMemcpy(d, host.data(), host.size(), cudaMemcpyHostToDevice) != cudaSuccess)
        std::exit(3);
    return d;
}

void emit_vec(kase &k, const char *name, void *d)
{
    field &f = get(k, name);
    const size_t es = f32 ? 4 : 8, n = f.v.empty() ? 1 : f.v.size();
    std::vector<unsigned char> host(n * es);
    if (cudaMemcpy(host.data(), d, host.size(), cudaMemcpyDeviceToHost) != cudaSuccess) std::exit(3);
    std::printf("v %s %zu", name, f.v.size());
    for (size_t j = 0; j < f.v.size(); ++j) {
        double v;
        if (f32) { float x; std::memcpy(&x, &host[j * 4], 4); v = x; }
        else std::memcpy(&v, &host[j * 8], 8);
        std::printf(" %a", v);
    }
    std::printf("\n");
}

cublasOperation_t op(char c) { return c == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T; }

/* Wait for queued work; a stream error turns into a failed status. */
int finish(cublasStatus_t st)
{
    if (st == CUBLAS_STATUS_SUCCESS && cudaStreamSynchronize(S) != cudaSuccess)
        st = CUBLAS_STATUS_EXECUTION_FAILED;
    return int(st);
}

void run(kase &k)
{
    const char *fn = k.fn;
    if (!f32 && std::strcmp(fn, "gemm") != 0) {
        std::fprintf(stderr, "driver: no %s for %s\n", fn, k.kind);
        std::exit(2);
    }
    int st = 0;
    const char *out_name = nullptr;
    void *out = nullptr;
    std::vector<void *> temps;
    double rres = 0;
    bool has_r = false;
    if (std::strcmp(fn, "axpy") == 0) {
        void *x = to_dev(k, "x"), *y = to_dev(k, "y");
        const float a = float(R(k, "alpha"));
        st = finish(cublasSaxpy(H, int(I(k, "n")), &a, (float *)x, int(I(k, "incx")), (float *)y, int(I(k, "incy"))));
        out_name = "y"; out = y; temps.push_back(x);
    } else if (std::strcmp(fn, "dot") == 0) {
        void *x = to_dev(k, "x"), *y = to_dev(k, "y");
        float r = 0;
        st = int(cublasSdot(H, int(I(k, "n")), (float *)x, int(I(k, "incx")), (float *)y, int(I(k, "incy")), &r));
        rres = r; has_r = true; temps.push_back(x); temps.push_back(y);
    } else if (std::strcmp(fn, "scal") == 0) {
        void *x = to_dev(k, "x");
        const float a = float(R(k, "alpha"));
        st = finish(cublasSscal(H, int(I(k, "n")), &a, (float *)x, int(I(k, "incx"))));
        out_name = "x"; out = x;
    } else if (std::strcmp(fn, "nrm2") == 0) {
        void *x = to_dev(k, "x");
        float r = 0;
        st = int(cublasSnrm2(H, int(I(k, "n")), (float *)x, int(I(k, "incx")), &r));
        rres = r; has_r = true; temps.push_back(x);
    } else if (std::strcmp(fn, "gemv") == 0) {
        void *a = to_dev(k, "a"), *x = to_dev(k, "x"), *y = to_dev(k, "y");
        const float al = float(R(k, "alpha")), be = float(R(k, "beta"));
        st = finish(cublasSgemv(H, op(C(k, "trans")), int(I(k, "m")), int(I(k, "n")), &al, (float *)a,
                                int(I(k, "lda")), (float *)x, int(I(k, "incx")), &be, (float *)y, int(I(k, "incy"))));
        out_name = "y"; out = y; temps.push_back(a); temps.push_back(x);
    } else if (std::strcmp(fn, "gemm") == 0) {
        void *a = to_dev(k, "a"), *b = to_dev(k, "b"), *c = to_dev(k, "c");
        const int m = int(I(k, "m")), n = int(I(k, "n")), kk = int(I(k, "k"));
        const int lda = int(I(k, "lda")), ldb = int(I(k, "ldb")), ldc = int(I(k, "ldc"));
        if (f32) {
            const float al = float(R(k, "alpha")), be = float(R(k, "beta"));
            st = finish(cublasSgemm(H, op(C(k, "transa")), op(C(k, "transb")), m, n, kk, &al, (float *)a, lda,
                                    (float *)b, ldb, &be, (float *)c, ldc));
        } else {
            const double al = R(k, "alpha"), be = R(k, "beta");
            st = finish(cublasDgemm(H, op(C(k, "transa")), op(C(k, "transb")), m, n, kk, &al, (double *)a, lda,
                                    (double *)b, ldb, &be, (double *)c, ldc));
        }
        out_name = "c"; out = c; temps.push_back(a); temps.push_back(b);
    } else if (std::strcmp(fn, "trsm") == 0) { // a batch of one
        void *a = to_dev(k, "a"), *b = to_dev(k, "b");
        void **arr = nullptr;
        void *host_arr[2] = {a, b};
        if (cudaMalloc(&arr, sizeof host_arr) != cudaSuccess ||
            cudaMemcpy(arr, host_arr, sizeof host_arr, cudaMemcpyHostToDevice) != cudaSuccess)
            std::exit(3);
        const float al = float(R(k, "alpha"));
        st = finish(cublasStrsmBatched(H, C(k, "side") == 'L' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
                                       C(k, "uplo") == 'U' ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
                                       op(C(k, "trans")), C(k, "diag") == 'U' ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
                                       int(I(k, "m")), int(I(k, "n")), &al, (const float *const *)arr, int(I(k, "lda")),
                                       (float *const *)arr + 1, int(I(k, "ldb")), 1));
        out_name = "b"; out = b; temps.push_back(a); temps.push_back(arr);
    } else {
        std::fprintf(stderr, "driver: no %s in this library\n", fn);
        std::exit(2);
    }
    std::printf("out %s %d\n", k.id, st);
    if (has_r) std::printf("r result %a\n", rres);
    if (out) { emit_vec(k, out_name, out); cudaFree(out); }
    for (void *t : temps) cudaFree(t);
    std::printf("end\n");
}

} // namespace

int main()
{
    if (cublasCreate(&H) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "driver: no device\n");
        return 3;
    }
    const char *smode = std::getenv("OCU_DRIVER_STREAM");
    if (smode && std::strcmp(smode, "1") == 0) {
        if (cudaStreamCreate(&S) != cudaSuccess) return 3;
        cublasSetStream(H, S);
    }
    static kase k;
    while (read_case(k)) {
        f32 = std::strcmp(k.kind, "f32") == 0;
        run(k);
    }
    cublasDestroy(H);
    if (S) cudaStreamDestroy(S);
    return 0;
}
