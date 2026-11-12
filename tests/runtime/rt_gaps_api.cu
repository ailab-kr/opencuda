// Runtime API items of card runtime-llama-gaps (items 10 to 15):
// cudaDataType values, device attribute policy, PCI bus id, peer access
// errors and cooperative launch.
//
// Every numeric value below was read from the public CUDA Runtime API
// documentation (enumerations cudaDeviceAttr and cudaError, "Data types
// used by CUDA Runtime") and the library-types documentation, not from the
// headers of this project:
//   cudaDevAttrWarpSize 10, MultiProcessorCount 16, PciBusId 33,
//   PciDeviceId 34, PciDomainId 50, ComputeCapabilityMajor 75,
//   ComputeCapabilityMinor 76, CooperativeLaunch 95,
//   MaxSharedMemoryPerBlockOptin 97;
//   cudaErrorInvalidValue 1, InvalidDevice 101, PeerAccessUnsupported 217,
//   LaunchFailure 719, PeerAccessAlreadyEnabled 704,
//   PeerAccessNotEnabled 705, NotSupported 801.
// cudaDevAttrCooperativeMultiDeviceLaunch (96, deprecated) could not be
// confirmed on the fetched pages; it is checked against 96 as the card
// states and the report says so.
//
// Usage: rt_gaps_api [major minor]   expected compute capability (default 5 0)
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == cudaSuccess)
#define ERR(call, want) do { cudaError_t e_ = (call); ++g_checks; if (e_ != (want)) { std::printf("FAIL %s:%d: %s returned %d (%s), want %d\n", __FILE__, __LINE__, #call, (int)e_, cudaGetErrorName(e_), (int)(want)); ++g_fail; } } while (0)

__global__ void write_marker(int *p) { *p = 1234; }

// ---- item 10: cudaDataType ---------------------------------------------------

static void test_data_types()
{
    CHECK((int)CUDA_R_32F == 0);
    CHECK((int)CUDA_R_64F == 1);
    CHECK((int)CUDA_R_16F == 2);
    CHECK((int)CUDA_R_8I == 3);
    CHECK((int)CUDA_C_32F == 4);
    CHECK((int)CUDA_C_64F == 5);
    CHECK((int)CUDA_C_16F == 6);
    CHECK((int)CUDA_C_8I == 7);
    CHECK((int)CUDA_R_8U == 8);
    CHECK((int)CUDA_C_8U == 9);
    CHECK((int)CUDA_R_32I == 10);
    CHECK((int)CUDA_C_32I == 11);
    CHECK((int)CUDA_R_32U == 12);
    CHECK((int)CUDA_C_32U == 13);
    CHECK((int)CUDA_R_16BF == 14);
    CHECK((int)CUDA_C_16BF == 15);
    CHECK((int)CUDA_R_4I == 16);
    CHECK((int)CUDA_C_4I == 17);
    CHECK((int)CUDA_R_4U == 18);
    CHECK((int)CUDA_C_4U == 19);
    CHECK((int)CUDA_R_16I == 20);
    CHECK((int)CUDA_C_16I == 21);
    CHECK((int)CUDA_R_16U == 22);
    CHECK((int)CUDA_C_16U == 23);
    CHECK((int)CUDA_R_64I == 24);
    CHECK((int)CUDA_C_64I == 25);
    CHECK((int)CUDA_R_64U == 26);
    CHECK((int)CUDA_C_64U == 27);
    CHECK((int)CUDA_R_8F_E4M3 == 28);
    CHECK((int)CUDA_R_8F_E5M2 == 29);
    // the two names are one type
    cudaDataType a = CUDA_R_16BF;
    cudaDataType_t b = a;
    cudaDataType c = b;
    CHECK(c == CUDA_R_16BF);
    static_assert(__is_same(cudaDataType, cudaDataType_t), "cudaDataType and cudaDataType_t must be one type");
}

// ---- numeric values of enumerators the tests depend on -----------------------

static void test_enum_values()
{
    CHECK((int)cudaDevAttrWarpSize == 10);
    CHECK((int)cudaDevAttrMultiProcessorCount == 16);
    CHECK((int)cudaDevAttrPciBusId == 33);
    CHECK((int)cudaDevAttrPciDeviceId == 34);
    CHECK((int)cudaDevAttrPciDomainId == 50);
    CHECK((int)cudaDevAttrComputeCapabilityMajor == 75);
    CHECK((int)cudaDevAttrComputeCapabilityMinor == 76);
    CHECK((int)cudaDevAttrCooperativeLaunch == 95);
    CHECK((int)cudaDevAttrCooperativeMultiDeviceLaunch == 96);
    CHECK((int)cudaDevAttrMaxSharedMemoryPerBlockOptin == 97);
    CHECK((int)cudaErrorInvalidValue == 1);
    CHECK((int)cudaErrorInvalidDevice == 101);
    CHECK((int)cudaErrorPeerAccessUnsupported == 217);
    CHECK((int)cudaErrorLaunchFailure == 719);
    CHECK((int)cudaErrorPeerAccessAlreadyEnabled == 704);
    CHECK((int)cudaErrorPeerAccessNotEnabled == 705);
    CHECK((int)cudaErrorNotSupported == 801);
}

// ---- items 11 and 12: attributes ---------------------------------------------

static void test_attributes(int want_major, int want_minor)
{
    cudaDeviceProp p;
    OK(cudaGetDeviceProperties(&p, 0));
    CHECK(p.major == want_major);
    CHECK(p.minor == want_minor);
    CHECK(p.warpSize == 32);
    CHECK(p.sharedMemPerBlockOptin >= 48 * 1024);
    CHECK(p.multiProcessorCount > 0);

    int v = -1;
    OK(cudaDeviceGetAttribute(&v, cudaDevAttrComputeCapabilityMajor, 0));
    CHECK(v == p.major);
    OK(cudaDeviceGetAttribute(&v, cudaDevAttrComputeCapabilityMinor, 0));
    CHECK(v == p.minor);
    OK(cudaDeviceGetAttribute(&v, cudaDevAttrWarpSize, 0));
    CHECK(v == p.warpSize);
    OK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0));
    CHECK(v == p.multiProcessorCount);
    OK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    CHECK(v == (int)p.sharedMemPerBlockOptin);

    cudaFuncAttributes fa;
    OK(cudaFuncGetAttributes(&fa, write_marker));
    CHECK(fa.ptxVersion < 90);
    CHECK(fa.ptxVersion == 10 * p.major + p.minor);

    // cooperative launch is never available
    v = -1;
    OK(cudaDeviceGetAttribute(&v, cudaDevAttrCooperativeLaunch, 0));
    CHECK(v == 0);
    v = -1;
    OK(cudaDeviceGetAttribute(&v, cudaDevAttrCooperativeMultiDeviceLaunch, 0));
    CHECK(v == 0);
    v = -1;
    OK(cudaDeviceGetAttribute(&v, (cudaDeviceAttr)95, 0)); // the documented number itself
    CHECK(v == 0);
    v = -1;
    OK(cudaDeviceGetAttribute(&v, (cudaDeviceAttr)96, 0));
    CHECK(v == 0);

    // an invalid device number
    int n = 0;
    OK(cudaGetDeviceCount(&n));
    ERR(cudaDeviceGetAttribute(&v, cudaDevAttrCooperativeLaunch, n), cudaErrorInvalidDevice);
    (void)cudaGetLastError();
    ERR(cudaDeviceGetAttribute(&v, cudaDevAttrCooperativeMultiDeviceLaunch, 1000), cudaErrorInvalidDevice);
    (void)cudaGetLastError();
    ERR(cudaDeviceGetAttribute(&v, cudaDevAttrCooperativeLaunch, -1), cudaErrorInvalidDevice);
    (void)cudaGetLastError();

    // PCI attributes equal the property fields, for every device
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp q;
        OK(cudaGetDeviceProperties(&q, d));
        int bus = -1, dev = -1, dom = -1;
        OK(cudaDeviceGetAttribute(&bus, cudaDevAttrPciBusId, d));
        OK(cudaDeviceGetAttribute(&dev, cudaDevAttrPciDeviceId, d));
        OK(cudaDeviceGetAttribute(&dom, cudaDevAttrPciDomainId, d));
        CHECK(bus == q.pciBusID);
        CHECK(dev == q.pciDeviceID);
        CHECK(dom == q.pciDomainID);
    }
}

// ---- item 13: PCI bus id -------------------------------------------------------

static void test_pci_bus_id()
{
    int n = 0;
    OK(cudaGetDeviceCount(&n));
    char buf[64];
    std::memset(buf, 'X', sizeof buf);
    OK(cudaDeviceGetPCIBusId(buf, 64, 0));
    CHECK(std::strcmp(buf, "0000:01:00.0") == 0);
    char again[64];
    OK(cudaDeviceGetPCIBusId(again, 64, 0));
    CHECK(std::strcmp(buf, again) == 0);
    if (n >= 2) {
        OK(cudaDeviceGetPCIBusId(buf, 64, 1));
        CHECK(std::strcmp(buf, "0000:02:00.0") == 0);
    }
    // exactly 13 bytes: 12 characters and the terminating NUL
    char tight[16];
    std::memset(tight, 'X', sizeof tight);
    OK(cudaDeviceGetPCIBusId(tight, 13, 0));
    CHECK(std::strcmp(tight, "0000:01:00.0") == 0);
    CHECK(tight[12] == '\0');
    CHECK(tight[13] == 'X'); // nothing written past len
    // too short
    for (int len : {0, 1, 12, -1}) {
        ERR(cudaDeviceGetPCIBusId(tight, len, 0), cudaErrorInvalidValue);
        (void)cudaGetLastError();
    }
    ERR(cudaDeviceGetPCIBusId(nullptr, 64, 0), cudaErrorInvalidValue);
    (void)cudaGetLastError();
    // out of range device
    ERR(cudaDeviceGetPCIBusId(buf, 64, n), cudaErrorInvalidDevice);
    (void)cudaGetLastError();
    ERR(cudaDeviceGetPCIBusId(buf, 64, -1), cudaErrorInvalidDevice);
    (void)cudaGetLastError();
}

// ---- item 14: peer access -------------------------------------------------------

static void test_peer()
{
    int n = 0;
    OK(cudaGetDeviceCount(&n));
    // flags must be 0
    ERR(cudaDeviceEnablePeerAccess(0, 1), cudaErrorInvalidValue);
    CHECK(cudaGetLastError() == cudaErrorInvalidValue); // left for cudaGetLastError
    ERR(cudaDeviceEnablePeerAccess(0, 0x80), cudaErrorInvalidValue);
    (void)cudaGetLastError();
    // oneself and nonexistent devices
    int cur = -1;
    OK(cudaGetDevice(&cur));
    ERR(cudaDeviceEnablePeerAccess(cur, 0), cudaErrorInvalidDevice);
    CHECK(cudaGetLastError() == cudaErrorInvalidDevice);
    ERR(cudaDeviceEnablePeerAccess(n, 0), cudaErrorInvalidDevice);
    CHECK(cudaGetLastError() == cudaErrorInvalidDevice);
    ERR(cudaDeviceEnablePeerAccess(-1, 0), cudaErrorInvalidDevice);
    CHECK(cudaGetLastError() == cudaErrorInvalidDevice);
    ERR(cudaDeviceEnablePeerAccess(9999, 0), cudaErrorInvalidDevice);
    (void)cudaGetLastError();
    // another real device
    for (int d = 0; d < n; ++d) {
        if (d == cur) continue;
        ERR(cudaDeviceEnablePeerAccess(d, 0), cudaErrorPeerAccessUnsupported);
        CHECK(cudaGetLastError() == cudaErrorPeerAccessUnsupported);
        int can = -1;
        OK(cudaDeviceCanAccessPeer(&can, cur, d));
        CHECK(can == 0);
    }
    // disabling what was never enabled
    if (n >= 2) {
        ERR(cudaDeviceDisablePeerAccess(cur == 0 ? 1 : 0), cudaErrorPeerAccessNotEnabled);
        CHECK(cudaGetLastError() == cudaErrorPeerAccessNotEnabled);
    } else { // no peer other than the current device: it must fail, with either error
        cudaError_t de = cudaDeviceDisablePeerAccess(0);
        CHECK(de == cudaErrorPeerAccessNotEnabled || de == cudaErrorInvalidDevice);
        (void)cudaGetLastError();
    }
    // the runtime still works afterwards
    int *p = nullptr;
    OK(cudaMalloc(&p, 4));
    OK(cudaFree(p));
}

// ---- item 15: cooperative launch ---------------------------------------------------

static void test_cooperative_launch()
{
    int *d = nullptr;
    OK(cudaMalloc(&d, sizeof(int)));
    int sentinel = 77, got = 0;
    OK(cudaMemcpy(d, &sentinel, sizeof sentinel, cudaMemcpyHostToDevice));
    void *args[] = {&d};
    ERR(cudaLaunchCooperativeKernel((const void *)write_marker, dim3(1), dim3(1), args, 0, 0), cudaErrorNotSupported);
    (void)cudaGetLastError();
    OK(cudaDeviceSynchronize());
    OK(cudaMemcpy(&got, d, sizeof got, cudaMemcpyDeviceToHost));
    CHECK(got == 77); // the kernel did not run
    // the typed function-pointer overload is checked to compile in cg_compile_checks (a compile failure here would hide everything else)
    ERR(cudaLaunchCooperativeKernel((const void *)write_marker, dim3(2), dim3(32), args, 0, 0), cudaErrorNotSupported);
    (void)cudaGetLastError();
    OK(cudaDeviceSynchronize());
    OK(cudaMemcpy(&got, d, sizeof got, cudaMemcpyDeviceToHost));
    CHECK(got == 77);
    // on a stream too
    cudaStream_t s;
    OK(cudaStreamCreate(&s));
    ERR(cudaLaunchCooperativeKernel((const void *)write_marker, dim3(1), dim3(1), args, 0, s), cudaErrorNotSupported);
    (void)cudaGetLastError();
    OK(cudaStreamSynchronize(s));
    OK(cudaMemcpy(&got, d, sizeof got, cudaMemcpyDeviceToHost));
    CHECK(got == 77);
    OK(cudaStreamDestroy(s));
    // an ordinary launch still runs
    write_marker<<<1, 1>>>(d);
    OK(cudaDeviceSynchronize());
    OK(cudaMemcpy(&got, d, sizeof got, cudaMemcpyDeviceToHost));
    CHECK(got == 1234);
    OK(cudaFree(d));
}

int main(int argc, char **argv)
{
    int major = 5, minor = 0;
    if (argc >= 3) { major = std::atoi(argv[1]); minor = std::atoi(argv[2]); }
    test_data_types();
    test_enum_values();
    test_attributes(major, minor);
    test_pci_bus_id();
    test_peer();
    test_cooperative_launch();
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
