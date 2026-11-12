/* Item 1 of the card libs-cub-subset in plain C: the launch configuration
   types and the attribute ids compile in C and carry the documented values. */
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <string.h>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

int main(void)
{
    cudaLaunchConfig_t cfg = {0};
    cudaLaunchAttribute a;
    CHECK(cfg.numAttrs == 0 && cfg.attrs == 0 && cfg.dynamicSmemBytes == 0);
    memset(&a, 0, sizeof a);
    a.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    a.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &a;
    cfg.numAttrs = 1;
    CHECK(cfg.attrs[0].id == 6);
    CHECK(cfg.attrs[0].val.programmaticStreamSerializationAllowed == 1);
    a.val.clusterDim.x = 2; a.val.clusterDim.y = 1; a.val.clusterDim.z = 1;
    CHECK(a.val.clusterDim.x == 2 && a.val.clusterDim.y == 1 && a.val.clusterDim.z == 1);
    CHECK(cudaLaunchAttributeIgnore == 0);
    CHECK(cudaLaunchAttributeAccessPolicyWindow == 1);
    CHECK(cudaLaunchAttributeCooperative == 2);
    CHECK(cudaLaunchAttributeSynchronizationPolicy == 3);
    CHECK(cudaLaunchAttributeClusterDimension == 4);
    CHECK(cudaLaunchAttributeClusterSchedulingPolicyPreference == 5);
    CHECK(cudaLaunchAttributeProgrammaticEvent == 7);
    CHECK(cudaLaunchAttributePriority == 8);
    CHECK(cudaLaunchAttributeMemSyncDomainMap == 9);
    CHECK(cudaLaunchAttributeMemSyncDomain == 10);
    CHECK(cudaLaunchAttributePreferredClusterDimension == 11);
    CHECK(cudaLaunchAttributeLaunchCompletionEvent == 12);
    CHECK(cudaLaunchAttributeDeviceUpdatableKernelNode == 13);
    CHECK(cudaLaunchAttributePreferredSharedMemoryCarveout == 14);
    printf("rt_pdl_c: %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "ok", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
