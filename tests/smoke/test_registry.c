/*
 * Smoke test for the skeleton: the registry finds the CPU backend, rejects
 * bad tables, and the CPU backend reports a device.
 */
#include "ocu_core.h"

#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                  \
        }                                                                \
    } while (0)

int main(void)
{
    CHECK(ocu_backend_count() >= 1);
    CHECK(ocu_select_backend("cpu") == OCU_OK);
    CHECK(ocu_select_backend("no-such-backend") == OCU_ERR_INVALID_VALUE);
    CHECK(ocu_select_backend(NULL) == OCU_ERR_INVALID_VALUE);

    const ocu_backend *b = ocu_active_backend();
    CHECK(b != NULL);
    if (b == NULL) {
        return 1;
    }
    CHECK(b->abi == OCU_BACKEND_ABI);
    CHECK(strcmp(b->name, "cpu") == 0);

    uint32_t count = 0;
    CHECK(b->dev_count(&count) == OCU_OK);
    CHECK(count == 1);
    CHECK(b->dev_count(NULL) == OCU_ERR_INVALID_VALUE);

    ocu_dev_info info;
    CHECK(b->dev_info(0, &info) == OCU_OK);
    CHECK(info.warp_size == 32 || info.warp_size == 64);
    CHECK(info.name[0] != '\0');
    CHECK(b->dev_info(1, &info) == OCU_ERR_INVALID_DEVICE);

    /* Unimplemented entries must say so, never pretend to succeed. */
    void *p = NULL;
    CHECK(b->mem_alloc(NULL, 16, &p) == OCU_ERR_NOT_IMPLEMENTED);

    /* Registration rejects a wrong ABI, an incomplete table, a duplicate. */
    ocu_backend bad = *b;
    bad.name = "bad-abi";
    bad.abi = OCU_BACKEND_ABI + 1u;
    CHECK(ocu_register_backend(&bad) == OCU_ERR_NOT_SUPPORTED);

    static ocu_backend incomplete;
    incomplete = *b;
    incomplete.name = "incomplete";
    incomplete.kernel_launch = NULL;
    CHECK(ocu_register_backend(&incomplete) == OCU_ERR_INVALID_VALUE);

    CHECK(ocu_register_backend(b) == OCU_ERR_INVALID_VALUE);
    CHECK(ocu_register_backend(NULL) == OCU_ERR_INVALID_VALUE);

    CHECK(strcmp(ocu_status_name(OCU_ERR_NOT_READY), "OCU_ERR_NOT_READY") == 0);

    if (failures == 0) {
        printf("smoke_registry: ok\n");
    }
    return failures == 0 ? 0 : 1;
}
