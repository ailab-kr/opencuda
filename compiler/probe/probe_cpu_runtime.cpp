// PROBE ONLY: the device the CPU path launches on, taken from the backend
// registry (ocu_core). Link it with objects built by `ocu_cc.py --target cpu`
// unless the program supplies ocu_cpu::backend() and ocu_cpu::device().
#include "probe_cpu.h"
#include "ocu_core.h"

#include <cstdio>
#include <cstdlib>

namespace ocu_cpu {

namespace {
struct active {
    const ocu_backend *b = nullptr;
    ocu_device *d = nullptr;
    active()
    {
        if (ocu_select_backend("cpu") != OCU_OK || ocu_core_device(&b, &d) != OCU_OK) {
            std::fprintf(stderr, "probe: no CPU backend\n");
            std::abort();
        }
    }
};
active &get()
{
    static active a;
    return a;
}
} // namespace

const ocu_backend *backend() { return get().b; }
ocu_device *device() { return get().d; }

} // namespace ocu_cpu
