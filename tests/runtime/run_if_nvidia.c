/*
 * run_if_nvidia <program> [args...]: run the program when an NVIDIA device
 * is usable, else exit 77 (the test is then reported as skipped).
 */
#include "ocu_backend_nvidia.h"

#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    if (argc < 2) return 2;
    uint32_t n = 0;
    if (ocu_backend_nvidia()->dev_count(&n) != OCU_OK || n == 0) {
        printf("SKIP: no NVIDIA device\n");
        return 77;
    }
    execv(argv[1], argv + 1);
    perror("execv");
    return 1;
}
