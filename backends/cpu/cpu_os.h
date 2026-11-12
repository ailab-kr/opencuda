/*
 * cpu_os.h -- the operating-system services the CPU backend uses, for
 * POSIX systems (Linux, macOS) and Windows. Everything else in the backend
 * is plain C11 plus POSIX threads (on Windows: the MinGW-w64 winpthreads
 * library).
 */
#ifndef OCU_CPU_OS_H
#define OCU_CPU_OS_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_WIN32)
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601 /* Windows 7: fibers, GetPhysicallyInstalledSystemMemory */
#endif
#define WIN32_LEAN_AND_MEAN
#include <malloc.h>
#include <windows.h>
/* No fork(): the fork-safety code is left out. */
#define CPU_OS_HAS_FORK 0
/* Fibers run on stacks the system gives them (cpu_fiber.c), so the backend
   allocates no stack region and keeps no stack canaries. */
#define CPU_OS_OWN_STACKS 0
#else
#include <sys/mman.h>
#include <unistd.h>
#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define MAP_ANONYMOUS MAP_ANON
#endif
#if !defined(MAP_NORESERVE)
#define MAP_NORESERVE 0
#endif
#define CPU_OS_HAS_FORK 1
#define CPU_OS_OWN_STACKS 1
#endif

static inline size_t cpu_os_page_size(void)
{
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize ? (size_t)si.dwPageSize : 4096u;
#else
    long page = sysconf(_SC_PAGESIZE);
    return page > 0 ? (size_t)page : 4096u;
#endif
}

static inline long cpu_os_cpu_count(void)
{
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (long)si.dwNumberOfProcessors;
#else
    return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

/* Physical memory in bytes, 0 when unknown. */
static inline uint64_t cpu_os_phys_bytes(void)
{
#if defined(_WIN32)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? (uint64_t)ms.ullTotalPhys : 0;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page = sysconf(_SC_PAGESIZE);
    return (pages > 0 && page > 0) ? (uint64_t)pages * (uint64_t)page : 0;
#endif
}

#if CPU_OS_OWN_STACKS
/* Read-write memory for stacks, committed lazily; NULL on failure. */
static inline void *cpu_os_map(size_t bytes)
{
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                   -1, 0);
    return p == MAP_FAILED ? NULL : p;
}
static inline int cpu_os_guard(void *p, size_t bytes) { return mprotect(p, bytes, PROT_NONE); }
static inline void cpu_os_unmap(void *p, size_t bytes) { munmap(p, bytes); }
#endif

/* Aligned heap memory; free it with cpu_os_aligned_free. */
static inline void *cpu_os_aligned_alloc(size_t align, size_t bytes)
{
#if defined(_WIN32)
    return _aligned_malloc(bytes ? bytes : 1, align);
#else
    void *p = NULL;
    return posix_memalign(&p, align, bytes ? bytes : 1) == 0 ? p : NULL;
#endif
}

static inline void cpu_os_aligned_free(void *p)
{
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

#endif /* OCU_CPU_OS_H */
