/*
 * Fast paths of the CPU backend (card backend-cpu, "tester list" items 1-9),
 * written from the card, ARCHITECTURE.md section 7 and the public header
 * ocu_cpu_kernel.h only.
 *
 * Expected values are computed on the host from a plain model: a warp is
 * `warp_size` consecutive threads of a block by linear index, a shuffle reads
 * lane `src` of the warp (own value when it is not a lane of this warp), a
 * vote has bit i set for lane i with a nonzero predicate.
 *
 * Run it in the modes the CMake file lists: fixed and random order, warp 32
 * and 64, one and several OS threads, strict, small stack, spin 0.
 *
 *   test_cpu_fastpath               all groups
 *   test_cpu_fastpath stats_child   internal: the program whose stderr the
 *                                   statistics test inspects
 */
#define _GNU_SOURCE

#include "ocu_backend.h"
#include "ocu_backend_cpu.h"
#include "ocu_cpu_kernel.h"

#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define OCU_T_ASAN 1
#endif
#if __has_feature(thread_sanitizer)
#define OCU_T_TSAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define OCU_T_ASAN 1
#endif
#if defined(__SANITIZE_THREAD__)
#define OCU_T_TSAN 1
#endif

static int g_failures, g_checks;
static const char *g_test = "";

#define CHECK(cond)                                                          \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL [%s] %s:%d: %s\n", g_test, __FILE__,       \
                    __LINE__, #cond);                                        \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

/* A check with the numbers that matter when it fails. */
#define CHECK_EQ(got, want)                                                  \
    do {                                                                     \
        unsigned long long g_ = (unsigned long long)(got);                   \
        unsigned long long w_ = (unsigned long long)(want);                  \
        ++g_checks;                                                          \
        if (g_ != w_) {                                                      \
            fprintf(stderr, "FAIL [%s] %s:%d: %s == %s: got %llu want %llu\n", \
                    g_test, __FILE__, __LINE__, #got, #want, g_, w_);        \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

static const ocu_backend *B;
static ocu_device *D;
static ocu_module *M;
static uint32_t W;                  /* warp size the backend reports */
static int g_strict, g_random, g_sanitized;
static unsigned long g_stack_kb = 128;

static const char *g_self;          /* argv[0] */

/* ---- device-side helpers ------------------------------------------------ */

static inline uint32_t lin(void)
{
    ocu_cpu_dim3 t = ocu_cpu_thread_idx(), d = ocu_cpu_block_dim();
    return t.x + d.x * (t.y + d.y * t.z);
}
static inline uint32_t nthreads(void)
{
    ocu_cpu_dim3 d = ocu_cpu_block_dim();
    return d.x * d.y * d.z;
}
static inline uint64_t H(uint64_t t, uint64_t r)
{
    uint64_t x = t * 0x9E3779B97F4A7C15ull + r * 0xD1B54A32D192ED03ull + 12345u;
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 32;
    return x;
}
static inline int pred_of(uint64_t t, uint64_t r) { return (int)((H(t, r) >> 11) & 1u); }

/* Warp geometry of thread `l` in a block of n threads, warp size w. */
static inline uint32_t wbase(uint32_t l, uint32_t w) { return l - l % w; }
static inline uint32_t wcount(uint32_t l, uint32_t n, uint32_t w)
{
    uint32_t b = wbase(l, w);
    return n - b < w ? n - b : w;
}

#define TAG_SHUF 0xAA
#define TAG_VOTE 0xBB
#define TAG_GATH 0xCC

/* ---- item 1: consecutive warp operations --------------------------------
 * 5 kinds of rounds, constant tags per kind (as a loop at one call site has),
 * so only the sequence number tells consecutive operations apart. */
static void k_chain(void **a)
{
    int *out = *(int **)a[0];
    uint32_t rounds = *(uint32_t *)a[1];
    uint32_t barrier_every = *(uint32_t *)a[2];
    uint32_t l = lin(), n = nthreads(), w = ocu_cpu_warp_size();
    uint32_t lane = l % w, base = wbase(l, w), cnt = wcount(l, n, w);
    uint64_t all[OCU_CPU_MAX_WARP];
    int bad = 0;
    for (uint32_t r = 0; r < rounds; ++r) {
        switch (r % 5) {
        case 0: {
            uint32_t src = (lane * 7u + r * 3u) % 70u;
            uint64_t got = ocu_cpu_warp_shuffle_masked(H(l, r), TAG_SHUF, 0, src);
            uint64_t want = src < cnt ? H(base + src, r) : H(l, r);
            bad += got != want;
            break;
        }
        case 1: {
            uint64_t got = ocu_cpu_warp_vote_masked(pred_of(l, r), TAG_VOTE, 0);
            uint64_t want = 0;
            for (uint32_t i = 0; i < cnt; ++i) want |= (uint64_t)pred_of(base + i, r) << i;
            bad += got != want;
            break;
        }
        case 2: {
            uint32_t c = ocu_cpu_warp_gather_tagged(H(l, r), TAG_GATH, all);
            bad += c != cnt;
            for (uint32_t i = 0; i < cnt && i < c; ++i) bad += all[i] != H(base + i, r);
            break;
        }
        case 3: { /* two shuffles back to back, the second reads the first */
            uint64_t g1 = ocu_cpu_warp_shuffle_masked(H(l, r), TAG_SHUF, 0, (lane + 1u) % cnt);
            uint64_t g2 = ocu_cpu_warp_shuffle_masked(g1, TAG_SHUF, 0, (lane + cnt - 1u) % cnt);
            bad += g2 != H(l, r);
            break;
        }
        default: { /* two votes back to back with different predicates */
            uint64_t v1 = ocu_cpu_warp_vote_masked(pred_of(l, r), TAG_VOTE, 0);
            uint64_t v2 = ocu_cpu_warp_vote_masked(pred_of(l, r + 1000), TAG_VOTE, 0);
            uint64_t w1 = 0, w2 = 0;
            for (uint32_t i = 0; i < cnt; ++i) {
                w1 |= (uint64_t)pred_of(base + i, r) << i;
                w2 |= (uint64_t)pred_of(base + i, r + 1000) << i;
            }
            bad += v1 != w1;
            bad += v2 != w2;
            break;
        }
        }
        if (barrier_every != 0 && r % barrier_every == 0) ocu_cpu_barrier();
    }
    out[ocu_cpu_block_idx().x * nthreads() + l] = bad;
}

/* Disjoint masks with different operation counts: the two halves of each
 * 32-lane group run unrelated sequences of different length, and the upper
 * half starts with one extra operation so the parities of their sequence
 * numbers differ. Each half checks the values of its own half. */
static void k_halves_skew(void **a)
{
    int *out = *(int **)a[0];
    uint32_t l = lin(), n = nthreads(), w = ocu_cpu_warp_size();
    uint32_t lane = l % w, lane32 = l % 32u, g0 = lane - lane32; /* group start in warp */
    uint32_t base = wbase(l, w);
    int upper = lane32 >= 16;
    uint32_t mask = upper ? 0xffff0000u : 0x0000ffffu;
    uint32_t first = upper ? 16u : 0u; /* first lane32 of my half */
    uint32_t rounds = upper ? 37u : 50u;
    uint64_t tag = upper ? 0x2222 : 0x1111;
    int bad = 0;
    if (upper) { /* one extra operation: parity differs from the lower half */
        uint64_t got = ocu_cpu_warp_shuffle_masked(H(l, 999), 0x3333, mask, g0 + 16u + 3u);
        uint32_t src_lin = base + g0 + 16u + 3u;
        bad += src_lin < n ? got != H(src_lin, 999) : got != H(l, 999);
    }
    for (uint32_t r = 0; r < rounds; ++r) {
        uint32_t s = first + (lane32 * 5u + r) % 16u; /* a lane of my half */
        uint32_t src_lin = base + g0 + s;
        if (src_lin >= n) continue; /* only complete halves are used in the test */
        uint64_t got = ocu_cpu_warp_shuffle_masked(H(l, r), tag, mask, g0 + s);
        bad += got != H(src_lin, r);
        uint64_t bits = ocu_cpu_warp_vote_masked(pred_of(l, r), tag + 1, mask);
        uint32_t want = 0;
        for (uint32_t j = first; j < first + 16u; ++j) want |= (uint32_t)pred_of(base + g0 + j, r) << j;
        bad += (uint32_t)(bits >> g0) != want && ((bits >> g0) & mask) != want;
    }
    out[ocu_cpu_block_idx().x * nthreads() + l] = bad;
}

/* ---- item 5: masked shuffle / vote at edges ----------------------------- */
static void k_edge(void **a)
{
    int *out = *(int **)a[0];
    uint32_t mask_kind = *(uint32_t *)a[1]; /* 0: mask 0, 1: 0xffffffff */
    uint32_t l = lin(), n = nthreads(), w = ocu_cpu_warp_size();
    uint32_t base = wbase(l, w), cnt = wcount(l, n, w);
    uint32_t mask = mask_kind ? 0xffffffffu : 0u;
    static const uint32_t srcs[] = {0, 1, 2, 5, 31, 32, 33, 62, 63, 64, 65, 127, 0x7fffffffu,
                                    0x80000000u, 0xfffffffeu, 0xffffffffu};
    uint64_t all[OCU_CPU_MAX_WARP];
    int bad = 0;
    for (uint32_t k = 0; k < sizeof(srcs) / sizeof(srcs[0]); ++k) {
        uint32_t src = srcs[k];
        uint64_t got = ocu_cpu_warp_shuffle_masked(H(l, k), 0x51, mask, src);
        uint64_t want = src < cnt ? H(base + src, k) : H(l, k);
        bad += got != want;
    }
    /* src = last lane of this warp, and one past it */
    {
        uint64_t got = ocu_cpu_warp_shuffle_masked(H(l, 77), 0x52, mask, cnt - 1);
        bad += got != H(base + cnt - 1, 77);
        got = ocu_cpu_warp_shuffle_masked(H(l, 78), 0x52, mask, cnt);
        bad += got != H(l, 78);
    }
    /* the shuffle equals the gather */
    for (uint32_t k = 0; k < 4; ++k) {
        uint32_t c = ocu_cpu_warp_gather_masked(H(l, 500 + k), 0x53, mask, all);
        bad += c != cnt;
        uint32_t src = (l * 3u + k) % (w + 3u);
        uint64_t got = ocu_cpu_warp_shuffle_masked(H(l, 500 + k), 0x54, mask, src);
        bad += got != (src < c ? all[src] : H(l, 500 + k));
    }
    /* predicates: any nonzero int counts */
    static const int preds[] = {0, 1, -1, 2, 256, INT_MIN, INT_MAX};
    for (uint32_t k = 0; k < sizeof(preds) / sizeof(preds[0]); ++k) {
        uint64_t got = ocu_cpu_warp_vote_masked(preds[(l + k) % 7], 0x55, mask);
        uint64_t want = 0;
        for (uint32_t i = 0; i < cnt; ++i) want |= (uint64_t)(preds[(base + i + k) % 7] != 0) << i;
        bad += got != want;
        bad += (cnt < 64) && (got >> cnt) != 0; /* lanes the warp does not have read 0 */
    }
    out[ocu_cpu_block_idx().x * nthreads() + l] = bad;
}

/* Partial masks: halves, even/odd, single-lane. Lanes use 32-lane groups. */
static void k_partial(void **a)
{
    int *out = *(int **)a[0];
    uint32_t mode = *(uint32_t *)a[1];
    uint32_t l = lin(), n = nthreads(), w = ocu_cpu_warp_size();
    uint32_t base = wbase(l, w), lane = l % w, lane32 = l % 32u, g0 = lane - lane32;
    uint32_t mask;
    switch (mode) {
    case 0: mask = lane32 < 16 ? 0x0000ffffu : 0xffff0000u; break;
    case 1: mask = (lane32 & 1u) ? 0xAAAAAAAAu : 0x55555555u; break;
    default: mask = 1u << lane32; break;
    }
    int bad = 0;
    for (uint32_t r = 0; r < 6; ++r) {
        uint32_t lowest = (uint32_t)__builtin_ctz(mask);
        uint32_t src = g0 + lowest;
        uint64_t got = ocu_cpu_warp_shuffle_masked(H(l, r), 0x61 + mode, mask, src);
        bad += got != H(base + src, r);
        uint64_t bits = ocu_cpu_warp_vote_masked(pred_of(l, r), 0x71 + mode, mask);
        uint32_t want = 0;
        for (uint32_t j = 0; j < 32; ++j) {
            if (((mask >> j) & 1u) && base + g0 + j < n) want |= (uint32_t)pred_of(base + g0 + j, r) << j;
        }
        bad += (uint32_t)((bits >> g0) & mask) != want;
        /* a gather with the same mask: named lanes' values */
        uint64_t all[OCU_CPU_MAX_WARP];
        ocu_cpu_warp_gather_masked(H(l, r + 50), 0x81 + mode, mask, all);
        for (uint32_t j = 0; j < 32; ++j) {
            if (((mask >> j) & 1u) && base + g0 + j < n) bad += all[g0 + j] != H(base + g0 + j, r + 50);
        }
    }
    out[ocu_cpu_block_idx().x * nthreads() + l] = bad;
}

/* ---- item 6: lane id and warp size ------------------------------------- */
static void k_lane(void **a)
{
    int *out = *(int **)a[0];
    uint32_t want_w = *(uint32_t *)a[1];
    uint32_t l = lin();
    int bad = 0;
    bad += ocu_cpu_warp_size() != want_w;
    bad += (uint32_t)ocu_cpu_tls_warp_size != want_w;
    bad += ocu_cpu_lane_id() != l % want_w;
    ocu_cpu_barrier();
    bad += ocu_cpu_lane_id() != l % want_w;
    ocu_cpu_warp_sync();
    bad += ocu_cpu_lane_id() != l % want_w;
    bad += ocu_cpu_warp_size() != want_w;
    out[ocu_cpu_block_idx().x * nthreads() + l] = bad;
}

/* ---- item 2: mismatch detection ---------------------------------------- */
/* op kinds: 0 shuffle_masked, 1 vote_masked, 2 gather_tagged, 3 gather_masked */
static void warp_op(uint32_t kind, uint64_t tag, uint32_t mask, uint64_t v)
{
    uint64_t all[OCU_CPU_MAX_WARP];
    switch (kind) {
    case 0: (void)ocu_cpu_warp_shuffle_masked(v, tag, mask, 0); break;
    case 1: (void)ocu_cpu_warp_vote_masked((int)(v & 1u), tag, mask); break;
    case 2: (void)ocu_cpu_warp_gather_tagged(v, tag, all); break;
    default: (void)ocu_cpu_warp_gather_masked(v, tag, mask, all); break;
    }
}

/* One lane of each warp comes with another tag. */
static void k_tag_mismatch(void **a)
{
    uint32_t kind = *(uint32_t *)a[0];
    uint32_t odd = *(uint32_t *)a[1]; /* lane (of the warp) that differs */
    uint32_t round = *(uint32_t *)a[2]; /* which of 4 consecutive operations */
    uint32_t l = lin(), w = ocu_cpu_warp_size();
    for (uint32_t r = 0; r < 4; ++r) {
        uint64_t tag = 0x900 + kind;
        if (r == round && l % w == odd) tag += 1;
        warp_op(kind, tag, 0, H(l, r));
    }
}

/* One lane comes with another mask (same tag): the named lanes disagree. */
static void k_mask_mismatch(void **a)
{
    (void)a;
    uint32_t l = lin() % 32u;
    if (l >= 16) return;
    uint32_t mask = l == 15 ? 0x0000fffeu : 0x0000ffffu;
    warp_op(0, 0x9100, mask, H(l, 1));
}

/* Lane `odd` does one more call than the others at the same site and then
 * waits at a block barrier. */
static void k_extra_call(void **a)
{
    uint32_t odd = *(uint32_t *)a[0];
    uint32_t l = lin();
    uint64_t all[OCU_CPU_MAX_WARP];
    for (int it = 0; it < 3; ++it) {
        if (l == odd && it == 1) continue; /* skips an occurrence */
        ocu_cpu_warp_gather_masked(H(l, it), 0x9200, 0, all);
    }
    ocu_cpu_barrier();
}

/* Lane `gone` returns before the whole-warp operations of the others. */
static void k_named_lane_gone(void **a)
{
    int *out = *(int **)a[0];
    uint32_t gone = *(uint32_t *)a[1];
    uint32_t mask = *(uint32_t *)a[2];
    uint32_t l = lin(), n = nthreads(), w = ocu_cpu_warp_size();
    uint32_t base = wbase(l, w), cnt = wcount(l, n, w);
    out[ocu_cpu_block_idx().x * n + l] = 0;
    if (l % w == gone) return;
    uint64_t all[OCU_CPU_MAX_WARP];
    int bad = 0;
    for (uint32_t r = 0; r < 3; ++r) {
        ocu_cpu_warp_gather_masked(H(l, r), 0x9300, mask, all);
        uint32_t lo = mask ? (l % w) - l % 32u : 0u;   /* a 32-bit mask names one 32-lane group */
        uint32_t hi = mask && lo + 32u < cnt ? lo + 32u : cnt;
        for (uint32_t i = lo; i < hi; ++i) {
            if (i == gone) continue;
            bad += all[i] != H(base + i, r);
        }
    }
    out[ocu_cpu_block_idx().x * n + l] = bad;
}

/* Everyone finishes an operation and returns at once. */
static void k_op_then_return(void **a)
{
    int *out = *(int **)a[0];
    uint32_t kind = *(uint32_t *)a[1];
    uint32_t l = lin(), n = nthreads(), w = ocu_cpu_warp_size();
    uint32_t cnt = wcount(l, n, w), base = wbase(l, w);
    int bad = 0;
    switch (kind) {
    case 0: bad += ocu_cpu_warp_shuffle_masked(H(l, 3), 0x9400, 0, cnt - 1) != H(base + cnt - 1, 3); break;
    case 1: ocu_cpu_warp_sync(); break;
    case 2: ocu_cpu_warp_sync_masked(0); break;
    default: {
        uint64_t v = ocu_cpu_warp_vote_masked(1, 0x9401, 0);
        bad += v != (cnt >= 64 ? ~0ull : ((1ull << cnt) - 1u));
        break;
    }
    }
    out[ocu_cpu_block_idx().x * n + l] = bad;
}

/* Disjoint halves: each half runs its own mask op and returns at once; the
 * upper half continues with a second op on its own mask. Legal in strict. */
static void k_half_then_return(void **a)
{
    int *out = *(int **)a[0];
    uint32_t l = lin(), n = nthreads(), w = ocu_cpu_warp_size();
    uint32_t lane = l % w, lane32 = l % 32u, g0 = lane - lane32, base = wbase(l, w);
    int upper = lane32 >= 16;
    uint32_t mask = upper ? 0xffff0000u : 0x0000ffffu;
    int bad = 0;
    uint64_t got = ocu_cpu_warp_shuffle_masked(H(l, 1), 0x9500 + upper, mask, g0 + (upper ? 16u : 0u));
    bad += got != H(base + g0 + (upper ? 16u : 0u), 1);
    if (upper) {
        got = ocu_cpu_warp_shuffle_masked(H(l, 2), 0x9502, mask, g0 + 31u);
        bad += got != H(base + g0 + 31u, 2);
    }
    out[ocu_cpu_block_idx().x * n + l] = bad;
}

/* After a full-mask op, the lower half returns; the upper half does a second
 * op with mask 0xffffffff (names the finished lower half): strict fails.
 * With *the upper mask* it is legal. */
static void k_second_op_names_gone(void **a)
{
    uint32_t use_full = *(uint32_t *)a[0];
    uint32_t l = lin(), lane32 = l % 32u;
    warp_op(0, 0x9600, 0xffffffffu, H(l, 1));
    if (lane32 < 16) return;
    warp_op(0, 0x9601, use_full ? 0xffffffffu : 0xffff0000u, H(l, 2));
}

/* ---- item 3: start order, mixed sync ----------------------------------- */
static void k_start(void **a)
{
    uint32_t *ticket = *(uint32_t **)a[0];
    uint32_t *ctr = *(uint32_t **)a[1];
    uint32_t variant = *(uint32_t *)a[2];
    uint32_t l = lin(), n = nthreads(), b = ocu_cpu_block_idx().x;
    ticket[b * n + l] = __atomic_fetch_add(&ctr[b], 1u, __ATOMIC_SEQ_CST);
    switch (variant) {
    case 0: break;
    case 1: ocu_cpu_barrier(); break;
    case 2: ocu_cpu_warp_sync(); break;
    case 3: /* even threads end at once, odd ones wait */
        if (l % 2 == 0) return;
        ocu_cpu_barrier();
        break;
    case 4: {
        uint64_t all[OCU_CPU_MAX_WARP];
        ocu_cpu_warp_gather(l, all);
        break;
    }
    case 5: /* threads 0 and 3 mod 4 wait at the warp, the rest return */
        if (l % 4 == 1 || l % 4 == 2) return;
        (void)ocu_cpu_warp_shuffle_masked(l, 0x9700, 0x99999999u, 0);
        break;
    default: break;
    }
}

/* Reduction mixing warp operations and block barriers; order independent. */
static void k_mixed(void **a)
{
    const uint32_t *in = *(const uint32_t **)a[0];
    uint64_t *out = *(uint64_t **)a[1];
    uint64_t *part = ocu_cpu_dynamic_shared();
    uint32_t l = lin(), n = nthreads(), w = ocu_cpu_warp_size(), b = ocu_cpu_block_idx().x;
    uint32_t lane = l % w, cnt = wcount(l, n, w), nw = (n + w - 1) / w;
    uint64_t v = in[b * n + l];
    uint64_t total = 0;
    for (int rep = 0; rep < 3; ++rep) {
        /* warp sum by a tree of shuffles (src = lane + offset, own if out) */
        uint64_t s = v;
        for (uint32_t off = 1; off < w; off <<= 1) {
            uint64_t o = ocu_cpu_warp_shuffle_masked(s, 0x9800, 0, lane + off);
            s += lane + off < cnt ? o : 0;
        }
        /* lane 0 now holds the warp sum (hillis-steele from above is a
           suffix scan: lane 0 has everything) */
        if (lane == 0) part[l / w] = s;
        ocu_cpu_barrier();
        uint64_t sum = 0;
        for (uint32_t k = 0; k < nw; ++k) sum += part[k];
        /* everyone broadcasts the total from the lane 0 of its warp */
        uint64_t t2 = ocu_cpu_warp_shuffle_masked(sum, 0x9801, 0, 0);
        total += t2 + (uint64_t)rep;
        ocu_cpu_barrier();
    }
    out[b * n + l] = total;
}

/* ---- item 4: stacks ------------------------------------------------------ */
__attribute__((noinline)) static uintptr_t frame_addr(void)
{
    return (uintptr_t)__builtin_frame_address(0);
}
static void k_frame(void **a)
{
    uintptr_t *out = *(uintptr_t **)a[0];
    uint32_t wait = *(uint32_t *)a[1];
    uint32_t l = lin();
    uintptr_t f = frame_addr();
    if (wait) ocu_cpu_barrier();
    out[ocu_cpu_block_idx().x * nthreads() + l] = f;
}
/* locals alive across a barrier, mixed with threads that never wait */
static void k_locals(void **a)
{
    int *out = *(int **)a[0];
    uint32_t uniform = *(uint32_t *)a[1];
    uint32_t l = lin();
    volatile uint32_t buf[64];
    for (uint32_t i = 0; i < 64; ++i) buf[i] = l * 100u + i;
    if (uniform || l % 3 != 0) ocu_cpu_barrier();
    int bad = 0;
    for (uint32_t i = 0; i < 64; ++i) bad += buf[i] != l * 100u + i;
    out[ocu_cpu_block_idx().x * nthreads() + l] = bad;
}
/* thread 0 overflows its stack by `extra` bytes beyond the configured size */
__attribute__((noinline)) static void smash(volatile char *p, size_t n)
{
    for (size_t i = 0; i < n; ++i) p[i] = (char)0x5a;
}
static void k_overflow(void **a)
{
    size_t bytes = *(size_t *)a[0];
    if (lin() != 0) return;
    volatile char *buf = __builtin_alloca(bytes);
    smash(buf, bytes);
    __asm__ volatile("" : : "r"(buf) : "memory");
}
static void k_ok(void **a)
{
    int *out = *(int **)a[0];
    uint32_t l = lin();
    ocu_cpu_barrier();
    out[ocu_cpu_block_idx().x * nthreads() + l] = (int)l + 1;
}

/* ---- item 7: spin, idle, fork ------------------------------------------- */
static void k_fill(void **a)
{
    uint32_t *out = *(uint32_t **)a[0];
    out[ocu_cpu_block_idx().x * ocu_cpu_block_dim().x + ocu_cpu_thread_idx().x] =
        ocu_cpu_block_idx().x * 1000u + ocu_cpu_thread_idx().x;
}

/* ---- item 9: x86 control registers --------------------------------------- */
#if defined(__x86_64__)
#define FPU_CFGS 24 /* 4 rounding modes x ftz/daz on/off x x87 precision 24/64 bit */
static const int g_modes[4] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};

static void apply_cfg(uint32_t cfg)
{
    int mode = g_modes[cfg % 4u];
    int ftz = (cfg / 4u) % 2u;
    int pc24 = (cfg / 8u) % 3u == 1;
    fesetround(mode);
    unsigned csr = _mm_getcsr();
    csr = ftz ? (csr | 0x8040u) : (csr & ~0x8040u);
    _mm_setcsr(csr);
    uint16_t cw;
    __asm__ volatile("fnstcw %0" : "=m"(cw));
    cw = (uint16_t)(pc24 ? (cw & ~0x300u) : (cw | 0x300u));
    __asm__ volatile("fldcw %0" : : "m"(cw));
}
/* The two readbacks that must stay with the configuration. */
static int cfg_readback_bad(uint32_t cfg)
{
    int bad = 0;
    int mode = g_modes[cfg % 4u];
    int ftz = (cfg / 4u) % 2u;
    int pc24 = (cfg / 8u) % 3u == 1;
    bad += fegetround() != mode;
    unsigned csr = _mm_getcsr();
    bad += ((csr >> 15) & 1u) != (unsigned)ftz;
    bad += ((csr >> 6) & 1u) != (unsigned)ftz;
    uint16_t cw;
    __asm__ volatile("fnstcw %0" : "=m"(cw));
    bad += ((cw >> 8) & 3u) != (pc24 ? 0u : 3u);
    return bad;
}
__attribute__((noinline)) static void signature(uint64_t *sig)
{
    volatile float one = 1.0f, three = 3.0f, none = -1.0f, tiny = 1.0e-39f, two = 2.0f;
    volatile float c1 = one / three, c2 = none / three, c3 = tiny * one, c4 = two / three;
    uint32_t u;
    memcpy(&u, (const void *)&c1, 4); sig[0] = u;
    memcpy(&u, (const void *)&c2, 4); sig[1] = u;
    memcpy(&u, (const void *)&c3, 4); sig[2] = u;
    memcpy(&u, (const void *)&c4, 4); sig[3] = u;
    volatile long double lo = 1.0L, lt = 3.0L, ln = -1.0L;
    volatile long double d1 = lo / lt, d2 = ln / lt;
    uint64_t m = 0;
    long double t1 = d1, t2 = d2;
    memcpy(&m, &t1, 8); sig[4] = m;
    memcpy(&m, &t2, 8); sig[5] = m;
}
static void k_fpu(void **a)
{
    int *out = *(int **)a[0];
    const uint64_t *table = *(const uint64_t **)a[1]; /* FPU_CFGS x 6 */
    uint32_t l = lin(), n = nthreads();
    int bad = 0;
    uint32_t cfg = (l * 5u + ocu_cpu_block_idx().x) % FPU_CFGS;
    apply_cfg(cfg);
    for (int r = 0; r < 6; ++r) {
        if (r % 2 == 0) ocu_cpu_barrier();
        else (void)ocu_cpu_warp_shuffle_masked(l, 0x9900, 0, 0);
        bad += cfg_readback_bad(cfg);
    }
    ocu_cpu_barrier();
    uint64_t sig[6];
    signature(sig);
    for (int i = 0; i < 6; ++i) bad += sig[i] != table[cfg * 6u + (uint32_t)i];
    out[ocu_cpu_block_idx().x * n + l] = bad;
}
/* a thread that ends in an odd state, followed by threads that must not see it */
static void k_fpu_dirty(void **a)
{
    int *out = *(int **)a[0];
    uint32_t l = lin();
    int bad = 0;
    bad += fegetround() != FE_TONEAREST;
    bad += (_mm_getcsr() & 0xE040u) != 0;
    if (l % 2 == 0) {
        fesetround(FE_DOWNWARD);
        _mm_setcsr(_mm_getcsr() | 0x8040u);
    }
    out[ocu_cpu_block_idx().x * nthreads() + l] = bad;
}
#endif

/* ---- statistics child ---------------------------------------------------- */
static void k_stat_a(void **a)
{
    (void)a;
    ocu_cpu_barrier();
}
static void k_stat_b(void **a)
{
    (void)a;
    ocu_cpu_warp_sync();
}

static const ocu_cpu_kernel_entry g_entries[] = {
    {"chain", k_chain},
    {"halves_skew", k_halves_skew},
    {"edge", k_edge},
    {"partial", k_partial},
    {"lane", k_lane},
    {"tag_mismatch", k_tag_mismatch},
    {"mask_mismatch", k_mask_mismatch},
    {"extra_call", k_extra_call},
    {"named_lane_gone", k_named_lane_gone},
    {"op_then_return", k_op_then_return},
    {"half_then_return", k_half_then_return},
    {"second_op_names_gone", k_second_op_names_gone},
    {"start", k_start},
    {"mixed", k_mixed},
    {"frame", k_frame},
    {"locals", k_locals},
    {"overflow", k_overflow},
    {"ok", k_ok},
    {"fill", k_fill},
#if defined(__x86_64__)
    {"fpu", k_fpu},
    {"fpu_dirty", k_fpu_dirty},
#endif
    {"stat_alpha_kernel", k_stat_a},
    {"stat_beta_kernel", k_stat_b},
};
static const ocu_cpu_image g_image = {OCU_CPU_IMAGE_MAGIC, OCU_CPU_IMAGE_VERSION,
                                      (uint32_t)(sizeof(g_entries) / sizeof(g_entries[0])),
                                      g_entries};

/* ---- host helpers --------------------------------------------------------- */
static ocu_kernel *kernel(const char *name)
{
    ocu_kernel *k = NULL;
    if (B->module_get_kernel(M, name, &k) != OCU_OK) {
        fprintf(stderr, "no kernel %s\n", name);
        exit(2);
    }
    return k;
}

static ocu_status run(const char *name, uint32_t grid, uint32_t bx, uint32_t by, uint32_t bz,
                      size_t shared, void **args, const size_t *sizes, uint32_t nargs)
{
    ocu_launch_params p;
    memset(&p, 0, sizeof(p));
    p.grid[0] = grid; p.grid[1] = 1; p.grid[2] = 1;
    p.block[0] = bx; p.block[1] = by; p.block[2] = bz;
    p.shared_memory_bytes = shared;
    p.args = args; p.arg_sizes = sizes; p.num_args = nargs;
    return B->kernel_launch(NULL, kernel(name), &p);
}

static void *dalloc(size_t bytes)
{
    void *p = NULL;
    if (B->mem_alloc(D, bytes, &p) != OCU_OK) {
        fprintf(stderr, "mem_alloc failed\n");
        exit(2);
    }
    memset(p, 0xff, bytes);
    return p;
}

static int count_nonzero(const int *v, size_t n)
{
    int bad = 0;
    for (size_t i = 0; i < n; ++i) bad += v[i] != 0;
    return bad;
}

/* Block sizes: a thread, small, one warp, short last warps, exact multiples. */
static const uint32_t g_sizes[] = {1, 2, 7, 31, 32, 33, 40, 63, 64, 65, 96, 130, 257, 1024};
#define NSIZES ((int)(sizeof(g_sizes) / sizeof(g_sizes[0])))

/* ---- item 1 -------------------------------------------------------------- */
static void test_chain(void)
{
    g_test = "item1_consecutive_warp_ops";
    for (int s = 0; s < NSIZES; ++s) {
        for (uint32_t be = 0; be < 3; ++be) { /* 0: none, 1: every round, 2: every 2nd */
            uint32_t n = g_sizes[s], blocks = 3, rounds = 60;
            int *out = dalloc(blocks * n * sizeof(int));
            void *args[] = {&out, &rounds, &be};
            size_t sz[] = {sizeof(out), 4, 4};
            ocu_status st = run("chain", blocks, n, 1, 1, 0, args, sz, 3);
            CHECK_EQ(st, OCU_OK);
            int bad = count_nonzero(out, blocks * n);
            if (bad != 0) fprintf(stderr, "  chain n=%u barrier_every=%u: %d bad threads\n", n, be, bad);
            CHECK_EQ(bad, 0);
            B->mem_free(D, out);
        }
    }
    /* 3D block, and a long chain (the sequence numbers run past any small counter) */
    {
        uint32_t rounds = 1000, be = 0;
        int *out = dalloc(2 * 5 * 3 * 4 * sizeof(int));
        void *args[] = {&out, &rounds, &be};
        size_t sz[] = {sizeof(out), 4, 4};
        CHECK_EQ(run("chain", 2, 5, 3, 4, 0, args, sz, 3), OCU_OK);
        CHECK_EQ(count_nonzero(out, 2 * 60), 0);
        B->mem_free(D, out);
    }
}

static void test_halves_skew(void)
{
    g_test = "item1_disjoint_masks_different_counts";
    const uint32_t sizes[] = {32, 64, 128, 96};
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
        uint32_t n = sizes[s];
        int *out = dalloc(2 * n * sizeof(int));
        void *args[] = {&out};
        size_t sz[] = {sizeof(out)};
        CHECK_EQ(run("halves_skew", 2, n, 1, 1, 0, args, sz, 1), OCU_OK);
        CHECK_EQ(count_nonzero(out, 2 * n), 0);
        B->mem_free(D, out);
    }
}

/* ---- item 5 -------------------------------------------------------------- */
static void test_edges(void)
{
    g_test = "item5_masked_shuffle_vote_edges";
    for (int s = 0; s < NSIZES; ++s) {
        for (uint32_t mk = 0; mk < 2; ++mk) {
            uint32_t n = g_sizes[s];
            if (mk == 1 && W == 64 && n > 32) continue; /* a 32-bit mask names one 32-lane group only */
            int *out = dalloc(2 * n * sizeof(int));
            void *args[] = {&out, &mk};
            size_t sz[] = {sizeof(out), 4};
            CHECK_EQ(run("edge", 2, n, 1, 1, 0, args, sz, 2), OCU_OK);
            int bad = count_nonzero(out, 2 * n);
            if (bad) fprintf(stderr, "  edge n=%u mask_kind=%u: %d bad\n", n, mk, bad);
            CHECK_EQ(bad, 0);
            B->mem_free(D, out);
        }
    }
    /* partial masks need whole 32-lane groups */
    const uint32_t full[] = {32, 64, 96, 128, 256};
    for (size_t s = 0; s < sizeof(full) / sizeof(full[0]); ++s) {
        for (uint32_t mode = 0; mode < 3; ++mode) {
            uint32_t n = full[s];
            int *out = dalloc(2 * n * sizeof(int));
            void *args[] = {&out, &mode};
            size_t sz[] = {sizeof(out), 4};
            CHECK_EQ(run("partial", 2, n, 1, 1, 0, args, sz, 2), OCU_OK);
            int bad = count_nonzero(out, 2 * n);
            if (bad) fprintf(stderr, "  partial n=%u mode=%u: %d bad\n", n, mode, bad);
            CHECK_EQ(bad, 0);
            B->mem_free(D, out);
        }
    }
}

/* ---- item 6 -------------------------------------------------------------- */
static void *outside_thread(void *arg)
{
    uint32_t *r = arg;
    r[0] = ocu_cpu_warp_size();
    return NULL;
}

static void test_lane_and_warp_size(void)
{
    g_test = "item6_lane_id_warp_size";
    const char *env = getenv("OCU_CPU_WARP");
    uint32_t want = env ? (uint32_t)strtoul(env, NULL, 10) : 32u;
    CHECK_EQ(W, want);
    CHECK_EQ(ocu_cpu_warp_size(), want);        /* outside a kernel: the setting */
    for (int s = 0; s < NSIZES; ++s) {
        uint32_t n = g_sizes[s];
        int *out = dalloc(2 * n * sizeof(int));
        void *args[] = {&out, &want};
        size_t sz[] = {sizeof(out), 4};
        CHECK_EQ(run("lane", 2, n, 1, 1, 0, args, sz, 2), OCU_OK);
        CHECK_EQ(count_nonzero(out, 2 * n), 0);
        B->mem_free(D, out);
        /* and again outside: nothing stale from the last fiber */
        CHECK_EQ(ocu_cpu_warp_size(), want);
    }
    /* 3D block: the lane follows the linear index (x fastest) */
    {
        int *out = dalloc(2 * 8 * 5 * 3 * sizeof(int));
        void *args[] = {&out, &want};
        size_t sz[] = {sizeof(out), 4};
        CHECK_EQ(run("lane", 2, 8, 5, 3, 0, args, sz, 2), OCU_OK);
        CHECK_EQ(count_nonzero(out, 2 * 120), 0);
        B->mem_free(D, out);
    }
    /* another host thread, outside any kernel */
    uint32_t r[2] = {0xdead, 0xbeef};
    pthread_t t;
    pthread_create(&t, NULL, outside_thread, r);
    pthread_join(t, NULL);
    CHECK_EQ(r[0], want);
}

/* ---- item 2 -------------------------------------------------------------- */
static void test_mismatch(void)
{
    g_test = "item2_mismatch_detection";
    const uint32_t sizes[] = {2, 32, 40, 64, 130};
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
        uint32_t n = sizes[si];
        for (uint32_t kind = 0; kind < 4; ++kind) {
            const uint32_t odds[] = {0, 1, 30, 31, 63};
            for (size_t oi = 0; oi < sizeof(odds) / sizeof(odds[0]); ++oi) {
                uint32_t odd = odds[oi];
                if (odd >= n || odd >= W) continue;
                if (n <= W && odd == 0 && n == 1) continue;
                for (uint32_t round = 0; round < 4; round += 3) {
                    void *args[] = {&kind, &odd, &round};
                    size_t sz[] = {4, 4, 4};
                    ocu_status st = run("tag_mismatch", 1, n, 1, 1, 0, args, sz, 3);
                    if (st != OCU_ERR_LAUNCH_FAILED)
                        fprintf(stderr, "  tag_mismatch n=%u kind=%u odd=%u round=%u: status %d\n",
                                n, kind, odd, round, (int)st);
                    CHECK_EQ(st, OCU_ERR_LAUNCH_FAILED);
                }
            }
        }
    }
    /* a lane alone in its warp has nobody to disagree with */
    {
        uint32_t kind = 0, odd = 0, round = 1;
        void *args[] = {&kind, &odd, &round};
        size_t sz[] = {4, 4, 4};
        CHECK_EQ(run("tag_mismatch", 1, 1, 1, 1, 0, args, sz, 3), OCU_OK);
    }
    /* mask mismatch with the same tag */
    {
        void *args[] = {NULL};
        size_t sz[] = {0};
        CHECK_EQ(run("mask_mismatch", 1, 32, 1, 1, 0, args, sz, 0), OCU_ERR_LAUNCH_FAILED);
        CHECK_EQ(run("mask_mismatch", 2, 64, 1, 1, 0, args, sz, 0), OCU_ERR_LAUNCH_FAILED);
    }
    /* a lane skips an occurrence and then waits at a block barrier: the others
       end up at a warp operation, it at the barrier */
    {
        uint32_t odd = 2;
        void *args[] = {&odd};
        size_t sz[] = {4};
        CHECK_EQ(run("extra_call", 1, 8, 1, 1, 0, args, sz, 1), OCU_ERR_LAUNCH_FAILED);
        odd = 31;
        CHECK_EQ(run("extra_call", 1, 32, 1, 1, 0, args, sz, 1), OCU_ERR_LAUNCH_FAILED);
    }
    /* the backend keeps working after each failure */
    {
        uint32_t rounds = 20, be = 0, n = 70;
        int *out = dalloc(n * sizeof(int));
        void *args[] = {&out, &rounds, &be};
        size_t sz[] = {sizeof(out), 4, 4};
        CHECK_EQ(run("chain", 1, n, 1, 1, 0, args, sz, 3), OCU_OK);
        CHECK_EQ(count_nonzero(out, n), 0);
        B->mem_free(D, out);
    }
}

static void test_strict_mask0(void)
{
    g_test = "item2_strict_mask0_names_whole_warp";
    const uint32_t sizes[] = {2, 32, 64, 70};
    for (size_t si = 0; si < 4; ++si) {
        uint32_t n = sizes[si], mask = 0;
        const uint32_t gones[] = {0, 1, 31};
        for (size_t gi = 0; gi < 3; ++gi) {
            uint32_t gone = gones[gi];
            if (gone >= n) continue;
            int *out = dalloc(2 * n * sizeof(int));
            void *args[] = {&out, &gone, &mask};
            size_t sz[] = {sizeof(out), 4, 4};
            ocu_status st = run("named_lane_gone", 2, n, 1, 1, 0, args, sz, 3);
            if (st != OCU_ERR_LAUNCH_FAILED)
                fprintf(stderr, "  strict, mask 0, n=%u, lane %u returned first: status %d (want %d)\n",
                        n, gone, (int)st, (int)OCU_ERR_LAUNCH_FAILED);
            CHECK_EQ(st, OCU_ERR_LAUNCH_FAILED);
            B->mem_free(D, out);
        }
    }
}

static void test_strict(void)
{
    g_test = g_strict ? "item2_strict_named_lane_finished" : "item2_default_named_lane_finished";
    const ocu_status named_gone = g_strict ? OCU_ERR_LAUNCH_FAILED : OCU_OK;
    /* a named lane returned before the whole-warp operation */
    const uint32_t sizes[] = {2, 32, 64, 70};
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
        uint32_t n = sizes[si];
        const uint32_t masks[] = {0xffffffffu, 0};
        /* Mask 0 names the whole warp; in strict mode it should be an error
           too, but it is not (see test_strict_mask0, expected to fail). */
        for (int mi = 0; mi < (g_strict ? 1 : 2); ++mi) {
        uint32_t mask = masks[mi];
        const uint32_t gones[] = {0, 1, 31};
        for (size_t gi = 0; gi < sizeof(gones) / sizeof(gones[0]); ++gi) {
            uint32_t gone = gones[gi];
            if (gone >= n) continue;
            int *out = dalloc(2 * n * sizeof(int));
            void *args[] = {&out, &gone, &mask};
            size_t sz[] = {sizeof(out), 4, 4};
            ocu_status st = run("named_lane_gone", 2, n, 1, 1, 0, args, sz, 3);
            if (st != named_gone) fprintf(stderr, "  named_lane_gone mask=0x%x n=%u gone=%u: status %d\n", mask, n, gone, (int)st);
            CHECK_EQ(st, named_gone);
            if (st == OCU_OK) CHECK_EQ(count_nonzero(out, 2 * n), 0);
            B->mem_free(D, out);
        }
        }
    }
    /* lanes that finish the operation and return at once are fine in both modes */
    for (int si = 0; si < NSIZES; ++si) {
        uint32_t n = g_sizes[si];
        for (uint32_t kind = 0; kind < 4; ++kind) {
            int *out = dalloc(2 * n * sizeof(int));
            void *args[] = {&out, &kind};
            size_t sz[] = {sizeof(out), 4};
            ocu_status st = run("op_then_return", 2, n, 1, 1, 0, args, sz, 2);
            if (st != OCU_OK) fprintf(stderr, "  op_then_return n=%u kind=%u: status %d\n", n, kind, (int)st);
            CHECK_EQ(st, OCU_OK);
            CHECK_EQ(count_nonzero(out, 2 * n), 0);
            B->mem_free(D, out);
        }
    }
    /* disjoint half masks, each half returns right after its own operation */
    {
        const uint32_t sizes2[] = {32, 64, 128};
        for (size_t si = 0; si < 3; ++si) {
            uint32_t n = sizes2[si];
            int *out = dalloc(2 * n * sizeof(int));
            void *args[] = {&out};
            size_t sz[] = {sizeof(out)};
            ocu_status st = run("half_then_return", 2, n, 1, 1, 0, args, sz, 1);
            if (st != OCU_OK) fprintf(stderr, "  half_then_return n=%u: status %d\n", n, (int)st);
            CHECK_EQ(st, OCU_OK);
            CHECK_EQ(count_nonzero(out, 2 * n), 0);
            B->mem_free(D, out);
        }
    }
    /* the second operation names lanes that have returned: strict fails */
    {
        uint32_t use_full;
        void *args[] = {&use_full};
        size_t sz[] = {4};
        use_full = 0;
        CHECK_EQ(run("second_op_names_gone", 1, 32, 1, 1, 0, args, sz, 1), OCU_OK);
        CHECK_EQ(run("second_op_names_gone", 1, 64, 1, 1, 0, args, sz, 1), OCU_OK);
        use_full = 1;
        CHECK_EQ(run("second_op_names_gone", 1, 32, 1, 1, 0, args, sz, 1), named_gone);
        CHECK_EQ(run("second_op_names_gone", 1, 64, 1, 1, 0, args, sz, 1), named_gone);
    }
    /* After any failure the next launch of the same shape works. */
    {
        uint32_t rounds = 10, be = 1, n = 64;
        int *out = dalloc(n * sizeof(int));
        void *args[] = {&out, &rounds, &be};
        size_t sz[] = {sizeof(out), 4, 4};
        CHECK_EQ(run("chain", 1, n, 1, 1, 0, args, sz, 3), OCU_OK);
        CHECK_EQ(count_nonzero(out, n), 0);
        B->mem_free(D, out);
    }
}

/* ---- item 3 -------------------------------------------------------------- */
static void check_start(uint32_t blocks, uint32_t bx, uint32_t by, uint32_t bz, uint32_t variant)
{
    uint32_t n = bx * by * bz;
    uint32_t *ticket = dalloc(blocks * n * sizeof(uint32_t));
    uint32_t *ctr = dalloc(blocks * sizeof(uint32_t));
    memset(ctr, 0, blocks * sizeof(uint32_t));
    void *args[] = {&ticket, &ctr, &variant};
    size_t sz[] = {sizeof(ticket), sizeof(ctr), 4};
    ocu_status st = run("start", blocks, bx, by, bz, 0, args, sz, 3);
    if (g_strict && variant == 3 && n >= 2) { /* threads end before the barrier: an error here */
        CHECK_EQ(st, OCU_ERR_LAUNCH_FAILED);
        B->mem_free(D, ticket);
        B->mem_free(D, ctr);
        return;
    }
    CHECK_EQ(st, OCU_OK);
    int not_identity = 0, not_perm = 0;
    for (uint32_t b = 0; b < blocks; ++b) {
        char *seen = calloc(n, 1);
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t t = ticket[b * n + i];
            not_identity += t != i;
            if (t >= n || seen[t]) ++not_perm;
            else seen[t] = 1;
        }
        free(seen);
    }
    CHECK_EQ(not_perm, 0); /* every order: tickets 0..n-1 each once */
    if (!g_random) {
        if (not_identity)
            fprintf(stderr, "  start order not 0,1,2.. : block %ux%ux%u variant %u: %d threads off\n",
                    bx, by, bz, variant, not_identity);
        CHECK_EQ(not_identity, 0);
    }
    B->mem_free(D, ticket);
    B->mem_free(D, ctr);
}

static void test_start_order(void)
{
    g_test = g_random ? "item3_start_permutation_random" : "item3_start_order_fixed";
    for (uint32_t variant = 0; variant < 6; ++variant) {
        for (int s = 0; s < NSIZES; ++s) check_start(3, g_sizes[s], 1, 1, variant);
        check_start(2, 5, 3, 4, variant);
    }
}

static void test_mixed(void)
{
    g_test = "item3_block_sync_and_warp_ops_mixed";
    const uint32_t sizes[] = {1, 5, 32, 33, 64, 100, 130, 257, 512};
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
        uint32_t n = sizes[si], blocks = 4;
        uint32_t *in = dalloc(blocks * n * sizeof(uint32_t));
        uint64_t *out = dalloc(blocks * n * sizeof(uint64_t));
        for (uint32_t i = 0; i < blocks * n; ++i) in[i] = (uint32_t)(H(i, 9) & 0xffffu);
        void *args[] = {&in, &out};
        size_t sz[] = {sizeof(in), sizeof(out)};
        size_t shared = ((n + W - 1) / W) * sizeof(uint64_t);
        CHECK_EQ(run("mixed", blocks, n, 1, 1, shared, args, sz, 2), OCU_OK);
        int bad = 0;
        for (uint32_t b = 0; b < blocks; ++b) {
            uint64_t sum = 0;
            for (uint32_t i = 0; i < n; ++i) sum += in[b * n + i];
            uint64_t want = 3 * sum + (0 + 1 + 2);
            for (uint32_t i = 0; i < n; ++i) bad += out[b * n + i] != want;
        }
        if (bad) fprintf(stderr, "  mixed n=%u: %d threads differ from the host sum\n", n, bad);
        CHECK_EQ(bad, 0);
        B->mem_free(D, in);
        B->mem_free(D, out);
    }
}

/* ---- item 4 -------------------------------------------------------------- */
static int distinct_u(const uintptr_t *v, size_t n)
{
    int d = 0;
    for (size_t i = 0; i < n; ++i) {
        int dup = 0;
        for (size_t j = 0; j < i && !dup; ++j) dup = v[i] == v[j];
        d += !dup;
    }
    return d;
}

static void test_stacks(void)
{
    g_test = "item4_stack_reuse_tops_overflow";
    /* no waiting: one stack serves every thread (fixed order, plain build) */
    {
        uint32_t n = 200, wait = 0;
        uintptr_t *out = dalloc(n * sizeof(uintptr_t));
        void *args[] = {&out, &wait};
        size_t sz[] = {sizeof(out), 4};
        CHECK_EQ(run("frame", 1, n, 1, 1, 0, args, sz, 2), OCU_OK);
        int d = distinct_u(out, n);
        if (!g_random && !g_sanitized) CHECK_EQ(d, 1);
        else CHECK(d >= 1 && d <= (int)n);
        B->mem_free(D, out);
    }
    /* waiting: each thread keeps its own stack; tops differ by 64-byte steps */
    {
        uint32_t n = 128, wait = 1;
        uintptr_t *out = dalloc(n * sizeof(uintptr_t));
        void *args[] = {&out, &wait};
        size_t sz[] = {sizeof(out), 4};
        CHECK_EQ(run("frame", 1, n, 1, 1, 0, args, sz, 2), OCU_OK);
        CHECK_EQ(distinct_u(out, n), n);
        int off_bad = 0;
        uintptr_t res[128];
        for (uint32_t i = 0; i < n; ++i) {
            off_bad += (out[i] % 16) != (out[0] % 16); /* same code, same frame offset */
            res[i] = out[i] % 4096;
        }
        CHECK_EQ(off_bad, 0);
        /* tops are staggered: the page offsets of the 128 stacks are not all one value */
        int dres = distinct_u(res, n);
        if (!g_sanitized) {
            if (dres < 4) fprintf(stderr, "  only %d distinct page offsets among %u stacks\n", dres, n);
            CHECK(dres >= 4);
            /* frames of different threads differ in multiples of 64 bytes */
            int step_bad = 0;
            for (uint32_t i = 1; i < n; ++i) step_bad += ((out[i] ^ out[0]) & 63u) != 0;
            CHECK_EQ(step_bad, 0);
        }
        B->mem_free(D, out);
    }
    /* locals survive across a barrier next to threads that never wait */
    for (int s = 0; s < NSIZES; ++s) {
        uint32_t n = g_sizes[s];
        int *out = dalloc(2 * n * sizeof(int));
        uint32_t uniform = g_strict; /* strict mode: threads ending before a barrier are errors */
        void *args[] = {&out, &uniform};
        size_t sz[] = {sizeof(out), 4};
        CHECK_EQ(run("locals", 2, n, 1, 1, 0, args, sz, 2), OCU_OK);
        CHECK_EQ(count_nonzero(out, 2 * n), 0);
        B->mem_free(D, out);
    }
    /* overflow of the configured stack: reported unless the block has 1 thread;
       the next launch of the same shape is correct */
    {
        size_t over = g_stack_kb * 1024u + 8192u;
        const uint32_t sizes[] = {2, 3, 32, 64};
        for (size_t si = 0; si < 4; ++si) {
            uint32_t n = sizes[si];
            void *args[] = {&over};
            size_t sz[] = {sizeof(over)};
            ocu_status st = run("overflow", 1, n, 1, 1, 0, args, sz, 1);
            if (st != OCU_ERR_LAUNCH_FAILED) fprintf(stderr, "  overflow n=%u: status %d\n", n, (int)st);
            CHECK_EQ(st, OCU_ERR_LAUNCH_FAILED);
            int *out = dalloc(n * sizeof(int));
            void *args2[] = {&out};
            size_t sz2[] = {sizeof(out)};
            CHECK_EQ(run("ok", 1, n, 1, 1, 0, args2, sz2, 1), OCU_OK);
            int bad = 0;
            for (uint32_t i = 0; i < n; ++i) bad += out[i] != (int)i + 1;
            CHECK_EQ(bad, 0);
            /* the very same shape and kernel again, without the overflow */
            size_t small = 1024;
            void *args3[] = {&small};
            CHECK_EQ(run("overflow", 1, n, 1, 1, 0, args3, sz, 1), OCU_OK);
            CHECK_EQ(run("overflow", 1, n, 1, 1, 0, args3, sz, 1), OCU_OK);
            B->mem_free(D, out);
        }
    }
}

/* ---- item 7: spin / idle / fork ----------------------------------------- */
static double cpu_seconds(void)
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double)ru.ru_utime.tv_sec + (double)ru.ru_stime.tv_sec +
           1e-6 * ((double)ru.ru_utime.tv_usec + (double)ru.ru_stime.tv_usec);
}
static void sleep_ms(long ms)
{
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
    while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {}
}

static int fill_ok(uint32_t blocks, uint32_t bx)
{
    uint32_t *out = dalloc(blocks * bx * sizeof(uint32_t));
    void *args[] = {&out};
    size_t sz[] = {sizeof(out)};
    int bad = run("fill", blocks, bx, 1, 1, 0, args, sz, 1) != OCU_OK;
    for (uint32_t b = 0; b < blocks && !bad; ++b)
        for (uint32_t t = 0; t < bx; ++t) bad += out[b * bx + t] != b * 1000u + t;
    B->mem_free(D, out);
    return bad;
}

static void test_spin_and_idle(void)
{
    g_test = "item7_spin_results_and_idle_cpu";
    int bad = 0;
    for (int i = 0; i < 300; ++i) bad += fill_ok(1 + (uint32_t)(i % 9), 32 + (uint32_t)(i % 5) * 32);
    CHECK_EQ(bad, 0);
    /* the chain kernel gives the same answers whatever the spin setting */
    uint32_t rounds = 30, be = 1, n = 96;
    int *out = dalloc(8 * n * sizeof(int));
    void *args[] = {&out, &rounds, &be};
    size_t sz[] = {sizeof(out), 4, 4};
    for (int i = 0; i < 50; ++i) {
        CHECK_EQ(run("chain", 8, n, 1, 1, 0, args, sz, 3), OCU_OK);
        if (count_nonzero(out, 8 * n)) { CHECK(0); break; }
    }
    B->mem_free(D, out);
    /* after the launches the process goes quiet */
    sleep_ms(300);
    double t0 = cpu_seconds();
    sleep_ms(700);
    double used = cpu_seconds() - t0;
    fprintf(stderr, "  idle: %.1f ms CPU in 700 ms\n", used * 1000.0);
    CHECK(used < 0.035);
    /* and it still works after being idle */
    CHECK_EQ(fill_ok(7, 64), 0);
    /* a gap between launches shorter than the watch time and one longer */
    for (int i = 0; i < 20; ++i) {
        CHECK_EQ(fill_ok(4, 64), 0);
        sleep_ms(i % 2 ? 1 : 20);
    }
}

#if !defined(OCU_T_TSAN)
static void test_fork_launch(void)
{
    g_test = "item7_fork_then_launch_in_child";
    for (int round = 0; round < 24; ++round) {
        CHECK_EQ(fill_ok(8, 64), 0);        /* pool threads exist and may be watching */
        if (round % 3 == 1) sleep_ms(1);
        if (round % 3 == 2) sleep_ms(30);
        pid_t pid = fork();
        if (pid == 0) {
            alarm(40);
            int bad = 0;
            for (int i = 0; i < 5; ++i) bad += fill_ok(1 + (uint32_t)i, 96);
            uint32_t rounds = 20, be = 2, n = 70;
            int *out = dalloc(3 * n * sizeof(int));
            void *args[] = {&out, &rounds, &be};
            size_t sz[] = {sizeof(out), 4, 4};
            bad += run("chain", 3, n, 1, 1, 0, args, sz, 3) != OCU_OK;
            bad += count_nonzero(out, 3 * n);
            _exit(bad == 0 ? 0 : 1);
        }
        CHECK(pid > 0);
        /* the parent keeps launching while the child runs */
        CHECK_EQ(fill_ok(8, 64), 0);
        int status = 0;
        pid_t got = waitpid(pid, &status, 0);
        CHECK(got == pid);
        if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0))
            fprintf(stderr, "  child %d status 0x%x\n", round, status);
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
    CHECK_EQ(fill_ok(8, 64), 0);
}
#endif

/* ---- item 8: statistics --------------------------------------------------- */
static int run_child(const char *stats, char *err, size_t cap, int *code)
{
    int fds[2];
    if (pipe(fds) != 0) return -1;
    pid_t pid = fork();
    if (pid == 0) {
        close(fds[0]);
        dup2(fds[1], 2);
        close(fds[1]);
        int nul = open("/dev/null", O_WRONLY);
        dup2(nul, 1);
        if (stats) setenv("OCU_CPU_STATS", stats, 1);
        else unsetenv("OCU_CPU_STATS");
        execl(g_self, g_self, "stats_child", (char *)NULL);
        _exit(127);
    }
    close(fds[1]);
    size_t len = 0;
    ssize_t r;
    while ((r = read(fds[0], err + len, cap - 1 - len)) > 0) len += (size_t)r;
    err[len] = 0;
    close(fds[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    *code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    return 0;
}

static int has_word(const char *line, const char *word)
{
    size_t wl = strlen(word);
    for (const char *p = line; (p = strstr(p, word)) != NULL; ++p) {
        int pre = p == line || (!(p[-1] >= '0' && p[-1] <= '9') && !(p[-1] >= 'a' && p[-1] <= 'z'));
        int post = !(p[wl] >= '0' && p[wl] <= '9') && !(p[wl] >= 'a' && p[wl] <= 'z');
        if (pre && post) return 1;
    }
    return 0;
}

static void test_stats(void)
{
    g_test = "item8_stats_output";
    static char err[65536];
    int code = 0;
    const char *offs[] = {NULL, "0", ""};
    for (size_t i = 0; i < 3; ++i) {
        CHECK_EQ(run_child(offs[i], err, sizeof(err), &code), 0);
        CHECK_EQ(code, 0);
        if (err[0]) fprintf(stderr, "  stats off (%s) printed: [%.200s]\n", offs[i] ? offs[i] : "unset", err);
        CHECK_EQ(strlen(err), 0);
    }
    CHECK_EQ(run_child("1", err, sizeof(err), &code), 0);
    CHECK_EQ(code, 0);
    fprintf(stderr, "---- stats sample (OCU_CPU_STATS=1) ----\n%.1500s---- end sample ----\n", err);
    CHECK(strlen(err) > 0);
    /* The table names its rows by an offset, not by the kernel name: find the
       rows by tpb / launches / blocks (alpha: 32 threads, 3 launches, 21
       blocks; beta: 32 threads, 2 launches, 10 blocks). */
    char *copy = strdup(err);
    int alpha_ok = 0, beta_ok = 0, lines = 0, prefixed = 0;
    for (char *ln = strtok(copy, "\n"); ln; ln = strtok(NULL, "\n")) {
        ++lines;
        prefixed += strncmp(ln, "ocu cpu stats:", 14) == 0;
        if (has_word(ln, "32") && has_word(ln, "3") && has_word(ln, "21")) alpha_ok = 1;
        if (has_word(ln, "32") && has_word(ln, "2") && has_word(ln, "10")) beta_ok = 1;
    }
    free(copy);
    CHECK(lines >= 4);        /* a one-line summary, a header and a row per kernel */
    CHECK_EQ(prefixed, lines); /* nothing but statistics lines on stderr */
    CHECK(alpha_ok);
    CHECK(beta_ok);
    CHECK(strstr(err, "5 launches") != NULL);
    CHECK(strstr(err, "31 blocks") != NULL);
}

static int stats_child_main(void)
{
    B = ocu_backend_cpu();
    if (B->dev_open(0, &D) != OCU_OK || B->module_load(D, &g_image, sizeof(g_image), &M) != OCU_OK) return 3;
    void *args[] = {NULL};
    size_t sz[] = {0};
    for (int i = 0; i < 3; ++i) if (run("stat_alpha_kernel", 7, 32, 1, 1, 0, args, sz, 0) != OCU_OK) return 4;
    for (int i = 0; i < 2; ++i) if (run("stat_beta_kernel", 5, 32, 1, 1, 0, args, sz, 0) != OCU_OK) return 5;
    B->module_unload(M);
    return 0;
}

/* ---- item 9: MXCSR / x87 -------------------------------------------------- */
#if defined(__x86_64__)
static void test_fpu(void)
{
    g_test = "item9_mxcsr_x87_per_fiber";
    fenv_t saved;
    fegetenv(&saved);
    uint64_t *table = dalloc(FPU_CFGS * 6 * sizeof(uint64_t));
    for (uint32_t cfg = 0; cfg < FPU_CFGS; ++cfg) {
        fesetenv(&saved);
        apply_cfg(cfg);
        signature(&table[cfg * 6u]);
    }
    fesetenv(&saved);
    /* the table must actually tell configurations apart, or the check is empty */
    int distinct = 0;
    for (uint32_t a = 0; a < FPU_CFGS; ++a) {
        int dup = 0;
        for (uint32_t b = 0; b < a && !dup; ++b) dup = memcmp(&table[a * 6u], &table[b * 6u], 48) == 0;
        distinct += !dup;
    }
    CHECK(distinct >= 8);
    const uint32_t sizes[] = {1, 2, 7, 24, 48, 100, 257};
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
        uint32_t n = sizes[si];
        int *out = dalloc(4 * n * sizeof(int));
        void *args[] = {&out, &table};
        size_t sz[] = {sizeof(out), sizeof(table)};
        CHECK_EQ(run("fpu", 4, n, 1, 1, 0, args, sz, 2), OCU_OK);
        int bad = count_nonzero(out, 4 * n);
        if (bad) fprintf(stderr, "  fpu n=%u: %d threads saw a foreign control state\n", n, bad);
        CHECK_EQ(bad, 0);
        B->mem_free(D, out);
        /* the launching thread is back in its own state */
        CHECK_EQ(fegetround(), FE_TONEAREST);
        CHECK_EQ(_mm_getcsr() & 0xE040u, 0);
    }
    B->mem_free(D, table);
}

static void test_fpu_dirty(void)
{
    g_test = "item9_fpu_state_does_not_reach_later_threads";
    /* threads that end dirty do not hand their state to the next thread */
    for (int s = 0; s < NSIZES; ++s) {
        uint32_t n = g_sizes[s];
        int *out = dalloc(3 * n * sizeof(int));
        void *args[] = {&out};
        size_t sz[] = {sizeof(out)};
        CHECK_EQ(run("fpu_dirty", 3, n, 1, 1, 0, args, sz, 1), OCU_OK);
        int bad = count_nonzero(out, 3 * n);
        if (bad) fprintf(stderr, "  fpu_dirty n=%u: %d threads started in a dirty state\n", n, bad);
        CHECK_EQ(bad, 0);
        B->mem_free(D, out);
        CHECK_EQ(fegetround(), FE_TONEAREST);
        CHECK_EQ(_mm_getcsr() & 0xE040u, 0);
    }
}

#endif

#define RUN(f) do { if (getenv("OCU_T_VERBOSE")) fprintf(stderr, "-- %s\n", #f); f(); } while (0)

int main(int argc, char **argv)
{
    g_self = argv[0];
    if (argc > 1 && strcmp(argv[1], "stats_child") == 0) return stats_child_main();
#if defined(__x86_64__)
    if (argc > 1 && strcmp(argv[1], "fpu_dirty") == 0) {
        B = ocu_backend_cpu();
        if (B->dev_open(0, &D) != OCU_OK || B->module_load(D, &g_image, sizeof(g_image), &M) != OCU_OK) return 2;
        ocu_dev_info inf;
        B->dev_info(0, &inf);
        W = inf.warp_size;
        test_fpu_dirty();
        return g_failures == 0 ? 0 : 1;
    }
#endif
    if (argc > 1 && strcmp(argv[1], "strict_mask0") == 0) {
        B = ocu_backend_cpu();
        if (B->dev_open(0, &D) != OCU_OK || B->module_load(D, &g_image, sizeof(g_image), &M) != OCU_OK) return 2;
        test_strict_mask0();
        return g_failures == 0 ? 0 : 1;
    }
    if (argc > 1 && strcmp(argv[1], "lane_outside") == 0) {
        /* card item 6: the lane id is right outside a kernel too (thread 0: lane 0) */
        B = ocu_backend_cpu();
        pid_t pid = fork();
        if (pid == 0) _exit(ocu_cpu_lane_id() == 0 ? 0 : 1);
        int st = 0;
        waitpid(pid, &st, 0);
        return WIFEXITED(st) && WEXITSTATUS(st) == 0 ? 0 : 1;
    }
    const char *e = getenv("OCU_CPU_STRICT");
    g_strict = e && strcmp(e, "1") == 0;
    e = getenv("OCU_CPU_ORDER");
    g_random = e && strcmp(e, "random") == 0;
    e = getenv("OCU_CPU_STACK_KB");
    if (e) g_stack_kb = strtoul(e, NULL, 10);
#if defined(OCU_T_ASAN) || defined(OCU_T_TSAN)
    g_sanitized = 1;
#endif
    B = ocu_backend_cpu();
    if (B->dev_open(0, &D) != OCU_OK || B->module_load(D, &g_image, sizeof(g_image), &M) != OCU_OK) {
        fprintf(stderr, "cannot open the CPU backend\n");
        return 1;
    }
    ocu_dev_info info;
    B->dev_info(0, &info);
    W = info.warp_size;

    RUN(test_chain);
    RUN(test_halves_skew);
    RUN(test_edges);
    RUN(test_lane_and_warp_size);
    RUN(test_mismatch);
    RUN(test_strict);
    RUN(test_start_order);
    RUN(test_mixed);
    RUN(test_stacks);
    RUN(test_spin_and_idle);
#if !defined(OCU_T_TSAN)
    RUN(test_fork_launch);
#endif
    RUN(test_stats);
#if defined(__x86_64__)
    RUN(test_fpu);
#endif
    B->module_unload(M);
    if (g_failures == 0) {
        printf("test_cpu_fastpath: ok (%d checks, warp %u, %s order%s)\n", g_checks, W,
               g_random ? "random" : "fixed", g_strict ? ", strict" : "");
        return 0;
    }
    fprintf(stderr, "test_cpu_fastpath: %d of %d checks failed\n", g_failures, g_checks);
    return 1;
}
