/*
 * amd_meta.c -- kernel parameter sizes from an amdgcn code object.
 *
 * The code object (ELF64) carries a note of type NT_AMDGPU_METADATA (32,
 * owner "AMDGPU") whose descriptor is a MessagePack map, as the LLVM
 * AMDGPU usage documentation describes ("Code Object V3 and later
 * Metadata"): "amdhsa.kernels" is an array of maps, each with ".name" and
 * ".args", an array of maps with ".size" and ".value_kind". Arguments whose
 * value kind starts with "hidden_" are added by the compiler, not declared
 * by the source, and are left out.
 *
 * The reader below handles the MessagePack forms that note uses (maps,
 * arrays, strings, unsigned and signed integers, booleans, nil) and
 * rejects anything else.
 */
#include "amd_hip.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

typedef struct mp {
    const uint8_t *p, *end;
    bool bad;
} mp;

static uint64_t be(mp *m, int n)
{
    if (m->end - m->p < n) {
        m->bad = true;
        return 0;
    }
    uint64_t v = 0;
    for (int i = 0; i < n; ++i) v = (v << 8) | m->p[i];
    m->p += n;
    return v;
}

typedef enum { MP_INT, MP_STR, MP_MAP, MP_ARR, MP_OTHER } mp_kind;

typedef struct mp_val {
    mp_kind        kind;
    uint64_t       n;     /* integer value, string length, element count */
    const uint8_t *str;
} mp_val;

static mp_val next(mp *m)
{
    mp_val v = {MP_OTHER, 0, NULL};
    if (m->p >= m->end) {
        m->bad = true;
        return v;
    }
    uint8_t t = *m->p++;
    if (t <= 0x7f) { v.kind = MP_INT; v.n = t; }
    else if (t >= 0xe0) { v.kind = MP_INT; v.n = 0; } /* negative fixint: not a size */
    else if ((t & 0xf0) == 0x80) { v.kind = MP_MAP; v.n = t & 15u; }
    else if ((t & 0xf0) == 0x90) { v.kind = MP_ARR; v.n = t & 15u; }
    else if ((t & 0xe0) == 0xa0) { v.kind = MP_STR; v.n = t & 31u; }
    else switch (t) {
    case 0xc0: case 0xc2: case 0xc3: break;
    case 0xcc: v.kind = MP_INT; v.n = be(m, 1); break;
    case 0xcd: v.kind = MP_INT; v.n = be(m, 2); break;
    case 0xce: v.kind = MP_INT; v.n = be(m, 4); break;
    case 0xcf: v.kind = MP_INT; v.n = be(m, 8); break;
    case 0xd0: be(m, 1); v.kind = MP_INT; break;
    case 0xd1: be(m, 2); v.kind = MP_INT; break;
    case 0xd2: be(m, 4); v.kind = MP_INT; break;
    case 0xd3: be(m, 8); v.kind = MP_INT; break;
    case 0xd9: v.kind = MP_STR; v.n = be(m, 1); break;
    case 0xda: v.kind = MP_STR; v.n = be(m, 2); break;
    case 0xdb: v.kind = MP_STR; v.n = be(m, 4); break;
    case 0xdc: v.kind = MP_ARR; v.n = be(m, 2); break;
    case 0xdd: v.kind = MP_ARR; v.n = be(m, 4); break;
    case 0xde: v.kind = MP_MAP; v.n = be(m, 2); break;
    case 0xdf: v.kind = MP_MAP; v.n = be(m, 4); break;
    default: m->bad = true; break;
    }
    if (v.kind == MP_STR) {
        if ((uint64_t)(m->end - m->p) < v.n) {
            m->bad = true;
        } else {
            v.str = m->p;
            m->p += v.n;
        }
    }
    return v;
}

/* Skips one complete value (its elements too). */
static void skip(mp *m, int depth)
{
    if (depth > 32) {
        m->bad = true;
        return;
    }
    mp_val v = next(m);
    uint64_t items = v.kind == MP_MAP ? 2 * v.n : v.kind == MP_ARR ? v.n : 0;
    for (uint64_t i = 0; i < items && !m->bad; ++i) skip(m, depth + 1);
}

static bool is(mp_val v, const char *s)
{
    size_t n = strlen(s);
    return v.kind == MP_STR && v.n == n && memcmp(v.str, s, n) == 0;
}

/* One element of ".args": its size, and whether it is a hidden argument. */
static bool read_arg(mp *m, uint64_t *size, bool *hidden)
{
    mp_val map = next(m);
    if (map.kind != MP_MAP) return false;
    *size = 0;
    *hidden = false;
    for (uint64_t i = 0; i < map.n && !m->bad; ++i) {
        mp_val key = next(m);
        if (is(key, ".size")) {
            mp_val v = next(m);
            if (v.kind != MP_INT) return false;
            *size = v.n;
        } else if (is(key, ".value_kind")) {
            mp_val v = next(m);
            if (v.kind != MP_STR) return false;
            *hidden = v.n >= 7 && memcmp(v.str, "hidden_", 7) == 0;
        } else {
            skip(m, 1);
        }
    }
    return !m->bad;
}

/* One element of "amdhsa.kernels": fills sizes when its .name is `kernel`. */
static int read_kernel(mp *m, const char *kernel, size_t *sizes, int max)
{
    mp_val map = next(m);
    if (map.kind != MP_MAP) return -2;
    bool match = false;
    int count = 0;
    size_t tmp[256];
    for (uint64_t i = 0; i < map.n && !m->bad; ++i) {
        mp_val key = next(m);
        if (is(key, ".name")) {
            match = is(next(m), kernel);
        } else if (is(key, ".args")) {
            mp_val arr = next(m);
            if (arr.kind != MP_ARR) return -2;
            for (uint64_t k = 0; k < arr.n && !m->bad; ++k) {
                uint64_t sz = 0;
                bool hidden = false;
                if (!read_arg(m, &sz, &hidden)) return -2;
                if (!hidden && count < 256) tmp[count++] = (size_t)sz;
            }
        } else {
            skip(m, 1);
        }
    }
    if (m->bad) return -2;
    if (!match) return -1;
    for (int i = 0; i < count && i < max; ++i) sizes[i] = tmp[i];
    return count;
}

static int params_in_note(const uint8_t *desc, size_t len, const char *kernel, size_t *sizes, int max)
{
    mp m = {desc, desc + len, false};
    mp_val top = next(&m);
    if (top.kind != MP_MAP) return -1;
    for (uint64_t i = 0; i < top.n && !m.bad; ++i) {
        mp_val key = next(&m);
        if (!is(key, "amdhsa.kernels")) {
            skip(&m, 1);
            continue;
        }
        mp_val arr = next(&m);
        if (arr.kind != MP_ARR) return -1;
        for (uint64_t k = 0; k < arr.n; ++k) {
            int r = read_kernel(&m, kernel, sizes, max);
            if (r >= 0) return r;
            if (r == -2) return -1;
        }
        return -1;
    }
    return -1;
}

static uint64_t le(const uint8_t *p, int n)
{
    uint64_t v = 0;
    for (int i = n - 1; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

int amd_kernel_params(const void *elf, size_t bytes, const char *kernel, size_t *sizes, int max)
{
    const uint8_t *b = elf;
    if (elf == NULL || kernel == NULL || bytes < 64 || memcmp(b, "\x7f" "ELF", 4) != 0 || b[4] != 2) return -1;
    uint64_t shoff = le(b + 0x28, 8);
    uint64_t shentsize = le(b + 0x3a, 2), shnum = le(b + 0x3c, 2);
    if (shentsize < 64 || shoff > bytes || shnum > (bytes - shoff) / shentsize) return -1;
    for (uint64_t s = 0; s < shnum; ++s) {
        const uint8_t *sh = b + shoff + s * shentsize;
        if (le(sh + 4, 4) != 7) continue; /* SHT_NOTE */
        uint64_t off = le(sh + 24, 8), size = le(sh + 32, 8);
        if (off > bytes || size > bytes - off) return -1;
        uint64_t at = off, end = off + size;
        while (at + 12 <= end) {
            uint64_t namesz = le(b + at, 4), descsz = le(b + at + 4, 4), type = le(b + at + 8, 4);
            uint64_t desc = at + 12 + (namesz + 3) / 4 * 4;
            if (desc > end || descsz > end - desc) return -1;
            if (type == 32 && namesz >= 6 && memcmp(b + at + 12, "AMDGPU", 6) == 0) {
                return params_in_note(b + desc, (size_t)descsz, kernel, sizes, max);
            }
            at = desc + (descsz + 3) / 4 * 4;
        }
    }
    return -1;
}
