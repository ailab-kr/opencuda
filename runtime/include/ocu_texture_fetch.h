/*
 * ocu_texture_fetch.h -- texture fetches in device code: tex1Dfetch, tex1D,
 * tex2D, tex3D, tex1DLayered, tex2DLayered and their Lod forms, for texture
 * objects (template<class T> T texNN(cudaTextureObject_t, ...), and the
 * forms that store through a pointer) and, on the CPU path, for legacy
 * texture references (texNN(texture<T, type, mode>, ...)).
 *
 * Written for this project from the public CUDA C++ Programming Guide
 * ("Texture Memory", "Texture Fetching", "Texture Functions") and the
 * Runtime API reference (cudaCreateTextureObject: cudaTextureDesc). Not
 * derived from any vendor header. The sampling is done in software from the
 * descriptor (ocu_texture_desc.h), so it is the same on every path:
 *
 *   point:  T[floor(x)] (per coordinate)
 *   linear: xB = x - 0.5; xB is rounded to 8 fractional bits (the 9-bit
 *           fixed-point weights of the guide, 1.0 exactly representable:
 *           a fraction of 0.99805 becomes 1, as the guide's tex2Dgather note
 *           describes); i = floor(xB), alpha = frac(xB); the weighted sum of
 *           the 2, 4 or 8 neighbours in the guide's order.
 *   normalized coordinates: x is scaled by the size after the wrap or
 *           mirror step; wrap and mirror without normalized coordinates act
 *           as clamp (API reference, cudaTextureDesc::addressMode).
 *   clamp:  indices clamped to [0, N-1] (coordinates to [0, N)).
 *   border: a coordinate outside [0, N) gives the border value (zero for
 *           texture references, cudaTextureDesc::borderColor for objects);
 *           a linear-filter neighbour outside the texture contributes the
 *           border value.
 *   wrap:   x -> frac(x); neighbours wrap around.
 *   mirror: x -> frac(x) when floor(x) is even, 1 - frac(x) when odd;
 *           neighbours outside reflect back (-1 -> 0, N -> N-1).
 *   read mode normalized float: 8- and 16-bit integers map to [0, 1]
 *           (unsigned, v / (2^n - 1)) or [-1, 1] (signed, max(v / (2^(n-1) - 1), -1));
 *           32-bit integers are not promoted. binary16 is promoted to float
 *           before filtering.
 *   linear filtering is done only when the result type has float
 *           components; otherwise the fetch is point sampled.
 *   tex1Dfetch: integer index into linear memory, no filtering, no
 *           addressing; an index outside [0, N) reads zero.
 *
 * Included by the language headers after they define OCU_TEX_FN.
 */
#ifndef OCU_TEXTURE_FETCH_H
#define OCU_TEXTURE_FETCH_H

#ifndef OCU_TEX_FN
#error "define OCU_TEX_FN before including ocu_texture_fetch.h"
#endif
/* Static member functions: the qualifiers of OCU_TEX_FN without `static`. */
#ifndef OCU_TEX_MFN
#define OCU_TEX_MFN inline
#endif

namespace ocu_tex {

/* ---- result types ---------------------------------------------------------- */

/* n: components (0: not a texel type, read as raw bytes); C: component type;
   F: the matching floating-point type (normalized-float read mode). */
template <class T> struct traits { static const int n = 0; typedef unsigned char C; typedef T F; };

#define OCU_TEX_SCALAR(T, Ft)                                                        \
    template <> struct traits<T> {                                                   \
        static const int n = 1; typedef T C; typedef Ft F;                           \
        OCU_TEX_MFN static T make(const C *c) { return c[0]; }                        \
    };
#define OCU_TEX_VEC(V, T, k, Ft)                                                     \
    template <> struct traits<V> {                                                   \
        static const int n = k; typedef T C; typedef Ft F;                           \
        OCU_TEX_MFN static V make(const C *c)                                        \
        {                                                                            \
            V v;                                                                     \
            T *p = &v.x;                                                             \
            for (int i = 0; i < k; ++i) p[i] = c[i];                                 \
            return v;                                                                \
        }                                                                            \
    };
#define OCU_TEX_FAMILY(T, base, fl)                                                  \
    OCU_TEX_SCALAR(T, fl)                                                            \
    OCU_TEX_VEC(base##1, T, 1, float1)                                               \
    OCU_TEX_VEC(base##2, T, 2, float2)                                               \
    OCU_TEX_VEC(base##4, T, 4, float4)

OCU_TEX_SCALAR(char, float)
OCU_TEX_FAMILY(signed char, char, float)
OCU_TEX_FAMILY(unsigned char, uchar, float)
OCU_TEX_FAMILY(short, short, float)
OCU_TEX_FAMILY(unsigned short, ushort, float)
OCU_TEX_FAMILY(int, int, float)
OCU_TEX_FAMILY(unsigned int, uint, float)
OCU_TEX_FAMILY(long, long, float)
OCU_TEX_FAMILY(unsigned long, ulong, float)
OCU_TEX_FAMILY(float, float, float)
#undef OCU_TEX_FAMILY
#undef OCU_TEX_VEC
#undef OCU_TEX_SCALAR

template <class C> struct is_float { static const bool value = false; };
template <> struct is_float<float> { static const bool value = true; };

/* What texNN(texture<T, type, mode>, ...) returns. */
template <class T, int mode> struct ref_result { typedef T type; };
template <class T> struct ref_result<T, cudaReadModeNormalizedFloat> {
    typedef typename traits<T>::F type;
};

/* The sampling state of one fetch. */
struct sampler {
    int address[3];
    int filter;
    int read_norm;
    int normalized;
    const float *border; /* NULL: zero */
};

OCU_TEX_FN sampler of_object(const ocu_tex_desc &d)
{
    sampler s;
    for (int i = 0; i < 3; ++i) s.address[i] = d.address[i];
    s.filter = d.filter;
    s.read_norm = d.read_norm;
    s.normalized = d.normalized;
    s.border = d.border;
    return s;
}

OCU_TEX_FN sampler of_reference(const textureReference &r, int read_mode)
{
    sampler s;
    for (int i = 0; i < 3; ++i) s.address[i] = (int)r.addressMode[i];
    s.filter = (int)r.filterMode;
    s.read_norm = read_mode == cudaReadModeNormalizedFloat;
    s.normalized = r.normalized != 0;
    s.border = 0;
    return s;
}

/* ---- texel decoding --------------------------------------------------------- */

/* The format of the fetch: the descriptor's, or for raw (kind None) data
   the fetch type's own. */
struct format {
    int channels, channel_bytes, kind, texel_bytes, norm;
};

template <class T>
OCU_TEX_FN format format_of(const ocu_tex_desc &d, const sampler &s)
{
    typedef traits<T> tr;
    format f;
    if (d.kind == OCU_TEXK_RAW || d.texel_bytes == 0) {
        f.channels = tr::n;
        f.channel_bytes = tr::n ? (int)sizeof(typename tr::C) : 0;
        f.kind = is_float<typename tr::C>::value ? OCU_TEXK_FLOAT
               : ((typename tr::C)(-1) < (typename tr::C)0) ? OCU_TEXK_SIGNED : OCU_TEXK_UNSIGNED;
        if (tr::n == 0) f.kind = OCU_TEXK_RAW;
        f.texel_bytes = (int)sizeof(T);
    } else {
        f.channels = d.channels;
        f.channel_bytes = d.channel_bytes;
        f.kind = d.kind;
        f.texel_bytes = (int)d.texel_bytes;
    }
    f.norm = (s.read_norm || d.always_norm) && f.channel_bytes <= 2 &&
             (f.kind == OCU_TEXK_SIGNED || f.kind == OCU_TEXK_UNSIGNED);
    return f;
}

OCU_TEX_FN float half_to_float(unsigned short h)
{
    unsigned int sign = (unsigned int)(h >> 15) << 31;
    int e = (h >> 10) & 0x1f;
    unsigned int m = h & 0x3ffu;
    unsigned int bits;
    if (e == 0x1f) {
        bits = sign | 0x7f800000u | (m << 13);
    } else if (e == 0) {
        if (m == 0) {
            bits = sign;
        } else { /* subnormal: normalise */
            int sh = 0;
            while (!(m & 0x400u)) { m <<= 1; ++sh; }
            m &= 0x3ffu;
            bits = sign | ((unsigned int)(127 - 15 + 1 - sh) << 23) | (m << 13);
        }
    } else {
        bits = sign | ((unsigned int)(e - 15 + 127) << 23) | (m << 13);
    }
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

/* Integer value of component c at p (sign- or zero-extended). */
OCU_TEX_FN long long load_int(const unsigned char *p, int bytes, bool is_signed)
{
    if (bytes == 1) return is_signed ? (long long)*(const signed char *)p : (long long)*p;
    if (bytes == 2) {
        unsigned short v;
        __builtin_memcpy(&v, p, 2);
        return is_signed ? (long long)(short)v : (long long)v;
    }
    unsigned int v;
    __builtin_memcpy(&v, p, 4);
    return is_signed ? (long long)(int)v : (long long)v;
}

/* Components of one texel as floats (missing components 0). */
OCU_TEX_FN void decode_float(const unsigned char *p, const format &f, float out[4])
{
    for (int c = 0; c < 4; ++c) out[c] = 0.0f;
    for (int c = 0; c < f.channels && c < 4; ++c) {
        const unsigned char *q = p + c * f.channel_bytes;
        if (f.kind == OCU_TEXK_FLOAT && f.channel_bytes == 4) {
            __builtin_memcpy(&out[c], q, 4);
        } else if (f.kind == OCU_TEXK_HALF || (f.kind == OCU_TEXK_FLOAT && f.channel_bytes == 2)) {
            unsigned short h;
            __builtin_memcpy(&h, q, 2);
            out[c] = half_to_float(h);
        } else {
            bool sg = f.kind == OCU_TEXK_SIGNED;
            long long v = load_int(q, f.channel_bytes, sg);
            if (f.norm) {
                float scale = sg ? (f.channel_bytes == 1 ? 127.0f : 32767.0f)
                                 : (f.channel_bytes == 1 ? 255.0f : 65535.0f);
                float r = (float)v / scale;
                out[c] = r < -1.0f ? -1.0f : r;
            } else {
                out[c] = (float)v;
            }
        }
    }
}

/* Components of one texel as integers. */
OCU_TEX_FN void decode_int(const unsigned char *p, const format &f, long long out[4])
{
    for (int c = 0; c < 4; ++c) out[c] = 0;
    for (int c = 0; c < f.channels && c < 4; ++c) {
        out[c] = load_int(p + c * f.channel_bytes, f.channel_bytes, f.kind == OCU_TEXK_SIGNED);
    }
}

/* ---- addressing ---------------------------------------------------------------- */

OCU_TEX_FN float floor_(float x) { return __builtin_floorf(x); }

/* The coordinate in texels after the normalized-coordinate wrap and mirror
   steps; also the address mode left to apply to indices. */
OCU_TEX_FN float to_texels(float c, unsigned int n, int norm, int *mode)
{
    if (!norm) {
        if (*mode == cudaAddressModeWrap || *mode == cudaAddressModeMirror) {
            *mode = cudaAddressModeClamp;
        }
        return c;
    }
    if (*mode == cudaAddressModeWrap) {
        c = c - floor_(c);
    } else if (*mode == cudaAddressModeMirror) {
        float fl = floor_(c);
        float fr = c - fl;
        float half = fl * 0.5f;
        bool odd = half != floor_(half);
        c = odd ? 1.0f - fr : fr;
    }
    return c * (float)n;
}

/* Index i on an axis of n texels under `mode`; false when it falls on the
   border. */
OCU_TEX_FN bool index_ok(long long *i, unsigned int n, int mode)
{
    long long v = *i, N = (long long)n;
    if (v >= 0 && v < N) return true;
    switch (mode) {
    case cudaAddressModeBorder:
        return false;
    case cudaAddressModeWrap:
        v %= N;
        if (v < 0) v += N;
        break;
    case cudaAddressModeMirror:
        if (v < 0) v = -v - 1;
        if (v >= N) v = 2 * N - 1 - v;
        if (v < 0) v = 0;
        if (v >= N) v = N - 1;
        break;
    default: /* clamp */
        v = v < 0 ? 0 : N - 1;
        break;
    }
    *i = v;
    return true;
}

/* floor of a coordinate as an index, NaN as 0, saturated far outside. */
OCU_TEX_FN long long floor_index(float x)
{
    if (!(x == x)) return 0;
    if (x < -1.0e18f) return -(1LL << 60);
    if (x > 1.0e18f) return 1LL << 60;
    return (long long)floor_(x);
}

/* One axis of a fetch: the first index, its linear weight alpha (0 for
   point sampling) and whether the whole fetch is on the border. */
struct axis {
    long long i;
    float a;
    int mode;
    unsigned int n;
};

OCU_TEX_FN bool setup_axis(axis *ax, float c, unsigned int n, int mode, int norm, bool linear)
{
    ax->mode = mode;
    ax->n = n;
    float x = to_texels(c, n, norm, &ax->mode);
    if (ax->mode == cudaAddressModeBorder && !(x >= 0.0f && x < (float)n)) return false;
    if (!linear) {
        ax->i = floor_index(x);
        ax->a = 0.0f;
        return true;
    }
    /* xB = x - 0.5 with 8 fractional bits, rounded to nearest. */
    float xb = x - 0.5f;
    float q = floor_(xb * 256.0f + 0.5f);
    long long qi = (q == q) ? (q < -1.0e18f ? -(1LL << 60) : q > 1.0e18f ? (1LL << 60) : (long long)q)
                            : 0;
    long long i = qi >= 0 ? qi / 256 : -((-qi + 255) / 256);
    ax->i = i;
    ax->a = (float)(qi - i * 256) / 256.0f;
    return true;
}

/* ---- fetches -------------------------------------------------------------------- */

OCU_TEX_FN const unsigned char *texel_at(const ocu_tex_desc &d, const format &f, long long i,
                                         long long j, long long k)
{
    return (const unsigned char *)d.base + (unsigned long long)k * d.slice +
           (unsigned long long)j * d.pitch + (unsigned long long)i * (unsigned long long)f.texel_bytes;
}

template <class T>
OCU_TEX_FN T zero_value()
{
    T v;
    __builtin_memset(&v, 0, sizeof(T));
    return v;
}

template <class T, int n> struct maker {
    OCU_TEX_MFN static T make(const typename traits<T>::C *c) { return traits<T>::make(c); }
};
template <class T> struct maker<T, 0> {
    OCU_TEX_MFN static T make(const typename traits<T>::C *) { return zero_value<T>(); }
};

template <class T>
OCU_TEX_FN T make_(const typename traits<T>::C *c)
{
    return maker<T, traits<T>::n>::make(c);
}

/* The border value as the result type: the floats themselves, or for
   integer results their bit patterns (the API reference: integer border
   colours are passed reinterpreted as float). */
template <class T>
OCU_TEX_FN T border_value(const sampler &s)
{
    typedef traits<T> tr;
    if (tr::n == 0 || s.border == 0) return zero_value<T>();
    typename tr::C c[4];
    for (int i = 0; i < 4; ++i) {
        if (is_float<typename tr::C>::value) {
            c[i] = (typename tr::C)s.border[i];
        } else {
            unsigned int b;
            __builtin_memcpy(&b, &s.border[i], 4);
            c[i] = (typename tr::C)b;
        }
    }
    return maker<T, traits<T>::n>::make(c);
}

/* Point sample at indices (i, j, k) already addressed. */
template <class T>
OCU_TEX_FN T point_value(const ocu_tex_desc &d, const format &f, long long i, long long j,
                         long long k)
{
    typedef traits<T> tr;
    const unsigned char *p = texel_at(d, f, i, j, k);
    if (tr::n == 0 || f.kind == OCU_TEXK_RAW) {
        T v = zero_value<T>();
        __builtin_memcpy(&v, p, sizeof(T) < (unsigned)f.texel_bytes ? sizeof(T) : (size_t)f.texel_bytes);
        return v;
    }
    typename tr::C c[4];
    if (is_float<typename tr::C>::value) {
        float v[4];
        decode_float(p, f, v);
        for (int q = 0; q < 4; ++q) c[q] = (typename tr::C)v[q];
    } else {
        long long v[4];
        decode_int(p, f, v);
        for (int q = 0; q < 4; ++q) c[q] = (typename tr::C)v[q];
    }
    return maker<T, tr::n>::make(c);
}

/* A fetch at coordinates (x, y, z); dims = 1, 2 or 3 coordinates, `layer`
   >= 0 selects a layer (layered textures). */
template <class T>
OCU_TEX_FN T sample(const ocu_tex_desc &d, const sampler &s, int dims, float x, float y, float z,
                    int layer)
{
    typedef traits<T> tr;
    if (d.base == 0) return zero_value<T>();
    format f = format_of<T>(d, s);
    bool linear = s.filter == cudaFilterModeLinear && tr::n != 0 &&
                  is_float<typename tr::C>::value && f.kind != OCU_TEXK_RAW &&
                  d.resource != OCU_TEXR_LINEAR;
    unsigned int size[3] = {d.width, dims > 1 ? d.height : 1u, dims > 2 ? d.depth : 1u};
    float co[3] = {x, y, z};
    axis ax[3];
    for (int a = 0; a < 3; ++a) {
        if (a < dims) {
            if (!setup_axis(&ax[a], co[a], size[a], s.address[a], s.normalized, linear)) {
                return border_value<T>(s);
            }
        } else {
            ax[a].i = 0; ax[a].a = 0.0f; ax[a].mode = cudaAddressModeClamp; ax[a].n = 1;
        }
    }
    long long kbase = 0;
    if (layer >= 0) {
        long long L = d.depth ? (long long)d.depth : 1;
        kbase = layer < L ? layer : L - 1;
    }
    if (!linear) {
        long long idx[3];
        for (int a = 0; a < 3; ++a) {
            idx[a] = ax[a].i;
            if (!index_ok(&idx[a], ax[a].n, ax[a].mode)) return border_value<T>(s);
        }
        return point_value<T>(d, f, idx[0], idx[1], layer >= 0 ? kbase : idx[2]);
    }
    /* Linear: the 2^dims neighbours in the guide's order (x fastest). */
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    T bv = border_value<T>(s);
    const typename tr::C *bc = &((const typename tr::C *)&bv)[0];
    int taps = 1 << dims;
    for (int t = 0; t < taps; ++t) {
        float w = 1.0f;
        long long idx[3];
        bool in = true;
        for (int a = 0; a < 3; ++a) {
            int up = a < dims ? (t >> a) & 1 : 0;
            idx[a] = ax[a].i + up;
            if (a < dims) w *= up ? ax[a].a : 1.0f - ax[a].a;
            if (!index_ok(&idx[a], ax[a].n, ax[a].mode)) in = false;
        }
        float v[4];
        if (in) {
            decode_float(texel_at(d, f, idx[0], idx[1], layer >= 0 ? kbase : idx[2]), f, v);
        } else {
            for (int q = 0; q < 4; ++q) v[q] = q < tr::n ? (float)bc[q] : 0.0f;
        }
        for (int q = 0; q < 4; ++q) acc[q] += w * v[q];
    }
    typename tr::C c[4];
    for (int q = 0; q < 4; ++q) c[q] = (typename tr::C)acc[q];
    return make_<T>(c);
}

/* tex1Dfetch: integer index, no filtering, no addressing. */
template <class T>
OCU_TEX_FN T fetch_index(const ocu_tex_desc &d, const sampler &s, int x)
{
    if (d.base == 0) return zero_value<T>();
    format f = format_of<T>(d, s);
    if (x < 0 || (unsigned int)x >= d.width) return zero_value<T>();
    if (f.texel_bytes == 0 ||
        ((unsigned long long)x + 1u) * (unsigned long long)f.texel_bytes > d.bytes) {
        return zero_value<T>();
    }
    return point_value<T>(d, f, x, 0, 0);
}

OCU_TEX_FN const ocu_tex_desc &object(cudaTextureObject_t o)
{
    return *(const ocu_tex_desc *)(__UINTPTR_TYPE__)o;
}

} // namespace ocu_tex

/* ---- texture objects -------------------------------------------------------------- */

template <class T>
OCU_TEX_FN T tex1Dfetch(cudaTextureObject_t o, int x)
{
    const ocu_tex_desc &d = ocu_tex::object(o);
    return ocu_tex::fetch_index<T>(d, ocu_tex::of_object(d), x);
}
template <class T>
OCU_TEX_FN T tex1D(cudaTextureObject_t o, float x)
{
    const ocu_tex_desc &d = ocu_tex::object(o);
    return ocu_tex::sample<T>(d, ocu_tex::of_object(d), 1, x, 0.0f, 0.0f, -1);
}
template <class T>
OCU_TEX_FN T tex2D(cudaTextureObject_t o, float x, float y)
{
    const ocu_tex_desc &d = ocu_tex::object(o);
    return ocu_tex::sample<T>(d, ocu_tex::of_object(d), 2, x, y, 0.0f, -1);
}
template <class T>
OCU_TEX_FN T tex3D(cudaTextureObject_t o, float x, float y, float z)
{
    const ocu_tex_desc &d = ocu_tex::object(o);
    return ocu_tex::sample<T>(d, ocu_tex::of_object(d), 3, x, y, z, -1);
}
template <class T>
OCU_TEX_FN T tex1DLayered(cudaTextureObject_t o, float x, int layer)
{
    const ocu_tex_desc &d = ocu_tex::object(o);
    return ocu_tex::sample<T>(d, ocu_tex::of_object(d), 1, x, 0.0f, 0.0f, layer < 0 ? 0 : layer);
}
template <class T>
OCU_TEX_FN T tex2DLayered(cudaTextureObject_t o, float x, float y, int layer)
{
    const ocu_tex_desc &d = ocu_tex::object(o);
    return ocu_tex::sample<T>(d, ocu_tex::of_object(d), 2, x, y, 0.0f, layer < 0 ? 0 : layer);
}
/* Level of detail: there are no mipmapped arrays, so every level is level 0. */
template <class T>
OCU_TEX_FN T tex1DLod(cudaTextureObject_t o, float x, float) { return tex1D<T>(o, x); }
template <class T>
OCU_TEX_FN T tex2DLod(cudaTextureObject_t o, float x, float y, float) { return tex2D<T>(o, x, y); }
template <class T>
OCU_TEX_FN T tex3DLod(cudaTextureObject_t o, float x, float y, float z, float)
{
    return tex3D<T>(o, x, y, z);
}

/* The forms that store the result through a pointer. */
template <class T>
OCU_TEX_FN void tex1Dfetch(T *r, cudaTextureObject_t o, int x) { *r = tex1Dfetch<T>(o, x); }
template <class T>
OCU_TEX_FN void tex1D(T *r, cudaTextureObject_t o, float x) { *r = tex1D<T>(o, x); }
template <class T>
OCU_TEX_FN void tex2D(T *r, cudaTextureObject_t o, float x, float y) { *r = tex2D<T>(o, x, y); }
template <class T>
OCU_TEX_FN void tex3D(T *r, cudaTextureObject_t o, float x, float y, float z)
{
    *r = tex3D<T>(o, x, y, z);
}
template <class T>
OCU_TEX_FN void tex1DLayered(T *r, cudaTextureObject_t o, float x, int layer)
{
    *r = tex1DLayered<T>(o, x, layer);
}
template <class T>
OCU_TEX_FN void tex2DLayered(T *r, cudaTextureObject_t o, float x, float y, int layer)
{
    *r = tex2DLayered<T>(o, x, y, layer);
}

/* ---- texture references -------------------------------------------------------------- */

#if defined(OCU_CPU_PATH)
#define OCU_TEX_REF_RESULT typename ocu_tex::ref_result<T, m>::type
template <class T, int t, enum cudaTextureReadMode m>
OCU_TEX_FN OCU_TEX_REF_RESULT tex1Dfetch(const texture<T, t, m> &r, int x)
{
    return ocu_tex::fetch_index<OCU_TEX_REF_RESULT>(r.ocu_bound, ocu_tex::of_reference(r, m), x);
}
template <class T, int t, enum cudaTextureReadMode m>
OCU_TEX_FN OCU_TEX_REF_RESULT tex1D(const texture<T, t, m> &r, float x)
{
    return ocu_tex::sample<OCU_TEX_REF_RESULT>(r.ocu_bound, ocu_tex::of_reference(r, m), 1, x,
                                               0.0f, 0.0f, -1);
}
template <class T, int t, enum cudaTextureReadMode m>
OCU_TEX_FN OCU_TEX_REF_RESULT tex2D(const texture<T, t, m> &r, float x, float y)
{
    return ocu_tex::sample<OCU_TEX_REF_RESULT>(r.ocu_bound, ocu_tex::of_reference(r, m), 2, x, y,
                                               0.0f, -1);
}
template <class T, int t, enum cudaTextureReadMode m>
OCU_TEX_FN OCU_TEX_REF_RESULT tex3D(const texture<T, t, m> &r, float x, float y, float z)
{
    return ocu_tex::sample<OCU_TEX_REF_RESULT>(r.ocu_bound, ocu_tex::of_reference(r, m), 3, x, y, z,
                                               -1);
}
template <class T, int t, enum cudaTextureReadMode m>
OCU_TEX_FN OCU_TEX_REF_RESULT tex1DLayered(const texture<T, t, m> &r, float x, int layer)
{
    return ocu_tex::sample<OCU_TEX_REF_RESULT>(r.ocu_bound, ocu_tex::of_reference(r, m), 1, x,
                                               0.0f, 0.0f, layer < 0 ? 0 : layer);
}
template <class T, int t, enum cudaTextureReadMode m>
OCU_TEX_FN OCU_TEX_REF_RESULT tex2DLayered(const texture<T, t, m> &r, float x, float y, int layer)
{
    return ocu_tex::sample<OCU_TEX_REF_RESULT>(r.ocu_bound, ocu_tex::of_reference(r, m), 2, x, y,
                                               0.0f, layer < 0 ? 0 : layer);
}
#undef OCU_TEX_REF_RESULT
#else
/* NVIDIA path: device code cannot read a host variable, and the vendor
   runtime removed texture references in 12.0. A fetch through one stops
   the build with this message. */
template <class T> struct ocu_tex_ref_unsupported { static const bool value = false; };
#define OCU_TEX_REF_UNSUPPORTED(name, ...)                                                    \
    template <class T, int t, enum cudaTextureReadMode m>                                     \
    OCU_TEX_FN T name(const texture<T, t, m> &, __VA_ARGS__)                                  \
    {                                                                                         \
        static_assert(ocu_tex_ref_unsupported<T>::value,                                      \
                      "texture references are not supported on this path (removed in CUDA "   \
                      "12): use texture objects");                                            \
        return ocu_tex::zero_value<T>();                                                      \
    }
OCU_TEX_REF_UNSUPPORTED(tex1Dfetch, int)
OCU_TEX_REF_UNSUPPORTED(tex1D, float)
OCU_TEX_REF_UNSUPPORTED(tex2D, float, float)
OCU_TEX_REF_UNSUPPORTED(tex3D, float, float, float)
#undef OCU_TEX_REF_UNSUPPORTED
#endif

#endif /* OCU_TEXTURE_FETCH_H */
