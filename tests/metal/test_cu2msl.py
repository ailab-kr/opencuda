#!/usr/bin/env python3
"""Host-side tests of compiler/metal/cu2msl.py (no Metal device needed):
what it refuses, with which message, and the shape of what it emits.
Generated MSL is checked with msl_check.py (C++ for OpenCL against
msl_stub.h).

    OCU_CLANG=<clang++ with CUDA support> python3 -m unittest test_cu2msl
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TOOL = REPO / "compiler" / "metal" / "cu2msl.py"
CLANG = os.environ.get("OCU_CLANG", "clang++")
sys.path.insert(0, str(HERE))
import msl_check  # noqa: E402


def translate(src: str, *extra):
    """(returncode, image text or None, stderr)."""
    with tempfile.TemporaryDirectory() as tmp:
        cu = Path(tmp) / "t.cu"
        cu.write_text(src)
        img = Path(tmp) / "t.img"
        r = subprocess.run([sys.executable, str(TOOL), str(cu), "-o", str(img), "--clang", CLANG, *extra],
                           capture_output=True, text=True)
        text = img.read_text() if img.exists() else None
        ok = None
        if text is not None:
            ok, msg, _ = msl_check.check(img, CLANG)
            if not ok:
                text = "SYNTAX CHECK FAILED\n" + msg + "\n" + text
        return r.returncode, text, r.stderr


class Refusals(unittest.TestCase):
    def refused(self, src, words):
        rc, img, err = translate(src)
        self.assertNotEqual(rc, 0, img)
        self.assertIsNone(img)
        self.assertIn("cannot translate for Metal", err)
        self.assertIn(words, err)

    def test_double(self):
        self.refused("__global__ void k(double *p) { p[0] = 1.0; }", "double")

    def test_one_declaration_two_spaces(self):
        self.refused("""
__global__ void k(float *g) {
    __shared__ float s[32];
    float *a = s, *b = g;
    a[0] = b[0];
}""", "split it")

    def test_recursion(self):
        self.refused("""
__device__ int f(int n) { return n ? f(n - 1) + 1 : 0; }
__global__ void k(int *p) { p[0] = f(p[1]); }""", "recursive")

    def test_namespace(self):
        self.refused("""
namespace ns { __device__ int f(int n) { return n; } }
__global__ void k(int *p) { p[0] = ns::f(p[1]); }""", "namespace")

    def test_shared_in_device_function(self):
        self.refused("""
__device__ float f(float v) { __shared__ float s[32]; s[0] = v; return s[1]; }
__global__ void k(float *p) { p[0] = f(p[1]); }""", "__shared__ in a device function")

    def test_pointer_to_pointer(self):
        self.refused("__global__ void k(float **p) { p[0][0] = 1.0f; }", "pointers to pointers")

    def test_global_in_macro_body(self):
        self.refused("""
__constant__ float c[4];
#define FIRST (c[0])
__global__ void k(float *p) { p[0] = FIRST; }""", "inside a macro")

    def test_struct_with_float3(self):
        self.refused("""
struct S { float3 v; int n; };
__global__ void k(S s, float *p) { p[0] = s.v.x; }""", "laid out differently")


class Emitted(unittest.TestCase):
    def ok(self, src, *extra):
        rc, img, err = translate(src, *extra)
        self.assertEqual(rc, 0, err)
        self.assertNotIn("SYNTAX CHECK FAILED", img)
        return img

    def test_overloads_per_address_space(self):
        img = self.ok("""
__device__ float first(const float *p) { return p[0]; }
__global__ void k(float *g) {
    __shared__ float s[32];
    float t[2] = {1.0f, 2.0f};
    s[threadIdx.x] = g[threadIdx.x];
    g[0] = first(s) + first(g) + first(t);
}""")
        for space in ("device", "threadgroup", "thread"):
            self.assertRegex(img, r"static float first\(thread const ocu_context &ocu_ctx, " + space + r" const float \* p\) \{")

    def test_pointer_return(self):
        img = self.ok("""
__device__ float *at(float *p, int i) { return p + i; }
__global__ void k(float *g) {
    __shared__ float s[32];
    float *q = at(s, 1);
    float *r = at(g, 2);
    *r = *q;
}""")
        self.assertIn("static threadgroup float * at(thread const ocu_context &ocu_ctx, threadgroup float * p, int i)", img)
        self.assertIn("threadgroup float *q = at(ocu_ctx, s, 1);", img)
        self.assertIn("device float *r = at(ocu_ctx, g, 2);", img)

    def test_template_kernel_instances(self):
        img = self.ok("""
template <typename T, int N> __global__ void fill(T *p, T v) { if (threadIdx.x < N) p[threadIdx.x] = v; }
template __global__ void fill<float, 8>(float *, float);
void host(int *p) { fill<int, 4><<<1, 32>>>(p, 7); }""")
        self.assertIn("param _Z4fillIfLi8EEvPT_S0_ 8 4", img)
        self.assertIn("param _Z4fillIiLi4EEvPT_S0_ 8 4", img)
        self.assertIn("typedef float T;", img)
        self.assertIn("constexpr int N = 4;", img)

    def test_globals_layout_and_init(self):
        img = self.ok("""
__constant__ int k[3] = {1, -2};
__device__ float f = 0.5f;
__device__ unsigned char flag;
__global__ void kern(float *p) { p[0] = k[1] + f + flag; }""")
        self.assertIn("global k 0 12", img)
        self.assertIn("init 0 01000000feffffff00000000", img)
        self.assertIn("global f 12 4", img)
        self.assertIn("init 12 0000003f", img)
        self.assertIn("global flag 16 1", img)
        self.assertNotIn("init 16", img)
        self.assertIn("device uchar *ocu_glob [[buffer(2)]]", img)

    def test_struct_argument(self):
        img = self.ok("""
struct P { char c; double2 *unused_never; };
struct Q { const float *src; char c; short s; float f; float2 v; };
__global__ void k(Q q, float *out) { out[0] = q.src[0] + q.c + q.s + q.f + q.v.y; }""")
        self.assertIn("param _Z1k1QPf 24 8", img)
        self.assertIn("device const float *src;", img)
        self.assertIn('static_assert(sizeof(Q) == 24', img)
        self.assertNotIn("struct P", img)  # not named by the device code

    def test_keep_going(self):
        rc, img, err = translate("""
__global__ void good(float *p) { p[0] = 1.0f; }
__global__ void bad(double *p) { p[0] = 1.0; }""", "--keep-going", "--report")
        self.assertEqual(rc, 0, err)
        self.assertIn("param _Z4goodPf", img)
        self.assertNotIn("_Z3badPd", img)
        self.assertIn("skipping kernel 'bad'", err)

    def test_call_written_in_a_macro(self):
        rc, img, err = translate("""
__device__ float twice(float v) { return 2.0f * v; }
#define APPLY(f, x) f(x)
__global__ void k(float *p) { p[0] = APPLY(twice, p[1]); }""")
        self.assertNotEqual(rc, 0)
        self.assertIn("through a macro", err)

    def test_call_in_a_macro_argument(self):
        img = self.ok("""
__device__ float twice(float v) { return 2.0f * v; }
#define SQUARE(x) ((x) * (x))
__global__ void k(float *p) { p[0] = SQUARE(twice(p[1])); }""")
        self.assertIn("SQUARE(twice(ocu_ctx, p[1]))", img)

    def test_texture_declaration_does_not_stop_the_file(self):
        src = """
texture<float, 1, cudaReadModeElementType> tex;
__global__ void plain(float *p) { p[0] = 1.0f; }
__global__ void textured(float *p) { p[0] = tex1Dfetch(tex, 0); }"""
        rc, img, err = translate(src, "--keep-going")
        self.assertEqual(rc, 0, err)
        self.assertIn("plain", img)
        self.assertIn("texture reference 'tex' is not supported", err)

    def test_constant_float3_global(self):
        img = self.ok("""
__constant__ float3 c[1];
__global__ void k(float *p) { p[0] = c[0].y; }""")
        self.assertIn("packed_float3 ocu_gt_c[1]", img)

    def test_constant_struct_global_without_initializer(self):
        self.ok("""
struct P { float *d; int n; };
__constant__ P cp;
__global__ void k(float *p) { p[cp.n] = 1.0f; }""")

    def test_helper_headers(self):
        self.ok("""
#include <helper_cuda.h>
#include <helper_timer.h>
__global__ void k(int *p) { p[0] = 1; }
void host(int *p) { checkCudaErrors(cudaDeviceSynchronize()); getLastCudaError("k"); }""")

    def test_spaced_chevrons(self):
        self.ok("""
__global__ void k(int *p) { p[0] = 1; }
void host(int *p) { k << < 1, 1 >> > (p); }""")


if __name__ == "__main__":
    unittest.main()
