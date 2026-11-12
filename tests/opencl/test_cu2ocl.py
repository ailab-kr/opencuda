#!/usr/bin/env python3
"""Host-side tests of compiler/opencl/cu2ocl.py (no OpenCL device needed):
what it accepts that the Metal translation refuses, what it refuses, and
the shape of the image (header lines, source length, SPIR-V module).

    OCU_CLANG=<clang++ with CUDA and C++ for OpenCL> [OCU_LLVM_SPIRV=<llvm-spirv>]
        python3 -m unittest test_cu2ocl
"""
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TOOL = REPO / "compiler" / "opencl" / "cu2ocl.py"
CLANG = os.environ.get("OCU_CLANG", "clang++")
SPIRV = os.environ.get("OCU_LLVM_SPIRV", "none")


def translate(src: str):
    """(returncode, image bytes or None, stderr)."""
    with tempfile.TemporaryDirectory() as tmp:
        cu = Path(tmp) / "t.cu"
        cu.write_text(src)
        img = Path(tmp) / "t.img"
        r = subprocess.run([sys.executable, str(TOOL), str(cu), "-o", str(img), "--clang", CLANG],
                           capture_output=True, text=True)
        return r.returncode, (img.read_bytes() if img.exists() else None), r.stderr


def parse(image: bytes):
    """(header lines, source text, SPIR-V bytes)."""
    assert image.startswith(b"OCUOCL1\n")
    head, body = image[8:].split(b"\n---\n", 1)
    lines = head.decode().splitlines()
    sizes = {l.split()[0]: int(l.split()[1]) for l in lines if l.split()[0] in ("source", "spirv")}
    src = body[:sizes["source"]].decode("latin-1")
    spv = body[sizes["source"]:sizes["source"] + sizes.get("spirv", 0)]
    assert len(body) == sizes["source"] + sizes.get("spirv", 0)
    return lines, src, spv


class Accepts(unittest.TestCase):
    def test_double(self):
        rc, img, err = translate("__global__ void k(double *p, double s) { p[threadIdx.x] *= s; }")
        self.assertEqual(rc, 0, err)
        lines, src, _ = parse(img)
        self.assertIn("param _Z1kPdd 8 8", lines)
        self.assertIn("ocu_arg<double>", src)

    def test_vec3_argument_read_as_three_scalars(self):
        rc, img, err = translate("__global__ void k(float3 v, float *o) { o[0] = v.x + v.y + v.z; }")
        self.assertEqual(rc, 0, err)
        lines, src, _ = parse(img)
        self.assertIn("param _Z1k6float3Pf 12 8", lines)
        self.assertIn("ocu_arg3<float3, float>", src)

    def test_kernel_signature_and_variables(self):
        rc, img, err = translate("""
__device__ int counter = 5;
__constant__ float scale[2] = {1.5f, 2.0f};
__global__ void k(float *o) { o[threadIdx.x] = scale[1] * counter; }""")
        self.assertEqual(rc, 0, err)
        lines, src, _ = parse(img)
        self.assertTrue(any(l.startswith("global counter ") for l in lines), lines)
        self.assertTrue(any(l.startswith("init ") for l in lines), lines)
        self.assertIn("device uchar *ocu_glob,", src)
        self.assertIn("OCU_U3(get_local_id)", src)

    def test_long_long_is_64_bits(self):
        # OpenCL's `long long` is reserved for 128 bits; CUDA's is OpenCL's long.
        rc, img, err = translate("__global__ void k(unsigned long long *p, long long s) {"
                                 " unsigned long long v = p[0]; p[0] = (v << 3) + s; }")
        self.assertEqual(rc, 0, err)
        lines, src, _ = parse(img)
        user = src[src.index("// ---- macros"):]
        self.assertNotRegex(user, r"\blong\s+long\b")
        self.assertIn("param _Z1kPyx 8 8", lines)

    def test_inner_block_shared_moves_to_kernel_scope(self):
        rc, img, err = translate("""
__global__ void k(float *o) {
    if (blockIdx.x == 0) {
        __shared__ float tile[64];
        tile[threadIdx.x] = 1.0f;
        __syncthreads();
        o[threadIdx.x] = tile[63 - threadIdx.x];
    }
}""")
        self.assertEqual(rc, 0, err)
        _, src, _ = parse(img)
        body = src[src.index("kernel void _Z1kPf"):]
        self.assertLess(body.index("threadgroup float tile[64];"), body.index("if (blockIdx.x == 0)"))

    @unittest.skipIf(SPIRV == "none", "no llvm-spirv")
    def test_spirv_module(self):
        rc, img, err = translate("__global__ void k(int *p) { p[blockIdx.x] = 1; }")
        self.assertEqual(rc, 0, err)
        _, _, spv = parse(img)
        self.assertGreater(len(spv), 20)
        self.assertEqual(struct.unpack("<I", spv[:4])[0], 0x07230203)  # SPIR-V magic number


class Refusals(unittest.TestCase):
    def refused(self, src, words):
        rc, img, err = translate(src)
        self.assertNotEqual(rc, 0, img)
        self.assertIsNone(img)
        self.assertIn("cannot translate for OpenCL", err)
        self.assertIn(words, err)

    def test_function_pointers(self):
        self.refused("__device__ int f(int x) { return x + 1; }\n"
                     "__global__ void k(int *o) { int (*fp)(int) = f; o[0] = fp(1); }",
                     "not available on OpenCL")

    def test_warp_functions_have_no_fixed_width(self):
        # Sub-groups are not 32 wide everywhere; the prelude leaves the warp
        # functions out, and the generated code does not compile.
        self.refused("__global__ void k(int *p) { p[threadIdx.x] = __shfl_down_sync(0xffffffffu, p[0], 1); }",
                     "does not compile")

    def test_one_declaration_two_spaces(self):
        self.refused("""
__global__ void k(float *g) {
    __shared__ float s[32];
    float *a = s, *b = g;
    a[0] = b[0];
}""", "split it")


if __name__ == "__main__":
    unittest.main()
