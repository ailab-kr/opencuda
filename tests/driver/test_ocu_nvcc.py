"""ocu-nvcc's command line, checked from outside (subprocess), as a build
system uses it.

    OCU_NVCC=<toolkit>/bin/ocu-nvcc python3 -m unittest test_ocu_nvcc
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

DRIVER = os.environ.get("OCU_NVCC", "")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SAXPY = REPO / "compiler" / "probe" / "probe_kernel.cu"
MAIN = REPO / "tests" / "probe" / "main_saxpy.cu"

KERNEL = """
__global__ void twice(int *p) { p[threadIdx.x] *= 2; }
int main() {
    int *d = nullptr, h[4] = {1, 2, 3, 4};
    cudaMalloc(&d, sizeof h);
    cudaMemcpy(d, h, sizeof h, cudaMemcpyHostToDevice);
    twice<<<1, 4>>>(d);
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    cudaFree(d);
    return h[0] + h[1] + h[2] + h[3] == 20 ? 0 : 1;
}
"""


def host_openmp() -> bool:
    """True when the host compiler the driver picks (the newest
    /usr/lib/llvm-N clang++, else clang++, else c++) compiles and links an
    OpenMP program by itself. Without the OpenMP runtime installed, or with
    a compiler that has no -fopenmp (Apple clang), the driver's OpenMP test
    cannot say anything about the driver."""
    cxx = next((p for p in (f"/usr/lib/llvm-{v}/bin/clang++" for v in (20, 19, 18)) if os.path.exists(p)),
               shutil.which("clang++") or shutil.which("c++"))
    if not cxx:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "p.cpp"
        src.write_text("#include <omp.h>\nint main() { return omp_get_max_threads() > 0 ? 0 : 1; }\n")
        done = subprocess.run([cxx, "-fopenmp", str(src), "-o", str(Path(tmp) / "p")], capture_output=True)
        return done.returncode == 0


def nvcc(*args, cwd=None, check=True):
    done = subprocess.run([DRIVER, *map(str, args)], cwd=cwd, capture_output=True, text=True)
    if check and done.returncode != 0:
        raise AssertionError(f"ocu-nvcc {' '.join(map(str, args))} failed "
                             f"({done.returncode}):\n{done.stdout}\n{done.stderr}")
    return done


@unittest.skipUnless(DRIVER, "set OCU_NVCC to the driver")
class CommandLine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ocu-nvcc-test-")
        self.dir = Path(self.tmp.name)
        self.src = self.dir / "k.cu"
        self.src.write_text(KERNEL)

    def tearDown(self):
        self.tmp.cleanup()

    def test_version_is_ours(self):
        for flag in ("--version", "-V"):
            out = nvcc(flag).stdout
            self.assertIn("ocu-nvcc", out)
            self.assertNotIn("NVIDIA", out)
            self.assertRegex(out, r"V\d+\.\d+\.\d+")

    def test_compile_link_run_one_step(self):
        nvcc(self.src, "-o", self.dir / "a")
        self.assertEqual(subprocess.run([self.dir / "a"]).returncode, 0)

    def test_default_output_names(self):
        nvcc("-c", self.src, cwd=self.dir)
        self.assertTrue((self.dir / "k.o").is_file())
        nvcc("k.o", cwd=self.dir)
        self.assertTrue((self.dir / "a.out").is_file())
        self.assertEqual(subprocess.run([self.dir / "a.out"]).returncode, 0)

    def test_option_spellings(self):
        # long and short forms, "=" and separate values, attached values
        nvcc("--compile", "--x=cu", self.src, "--output-file", self.dir / "a.o",
             "--gpu-architecture=sm_75", "--gpu-code=sm_75,compute_75",
             "-gencode=arch=compute_80,code=[sm_80,compute_80]", "--optimize", "2",
             "-O3", "-g", "-G", "--generate-line-info", "-std=c++17", "--std", "c++20",
             "--define-macro", "OCU_T_A=1", "-DOCU_T_B", "-UOCU_T_B", "--include-path", self.dir,
             "-isystem", self.dir, "--compiler-options", "-Wall,-Wextra",
             "-Xcompiler", "-Wno-unused -Wno-unused-parameter", "-Xptxas", "-v",
             "-Xnvlink", "--verbose", "--use_fast_math", "--expt-relaxed-constexpr",
             "--extended-lambda", "-extended-lambda", "-ccbin", "clang++",
             "-Wno-deprecated-gpu-targets", "--threads", "4", "-diag-suppress", "177",
             "-cudart", "static", "-rdc=false", "-fPIC", "-m64")
        self.assertTrue((self.dir / "a.o").is_file())

    def test_ignored_backend_tool_options_are_noted(self):
        err = nvcc("-c", self.src, "-o", self.dir / "a.o", "-Xptxas", "-O3", "-v").stderr
        self.assertIn("-Xptxas -O3", err)

    def test_unknown_option_fails(self):
        done = nvcc("-c", self.src, "--no-such-option", check=False)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("unknown option", done.stderr)

    def test_dependencies(self):
        nvcc("-M", self.src, "-o", self.dir / "k.d")
        text = (self.dir / "k.d").read_text()
        self.assertIn("k.o:", text)
        self.assertIn("cuda_runtime.h", text)
        out = nvcc("-M", self.src, "-MT", "custom.o").stdout
        self.assertIn("custom.o:", out)
        nvcc("-c", self.src, "-o", self.dir / "k.o", "-MD", "-MF", self.dir / "x.d",
             "-MT", "obj/k.o")
        self.assertIn("obj/k.o:", (self.dir / "x.d").read_text())
        nvcc("-c", self.src, "-o", self.dir / "k2.o", "-MD")
        self.assertTrue((self.dir / "k2.d").is_file())

    def test_define_with_commas(self):
        # one macro whose value has commas, not three macros: "f16" stays an
        # ordinary name
        src = self.dir / "d.cu"
        src.write_text('static const char *s = OCU_LIST;\nint f16 = 3;\n'
                       'int main() { return s[0] == \'a\' && f16 == 3 ? 0 : 1; }\n')
        nvcc("-c", src, "-o", self.dir / "d.o", '-DOCU_LIST="a-a,f16-f16,b"')

    def test_preprocess(self):
        out = nvcc("-E", self.src, "-DMARK=12345", "-x", "cu").stdout
        self.assertIn("twice", out)
        nvcc("-E", self.src, "-o", self.dir / "k.ii")
        self.assertIn("twice", (self.dir / "k.ii").read_text())

    def test_dryrun_runs_nothing(self):
        out = nvcc("-dryrun", "-c", self.src, "-o", self.dir / "never.o").stdout
        self.assertIn("#$ ", out)
        self.assertFalse((self.dir / "never.o").exists())

    def test_verbose_records_architectures(self):
        err = nvcc("-v", "-c", self.src, "-o", self.dir / "k.o", "-arch=sm_86").stderr
        self.assertIn("-target-cpu sm_86", err)

    def test_host_sources_and_libraries(self):
        c = self.dir / "h.c"
        c.write_text("#include <cuda_runtime.h>\nint count(void) { int n = 0;"
                     " return cudaGetDeviceCount(&n) == cudaSuccess ? n : -1; }\n")
        cpp = self.dir / "m.cpp"
        cpp.write_text('extern "C" int count(void);\n#include <cstdio>\n'
                       'int main() { std::printf("%d\\n", count()); return count() > 0 ? 0 : 1; }\n')
        nvcc("-c", c, "-o", self.dir / "h.o")
        nvcc("-lib", self.dir / "h.o", "-o", self.dir / "libh.a")
        nvcc(cpp, "-L", self.dir, "-lh", "-lcudart", "-o", self.dir / "m")
        self.assertEqual(subprocess.run([self.dir / "m"]).returncode, 0)

    def test_host_openmp(self):
        # nvcc -Xcompiler -fopenmp: omp.h is found, and the program runs without
        # LD_LIBRARY_PATH with both OpenMP and the GPU working (the OpenMP
        # runtime's directory must not hide the device's libraries).
        if not host_openmp():
            self.skipTest("the host compiler has no OpenMP (no -fopenmp or no omp.h / runtime)")
        src = self.dir / "o.cu"
        src.write_text("#include <omp.h>\n" + KERNEL.replace(
            "int main() {", "int main() { int n = 0;\n#pragma omp parallel reduction(+:n)\n"
            " n += 1;\n if (n != omp_get_max_threads()) return 2;"))
        nvcc("-c", src, "-Xcompiler", "-fopenmp", "-o", self.dir / "o.o")
        nvcc(self.dir / "o.o", "-Xcompiler", "-fopenmp", "-o", self.dir / "o")
        env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
        self.assertEqual(subprocess.run([self.dir / "o"], env=env).returncode, 0)

    def test_shared_library(self):
        lib = self.dir / "k.cu"
        lib.write_text('__global__ void inc(int *p) { *p += 1; }\n'
                       'extern "C" int bump(void) { int *d, h = 41; cudaMalloc(&d, 4);\n'
                       ' cudaMemcpy(d, &h, 4, cudaMemcpyHostToDevice); inc<<<1, 1>>>(d);\n'
                       ' cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost); cudaFree(d); return h; }\n')
        nvcc("-shared", "-Xcompiler", "-fPIC", lib, "-o", self.dir / "libk.so")
        main = self.dir / "main.c"
        main.write_text("int bump(void);\nint main(void) { return bump() == 42 ? 0 : 1; }\n")
        nvcc(main, "-L", self.dir, "-lk", "-Xlinker", f"-rpath={self.dir}", "-o", self.dir / "m")
        self.assertEqual(subprocess.run([self.dir / "m"]).returncode, 0)

    def test_response_file(self):
        rsp = self.dir / "opts.rsp"
        rsp.write_text(f"-c {self.src} -o {self.dir / 'r.o'} -O2\n")
        nvcc(f"@{rsp}")
        self.assertTrue((self.dir / "r.o").is_file())
        nvcc("--options-file", rsp)

    def test_separate_files_like_a_makefile(self):
        nvcc("-c", SAXPY, "-o", self.dir / "k.o")
        nvcc("-c", MAIN, "-o", self.dir / "m.o")
        nvcc(self.dir / "k.o", self.dir / "m.o", "-o", self.dir / "saxpy", "-lcudart")
        done = subprocess.run([self.dir / "saxpy"], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout)
        self.assertIn("ok", done.stdout)

    def test_rdc_and_device_link(self):
        # CPU target: device code is host code, so relocatable device code
        # links as usual and the device-link step has nothing to do.
        nvcc("-dc", self.src, "-o", self.dir / "k.o", "--ocu-target=cpu")
        nvcc("-dlink", self.dir / "k.o", "-o", self.dir / "dl.o", "--ocu-target=cpu")
        nvcc(self.dir / "k.o", self.dir / "dl.o", "-o", self.dir / "a", "--ocu-target=cpu")
        self.assertEqual(subprocess.run([self.dir / "a"]).returncode, 0)
        # The nvidia target cannot link device code across files yet: it says so.
        for args in (["-rdc=true", "-c"], ["-dc"], ["-fgpu-rdc", "-c"]):
            done = nvcc(*args, self.src, "--ocu-target=nvidia", check=False)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("not support", done.stderr)

    def test_unsupported_outputs_say_so(self):
        for flag in ("-cubin", "-fatbin"):
            done = nvcc(flag, self.src, check=False)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("not supported", done.stderr)
        done = nvcc("-ptx", self.src, "--ocu-target=cpu", check=False)
        self.assertNotEqual(done.returncode, 0)

    def test_vendor_paths_refused(self):
        for opt in ("-I", "-isystem", "-L"):
            done = nvcc("-c", self.src, opt, "/usr/local/cuda/include", check=False)
            self.assertNotEqual(done.returncode, 0, opt)
            self.assertIn("vendor", done.stderr)

    def test_target_from_name(self):
        link = self.dir / "ocu-nvcc-nvidia"
        link.symlink_to(Path(DRIVER).resolve())
        out = subprocess.run([link, "--version"], capture_output=True, text=True).stdout
        self.assertIn("target nvidia", out)
        out = nvcc("--ocu-target=nvidia", "--version").stdout
        self.assertIn("target nvidia", out)

    def test_ptx_output(self):
        if not Path("/usr/lib/llvm-18/bin/clang++").exists():
            self.skipTest("no LLVM clang with NVPTX")
        nvcc("--ocu-target=nvidia", "-ptx", SAXPY, "-arch=sm_80", "-o", self.dir / "k.ptx")
        self.assertIn(".target sm_80", (self.dir / "k.ptx").read_text())


if __name__ == "__main__":
    unittest.main()
