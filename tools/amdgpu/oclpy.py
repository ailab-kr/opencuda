"""oclpy: the OpenCL backend's protocol (backends/opencl/ocl_backend.c) in
Python over ctypes, for checking translated images on a real GPU where a
native build of the backend cannot run yet.

Why it exists: on Windows with Smart App Control on, binaries built on the
machine are blocked, but the signed Python interpreter and the signed
OpenCL driver run. This module drives the driver the same way the C backend
does, so an image made by compiler/opencl/cu2ocl.py can be checked on the
GPU in front of us:

  - memory: coarse-grain SVM blocks, 256-byte aligned like the backend;
  - module: OCUOCL1 image, its SPIR-V module (devices with
    cl_khr_il_program);
  - module variables: one SVM block from the `global` / `init` lines;
  - launch: argument values packed at their natural alignment (at most 16)
    into a read-only buffer, offsets in a second one, the variable block,
    then dynamic local memory; every allocation named with
    CL_KERNEL_EXEC_INFO_SVM_PTRS.

It is a checking aid, not a backend: one device, one in-order queue,
synchronous launches.
"""
from __future__ import annotations

import ctypes as C
import os
import sys

CL_DEVICE_TYPE_GPU = 1 << 2
CL_DEVICE_NAME = 0x102B
CL_DEVICE_VERSION = 0x102F
CL_DEVICE_EXTENSIONS = 0x1030
CL_DEVICE_SVM_CAPABILITIES = 0x1053
CL_DEVICE_IL_VERSION = 0x105B
CL_DEVICE_BOARD_NAME_AMD = 0x4038
CL_PLATFORM_NAME = 0x0902
CL_PROGRAM_BUILD_LOG = 0x1183
CL_MEM_READ_WRITE = 1
CL_MEM_READ_ONLY = 1 << 2
CL_MEM_COPY_HOST_PTR = 1 << 5
CL_KERNEL_EXEC_INFO_SVM_PTRS = 0x11B6
CL_KERNEL_NUM_ARGS = 0x1191
CL_KERNEL_WORK_GROUP_SIZE = 0x11B0


class OclError(RuntimeError):
    pass


def _lib():
    name = os.environ.get("OCU_OPENCL_LIBRARY") or ("OpenCL.dll" if sys.platform == "win32" else "libOpenCL.so.1")
    lib = C.WinDLL(name) if sys.platform == "win32" else C.CDLL(name)
    vp = C.c_void_p
    for f in ("clCreateContext", "clCreateCommandQueueWithProperties", "clCreateProgramWithBinary",
              "clCreateProgramWithSource", "clCreateProgramWithIL", "clCreateKernel", "clCreateBuffer"):
        if hasattr(lib, f):
            getattr(lib, f).restype = vp
    lib.clSVMAlloc.restype = vp
    lib.clSVMAlloc.argtypes = [vp, C.c_uint64, C.c_uint, C.c_uint]
    lib.clSVMFree.argtypes = [vp, vp]
    lib.clEnqueueSVMMemcpy.argtypes = [vp, C.c_uint, vp, vp, C.c_size_t, C.c_uint, vp, vp]
    lib.clEnqueueSVMMemFill.argtypes = [vp, vp, vp, C.c_size_t, C.c_size_t, C.c_uint, vp, vp]
    lib.clSetKernelArg.argtypes = [vp, C.c_uint, C.c_size_t, vp]
    lib.clSetKernelArgSVMPointer.argtypes = [vp, C.c_uint, vp]
    lib.clSetKernelExecInfo.argtypes = [vp, C.c_uint, C.c_size_t, vp]
    lib.clEnqueueNDRangeKernel.argtypes = [vp, vp, C.c_uint, vp, vp, vp, C.c_uint, vp, vp]
    lib.clFinish.argtypes = [vp]
    lib.clBuildProgram.argtypes = [vp, C.c_uint, vp, C.c_char_p, vp, vp]
    lib.clGetProgramBuildInfo.argtypes = [vp, vp, C.c_uint, C.c_size_t, vp, vp]
    lib.clGetDeviceInfo.argtypes = [vp, C.c_uint, C.c_size_t, vp, vp]
    lib.clGetPlatformInfo.argtypes = [vp, C.c_uint, C.c_size_t, vp, vp]
    lib.clGetKernelInfo.argtypes = [vp, C.c_uint, C.c_size_t, vp, vp]
    lib.clGetKernelWorkGroupInfo.argtypes = [vp, vp, C.c_uint, C.c_size_t, vp, vp]
    return lib


def _info_str(fn, obj, param) -> str:
    n = C.c_size_t()
    if fn(obj, param, 0, None, C.byref(n)) != 0 or n.value == 0:
        return ""
    buf = C.create_string_buffer(n.value)
    fn(obj, param, n.value, buf, None)
    return buf.value.decode(errors="replace")


def _align(size: int) -> int:
    a = 1
    while a < 16 and size % (a * 2) == 0:
        a *= 2
    return a


class Kernel:
    def __init__(self, module, name, handle, params):
        self.module, self.name, self.k, self.params = module, name, handle, params


class Module:
    def __init__(self, dev, image: bytes):
        self.dev = dev
        cl = dev.cl
        if not image.startswith(b"OCUOCL1\n"):
            raise OclError("not an OCUOCL1 image")
        head, body = image[8:].split(b"\n---\n", 1)
        self.meta = head.decode("latin-1").splitlines()
        lens = {"source": 0, "spirv": 0, "clc": 0}
        for l in self.meta:
            w = l.split()
            if w and w[0] in lens:
                lens[w[0]] = int(w[1])
        err = C.c_int()
        self.form = None
        if dev.il and lens["spirv"]:
            spv = body[lens["source"]:lens["source"] + lens["spirv"]]
            self.prog = cl.clCreateProgramWithIL(dev.ctx, spv, len(spv), C.byref(err))
            self.form, options = "spirv", b""
        else:
            # AMD's OpenCL drivers take neither SPIR-V nor C++ for OpenCL
            # (checked on the 780M): use the AMD backend (HIP) for them.
            raise OclError(f"the image has no form {dev.name} ({dev.arch}) takes")
        if not self.prog or err.value:
            raise OclError(f"creating the program failed ({err.value})")
        r = cl.clBuildProgram(self.prog, 1, dev.dev_arr, options, None, None)
        if r != 0:
            n = C.c_size_t()
            cl.clGetProgramBuildInfo(self.prog, dev.dev, CL_PROGRAM_BUILD_LOG, 0, None, C.byref(n))
            log = C.create_string_buffer(n.value + 1)
            cl.clGetProgramBuildInfo(self.prog, dev.dev, CL_PROGRAM_BUILD_LOG, n.value, log, None)
            raise OclError(f"build failed ({r}):\n{log.value.decode(errors='replace')[-4000:]}")
        # module variables
        self.globals, size = {}, 0
        for l in self.meta:
            w = l.split()
            if w and w[0] == "global":
                off, sz = int(w[2]), int(w[3])
                self.globals[w[1]] = (off, sz)
                size = max(size, off + sz)
        self.glob = None
        if self.globals:
            self.glob = dev.alloc(max(size, 4))
            dev.fill(self.glob, 0, max(size, 4))
            for l in self.meta:
                w = l.split()
                if w and w[0] == "init":
                    data = bytes.fromhex(w[2])
                    dev.copy_in(self.glob + int(w[1]), data)
        self.kernels = {}

    def kernel(self, name: str) -> Kernel:
        if name in self.kernels:
            return self.kernels[name]
        cl, err = self.dev.cl, C.c_int()
        k = cl.clCreateKernel(self.prog, name.encode(), C.byref(err))
        if not k:
            raise OclError(f"no kernel {name} ({err.value})")
        params = []
        for l in self.meta:
            w = l.split()
            if w and w[0] == "param" and w[1] == name:
                params = [int(x) for x in w[2:]]
        nargs = C.c_uint()
        cl.clGetKernelInfo(k, CL_KERNEL_NUM_ARGS, 4, C.byref(nargs), None)
        kk = Kernel(self, name, C.c_void_p(k), params)
        kk.takes_glob = nargs.value == 4
        self.kernels[name] = kk
        return kk

    def global_ptr(self, name):
        off, sz = self.globals[name]
        return self.glob + off, sz


class Device:
    def __init__(self, index: int = 0):
        cl = self.cl = _lib()
        n = C.c_uint()
        cl.clGetPlatformIDs(0, None, C.byref(n))
        plats = (C.c_void_p * n.value)()
        cl.clGetPlatformIDs(n, plats, None)
        devs = []
        for p in plats:
            nd = C.c_uint()
            if cl.clGetDeviceIDs(C.c_void_p(p), CL_DEVICE_TYPE_GPU, 0, None, C.byref(nd)) != 0:
                continue
            arr = (C.c_void_p * nd.value)()
            cl.clGetDeviceIDs(C.c_void_p(p), CL_DEVICE_TYPE_GPU, nd, arr, None)
            devs += [(C.c_void_p(p), C.c_void_p(d)) for d in arr]
        if index >= len(devs):
            raise OclError("no OpenCL GPU")
        self.platform, self.dev = devs[index]
        self.dev_arr = (C.c_void_p * 1)(self.dev.value)
        self.arch = _info_str(cl.clGetDeviceInfo, self.dev, CL_DEVICE_NAME)
        board = _info_str(cl.clGetDeviceInfo, self.dev, CL_DEVICE_BOARD_NAME_AMD)
        self.name = board or self.arch
        self.version = _info_str(cl.clGetDeviceInfo, self.dev, CL_DEVICE_VERSION)
        ext = _info_str(cl.clGetDeviceInfo, self.dev, CL_DEVICE_EXTENSIONS).split()
        self.il = "cl_khr_il_program" in ext
        err = C.c_int()
        self.ctx = C.c_void_p(cl.clCreateContext(None, 1, self.dev_arr, None, None, C.byref(err)))
        self.q = C.c_void_p(cl.clCreateCommandQueueWithProperties(self.ctx, self.dev, None, C.byref(err)))
        self.allocs: dict[int, int] = {}

    # memory ------------------------------------------------------------------------------
    def alloc(self, nbytes: int) -> int:
        raw = self.cl.clSVMAlloc(self.ctx, CL_MEM_READ_WRITE, nbytes + 256, 0)
        if not raw:
            raise OclError("clSVMAlloc failed")
        p = (raw + 255) // 256 * 256
        self.allocs[p] = raw
        return p

    def free(self, p: int):
        self.cl.clFinish(self.q)
        self.cl.clSVMFree(self.ctx, C.c_void_p(self.allocs.pop(p)))

    def copy_in(self, dst: int, data: bytes):
        buf = C.create_string_buffer(data, len(data))
        self._check(self.cl.clEnqueueSVMMemcpy(self.q, 1, C.c_void_p(dst), buf, len(data), 0, None, None), "memcpy")

    def copy_out(self, src: int, nbytes: int) -> bytes:
        buf = C.create_string_buffer(nbytes)
        self._check(self.cl.clEnqueueSVMMemcpy(self.q, 1, buf, C.c_void_p(src), nbytes, 0, None, None), "memcpy")
        return buf.raw

    def fill(self, dst: int, byte: int, nbytes: int):
        pat = C.c_ubyte(byte)
        self._check(self.cl.clEnqueueSVMMemFill(self.q, C.c_void_p(dst), C.byref(pat), 1, nbytes, 0, None, None), "fill")

    def _check(self, r, what):
        if r != 0:
            raise OclError(f"{what} failed ({r})")

    # code ----------------------------------------------------------------------------------
    def load(self, image: bytes) -> Module:
        return Module(self, image)

    def launch(self, k: Kernel, grid, block, args: list[bytes], shared: int = 0):
        cl = self.cl
        grid = tuple(grid) + (1,) * (3 - len(grid))
        block = tuple(block) + (1,) * (3 - len(block))
        packed, offs, at = bytearray(), [], 0
        for a in args:
            al = _align(len(a))
            at = (at + al - 1) // al * al
            packed += b"\0" * (at - len(packed))
            offs.append(at)
            packed += a
            at += len(a)
        packed += b"\0" * 16
        offb = b"".join(o.to_bytes(4, "little") for o in offs + [0])
        err = C.c_int()
        pk = C.create_string_buffer(bytes(packed), len(packed))
        ob = C.create_string_buffer(offb, len(offb))
        argbuf = C.c_void_p(cl.clCreateBuffer(self.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, len(packed), pk, C.byref(err)))
        offbuf = C.c_void_p(cl.clCreateBuffer(self.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, len(offb), ob, C.byref(err)))
        i = 0
        self._check(cl.clSetKernelArg(k.k, i, C.sizeof(C.c_void_p), C.byref(argbuf)), "arg"); i += 1
        self._check(cl.clSetKernelArg(k.k, i, C.sizeof(C.c_void_p), C.byref(offbuf)), "arg"); i += 1
        if k.takes_glob:
            self._check(cl.clSetKernelArgSVMPointer(k.k, i, C.c_void_p(k.module.glob)), "glob arg"); i += 1
        dyn = (shared + 15) // 16 * 16
        self._check(cl.clSetKernelArg(k.k, i, max(dyn, 16), None), "local arg")
        ptrs = (C.c_void_p * max(len(self.allocs), 1))(*self.allocs.keys())
        if self.allocs:
            self._check(cl.clSetKernelExecInfo(k.k, CL_KERNEL_EXEC_INFO_SVM_PTRS, C.sizeof(ptrs), ptrs), "svm ptrs")
        gws = (C.c_size_t * 3)(*(g * b for g, b in zip(grid, block)))
        lws = (C.c_size_t * 3)(*block)
        self._check(cl.clEnqueueNDRangeKernel(self.q, k.k, 3, None, gws, lws, 0, None, None), "launch")
        self._check(cl.clFinish(self.q), "finish")
        cl.clReleaseMemObject(argbuf)
        cl.clReleaseMemObject(offbuf)
