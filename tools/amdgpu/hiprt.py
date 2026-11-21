"""hiprt: the AMD HIP runtime (amdhip64, shipped with the AMD GPU driver)
over ctypes: the calls backends/amd/README.md maps the backend interface to,
for checking AMD code objects on the GPU before a native backend build can
run on the machine. Only documented HIP runtime functions are called.
"""
from __future__ import annotations

import ctypes as C
import os
import sys

H2D, D2H, D2D = 1, 2, 3
VP = C.c_void_p


class HipError(RuntimeError):
    pass


def _find() -> str:
    explicit = os.environ.get("OCU_HIP_LIBRARY")
    if explicit:
        return explicit
    names = (["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"] if sys.platform == "win32"
             else ["libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"])
    for n in names:
        try:
            (C.WinDLL if sys.platform == "win32" else C.CDLL)(n)
            return n
        except OSError:
            continue
    raise HipError("HIP runtime not found (set OCU_HIP_LIBRARY)")


class Hip:
    def __init__(self, path: str | None = None, device: int = 0):
        self.path = path or _find()
        L = self.lib = (C.WinDLL if sys.platform == "win32" else C.CDLL)(self.path)
        L.hipGetErrorString.restype = C.c_char_p
        L.hipMalloc.argtypes = [C.POINTER(VP), C.c_size_t]
        L.hipFree.argtypes = [VP]
        L.hipMemcpy.argtypes = [VP, VP, C.c_size_t, C.c_int]
        L.hipMemset.argtypes = [VP, C.c_int, C.c_size_t]
        L.hipModuleLoadData.argtypes = [C.POINTER(VP), C.c_char_p]
        L.hipModuleUnload.argtypes = [VP]
        L.hipModuleGetFunction.argtypes = [C.POINTER(VP), VP, C.c_char_p]
        L.hipModuleGetGlobal.argtypes = [C.POINTER(VP), C.POINTER(C.c_size_t), VP, C.c_char_p]
        L.hipModuleLaunchKernel.argtypes = [VP, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint,
                                            C.c_uint, VP, C.POINTER(VP), C.POINTER(VP)]
        L.hipDeviceGetName.argtypes = [C.c_char_p, C.c_int, C.c_int]
        L.hipDeviceGetAttribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_int]
        L.hipRuntimeGetVersion.argtypes = [C.POINTER(C.c_int)]
        L.hipDriverGetVersion.argtypes = [C.POINTER(C.c_int)]
        self.ok(L.hipInit(0), "hipInit")
        n = C.c_int()
        self.ok(L.hipGetDeviceCount(C.byref(n)), "hipGetDeviceCount")
        if n.value <= device:
            raise HipError("no HIP device")
        self.ok(L.hipSetDevice(device), "hipSetDevice")
        name = C.create_string_buffer(256)
        L.hipDeviceGetName(name, 256, device)
        self.name = name.value.decode(errors="replace")
        rv = C.c_int()
        L.hipRuntimeGetVersion(C.byref(rv))
        self.runtime_version = rv.value
        self.device = device

    def ok(self, st, what):
        if st != 0:
            msg = self.lib.hipGetErrorString(st)
            raise HipError(f"{what}: {st} {msg.decode() if msg else ''}")

    def malloc(self, n: int) -> int:
        p = VP()
        self.ok(self.lib.hipMalloc(C.byref(p), n), "hipMalloc")
        return p.value

    def free(self, p: int):
        self.ok(self.lib.hipFree(VP(p)), "hipFree")

    def copy_in(self, dst: int, data: bytes):
        self.ok(self.lib.hipMemcpy(VP(dst), C.c_char_p(data), len(data), H2D), "hipMemcpy H2D")

    def copy_out(self, src: int, n: int) -> bytes:
        buf = C.create_string_buffer(n)
        self.ok(self.lib.hipMemcpy(C.cast(buf, VP), VP(src), n, D2H), "hipMemcpy D2H")
        return buf.raw

    def memset(self, dst: int, byte: int, n: int):
        self.ok(self.lib.hipMemset(VP(dst), byte, n), "hipMemset")

    def load(self, code_object: bytes) -> int:
        m = VP()
        self.ok(self.lib.hipModuleLoadData(C.byref(m), code_object), "hipModuleLoadData")
        return m.value

    def unload(self, m: int):
        self.ok(self.lib.hipModuleUnload(VP(m)), "hipModuleUnload")

    def function(self, m: int, name: str) -> int:
        f = VP()
        self.ok(self.lib.hipModuleGetFunction(C.byref(f), VP(m), name.encode()), f"hipModuleGetFunction {name}")
        return f.value

    def global_(self, m: int, name: str) -> tuple[int, int]:
        p, n = VP(), C.c_size_t()
        self.ok(self.lib.hipModuleGetGlobal(C.byref(p), C.byref(n), VP(m), name.encode()), f"hipModuleGetGlobal {name}")
        return p.value, n.value

    def launch(self, f: int, grid, block, args: list[bytes], shared: int = 0, sync: bool = True):
        """args: each argument value's bytes; the runtime gets one pointer per
        value, the backend interface's own argument form."""
        grid = tuple(grid) + (1,) * (3 - len(grid))
        block = tuple(block) + (1,) * (3 - len(block))
        bufs = [C.create_string_buffer(a, max(len(a), 1)) for a in args]
        ptrs = (VP * max(len(bufs), 1))(*[C.cast(b, VP) for b in bufs])
        self.ok(self.lib.hipModuleLaunchKernel(VP(f), *grid, *block, shared, None, ptrs, None),
                "hipModuleLaunchKernel")
        if sync:
            self.ok(self.lib.hipDeviceSynchronize(), "hipDeviceSynchronize")
