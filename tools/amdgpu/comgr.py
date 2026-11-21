"""comgr: AMD's code object manager (amd_comgr, shipped with the AMD GPU
driver) driven over ctypes: device code compiled for an AMD GPU without a
Clang executable on the machine.

The driver's OpenCL and HIP runtimes compile kernels through this library;
it embeds AMD's Clang/LLVM (driver 3686.0: AMD Clang 24). Calling it from
Python lets device code be compiled where no compiler binary may run (on
Windows with Smart App Control, binaries built or downloaded without a
signature are blocked; the driver's DLLs are signed).

Only the public C API described in AMD's amd_comgr documentation is used:
data sets, action info (ISA name, language, options, logging) and actions.

    from comgr import Comgr
    cg = Comgr()
    co = cg.hip_to_executable(source_text, "gfx1103", ["-O3"], includes={"k.h": b"..."})
"""
from __future__ import annotations

import ctypes as C
import os
import sys

# data kinds
SOURCE, INCLUDE, LOG, BC, RELOCATABLE, EXECUTABLE = 0x1, 0x2, 0x5, 0x6, 0x7, 0x8
# languages
LANG_NONE, LANG_OPENCL_1_2, LANG_OPENCL_2_0, LANG_HIP, LANG_LLVM_IR = 0, 1, 2, 3, 4
# actions. comgr 3.x numbers them differently from the 2.x headers (actions
# were removed and the rest renumbered); these are the 3.x values, found by
# running each number on a HIP source with the driver's amd_comgr_3.dll
# (API 3.5) and looking at what it produced:
#   0x0 source -> preprocessed source      0x2 source -> bitcode
#   0x4 bitcode -> relocatable (codegen)   0x6 relocatable -> relocatable (ld.lld -r)
#   0x7 relocatable -> executable          0xC source -> bitcode with device libraries
#   0xD source -> relocatable              0x14 source -> SPIR-V
ACT_SOURCE_TO_PREPROCESSOR = 0x0
ACT_COMPILE_SOURCE_TO_BC = 0x2
ACT_CODEGEN_BC_TO_RELOCATABLE = 0x4
ACT_LINK_RELOCATABLE_TO_RELOCATABLE = 0x6
ACT_LINK_RELOCATABLE_TO_EXECUTABLE = 0x7
ACT_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC = 0xC
ACT_COMPILE_SOURCE_TO_RELOCATABLE = 0xD

H = C.c_uint64  # every comgr handle is a struct holding one uint64


class ComgrError(RuntimeError):
    pass


def _find() -> str:
    explicit = os.environ.get("OCU_AMD_COMGR")
    if explicit:
        return explicit
    names = (["amd_comgr_3.dll", "amd_comgr_2.dll", "amd_comgr.dll"] if sys.platform == "win32"
             else ["libamd_comgr.so.3", "libamd_comgr.so.2"])
    for n in names:
        try:
            (C.WinDLL if sys.platform == "win32" else C.CDLL)(n)
            return n
        except OSError:
            continue
    raise ComgrError("amd_comgr not found (set OCU_AMD_COMGR)")


class Comgr:
    def __init__(self, path: str | None = None):
        self.path = path or _find()
        self.lib = (C.WinDLL if sys.platform == "win32" else C.CDLL)(self.path)
        maj, mnr = C.c_size_t(), C.c_size_t()
        self.lib.amd_comgr_get_version(C.byref(maj), C.byref(mnr))
        self.version = (maj.value, mnr.value)
        L = self.lib
        L.amd_comgr_set_data.argtypes = [H, C.c_size_t, C.c_char_p]
        L.amd_comgr_set_data_name.argtypes = [H, C.c_char_p]
        L.amd_comgr_data_set_add.argtypes = [H, H]
        L.amd_comgr_action_info_set_isa_name.argtypes = [H, C.c_char_p]
        L.amd_comgr_action_info_set_language.argtypes = [H, C.c_int]
        L.amd_comgr_action_info_set_option_list.argtypes = [H, C.POINTER(C.c_char_p), C.c_size_t]
        L.amd_comgr_action_info_set_logging.argtypes = [H, C.c_bool]
        L.amd_comgr_do_action.argtypes = [C.c_int, H, H, H]
        L.amd_comgr_action_data_count.argtypes = [H, C.c_int, C.POINTER(C.c_size_t)]
        L.amd_comgr_action_data_get_data.argtypes = [H, C.c_int, C.c_size_t, C.POINTER(H)]
        L.amd_comgr_get_data.argtypes = [H, C.POINTER(C.c_size_t), C.c_char_p]
        L.amd_comgr_release_data.argtypes = [H]
        L.amd_comgr_destroy_data_set.argtypes = [H]
        L.amd_comgr_destroy_action_info.argtypes = [H]

    # ---- small helpers ----------------------------------------------------------------
    def _ok(self, st, what):
        if st != 0:
            raise ComgrError(f"{what}: comgr status {st}")

    def _data(self, kind, name: str, data: bytes) -> int:
        h = H()
        self._ok(self.lib.amd_comgr_create_data(kind, C.byref(h)), "create_data")
        self._ok(self.lib.amd_comgr_set_data(h, len(data), data), "set_data")
        self._ok(self.lib.amd_comgr_set_data_name(h, name.encode()), "set_data_name")
        return h.value

    def _set(self, items=()) -> int:
        s = H()
        self._ok(self.lib.amd_comgr_create_data_set(C.byref(s)), "create_data_set")
        for kind, name, data in items:
            d = self._data(kind, name, data)
            self._ok(self.lib.amd_comgr_data_set_add(s, d), "data_set_add")
            self.lib.amd_comgr_release_data(d)
        return s.value

    def _get(self, dset, kind) -> list[bytes]:
        n = C.c_size_t()
        self.lib.amd_comgr_action_data_count(dset, kind, C.byref(n))
        out = []
        for i in range(n.value):
            d = H()
            self.lib.amd_comgr_action_data_get_data(dset, kind, i, C.byref(d))
            sz = C.c_size_t()
            self.lib.amd_comgr_get_data(d, C.byref(sz), None)
            buf = C.create_string_buffer(sz.value)
            self.lib.amd_comgr_get_data(d, C.byref(sz), buf)
            out.append(buf.raw[:sz.value])
            self.lib.amd_comgr_release_data(d)
        return out

    def action(self, kind, inputs: int, isa: str, lang: int, options: list[str], want: int) -> tuple[list[bytes], str]:
        info = H()
        self._ok(self.lib.amd_comgr_create_action_info(C.byref(info)), "create_action_info")
        try:
            self._ok(self.lib.amd_comgr_action_info_set_isa_name(info, isa.encode()), "set_isa_name")
            if lang:
                self._ok(self.lib.amd_comgr_action_info_set_language(info, lang), "set_language")
            arr = (C.c_char_p * max(len(options), 1))(*[o.encode() for o in options])
            self._ok(self.lib.amd_comgr_action_info_set_option_list(info, arr, len(options)), "set_option_list")
            self.lib.amd_comgr_action_info_set_logging(info, True)
            result = self._set()
            st = self.lib.amd_comgr_do_action(kind, info, inputs, result)
            log = b"\n".join(self._get(result, LOG)).decode(errors="replace")
            if st != 0:
                self.lib.amd_comgr_destroy_data_set(result)
                raise ComgrError(f"comgr action {kind:#x} failed (status {st}):\n{log[-6000:]}")
            got = self._get(result, want)
            self.lib.amd_comgr_destroy_data_set(result)
            return got, log
        finally:
            self.lib.amd_comgr_destroy_action_info(info)

    # ---- pipelines ---------------------------------------------------------------------
    def hip_to_executable(self, source: str | bytes, arch: str, options: list[str] | None = None,
                          includes: dict[str, bytes] | None = None, name: str = "module.hip") -> bytes:
        """HIP-mode device compile of one source file to a loadable code object
        (ELF executable) for amdgcn-amd-amdhsa--<arch>."""
        isa = f"amdgcn-amd-amdhsa--{arch}"
        src = source.encode("latin-1") if isinstance(source, str) else source
        items = [(SOURCE, name, src)] + [(INCLUDE, n, d) for n, d in (includes or {}).items()]
        inp = self._set(items)
        try:
            (rel,), _ = self.action(ACT_COMPILE_SOURCE_TO_RELOCATABLE, inp, isa, LANG_HIP, options or [],
                                    RELOCATABLE)
        finally:
            self.lib.amd_comgr_destroy_data_set(inp)
        rset = self._set([(RELOCATABLE, "module.o", rel)])
        try:
            (exe,), _ = self.action(ACT_LINK_RELOCATABLE_TO_EXECUTABLE, rset, isa, LANG_NONE, [], EXECUTABLE)
        finally:
            self.lib.amd_comgr_destroy_data_set(rset)
        return exe

    def preprocess(self, source: str, arch: str, options: list[str]) -> str:
        inp = self._set([(SOURCE, "module.hip", source.encode("latin-1"))])
        try:
            (out,), _ = self.action(ACT_SOURCE_TO_PREPROCESSOR, inp, f"amdgcn-amd-amdhsa--{arch}", LANG_HIP,
                                    options, SOURCE)
        finally:
            self.lib.amd_comgr_destroy_data_set(inp)
        return out.decode("latin-1")
