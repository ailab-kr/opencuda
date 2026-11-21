"""Kernel and variable symbols of an amdgcn code object (ELF64), read with
the standard library only: what hipModuleGetFunction / hipModuleGetGlobal
can look up."""
from __future__ import annotations

import struct


def symbols(elf: bytes) -> list[tuple[str, int, int]]:
    """(name, type, size) of every defined global symbol. type: 1 object, 2 function."""
    if elf[:4] != b"\x7fELF":
        raise ValueError("not ELF")
    shoff, = struct.unpack_from("<Q", elf, 0x28)
    shentsize, shnum, _ = struct.unpack_from("<HHH", elf, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", elf, shoff + i * shentsize) for i in range(shnum)]
    out = []
    for s in secs:
        if s[1] not in (2, 11):          # SHT_SYMTAB, SHT_DYNSYM
            continue
        strtab = secs[s[6]]
        for off in range(s[4], s[4] + s[5], 24):
            name, info, _other, shndx, _value, size = struct.unpack_from("<IBBHQQ", elf, off)
            if shndx == 0 or (info >> 4) != 1:   # undefined, or not global
                continue
            start = strtab[4] + name
            n = elf[start:elf.index(b"\0", start)].decode()
            if n and (n, info & 15, size) not in out:
                out.append((n, info & 15, size))
    return out


def _msgpack(b: bytes, i: int = 0):
    """A minimal MessagePack reader (maps, arrays, strings, integers,
    booleans): the format of the code object's AMDGPU metadata note."""
    t = b[i]
    if t <= 0x7f:
        return t, i + 1
    if 0x80 <= t <= 0x8f or t in (0xde, 0xdf):
        n, i = (t & 15, i + 1) if t <= 0x8f else ((struct.unpack_from(">H", b, i + 1)[0], i + 3) if t == 0xde
                                                   else (struct.unpack_from(">I", b, i + 1)[0], i + 5))
        d = {}
        for _ in range(n):
            k, i = _msgpack(b, i)
            v, i = _msgpack(b, i)
            d[k] = v
        return d, i
    if 0x90 <= t <= 0x9f or t in (0xdc, 0xdd):
        n, i = (t & 15, i + 1) if t <= 0x9f else ((struct.unpack_from(">H", b, i + 1)[0], i + 3) if t == 0xdc
                                                   else (struct.unpack_from(">I", b, i + 1)[0], i + 5))
        a = []
        for _ in range(n):
            v, i = _msgpack(b, i)
            a.append(v)
        return a, i
    if 0xa0 <= t <= 0xbf or t in (0xd9, 0xda, 0xdb):
        n, i = (t & 31, i + 1) if t <= 0xbf else {0xd9: (b[i + 1], i + 2),
                                                 0xda: (struct.unpack_from(">H", b, i + 1)[0], i + 3),
                                                 0xdb: (struct.unpack_from(">I", b, i + 1)[0], i + 5)}[t]
        return b[i:i + n].decode(), i + n
    if t in (0xc2, 0xc3):
        return t == 0xc3, i + 1
    if t == 0xc0:
        return None, i + 1
    fmt = {0xcc: ">B", 0xcd: ">H", 0xce: ">I", 0xcf: ">Q", 0xd0: ">b", 0xd1: ">h", 0xd2: ">i", 0xd3: ">q"}.get(t)
    if fmt:
        return struct.unpack_from(fmt, b, i + 1)[0], i + 1 + struct.calcsize(fmt)
    if t >= 0xe0:
        return t - 256, i + 1
    raise ValueError(f"msgpack type {t:#x}")


def metadata(elf: bytes) -> dict:
    """The AMDGPU metadata note (NT_AMDGPU_METADATA = 32, owner "AMDGPU")."""
    shoff, = struct.unpack_from("<Q", elf, 0x28)
    shentsize, shnum, _ = struct.unpack_from("<HHH", elf, 0x3A)
    for k in range(shnum):
        s = struct.unpack_from("<IIQQQQIIQQ", elf, shoff + k * shentsize)
        if s[1] != 7:                       # SHT_NOTE
            continue
        off, end = s[4], s[4] + s[5]
        while off + 12 <= end:
            namesz, descsz, ntype = struct.unpack_from("<III", elf, off)
            name = elf[off + 12:off + 12 + namesz].rstrip(b"\0")
            desc = off + 12 + (namesz + 3) // 4 * 4
            if name == b"AMDGPU" and ntype == 32:
                return _msgpack(elf, desc)[0]
            off = desc + (descsz + 3) // 4 * 4
    return {}


def kernel_params(elf: bytes) -> dict[str, list[int]]:
    """Kernel name -> sizes of the parameters the source declares (the
    hidden arguments the compiler adds are left out)."""
    out = {}
    for k in metadata(elf).get("amdhsa.kernels", []):
        args = [a for a in k.get(".args", []) if not str(a.get(".value_kind", "")).startswith("hidden_")]
        out[k[".name"]] = [a[".size"] for a in args]
    return out


def kernels(elf: bytes) -> list[str]:
    """Kernel entry points: functions with a matching '<name>.kd' descriptor."""
    syms = symbols(elf)
    kd = {n[:-3] for n, _, _ in syms if n.endswith(".kd")}
    return [n for n, t, _ in syms if t == 2 and n in kd]
