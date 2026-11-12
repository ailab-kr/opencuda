# Third-party components

Everything this repository uses but does not own, with its licence.
Add a row before the first use.

| Component | Used for | Licence | How it is used | Added |
|---|---|---|---|---|
| Clang / LLVM 18.1.3 | Compiling `.cu` sources for each target | Apache-2.0 WITH LLVM-exception | External tool, not vendored | 2026-11-12 |
| CMake | Build | BSD-3-Clause | External tool | 2026-11-12 |
| Python 3.10+ standard library | Conformance harness | PSF | External runtime | 2026-11-12 |

Not used, by rule (see `ALLOWLIST.md`): vendor CUDA SDK headers, libraries,
samples and tools.
