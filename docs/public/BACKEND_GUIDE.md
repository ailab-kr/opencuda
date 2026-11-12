# Backend guide

> DRAFT for review. Names with the `ocu` prefix are working names and will
> change with the public project name; the structure will not.

This guide is for chip vendors and community members who want to run
unmodified CUDA C++ programs on a new chip, GPU or NPU. A backend is one C
file set that fills in one function table. Nothing above the table is chip
specific, so a new backend does not change the runtime, the libraries or the
conformance suite.

## 1. The shape of a backend

```
CUDA C++ source and build files (unmodified)
        |
   compiler glue      runtime API (CUDA-named)      libraries (BLAS, ...)
        |                    |                            |
        +--------------------+----------------------------+
                             |
                  backends/interface/ocu_backend.h      <- the only chip boundary
                             |
        cpu | nvidia | metal | your backend
```

Rules that keep the boundary clean:

- A backend may use the interface header and its own chip's libraries or
  drivers. It may not use another backend, and nothing above the interface
  may branch on a chip name.
- The conformance suite uses only what a user program sees. It never includes
  internal headers.
- If a new backend seems to need a change above the interface, the interface
  has a gap. Propose the change through the RFC process (GOVERNANCE.md); an
  incompatible change bumps `OCU_BACKEND_ABI`.

## 2. The ABI: one header

The ABI is `backends/interface/ocu_backend.h` (self-contained C11, current
`OCU_BACKEND_ABI` is 2). A backend returns a pointer to a filled-in
`ocu_backend` table and registers it (`ocu_register_backend`, or a line in
the built-in list in `runtime/core/registry.c`). Registration with a
different ABI number is refused.

### Required entries

| Group | Entries | Contract in one line |
|---|---|---|
| identity | `abi`, `name` | `name` is lower case and stable (`"cpu"`, `"metal"`) |
| device | `dev_count`, `dev_info`, `dev_open`, `dev_close` | report the real warp/wavefront size (32 or 64), real limits, real memory |
| memory | `mem_alloc`, `mem_free`, `mem_copy`, `mem_set`, `mem_info`, `host_alloc_pinned`, `host_free_pinned` | device pointers are plain addresses: `ptr + n` is valid inside the allocation. A `NULL` queue means run synchronously. Free waits for queued work |
| queues | `queue_create`, `queue_destroy`, `queue_sync`, `queue_query` | work in one queue runs in submission order; an error raised by queued work is reported by the next sync or query, then cleared |
| code | `module_load`, `module_unload`, `module_get_kernel` | the image format is private to the backend (PTX text, MSL source, a function table, a vendor binary) |
| execution | `kernel_launch` | copies the argument values before returning; arguments total at most `OCU_MAX_ARG_BYTES` (32768) |
| markers | `marker_create`, `marker_destroy`, `marker_record`, `marker_wait`, `marker_query`, `marker_elapsed_ms` | a never-recorded marker is complete; wait waits for the record current at the call |

### Optional entries

Optional entries may be `NULL`; the runtime then returns "not supported" for
the API that needs them, or falls back to a portable path.

| Entry | Purpose | Without it |
|---|---|---|
| `get_provider` | returns a tuned library table (for example the BLAS provider in `ocu_blas_provider.h`) | the library layer uses its portable implementation |
| `module_get_global` | address and size of a device-code variable (`__device__`, `__constant__`) | symbol copies report "not supported" |
| `kernel_arg_sizes` | parameter count and sizes, for launches that hold only an array of argument pointers | such launches report "not supported" |
| `mem_alloc_shared`, `host_register`, `host_unregister` | memory that host and device address identically | managed memory and host registration are offered only where `supports_unified_memory` is set |

### Status codes

Return `OCU_ERR_NOT_SUPPORTED` when the chip cannot do something and
`OCU_ERR_NOT_IMPLEMENTED` when the backend does not do it yet. The scoreboard
shows them differently. Never report success for work that was not done.

### Concurrency and lifetime

Backend functions can be called from several threads at once. Calling
`kernel_launch`, `queue_sync`, `queue_destroy`, `marker_wait` or a free from
inside a running kernel returns `OCU_ERR_NOT_SUPPORTED`. Section 6 of
`ARCHITECTURE.md` lists the state rules.

## 3. Device code

The interface does not say how device code is produced; it says how it is
loaded. Each backend documents its image format. Existing examples:

| Backend | Image | Produced by |
|---|---|---|
| cpu | table of host functions | the CPU path rewrites launch syntax and compiles as C++ |
| nvidia | PTX text | an open-source Clang in CUDA mode targeting NVPTX; math functions come from the project's own device math header |
| metal | Metal Shading Language source | a translator that reads the Clang device-pass syntax tree and rewrites only what MSL needs (kernel heads, address spaces, shared memory, built-in variables) |

New GPU backends usually reuse the open-source compiler front end and add a
device-code generator or a translator for their ISA or shading language.

## 4. Conformance levels

A backend declares which level it passes. Levels are cumulative.

| Level | Evidence required |
|---|---|
| B0 interface | the backend tests (`tests/backend`) pass: memory, queues, markers, launch, error cases, sanitizer builds where the platform has them |
| B1 runtime | the runtime test programs (`tests/runtime`, ordinary CUDA C++ programs) build unmodified and pass on the backend |
| B2 language | device-code semantics: barriers, shared memory, atomics, warp shuffles and votes, math functions within tolerance (L3) |
| B3 libraries | L4 suites within tolerance (BLAS today), with a provider optional |
| B4 applications | L5: public programs (Rodinia today) produce the same result as an independent reference |

Report a result with the scoreboard tool:

```sh
ctest --test-dir build --output-on-failure > ctest.txt
python3 tools/scoreboard.py build/tests/blas_<backend>_result.json \
    --ctest <backend>=ctest.txt --rodinia <backend>=rodinia.txt --out-dir scoreboard/
```

The scoreboard lists skipped and unsupported tests; they are never counted as
passed. Today executable evidence exists for B0, B1, B3 (BLAS) and B4
(Rodinia); the L1 to L3 suites are being written, and a backend's B2 claim is
backed by the runtime and language test programs until they exist.

Speed is reported as a median and a slow-10 percent value. A backend is not
ranked by speed; it is described by it.

## 5. Steps to add a backend

1. Create `backends/<name>/` and implement the table. Start from the CPU
   backend for structure; use `OCU_ERR_NOT_IMPLEMENTED` for what is missing.
2. Add a build option in `backends/CMakeLists.txt` and one registration line
   in `runtime/core/registry.c`.
3. Document the device-code image format and how it is produced.
4. Add backend tests under `tests/backend` that skip cleanly when the device
   is absent.
5. Run the suites on real hardware and publish the result records.
6. Open an RFC if the interface needs a new entry.

Acceptance criteria are in GOVERNANCE.md.

## 6. Library and operator-level backends for NPUs (planned)

Status: planned. Nothing in this section exists in the tree yet.

Some accelerators have no SIMT cores and cannot execute arbitrary CUDA C++
kernels. They can still serve the large share of CUDA programs whose time is
spent in a few library calls (matrix multiply, convolution, reductions,
attention). The planned approach:

- The backend implements the memory, queue and marker entries as usual, and
  returns `OCU_ERR_NOT_SUPPORTED` from `module_load` and `kernel_launch` for
  arbitrary device code.
- It exposes its operators through `get_provider(<library>)`. The provider
  tables are defined per library next to the interface (BLAS exists; further
  tables, for example convolution and normalisation, would be added by RFC).
- Conformance would be reported per library level only (B3 without B2).
  Programs that launch their own kernels are reported as "not supported" at
  L1 and L3, not hidden.
- A graph or operator-fusion entry is expected to be needed. That is an
  interface extension and goes through an RFC before any code.

Vendors planning such a backend are encouraged to open an RFC early so the
provider tables fit their operator sets.

## 7. Contribution rules specific to backends

- Use public documentation only. Do not copy or transcribe another vendor's
  headers, examples or source; do not disassemble or trace vendor binaries.
  Record the public document a change is based on in the commit message.
- Vendor engineers contributing code for their own chip should say so in the
  pull request; the same review and test rules apply to everyone.
- Result records produced on private test sets are not accepted as evidence;
  use the public suites.
