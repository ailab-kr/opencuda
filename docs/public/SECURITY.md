# Security policy

**English** · [한국어](SECURITY.ko.md)

> DRAFT for review.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use the repository's
private vulnerability reporting (the "Security" tab, "Report a vulnerability"). Include the
affected version or commit, the
platform and backend, what you observed, and a way to reproduce it.

We aim to acknowledge a report within [3] working days, give an assessment
within [10] working days, and publish a fix and an advisory within [90] days
of the report. We will credit reporters who want to be credited. We ask for
coordinated disclosure within that window.

## Supported versions

Until the first stable release, only the latest commit of the main branch is
supported. After it, the latest release and the one before it receive
security fixes.

## Scope

In scope:

- memory-safety errors in the runtime, the backends, the libraries and the
  compiler glue (out-of-bounds copies, use after free, integer overflow in
  size calculations, unchecked pointers passed through the API);
- the backend loader and registration code, including run-time loading of
  vendor driver libraries;
- the translators and code generators, where crafted input could make them
  execute or write something other than their output;
- the scoreboard and conformance tools processing untrusted result records.

Out of scope:

- vulnerabilities in a chip vendor's driver or library; report those to the
  vendor (we will help route the report);
- the CPU backend executing arbitrary code that a user passes it: it runs
  kernels with the privileges of the calling process by design;
- denial of service by programs that deliberately never terminate.

## Hardening already in place

Builds are tested with address, undefined-behaviour and thread sanitizers;
CPU-backend kernel stacks have guard pages and overflow detection; the
memory tables of the GPU backends reject out-of-range copies and
mid-allocation frees instead of passing them to the driver.
