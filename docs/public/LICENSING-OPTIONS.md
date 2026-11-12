# Licensing options

> DRAFT for review. This is a neutral comparison to help the maintainers
> decide. It is not legal advice; the final choice should be reviewed by
> counsel for the steward organisation. No licence has been added to the
> repository.

## What matters for this project

- Chip vendors (GPU and NPU makers) are expected to contribute backends, and
  to ship the project inside their own products and SDKs. Their legal teams
  need a licence that is familiar and easy to approve.
- The project calls into other projects at run time (vendor drivers) and uses
  LLVM/Clang as an external tool. It currently vendors no third-party source
  (see `docs/third-party.md`).
- The project may later take code or build on LLVM components, whose licence
  is Apache-2.0 with LLVM exceptions.
- The conformance suite is meant to be reused, including by parties that
  compete with each other.

## The three candidates

| | Apache-2.0 | MIT | BSD-3-Clause |
|---|---|---|---|
| Length and familiarity | long; well known to corporate legal teams | very short; the most widely recognised | short; widely recognised |
| Express patent licence from contributors | yes, for each contribution and the work as combined | no express grant (an implied licence is often argued, but is not written down) | no express grant |
| Patent retaliation | yes: a licensee who sues over the work loses the patent licence it was granted | none | none |
| Attribution / notices | keep licence and NOTICE file, mark changes | keep copyright notice and licence text | keep notice and licence text; no use of names for endorsement |
| Trademark | explicitly no trademark licence | silent | no endorsement with contributors' names |
| Contribution terms | section 5: contributions are under the licence unless stated otherwise (a CLA or DCO is still common) | none in text | none in text |
| Combination with GPL-3.0 code | compatible (one direction) | compatible | compatible |
| Combination with GPL-2.0-only code | not compatible | compatible | compatible |
| Combination with LLVM (Apache-2.0 with LLVM exceptions) | same family; code can be exchanged both ways without friction | permissive; LLVM code copied into the project keeps its own licence terms | same as MIT |

## Points specific to a vendor-neutral compatibility layer

**Patents.** A compatibility layer sits near areas where companies hold
patents. Apache-2.0 gives users a written patent licence from each
contributor, which some adopters require and which makes contributions from
many companies safer to combine. The same clause is what some contributors'
counsel review most carefully, because a contributor grants a licence under
its own patents that read on its contribution. MIT and BSD-3-Clause avoid that
discussion by saying nothing, which leaves the question open for both sides.

**Vendor comfort.** Many vendors have standing approval to contribute to and
redistribute Apache-2.0 and MIT projects; BSD-3-Clause is similarly common.
The retaliation clause of Apache-2.0 sometimes triggers extra review at
companies with large patent portfolios. A short licence helps small vendors
that have no legal department.

**Compatibility with LLVM.** All three are permissive and can be combined
with LLVM's Apache-2.0 with LLVM exceptions as long as notices are kept. The
LLVM exceptions let binaries embedding LLVM-derived object code omit the
attribution that Apache-2.0 would otherwise require in binary form. Choosing
plain Apache-2.0 for the project keeps the same legal family; adding the same
exception text would also be possible and would make downstream binary
redistribution lighter. MIT and BSD-3-Clause require their notice in binary
distributions, too.

**Copyleft.** None of the three requires downstream users to publish their
changes. If keeping improvements to the conformance suite or the interface
open is a goal, that has to be handled by governance (for example, trademark
and certification rules for the scoreboard name) rather than by the licence.

**Trademarks.** None of the three grants rights to a project name. Naming the
project, and any rules for using the name together with scoreboard results,
are separate decisions for the steward.

**Dependencies of different licences.** Documentation, test data derived from
other works and any vendored code keep their own licences, which must be
listed in `docs/third-party.md`. Contributor Covenant text (CC BY 4.0) is
referenced rather than copied.

## Variants worth considering

- Apache-2.0 with LLVM exceptions: the closest to LLVM; some adopters will
  recognise the pattern from LLVM-hosted projects.
- Dual licence MIT OR Apache-2.0 (common in some ecosystems): lets each user
  pick, at the cost of two licence files and slightly more explanation.
- Different licences for different parts, for example a more permissive licence
  for the conformance suite and the interface header (to encourage reuse by
  every party) and Apache-2.0 for the rest. This adds clarity requirements
  for contributors.

## Contribution mechanism (independent of the licence)

- Developer Certificate of Origin sign-off (`Signed-off-by`) on every commit:
  light, no paperwork, widely used.
- Contributor licence agreement: gives the steward explicit rights to relicense
  and defend the project; adds friction for individuals and for vendors
  with approval processes.
- Because the project is written from public documentation only, a statement
  of provenance on each contribution is useful under either mechanism.

## Questions for the decision

1. Do vendor contributors need the written patent grant, or does it block
   their approval? (Ask two or three prospective vendors.)
2. Is exchange with LLVM code expected? If yes, prefer the Apache family.
3. Should the conformance suite be licensed differently from the rest?
4. DCO or CLA?
5. Who holds copyright and the name: the steward from day one, or the
   contributors?
