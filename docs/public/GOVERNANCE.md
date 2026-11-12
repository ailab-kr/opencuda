# Governance

> DRAFT for review. Bracketed items (`[...]`) are decisions still to be made.

## 1. Purpose

PROJECT_NAME exists to let CUDA C++ programs run on any chip whose maker
provides a backend, and to measure how well each does so with a public,
vendor-neutral test suite. The project is a shared piece of infrastructure.
No single vendor, including any contributor's employer, controls it.

## 2. Steward

The project is stewarded by a neutral non-profit organisation
[name of the steward, to be decided]. The steward:

- holds the project name, domains and any trademarks, and the copyright
  assignments or licences contributors grant (as the chosen licence requires);
- does not direct technical decisions beyond the process below;
- publishes its finances and the list of organisations that fund the project;
- may not accept funding or in-kind support on terms that give the funder
  influence over conformance results.

## 3. Roles

| Role | Who | Can |
|---|---|---|
| Contributor | anyone | open issues, pull requests, RFC proposals |
| Reviewer | contributors with a record of useful reviews | approve changes in an area |
| Maintainer | appointed by the maintainers (below) | merge, cut releases, vote on RFCs |
| Backend maintainer | a person nominated by a backend's vendor or community | merge changes inside `backends/<name>/` only |
| Steward board | the non-profit's directors | legal, financial, trademark, code of conduct appeals |

Maintainers are appointed by a two-thirds vote of the current maintainers,
based on sustained contribution. No more than [one third] of the maintainers
may be employed by the same organisation; when a change would break this,
the group pauses new appointments from that organisation. A maintainer who
is inactive for twelve months is moved to emeritus status. Maintainers
declare conflicts of interest on any decision that concerns their employer's
backend or product, and abstain from the vote on it.

## 4. Decision making

- Everyday changes: pull request, one maintainer approval, passing checks, and
  no unresolved objection after [3] working days.
- Anything touching the backend interface (`ocu_backend.h`), the result-record
  format, the scoring rules or the tolerances goes through an RFC.
- Disputes that lazy consensus cannot settle are decided by a simple majority
  of maintainers; an RFC needs a two-thirds majority.
- The steward board decides only the matters in the table above.

### Tests and implementation stay apart

A change may touch the conformance test paths or the implementation paths,
not both, and a feature's tests and implementation should be written by
different people. If a test looks wrong, the objection is recorded and
reviewed; it is not edited from the implementation side. This protects the
suite's neutrality across vendors.

## 5. RFC process

1. Open an issue titled `RFC: <summary>` and add a document under `rfcs/`
   using the template: motivation, design, effect on existing backends,
   effect on conformance, alternatives, unresolved questions.
2. Discussion stays open for at least [14] days. Backend maintainers of
   affected backends are asked for comment explicitly.
3. Maintainers vote once; the result and reasoning are recorded in the RFC.
4. An accepted RFC that changes the backend interface in an incompatible way
   increases `OCU_BACKEND_ABI` and keeps the previous version loadable for at
   least [one release].
5. Implementation follows in ordinary pull requests that cite the RFC.

## 6. Backend acceptance criteria (vendor neutral)

A backend is accepted into the main tree when:

1. It implements all required entries of the current ABI, and reports
   unsupported or unfinished features as such, never as success.
2. It builds from public sources and public documentation, with no
   proprietary file needed to build; run-time dependence on a driver or
   library is allowed and must be documented.
3. It passes level B0 (and B1 if it claims runtime support) of the backend
   levels in BACKEND_GUIDE.md, on real hardware, with result records
   attached from a machine others can describe (device, driver, OS).
4. It documents its device-code image format and how the image is produced.
5. It has at least [two] named backend maintainers, at least one of whom is
   reachable through the project's public channels.
6. Its code is under the project's licence, contributed under the same terms
   as all other code.

The same criteria apply to every vendor. The criteria do not rank
vendors, do not require any commercial arrangement, and do not require
hardware donation (though hardware for continuous testing is welcome and is
handled by the steward without influence on the result).

Removal: a backend with no maintainer for twelve months, or failing its
claimed level for two consecutive releases without a plan, moves to an
`attic/` area after notice.

## 7. Conformance results

- Scores come from public suites run in public, reproducible ways. Results
  from private test sets are not published as project results.
- Anyone can submit a result record; submissions are checked by re-running
  where possible, and flagged as unverified otherwise.
- A vendor may comment on its own results; it may not edit them.
- A published result is changed only for a mistake in the suite or in the
  record, with the correction visible in the history.

## 8. Code of conduct, security, licence

See CODE_OF_CONDUCT.md and SECURITY.md. The licence is the Apache License 2.0
(LICENSE at the repository root).

## 9. Changing this document

By RFC, with a two-thirds vote of the maintainers and no objection from the
steward board on legal grounds.
