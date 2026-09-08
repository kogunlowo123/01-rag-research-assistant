# Dependency vulnerability exceptions

CI fails on any HIGH or CRITICAL dependency vulnerability. Occasionally a
finding has no upstream fix and no viable replacement. Such a finding may be
time-boxed here rather than suppressed silently, and never by disabling the
scanner.

## Process

1. Confirm the finding is real and reachable from this codebase. A
   vulnerability in a code path the application never executes is still
   recorded, but with that fact stated.
2. Add a row below with the identifier, the package, why it cannot be fixed
   now, the compensating control, and a review date no more than 90 days out.
3. Add the identifier to `security/audit-ignores.txt`.
4. Remove both entries as soon as a fixed version is available.

An exception whose review date has passed is treated as a build failure by the
reviewer, not as a rubber stamp.

## Current exceptions

| Identifier | Package | Reachable | Why not fixed | Compensating control | Review by |
| --- | --- | --- | --- | --- | --- |
| _(none)_ | | | | | |

As of the last dependency scan, every package in `uv.lock` resolves without a
HIGH or CRITICAL finding, so this table is empty. That is the intended steady
state.
