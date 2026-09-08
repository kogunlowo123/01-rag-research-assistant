# Contributing

Thanks for taking the time to contribute. This document describes the local
workflow, the quality bar enforced in CI, and how changes are reviewed.

## Prerequisites

- Python 3.12 (the project pins `>=3.12,<3.13`)
- [uv](https://docs.astral.sh/uv/) 0.10 or newer
- Docker (optional, only needed for container work)

`make` is convenient on Linux and macOS but is not required. `python tasks.py`
is the cross-platform entry point and the source of truth; the `Makefile`
simply delegates to it.

## Getting set up

```bash
uv sync --all-extras --dev     # or: python tasks.py setup
cp .env.example .env
```

Never commit a populated `.env`. `.gitignore` excludes it and CI runs a secret
scan over both the working tree and the full git history.

## Development loop

| Task | Command |
| --- | --- |
| Format | `python tasks.py fmt` |
| Lint | `python tasks.py lint` |
| Type check | `python tasks.py typecheck` |
| Unit tests | `python tasks.py test-unit` |
| Integration tests | `python tasks.py test-integration` |
| Security tests | `python tasks.py test-security` |
| Full suite + coverage gate | `python tasks.py test` |
| Local security scans | `python tasks.py security` |
| Container image | `python tasks.py docker-build` |

Run `python tasks.py --list` for the full list.

## Quality bar

A change is mergeable when all of the following hold:

1. `ruff check` and `ruff format --check` are clean.
2. `mypy` reports no errors.
3. The full test suite passes and line coverage is at least 80%.
4. `bandit` and `pip-audit` report no unresolved findings. If a dependency
   vulnerability has no upstream fix, add a justified, dated entry to
   `security/audit-exceptions.md` and the identifier to
   `security/audit-ignores.txt`.
5. No secret scanner finding, in the tree or in history.

## Tests

Tests are grouped by pytest marker so each layer can run independently:

- `unit` — pure logic, no I/O, no network.
- `integration` — component wiring, real local storage, in-process HTTP client.
- `security` — adversarial cases. A failure here is a security regression.
- `e2e` — the full path through the running application.

New behaviour needs a test at the lowest layer that can express it. Tests that
assert nothing meaningful (`assert True`, smoke calls with no assertions) are
rejected in review.

## Commits and pull requests

- Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`,
  `fix:`, `docs:`, `refactor:`, `test:`, `build:`, `ci:`, `chore:`.
- Keep the subject line under 72 characters and use the imperative mood.
- One logical change per pull request.
- Describe the behaviour change, the risk, and how you verified it.
- Update `CHANGELOG.md` under `## [Unreleased]` for anything user-visible.

## Reporting security issues

Do not open a public issue for a vulnerability. Follow the process in
[SECURITY.md](SECURITY.md).
