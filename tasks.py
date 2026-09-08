#!/usr/bin/env python3
"""Cross-platform developer task runner.

Only the Python standard library is used so this works on a bare checkout
before dependencies are installed, and on Windows hosts where ``make`` is not
available. ``Makefile`` delegates here so both entry points stay in sync.

Usage:
    python tasks.py <task> [extra args passed through to the underlying tool]
    python tasks.py --list
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE = os.environ.get("IMAGE_NAME", ROOT.name)

# Task name -> (help text, list of argv lists executed in order)
Task = tuple[str, list[list[str]]]

UV = "uv"


def _uv(*args: str) -> list[str]:
    return [UV, *args]


def _uv_run(*args: str) -> list[str]:
    return [UV, "run", *args]


TASKS: dict[str, Task] = {
    "setup": (
        "Create the virtual environment and install all dependency groups.",
        [_uv("sync", "--all-extras", "--dev", "--group", "docs")],
    ),
    "lock": (
        "Refresh uv.lock from pyproject.toml.",
        [_uv("lock")],
    ),
    "fmt": (
        "Apply ruff formatting and import ordering fixes.",
        [_uv_run("ruff", "format", "."), _uv_run("ruff", "check", "--fix", ".")],
    ),
    "lint": (
        "Run ruff lint and formatting checks without modifying files.",
        [_uv_run("ruff", "check", "."), _uv_run("ruff", "format", "--check", ".")],
    ),
    "typecheck": (
        "Run mypy.",
        [_uv_run("mypy", ".")],
    ),
    "test": (
        "Run the whole test suite with the coverage gate.",
        [_uv_run("pytest", "--cov", "--cov-report=term-missing", "--cov-fail-under=80")],
    ),
    "test-unit": ("Run unit tests only.", [_uv_run("pytest", "-m", "unit")]),
    "test-integration": (
        "Run integration tests only.",
        [_uv_run("pytest", "-m", "integration")],
    ),
    "test-security": (
        "Run the adversarial/security test suite only.",
        [_uv_run("pytest", "-m", "security")],
    ),
    "security": (
        "Run static analysis and dependency vulnerability scanning locally.",
        [
            _uv_run("bandit", "-c", "pyproject.toml", "-r", "src", "-f", "screen"),
            _uv(
                "export",
                "--no-emit-project",
                "--no-hashes",
                "--format",
                "requirements-txt",
                "--output-file",
                "requirements.audit.txt",
            ),
            # --no-deps audits the locked set exactly as resolved, rather than
            # re-resolving it: uv.lock is the source of truth for what ships.
            [
                UV,
                "tool",
                "run",
                "pip-audit",
                "--strict",
                "--no-deps",
                "--requirement",
                "requirements.audit.txt",
            ],
        ],
    ),
    "build": ("Build the wheel and sdist.", [_uv("build")]),
    "site": (
        "Build the documentation site into _site/ and check its internal links.",
        [_uv_run("--group", "docs", "python", "scripts/build_site.py", "--output", "_site")],
    ),
    "docker-build": (
        "Build the container image.",
        [["docker", "build", "-t", f"{IMAGE}:local", "."]],
    ),
    "clean": ("Remove build and tool caches.", []),
}

# Tasks that a repository may add or override are declared in tasks_local.py.
_local = ROOT / "tasks_local.py"
if _local.exists():
    import importlib.util

    _spec = importlib.util.spec_from_file_location("tasks_local", _local)
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        TASKS.update(getattr(_mod, "TASKS", {}))

CLEAN_PATHS = (
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    ".coverage",
    "coverage.xml",
    "requirements.audit.txt",
    "_site",
)


def _clean() -> int:
    for name in CLEAN_PATHS:
        target = ROOT / name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()
    for cache in ROOT.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    for egg in ROOT.glob("*.egg-info"):
        shutil.rmtree(egg, ignore_errors=True)
    print("removed build and tool caches")
    return 0


def _require_uv() -> None:
    if shutil.which(UV) is None:
        sys.exit(
            "uv is required but was not found on PATH.\n"
            "Install it from https://docs.astral.sh/uv/getting-started/installation/"
        )


def run_task(name: str, passthrough: Sequence[str]) -> int:
    if name == "clean":
        return _clean()
    if name not in TASKS:
        known = ", ".join(sorted(TASKS))
        sys.exit(f"unknown task {name!r}. Known tasks: {known}")
    _, commands = TASKS[name]
    if commands and commands[0][0] == UV:
        _require_uv()
    for index, command in enumerate(commands):
        argv = list(command)
        if passthrough and index == len(commands) - 1:
            argv.extend(passthrough)
        print(f"$ {' '.join(argv)}", flush=True)
        result = subprocess.run(argv, cwd=ROOT, check=False)  # noqa: S603
        if result.returncode != 0:
            return result.returncode
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="tasks.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task", nargs="?", help="task to run")
    parser.add_argument("--list", action="store_true", help="list available tasks")
    args, passthrough = parser.parse_known_args()

    if args.list or not args.task:
        width = max(len(name) for name in TASKS)
        for name in sorted(TASKS):
            print(f"  {name:<{width}}  {TASKS[name][0]}")
        return 0
    return run_task(args.task, passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
