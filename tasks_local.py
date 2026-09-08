"""Repository-specific tasks, merged into the generic runner in ``tasks.py``."""

from __future__ import annotations

UV = "uv"


def _run(*args: str) -> list[str]:
    return [UV, "run", *args]


TASKS: dict[str, tuple[str, list[list[str]]]] = {
    "run": (
        "Run the HTTP service on http://127.0.0.1:8000.",
        [_run("rag-assistant", "serve")],
    ),
    "ingest-sample": (
        "Ingest the regression corpus into the local index.",
        [_run("rag-assistant", "ingest", "data/regression/corpus")],
    ),
    "evaluate": (
        "Run the regression dataset and enforce the quality gate.",
        [
            _run(
                "rag-assistant",
                "evaluate",
                "data/regression/dataset.jsonl",
                "--report",
                "var/evaluation-report.json",
            )
        ],
    ),
    "docker-run": (
        "Build and run the container image locally on port 8000.",
        [
            ["docker", "build", "-t", "rag-research-assistant:local", "."],
            [
                "docker",
                "run",
                "--rm",
                "-p",
                "8000:8000",
                "-e",
                "RAG_SECURITY__API_KEYS=acme:local-development-key",
                "rag-research-assistant:local",
            ],
        ],
    ),
    "smoke": (
        "Build the image and run the container smoke test against it.",
        [
            ["docker", "build", "-t", "rag-research-assistant:local", "."],
            ["bash", "scripts/smoke-test.sh", "rag-research-assistant:local"],
        ],
    ),
    "examples": (
        "Run every example script end to end.",
        [
            _run("python", "examples/quickstart.py"),
            _run("python", "examples/injection_demo.py"),
            _run("python", "examples/hybrid_retrieval.py"),
        ],
    ),
}
