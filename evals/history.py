"""
The trend: one line per accepted baseline, committed beside it.

`evals/history.jsonl` is append-only. Every `python -m evals accept` adds one JSON line --
the date, the code it was accepted on, the model and prompt versions, the completion, the
violations, the failure mix and the golden set's hash -- so the failure chart can show the
mix moving from one accepted baseline to the next, and git shows who accepted what. The gate
checks the file ends with the baseline it compares against: a history edited on its own, or
a baseline accepted without its line, fails.
"""

import json
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from app.graph.prompts import PROMPT_VERSIONS
from app.llm import DEFAULT_MODEL
from evals.golden import EVALS_DIR
from evals.scoring import Scoreboard, as_text

HISTORY = EVALS_DIR / "history.jsonl"
# What a line must agree on with the baseline it was written for. The date and the code are context, not measures.
COMPARED = ("cases", "completed", "task_completion", "safety_violations", "unsafe_cases", "failure_mix", "golden_sha256")


def code_version() -> str:
    """The commit the acceptance ran on, or "unknown" outside a git checkout."""
    try:
        found = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=EVALS_DIR.parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return found.stdout.strip() if found.returncode == 0 and found.stdout.strip() else "unknown"


def entry(board: Scoreboard, accepted_on: date | None = None) -> dict[str, Any]:
    """One history line for this scoreboard. The date is UTC, so two machines agree on it."""
    return {
        "accepted_on": (accepted_on or datetime.now(tz=UTC).date()).isoformat(),
        "code": code_version(),
        "model": DEFAULT_MODEL,
        "prompt_versions": dict(PROMPT_VERSIONS),
        "cases": board.cases,
        "completed": board.completed,
        "task_completion": as_text(board.task_completion),
        "safety_violations": board.safety_violations,
        "unsafe_cases": sorted(board.unsafe_cases),
        "failure_mix": dict(board.failure_mix),
        "golden_sha256": board.golden_sha256,
    }


def append(path: Path, board: Scoreboard) -> dict[str, Any]:
    """Add this scoreboard's line to the history and return it."""
    line = entry(board)
    with path.open("a", encoding="utf-8") as history:
        history.write(json.dumps(line, sort_keys=True) + "\n")
    return line


def last(path: Path) -> dict[str, Any] | None:
    """The most recent line, or None for an empty file."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    found = json.loads(lines[-1])
    return found if isinstance(found, dict) else None


def ends_with(path: Path, board: Scoreboard) -> bool:
    """Whether the history's last line was written for a scoreboard measuring the same as `board`."""
    latest = last(path)
    if latest is None:
        return False
    expected = entry(board)
    return all(latest.get(name) == expected[name] for name in COMPARED)
