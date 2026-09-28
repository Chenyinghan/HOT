#!/usr/bin/env python
"""Repo-root entrypoint for the Bilevel BASS/XML search."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _configure_parent_native_threads() -> None:
    numeric_threads = os.environ.get("BILEVEL_PARENT_NUMERIC_THREADS", "1")
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, numeric_threads)
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")


_configure_parent_native_threads()


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _prepend_default_args(argv: list[str]) -> list[str]:
    """Select the canonical sweep config unless explicitly overridden."""

    defaults = {
        "--task-json": "tasks/sweep_balls/config.json",
    }
    out: list[str] = []
    for flag, value in defaults.items():
        if flag not in argv and not any(item.startswith(flag + "=") for item in argv):
            out.extend([flag, value])
    out.extend(argv)
    return out


def main(argv: list[str] | None = None) -> int:
    # Keep relative paths stable even when the script is launched elsewhere.
    os.chdir(str(REPO_ROOT))

    from bilevel.search import main as search_main

    old_argv = sys.argv
    try:
        forwarded = _prepend_default_args(list(sys.argv[1:] if argv is None else argv))
        sys.argv = [str(REPO_ROOT / "run_bilevel_search.py")] + forwarded
        return int(search_main())
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
