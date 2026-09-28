#!/usr/bin/env python
"""Subprocess worker for one task-specific lower-level XML optimization."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bilevel.serialization import jsonable
from bilevel.runtime import (
    load_task_from_context,
    run_low_level_optimization,
)


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Avoid empty/partial low-level result JSONs if the process is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(str(tmp_path), str(path))


try:
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
except (AttributeError, ValueError):
    pass


class _SafeStdout:
    def __init__(self, wrapped):
        self._wrapped = wrapped
        self._broken = False

    def write(self, data):
        if self._broken:
            return 0
        try:
            return self._wrapped.write(data)
        except BrokenPipeError:
            self._broken = True
            try:
                devnull = open(os.devnull, "w", encoding="utf-8")
                sys.stdout = devnull
                sys.stderr = devnull
            except Exception:
                pass
            return 0

    def flush(self):
        if self._broken:
            return None
        try:
            return self._wrapped.flush()
        except BrokenPipeError:
            self._broken = True
            return None

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


sys.stdout = _SafeStdout(sys.stdout)
sys.stderr = _SafeStdout(sys.stderr)


def _configure_native_threads() -> dict[str, int | str]:
    raw = os.environ.get("BILEVEL_NUMERIC_THREADS") or os.environ.get("OMP_NUM_THREADS") or "1"
    try:
        num_threads = max(1, int(raw))
    except ValueError:
        num_threads = 1
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(num_threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    torch_threads = "unavailable"
    torch_interop_threads = "unavailable"
    try:
        import torch

        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(1)
        torch_threads = int(torch.get_num_threads())
        torch_interop_threads = int(torch.get_num_interop_threads())
    except Exception as exc:
        torch_threads = f"error:{exc}"
        torch_interop_threads = f"error:{exc}"
    return {
        "numeric_threads": num_threads,
        "torch_threads": torch_threads,
        "torch_interop_threads": torch_interop_threads,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one lower-level XML optimization")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--result-json", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    thread_config = _configure_native_threads()
    request_path = Path(args.request_json)
    result_path = Path(args.result_json)
    request: dict[str, Any] = json.loads(request_path.read_text(encoding="utf-8"))

    try:
        context = dict(request.get("context", {}))
        task = load_task_from_context(str(request["task_name"]), context)
        result = run_low_level_optimization(
            task,
            str(request["xml_path"]),
            context,
        )
        if isinstance(result, dict):
            result["thread_config"] = thread_config
        _write_json_atomic(result_path, jsonable(result))
        return 0
    except Exception as exc:
        _write_json_atomic(
            result_path,
            {
                "score": "inf",
                "loss": "inf",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "thread_config": thread_config,
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
