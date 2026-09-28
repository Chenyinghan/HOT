#!/usr/bin/env python3
"""Warm-start shape/action refinement for an existing tool structure."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEME_FLAGS = {"success-constrained-target-shell": ("--optimizer", "success_constrained_target_shell")}


def _unique(paths: Iterable[Path]) -> List[Path]:
    result = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def resolve_structure(value: str, search_roots: Iterable[Path]) -> Path:
    direct = Path(value).expanduser()
    if direct.is_file():
        return direct.resolve()
    token = Path(value).stem
    matches = []
    for root in search_roots:
        root = root.expanduser()
        if not root.is_dir():
            continue
        matches.extend(root.glob(f"cache/{token}.xml"))
        matches.extend(root.glob(f"**/cache/{token}.xml"))
    matches = _unique(path for path in matches if path.is_file())
    if not matches:
        raise FileNotFoundError(
            f"could not resolve structure {value!r}; pass an XML path or add "
            "--search-root containing cache/<hash>.xml"
        )
    if len(matches) != 1:
        listing = "\n  ".join(str(path) for path in matches)
        raise ValueError(
            f"structure hash {value!r} is ambiguous; matches:\n  {listing}"
        )
    return matches[0]


def resolve_action_source(
    explicit: Optional[str],
    structure_value: str,
    search_roots: Iterable[Path],
) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"action source does not exist: {path}")
        return path.resolve()
    token = Path(structure_value).stem
    matches = []
    for root in search_roots:
        root = root.expanduser()
        if not root.is_dir():
            continue
        matches.extend(root.glob(f"replay/{token}"))
        matches.extend(root.glob(f"**/replay/{token}"))
    matches = _unique(
        path for path in matches
        if path.is_dir()
        and any(
            (path / name).is_file()
            for name in (
                "params.npy",
                "finalized_state.npz",
                "low_level_result.json",
                "params_checkpoint.npy",
            )
        )
    )
    if not matches:
        raise FileNotFoundError(
            f"could not locate optimized action for {token!r}; pass "
            "--action-source with its rollout directory or artifact"
        )
    if len(matches) != 1:
        listing = "\n  ".join(str(path) for path in matches)
        raise ValueError(
            f"optimized action for {token!r} is ambiguous; pass "
            f"--action-source explicitly. Matches:\n  {listing}"
        )
    return matches[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve an BASS structure and its optimized action, then start a "
            "new morphology/action optimization."
        )
    )
    parser.add_argument("--task", required=True, help="run_main task name")
    parser.add_argument(
        "--structure",
        required=True,
        help="XML path or BASS XML/run hash",
    )
    parser.add_argument(
        "--scheme",
        default="success-constrained-target-shell",
        choices=tuple(SCHEME_FLAGS),
        help=(
            "Stage-2 optimizer scheme. Defaults to the shared "
            "success-preserving target-shell morphology strategy."
        ),
    )
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help=(
            "Artifact root searched for cache/<hash>.xml and replay/<hash>. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--action-source",
        help=(
            "Explicit rollout directory or params/finalized/result artifact. "
            "Required for a direct XML unless a matching replay hash exists."
        ),
    )
    parser.add_argument("--save-dir", help="Output directory for the new solve")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and print/write the command without executing it",
    )
    parser.add_argument(
        "run_main_args",
        nargs=argparse.REMAINDER,
        help="Additional run_main arguments after --",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    search_roots = [
        Path(value) for value in args.search_root
    ] or [REPO_ROOT / "workspace" / "bilevel"]
    structure = resolve_structure(args.structure, search_roots)
    action_source = resolve_action_source(
        args.action_source,
        args.structure,
        search_roots,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = (
        Path(args.save_dir).expanduser()
        if args.save_dir
        else REPO_ROOT
        / "workspace"
        / "bilevel"
        / "reoptimization"
        / f"{structure.stem}_{args.scheme}_{timestamp}"
    ).resolve()
    extra = list(args.run_main_args)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    forbidden = {
        "--task",
        "--model-xml",
        "--save-dir",
        "--initial-action",
        "--optimizer",
        "--design-optim",
        "--no-design-optim",
    }
    conflicts = sorted(
        token.split("=", 1)[0]
        for token in extra
        if token.split("=", 1)[0] in forbidden
    )
    if conflicts:
        raise ValueError(
            "managed run_main arguments cannot be overridden after --: "
            + ", ".join(conflicts)
        )
    command = [
        sys.executable,
        str(REPO_ROOT / "run_main.py"),
        "--task",
        args.task,
        "--model-xml",
        str(structure),
        "--save-dir",
        str(save_dir),
        "--initial-action",
        str(action_source),
        "--design-optim",
        *SCHEME_FLAGS[args.scheme],
        *extra,
    ]
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "bilevel_reoptimization_v1",
        "task": args.task,
        "scheme": args.scheme,
        "structure": str(structure),
        "action_source": str(action_source),
        "command": command,
    }
    with (save_dir / "reoptimization_manifest.json").open(
        "w", encoding="utf-8"
    ) as fp:
        json.dump(manifest, fp, indent=2, sort_keys=True)
    print("[reoptimize] structure:", structure, flush=True)
    print("[reoptimize] action source:", action_source, flush=True)
    print("[reoptimize] output:", save_dir, flush=True)
    print("[reoptimize] command:", " ".join(command), flush=True)
    if args.dry_run:
        return 0
    return int(subprocess.run(command, cwd=str(REPO_ROOT)).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
