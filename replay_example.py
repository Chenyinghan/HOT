import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "replay_examples"


def replay(args, manifest):
    os.environ["MESA_SHADER_CACHE_DISABLE"] = "true"
    for key in (
        "BILEVEL_NUMERIC_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"

    import numpy as np
    import run_main
    from bilevel.runner import SimRenderer

    task = manifest["tasks"][args.task]
    expected = task["stages"][args.stage]
    artifact = DATA / args.task / args.stage
    rtol, atol = manifest["rtol"], manifest["atol"]

    def evaluate(runner, params):
        objective, _ = runner.forward(params, backward_flag=False)
        actual_q = np.asarray(runner.sim.get_q()).copy()
        with np.load(artifact / "trajectory.npz", allow_pickle=False) as archive:
            expected_q = archive["q"][-1].copy()
        rms_percent = 0.0
        if runner.optimize_design:
            action, _ = runner.unpack_params(params)
            initial = runner.pack_params(action, runner.initial_morphology_parameters)
            deformation = runner._direct_planar_normalized_deformation(initial, params)
            rms_percent = 100.0 * deformation["normalized_rms"]
        checks = {
            "objective_matches": bool(np.isclose(
                objective, expected["objective"], rtol=rtol, atol=atol,
            )),
            "final_state_matches": bool(np.allclose(
                actual_q, expected_q, rtol=rtol, atol=atol,
            )),
            "rms_matches": bool(np.isclose(
                rms_percent, expected["rms_percent"], rtol=rtol, atol=atol,
            )),
            "task_success": bool(
                runner.task.rollout_diagnostics(runner, params).get("task_success", False)
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Replay verification failed: {checks}")
        print(f"{args.task} {args.stage}: PASS; objective={objective:.9f}; task_success=True", flush=True)
        if not args.verify_only:
            runner.forward(params, backward_flag=False)
            if runner.optimize_design:
                _, cage = runner.unpack_params(params)
                runner.apply_morphology(cage, generate_mesh=True)
            runner._apply_replay_camera()
            runner.sim.viewer_options.speed = args.speed
            if args.once:
                runner.sim.viewer_options.loop = False
                runner.sim.viewer_options.infinite = False
            SimRenderer.replay(runner.sim)
        return 0

    return run_main.main([
        "--task", args.task,
        "--model-xml", str(ROOT / "tasks" / args.task / "reference.xml"),
        "--load-dir", str(artifact),
        "--design-optim" if args.stage == "stage2" else "--no-design-optim",
        "--morphology-parameterization", "unified_connected_head_morphology",
        "--num-steps", str(task["num_steps"]),
        "--sub-steps", str(task["sub_steps"]),
        "--no-visualize" if args.verify_only else "--visualize",
        "--replay-speed", str(args.speed),
    ], replay_callback=evaluate)


def main(argv=None):
    manifest = json.loads((DATA / "manifest.json").read_text())
    parser = argparse.ArgumentParser(description="Replay saved examples with HOT; no optimization.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task", choices=list(manifest["tasks"]))
    selection.add_argument("--all", action="store_true", help="Verify all eight examples without a viewer")
    parser.add_argument("--stage", choices=["stage1", "stage2"], default="stage2")
    parser.add_argument("--verify-only", action="store_true", help="Evaluate without opening a viewer")
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument("--once", action="store_true", help="Play once instead of looping")
    args = parser.parse_args(argv)
    if not 0 < args.speed < float("inf"):
        parser.error("--speed must be finite and positive")
    if not (args.verify_only or args.all) and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        parser.error("No display available; use --verify-only or --all")
    if not args.all:
        return replay(args, manifest)

    failed = False
    for task in manifest["tasks"]:
        for stage in ["stage1", "stage2"]:
            result = subprocess.run([
                sys.executable, "-B", str(Path(__file__).resolve()),
                "--task", task, "--stage", stage, "--verify-only",
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if result.returncode:
                failed = True
                print(result.stdout, end="", flush=True)
            print(f"{task} {stage}: {'FAIL' if result.returncode else 'PASS'}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
