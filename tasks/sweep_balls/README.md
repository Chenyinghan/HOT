# SweepBalls

Canonical task identifier: `sweep_balls`. `config.json` specifies the search/evaluator settings, `reference.xml` the reference tool, `scene.xml` the task scene, and `objective.py` the numerical task. Physical gates live in `task_progress.py`.

The reference default uses 1800 simulation steps, 30 substeps per control and a 100-iteration action-only optimization budget.

Run Stage 1 and then Stage 2 from the repository root with the `hot` environment active. Start Stage 2 after Stage 1 succeeds:

```bash
python run_main.py --task sweep_balls --save-dir workspace/sweep_balls/stage1
python run_main.py --task sweep_balls --design-optim --initial-action workspace/sweep_balls/stage1 --save-dir workspace/sweep_balls/stage2
```

These commands run without a viewer and save each stage's parameters and diagnostics in its output directory. Check `task_success` in `diagnostics.json`. Stage 2 may retain the Stage 1 result without improvement. Use new output directories to preserve earlier runs, and update `--initial-action` to match.

Replay the saved Stage 1 and Stage 2 examples:

```bash
python replay_example.py --task sweep_balls --stage stage1
python replay_example.py --task sweep_balls --stage stage2
```

Stage 2 is the default. Replay opens an interactive viewer and writes no files. Add `--verify-only` for headless verification. These commands replay the bundled examples; fresh optimization outputs are stored separately under `workspace/`.
