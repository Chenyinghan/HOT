# Structural search demo

This demo walks through Stage 1 of HOT (Hierarchical Optimization for Tool design): building a canonical structural DAG, calibrating BASS (Behavior-Aware Structural Search), and running guided search. Stage 1 evaluates complete structures through action optimization at fixed library shapes. Use the `--mode` switch in `search_demo/run.py` to begin with DAG building, physical calibration, or the supplied calibration artifacts.

Run commands from the repository root after following the [installation instructions](../README.md#install). DAG building uses the Python dependencies; calibration and search use the compiled DiffRedMax simulator. Generated files go under `workspace/search_demo/` by default. Give each run a fresh output directory.

The supplied DAG is optional (approximately 1.4 GiB) and is excluded from automatic Git LFS downloads by default. Saved-example replay, reference-tool optimization, and building your own DAG do not require this download. To calibrate or search with the supplied DAG, install Git LFS and run:

```bash
git lfs pull --include="search_demo/dags/demo_dag/**" --exclude=""
```

The empty `--exclude` overrides the repository's default exclusion for this command only. When updating an older checkout that does not yet contain `.lfsconfig`, use `GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin main` once to receive the new default without downloading DAG data.

## Start from scratch: build a DAG

```bash
python search_demo/run.py --mode build
```

The default build allows **three Head links and one functional group**, using all enabled searchable assets in `assets/library/catalog.json` plus the fixed Handle. It stores roll, pitch and yaw as root actions in the DAG. The Handle is counted internally, so a three-link build reports `max_links: 4` in its metadata. The output is `workspace/search_demo/dags/3_links_1_functions/`. This full build can take substantial time and disk space. The supplied `search_demo/dags/demo_dag/` is a ready three-link, single-functional-group DAG if you want to begin with calibration.

Change the three structural inputs as needed:

```bash
python search_demo/run.py --mode build --links 2 --functions 1 \
  --asset-library assets/library/catalog.json
```

Set `--links` and `--functions` to positive integers. `--asset-library` accepts a JSON or YAML catalog, or a text list of asset IDs, through the shared loader. A custom library needs geometry and contact resources for the physical evaluator and the `root/universal_handle` asset. Use `--dag PATH` to choose a fresh output directory. The three structural inputs determine the DAG geometry, so build a new DAG after changing one. The supplied SweepBalls and TorqueBolt task configurations calibrate single-functional-group DAGs.

## Calibrate manually

```bash
python search_demo/run.py --mode calibrate
```

This runs **9,800 Stage 1 physical evaluations** of SweepBalls on `search_demo/dags/demo_dag/`, then fits BASS milestone thresholds and continuation priors. Calibration samples distinct complete-structure and root-rotation choices. The command ends after fitting. Its default run directory is `workspace/search_demo/calibration/sweep_balls/`.

For the other supplied task, set the task flag with `--task torque_bolt`:

```bash
python search_demo/run.py --mode calibrate --task torque_bolt <other parameters>
```

You may check detailed parameter definitions with the following command:

```bash
python search_demo/run.py --help
```

Calibration needs enough valid stage progress to fit thresholds. If fitting fails with a small run, inspect its `evals.csv` and increase `--runs`. To use a DAG you built, pass `--dag PATH` and its `--asset-library PATH`; calibration reads the link limit and checks the DAG's grammar and assets before evaluation. The supplied DAG works with both `sweep_balls` and `torque_bolt`.

The run directory contains `config.json` and `command.json` for reproduction, `output/<task>/evals.csv`, and the fitted `output/<task>/calibration.json`. The calibration artifact includes the fitted values, candidate outcomes for replay, calibration seed, and DAG metadata hash. The evaluation CSV remains the run record. Use `--dry-run` to inspect the resolved config and command.

## Search with ready calibration

```bash
python search_demo/run.py --mode search
```

This starts SweepBalls BASS with two-query Bayesian lookahead (N=2) on the supplied DAG, using `search_demo/calibration/sweep_balls/calibration.json`. The artifact contains 9,800 candidate outcomes alongside the fitted calibration. These outcomes pass through the production warmup path: BASS fits milestone thresholds and continuation priors, then replays each completed probe into its posterior and search frontiers. Guided physical evaluations begin at number 9,801 for the packaged artifacts, or one after the chosen manual calibration count. Search defaults to `--evaluations -1`, with no cap on guided evaluations; it runs until stopped or the DAG is exhausted. A positive `--evaluations` value caps new guided candidate evaluations.

Choose a task, use a calibration artifact you produced, or set an evaluation limit:

```bash
python search_demo/run.py --mode search --task torque_bolt \
  --workers 4 --output workspace/search_demo/search/torque_bolt
python search_demo/run.py --mode search --task sweep_balls \
  --calibration-artifact workspace/search_demo/calibration/sweep_balls/output/sweep_balls/calibration.json \
  --output workspace/search_demo/search/sweep_manual
python search_demo/run.py --mode search --evaluations 200 --output workspace/search_demo/search/sweep_200
```

Search writes `config.json` and `command.json` under `workspace/search_demo/search/<task>/` by default. Evaluation records appear in `output/<task>/evals.csv`, with calibration rows labeled `calibration_replay`. A run with evaluated candidates also writes `best.xml` and `best_run.json` at the run root. For a calibration winner, `best.xml` describes the reconstructed structure and `best_run.json` records its saved score and task evidence. Read `task_success` for the recorded physical outcome. Guided evaluations save action artifacts for fresh replay.

To search a DAG you built, pass `--dag`, its `--calibration-artifact`, and the matching `--asset-library` when applicable. The runner checks task grammar, assets, artifact schema, candidate count and DAG hash. Exact candidate replay uses the calibration seed stored in the artifact. Use `--dry-run` to inspect the resolved search configuration.

The ready SweepBalls and TorqueBolt artifacts are derived from the production calibration runs. Their recorded calibration seeds are `2033923799` and `16001`, respectively.
