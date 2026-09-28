# Canonical Task Contract

Every maintained task is an equal-status `TaskBase` plugin implemented in `tasks/<task>/task.py` and loaded by mission name through `tasks.load_task(...)`.

Each task package has the same layout:

```text
tasks/<task>/
├── __init__.py
├── config.json
├── objective.py
├── reference.xml
├── scene.xml
└── task.py
```

`config.json` defines the runtime configuration. `TaskBase.__init__` merges explicit overrides into `task_config`, constructs the numerical `TaskDynamics` from `objective.py`, and calls the task definition to construct the framework contracts.

## Shared `TaskBase` methods

```python
task_spec()
search_spec()
handle_spec()
morphology_spec()
runtime_spec()
bind_scene(model)
create_objective(bindings)
seed_action(context)
action_bounds(...)
initialize_morphology(model_path, simulation)
```

`validate_candidate(...)` and `diagnostics(...)` have shared defaults and may be specialized by a task.

The numerical object stored in `task.numerical_task` retains the RedMax-facing hooks used by `bilevel.runner.CoOptRunner`:

```python
num_steps()
sub_steps()
objective_weights()
init_task(sim)
init_action(ndof_u, num_ctrl_steps, seed)
init_design(model_path, sim)
bounds(ndof_u, num_ctrl_steps, ndof_cage, optimize_design)
compute_terms(...)
write_terminal_grads(...)
```

Numerical tasks inherit the causal-rollout hook `optimization_rollout_control_steps(num_ctrl_steps)` from `bilevel.runner.BaseTask`. It defaults to the complete horizon. A staged task may use a shorter prefix that contains every active loss and gate. Final evaluation and replay use the complete horizon.

`objective.py` defines task formulas, action seeds, loss weights, analytic gradients, scene bindings and diagnostics. The runtime constructs and dispatches the task.

## Runtime flow

```text
config.json
  -> tasks.load_task(mission_name)
  -> TaskBase
  -> bilevel.runtime
  -> bilevel.lower.engine
  -> numerical TaskDynamics + MountPreservingCoOptRunner + RedMax
```

BASS, isolated lower-level subprocesses, standalone optimization, and replay all use this same flow. The subprocess request identifies a task by `task_name` and records its canonical `task_module` for provenance.

## Required configuration fields

- `mission_name`
- `functions`
- `assets_json`
- `scene_xml`
- `root_asset_id`
- `root_blocked_face`
- `function_count`
- `task_config`
- `bass`
- `xml`

Each `functions` entry must contain `function_name`; generated XML names the corresponding marker `{function_name}_endeffector`.

All maintained tasks use `assets/library/catalog.json`, `root/universal_handle`, and the `unified_connected_head_morphology` parameterization.

## Adding a task

1. Create the package with the standard file layout.
2. Export `TaskDefinition(TaskBase)` and implement its `_configure()` hook.
3. Keep numerical formulas in `objective.py`.
4. Add the mission-to-module mapping in `tasks/registry.py`.
5. Add contract, golden numerical, short RedMax, BASS, and replay tests.
