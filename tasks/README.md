# Universal task API

Every task package exports the same public class:

```python
from tasks import TaskBase, load_task
from tasks.torque_bolt.task import TaskDefinition

assert issubclass(TaskDefinition, TaskBase)
task = load_task("torque_bolt", config={"num_steps": 1200})
# Equivalent direct construction:
task = TaskDefinition(config={"num_steps": 1200})
```

The four registered names are `sweep_balls`, `torque_bolt`, `scoop_balls`, and `hammer_extract_nail`. Their class names and public method names are identical.

## Responsibilities

| Location | Responsibility |
|---|---|
| `base.py::TaskBase` | Shared configuration, lifecycle, scene binding, objective construction, initialization and bounds |
| `<task>/task.py::TaskDefinition` | Task-specific configuration/specification and optional validity, initialization or milestone overrides |
| `<task>/objective.py::TaskDynamics` | Task-specific physics, loss equations, gradients, staged rollout policy and success diagnostics |
| `objective.py::TaskObjective` | Shared objective interface over the numerical equations |
| `contexts.py::SceneBindings` | Common scene binding type, with task-instance ownership validation |
| `registry.py::TaskRegistry` | Task names mapped to module locations |
| `loader.py::load_task` | Import the registered module, check inheritance, and instantiate `TaskDefinition` |

`TaskDynamics` defines the numerical equations and uses `bilevel.runner.BaseTask` for shared RedMax hooks. It is available as `task.numerical_task`. Callers use `load_task` to construct the public `TaskDefinition`.

## Common lifecycle

`TaskBase.__init__(config=None, *, payload=None)` reads the derived class's `config_path`, merges explicit overrides into `task_config`, and calls `_configure()`. `TaskDefinition._configure()` creates `numerical_task` and the five typed specifications: `_task_spec`, `_search_spec`, `_handle_spec`, `_morphology_spec`, and `_runtime_spec`.

Every task exposes:

```python
task_spec()
search_spec()
handle_spec()
morphology_spec()
runtime_spec()
bind_scene(model)
create_objective(bindings)
seed_action(context)
action_bounds(*, ndof_u, num_ctrl_steps, ndof_cage, optimize_design)
initialize_morphology(model_path, simulation)
validate_candidate(candidate)
diagnostics(rollout)
bass_evaluation(*, score, run_result)
```

Most methods are inherited directly. Optional overrides use the same signatures. For example, ScoopBalls overrides `seed_action` to provide its neutral, seed-independent initial action. Candidate checks and milestone rewards use each task's physical gates. Scene bindings belong to the task instance that created them.

## Registering a task

Create `tasks/<name>/task.py` with a concrete `TaskDefinition(TaskBase)` and a `config_path` pointing to its `config.json`. Implement `_configure()` and any required physical overrides. Keep numerical equations in `objective.py::TaskDynamics`.

Register its module location:

```python
from tasks import register_task, load_task

register_task("new_task", "tasks.new_task.task")
task = load_task("new_task")
```

Built-in locations are declared in `registry.py`. The loader imports the registered module, checks its exported `TaskDefinition` subclass, and verifies its task name.

## Numerical and configuration contracts

See [specs.md](specs.md) for configuration fields and the numerical runner interface. Interface changes should preserve reference XMLs, scenes, losses, action initialization, physical success criteria, cage/LBS deformation, connectivity, contact consistency, selected root motion, calibration and evaluation provenance.

Validate a new task or interface change with common contract tests, short physical evaluator and replay checks, and a bounded BASS search. Final evaluation and reference replay use the full task horizon, including when optimization stages use causal prefixes.

## Scene resources

Keep task-owned scene geometry and contact samples in `scene/meshes/` and `scene/contacts/` inside the task package. Both `scene.xml` and `reference.xml` use task-relative paths to these files. Shared Handle and Head assets remain in `assets/library/`. XML generation rebases scene resource paths for its output location.
