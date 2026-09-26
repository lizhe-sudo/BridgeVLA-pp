# Astra RLBench evaluator

`eval_astra.py` runs a closed loop absolute-pose policy through the existing
BridgeVLA++ RLBench action mode. The experiment is an **Astra-Direct-RGB**
policy: the model sees a language instruction, front/left-shoulder/
right-shoulder/wrist RGB, the current `Observation.gripper_pose`, and the
gripper-open flag. It does not see depth, point clouds, object poses, masks,
task internals, success-detector state, handles, or demonstration actions.
`reset_to_demo` still reads the selected demonstration to initialize an
episode. The model receives no demonstration actions or future trajectory.

This is a four-RGB-view plus robot-state direct-pose experiment. It does not
establish that Astra and point-cloud BridgeVLA receive identical information,
and its results must not be described as an input-matched comparison. Public
paper scores are separate from local results.

## Fixed experiment definition

The main task set is fixed and ordered:

1. `place_cups`
2. `place_shape_in_shape_sorter`
3. `put_groceries_in_cupboard`
4. `stack_blocks`
5. `stack_cups`

`meat_off_grill` is a separate sanity-check task and is never included in the
five-task macro average. A debug subset remains labeled `debug` and cannot
produce a full-suite macro average.

The policy emits one world-frame target per decision: XYZ in meters, absolute
unit quaternion XYZW, and gripper command (`0` close, `1` open). The reference
point is the RLBench arm tip used by the existing pose action mode. The action
mode moves the arm before applying the gripper command. Planner substeps do not
count as policy targets. The evaluator adds no action chunks, retries,
recovery actions, high-level skills, or cross-step memory. Simulation does not
advance while the policy is running.

The collision flag is `ignore_collisions`:

| Mode | Policy output | Applied flag | Comparison boundary |
| --- | --- | --- | --- |
| `fixed0` | 8 values | always `0` | Default; collision checking stays enabled. This is not the baseline's predictive collision action space. |
| `fixed1` | 8 values | always `1` | Planner is asked to ignore collisions. This is not the predictive action space. |
| `predict` | 9 values, including the flag | policy value `0` or `1` | Predictive flag is part of the policy action space. |

No mode changes automatically based on task, planner result, or model output.
Formal runs must state `--collision-mode` explicitly.

## Budgets, run modes, episodes, and repeats

Two explicit budget protocols are available:

- `uniform25`: 25 policy targets per selected task.
- `repo_step_limits`: reads `configs/eval_step_limit.yml` and saves the parsed
  table and final task budgets in `run_manifest.json`. At this revision,
  `place_cups` and `stack_blocks` resolve to 35; the other fixed main tasks
  resolve to 25.

`--max-waypoints` (also accepted as the old `--episode-length` spelling) is a
debug-only cap. It can lower a protocol budget but cannot raise it. There is no
implicit CLI-over-table precedence. Formal mode requires the entire ordered
five-task set, 25 episodes per task, 5 repeats, and no debug cap. Formal runs
are never started automatically.

Episode IDs can be selected with `--start-episode`, `--episode-ids`, or a JSON
file passed to `--episode-list` (`[0, 1, ...]` or `{"episode_ids": [...]}`).
The list length must match `--eval-episodes`. Repeat ID, demonstration episode
ID, and optional random seed are distinct fields. Repeats of the same episode
ID are repeated evaluations of the same demonstration scene, not new
independent scenes. Python and NumPy seeds are set when `--seed` is provided;
Astra does not set PyRep/CoppeliaSim's internal random seed and does not claim
full simulator or planner determinism.

## Model and policy identity

This stage's real Codex policy configuration is `gpt-6-luna` with reasoning
effort `max`; both remain configurable. `requested_model` records the request.
`resolved_model` is filled only from explicit reliable response metadata;
otherwise it remains null. CLI version and reasoning effort are recorded
separately. Mock and manual policies are labeled `mock` or `manual`, with
model/service fields null or not applicable.

The inspected Codex CLI is version `0.157.1` in the implementation environment.
It supports the model, reasoning, read-only sandbox, JSON event, schema, and
last-message options used here. Its help exposes no verified switch that
disables tools. The evaluator rejects tool-use events before accepting an
action, but this is post-hoc detection, not strong tool isolation. The CLI
uses a temporary working directory outside the repository and checks the
working directory and every ancestor for `AGENTS.md`, `AGENTS.override.md`, and
project `.codex/config.toml`; it refuses to call Codex if any are present. It
also passes the verified `--ignore-user-config` option and runs with a
read-only sandbox. The evaluator adds no service retries; lower-level retry
behavior is unknown. A CLI invocation is not assumed to equal one underlying
model request; that request count is recorded as unknown.

## Failure classes, denominators, and timing

Each episode records planned/attempted/evaluable status, successes, failure
class, termination reason, action budget, counters, timings, and recording
status. Policy/task failures include invalid policy actions, detected tool
use, planning/IK failures, environment terminal failure, and budget exhaustion.
Infrastructure errors include initialization/data errors, missing fresh
observations, and model service/process failures. Recording errors are tracked
separately and do not invalidate a saved task result. The implementation
distinguishes the configured Codex inference timeout from service/process
errors; ambiguous errors remain unknown rather than being attributed to robot
capability.

Success uses the existing sparse RLBench convention (`reward > 99`, verified
against `eval.py`). Success rates are fractions in `[0, 1]`. Task summaries
show planned, attempted, evaluable, and successful counts. Infrastructure
failures make a run or task incomplete; any temporary rate retains its
explicit evaluable denominator. A main-suite macro average is emitted only
when all five tasks are completely covered for that repeat. Repeat mean and
sample standard deviation are null when not computable. Sanity-task results
are reported separately.

Counters distinguish policy decisions, Codex CLI invocations, environment
target-action attempts, simulation steps/time, policy time, CLI time, action
execution time, and episode wall-clock time. The underlying model request
count and unavailable token fields remain null. Simulation time counts
successful `scene.step()` calls during action execution; video playback time
and video duration are separate. Episode wall-clock ends before video
encoding. Position and pose arrival checks are diagnostics only and never
replace RLBench task success.

## Dependencies and output layout

The standard entry point requires the same dedicated stacks selected by
`eval.sh`:

- `RLBENCH_SIM_STACK`, default
  `finetune/bridgevla/libs/RLBench_peract587`, containing `rlbench/`.
- `PYREP_SIM_STACK`, default
  `finetune/bridgevla/libs/PyRep_stepjam231`, containing a compiled
  `pyrep/backend/_sim_cffi*.so`.

Missing or uncompiled dedicated stacks fail explicitly. Shared dependencies
are allowed only with `--allow-shared-sim-stack`, which marks the run
nonstandard; formal mode disallows that option. If `rlbench` or `pyrep` has
already been imported from the wrong location, the evaluator asks for a fresh
process instead of changing paths and continuing.

Each run creates:

```text
<output-root>/<evaluation-id>/
  run_manifest.json       # resolved protocol, code/dependency/model identity
  run_summary.json        # episode/task/repeat aggregates
  run.jsonl               # step and summary events (unless --log-file overrides it)
  episodes/<episode-run>/
    episode_summary.json
    episode_log.jsonl
    run_meta.json
    steps/step_NNN/       # raw model RGB, policy/action metadata, execution record
    video/                 # composite video and intermediate frames when enabled
```

Run manifests avoid complete environment dumps and credentials. Episode results
are written before MP4 encoding. If MP4 setup, rendering, image I/O, or encoding
fails, `recording_error` is saved while the episode result remains available.
If a result file itself cannot be written, the run fails visibly.

## Commands

Run `eval_astra.sh` to activate `bridgevla_plus_rlbench` when needed and set the
same CoppeliaSim/Qt/Xvfb paths used by `eval.sh`. Leave the default dedicated
simulator paths in place for standard runs; the Python entry point verifies
them before importing RLBench or PyRep.

Debug mock run (one task, one demo, two targets):

```bash
cd finetune/RLBench
bash eval_astra.sh \
  --policy mock --run-mode debug --tasks stack_cups \
  --budget-protocol uniform25 --eval-episodes 1 --repeats 1 \
  --start-episode 0 --max-waypoints 2 --collision-mode fixed0 \
  --no-record-video --output-root outputs/astra_debug
```

`meat_off_grill` sanity check with the real model configuration (this command
uses development episode 0; once used, that scene is a development/test scene):

```bash
cd finetune/RLBench
bash eval_astra.sh \
  --policy codex --run-mode debug --tasks meat_off_grill \
  --budget-protocol uniform25 --eval-episodes 1 --repeats 1 \
  --start-episode 0 --max-waypoints 25 --collision-mode fixed0 \
  --codex-model gpt-6-luna --codex-reasoning max \
  --output-root outputs/astra_meat_off_grill_smoke
```

Formal five-task command (documented only; do not use it for a smoke test):

```bash
cd finetune/RLBench
bash eval_astra.sh \
  --policy codex --run-mode formal \
  --tasks place_cups place_shape_in_shape_sorter put_groceries_in_cupboard stack_blocks stack_cups \
  --budget-protocol repo_step_limits \
  --eval-episodes 25 --repeats 5 --start-episode 0 \
  --collision-mode fixed0 --codex-model gpt-6-luna --codex-reasoning max \
  --output-root outputs/astra_formal
```

The smoke command is one debug episode, not a success-rate estimate. The
evaluator does not automatically retry it or launch the formal batch.
